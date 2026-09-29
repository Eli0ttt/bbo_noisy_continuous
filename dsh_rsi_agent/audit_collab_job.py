#!/usr/bin/env python3
"""Build a compact tool-call audit from Harbor-collected DSH zstd traces.

Run this on the Harbor host after a job completes. The public task image does
not include the `zstd` executable, so trace decoding is intentionally outside
the task container and never affects solver evaluation.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def decode(path: Path) -> str:
    proc = subprocess.run(["zstd", "-dc", str(path)], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError((proc.stderr or "zstd decode failed").strip())
    return proc.stdout


def read_trace(path: Path, root: Path) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    tool_counts: collections.Counter[str] = collections.Counter()
    step_counts: collections.Counter[str] = collections.Counter()
    errors = 0
    pending: dict[str, dict[str, Any]] = {}
    calls_by_seq: dict[int, dict[str, Any]] = {}
    for line in decode(path).splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        kind = event.get("type")
        if kind == "tool/call":
            name = str(data.get("name", "unknown"))
            call_id = str(data.get("callId", ""))
            turn, step = data.get("turn"), data.get("step")
            item = {
                "ordinal": len(calls) + 1,
                "turn": turn,
                "step": step,
                "tool": name,
                "call_id": call_id or None,
                "event_seq": event.get("seq"),
                "event_time_ms": event.get("time"),
            }
            calls.append(item)
            if call_id:
                pending[call_id] = item
            if isinstance(event.get("seq"), int):
                calls_by_seq[event["seq"]] = item
            tool_counts[name] += 1
            step_counts[f"turn-{turn}-step-{step}"] += 1
        elif kind == "tool/result":
            msg = data.get("message") if isinstance(data.get("message"), dict) else {}
            source = msg.get("source") if isinstance(msg.get("source"), dict) else {}
            call_id = str(source.get("callId", ""))
            if not call_id:
                for item in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
                    if isinstance(item, dict) and item.get("toolCallId"):
                        call_id = str(item["toolCallId"])
                        break
            call_item = pending.get(call_id) if call_id else None
            if call_item is None:
                for source_seq in event.get("sourceEventSeqs", []) if isinstance(event.get("sourceEventSeqs"), list) else []:
                    if source_seq in calls_by_seq:
                        call_item = calls_by_seq[source_seq]
                        break
            error = data.get("error")
            if error:
                errors += 1
            if call_item is not None:
                call_item["result_error"] = error or None
                call_item["result_event_seq"] = event.get("seq")
                call_item["result_time_ms"] = event.get("time")
                if isinstance(call_item.get("event_time_ms"), (int, float)) and isinstance(event.get("time"), (int, float)):
                    call_item["duration_ms"] = max(0, event["time"] - call_item["event_time_ms"])
    trace_name = path.name.removesuffix(".jsonl.zstd")
    m = re.fullmatch(r"(v\d+)-(primary|reviewer(?:-[A-Z])?(?:-repair)?)", trace_name)
    return {
        "trace": str(path.relative_to(root)),
        "version": m.group(1) if m else None,
        "role": m.group(2) if m else None,
        "call_count": len(calls),
        "tool_call_errors": errors,
        "tool_counts": dict(sorted(tool_counts.items())),
        "calls_by_turn_step": dict(sorted(step_counts.items())),
        "tool_sequence": [c["tool"] for c in calls],
    }


def audit(job: Path) -> Path:
    trace_dirs = list(job.glob("*/artifacts/logs/artifacts/bbo-collab/traces"))
    traces: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for directory in sorted(trace_dirs):
        for path in sorted(directory.glob("*.jsonl.zstd")):
            try:
                traces[path.name.removesuffix(".jsonl.zstd")] = read_trace(path, job)
            except Exception as exc:
                failures[path.name] = f"{type(exc).__name__}: {exc}"
    if not traces and failures:
        raise RuntimeError("all traces failed to decode: " + json.dumps(failures))
    tool_totals: collections.Counter[str] = collections.Counter()
    role_totals: collections.Counter[str] = collections.Counter()
    rounds: dict[str, dict[str, Any]] = {}
    for _,item in sorted(traces.items()):
        tool_totals.update(item["tool_counts"])
        role=item.get("role") or "unknown"
        role_totals[role]+=item["call_count"]
        version=item.get("version") or "unknown"
        rounds.setdefault(version,{})[role]={
            "trace":item["trace"],
            "tool_call_count":item["call_count"],
            "tool_call_errors":item["tool_call_errors"],
            "tool_counts":item["tool_counts"],
            "tool_sequence":item["tool_sequence"],
        }

    expected=set()
    for history_path in job.rglob("history.json"):
        try: rows=json.loads(history_path.read_text(encoding="utf-8"))
        except Exception: continue
        if not isinstance(rows,list): continue
        for row in rows:
            if not isinstance(row,dict): continue
            version=row.get("version")
            if not version: continue
            if row.get("primary_trace"): expected.add(f"{version}-primary")
            rt=row.get("reviewer_traces")
            if isinstance(rt,dict):
                expected.update(f"{version}-reviewer-{rid}" for rid,tr in rt.items() if tr)
    missing_expected=sorted(expected-set(traces))

    harbor_exceptions=[]
    for result_path in job.glob("*/result.json"):
        try: result=json.loads(result_path.read_text(encoding="utf-8"))
        except Exception: continue
        info=result.get("exception_info") if isinstance(result,dict) else None
        if isinstance(info,dict):
            harbor_exceptions.append({"trial":result.get("trial_name"),"type":info.get("exception_type"),"message":info.get("exception_message")})

    collaboration={"rounds":{},"memo_delivered":0,"memo_missing":0,"reviewer_session_ok":0,"reviewer_session_failed":0,
                   "neutral_rounds":0,"exact_duplicate_reuses":0,"invalid_proposals":0,
                   "parent_mutations_detected":0,"integrity_warnings":0,"integrity_detector_failures":0}
    history_rows=[]
    for history_path in job.rglob("history.json"):
        try:
            candidate=json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(candidate,list) and len(candidate)>len(history_rows):
            history_rows=candidate
    for row in history_rows:
        if not isinstance(row,dict) or not row.get("version"):
            continue
        v=row["version"]
        memo=row.get("reviewer_memo_state") or {}
        sessions=row.get("reviewer_session_ok") or {}
        delivered=sum(1 for x in memo.values() if x=="delivered")
        missing=sum(1 for x in memo.values() if x!="delivered")
        ok=sum(1 for x in sessions.values() if x is True)
        failed=sum(1 for x in sessions.values() if x is False)
        collaboration["memo_delivered"]+=delivered
        collaboration["memo_missing"]+=missing
        collaboration["reviewer_session_ok"]+=ok
        collaboration["reviewer_session_failed"]+=failed
        collaboration["neutral_rounds"]+=int(row.get("status")=="neutral")
        collaboration["exact_duplicate_reuses"]+=int(bool(row.get("selfcheck_reused_from_exact_duplicate")))
        proposal=row.get("proposal") or {}
        collaboration["invalid_proposals"]+=int(not proposal.get("valid",False))
        collaboration["parent_mutations_detected"]+=int(bool(row.get("parent_mutation_detected")))
        integ=row.get("experiment_integrity") or {}
        collaboration["integrity_warnings"]+=int(integ.get("warning") not in (None,"clean_single_mechanism_candidate"))
        collaboration["integrity_detector_failures"]+=int(integ.get("warning")=="detector_failed")
        collaboration["rounds"][v]={
            "status":row.get("status"),
            "proposal_valid":proposal.get("valid"),
            "mechanism_id":proposal.get("mechanism_id"),
            "score":row.get("score"),
            "score_delta":row.get("score_delta"),
            "family_id":proposal.get("family_id"),
            "integrity_warning":integ.get("warning"),
            "changed_symbols":integ.get("changed_symbols"),
            "parent_mutation_detected":row.get("parent_mutation_detected",False),
            "memo_state":memo,
            "reviewer_session_ok":sessions,
            "selfcheck_reused_from_exact_duplicate":row.get("selfcheck_reused_from_exact_duplicate",False),
        }

    output={
        "job":job.name,
        "trace_collection_status":"ok" if trace_dirs else "no_trace_directory_agent_did_not_export_traces",
        "harbor_exceptions":harbor_exceptions,
        "trace_count":len(traces),
        "tool_call_count":sum(role_totals.values()),
        "tool_counts":dict(sorted(tool_totals.items())),
        "role_call_counts":dict(sorted(role_totals.items())),
        "collaboration":collaboration,
        "rounds":rounds,
        "expected_trace_count_from_history":len(expected),
        "missing_expected_traces":missing_expected,
        "trace_decode_errors":failures,
    }
    agent_dirs = list(job.glob("*/agent"))
    destination = (agent_dirs[0] if agent_dirs else job) / "audit.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"TOOL_AUDIT_WRITTEN={destination}")
    print(f"TRACE_COLLECTION_STATUS={output['trace_collection_status']}")
    print(f"TRACE_COUNT={len(traces)} TOOL_CALL_COUNT={output['tool_call_count']}")
    print(f"MISSING_EXPECTED_TRACE_COUNT={len(missing_expected)}")
    print("TOOLS=" + json.dumps(output["tool_counts"], ensure_ascii=False, sort_keys=True))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    if not args.job.is_dir():
        parser.error(f"job directory does not exist: {args.job}")
    audit(args.job)
