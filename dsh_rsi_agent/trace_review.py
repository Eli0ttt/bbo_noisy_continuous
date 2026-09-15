from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

METRIC_NAME = "oracle_normalized_auc70_final30"
VERSION_HEADING_RE = re.compile(r"^##\s+(v\d+)\s*(?:-\s*(.*))?$", re.MULTILINE)
SCORE_RE = re.compile(
    r"^-\s*Score:\s*([0-9.+-eE]+)(?:\s*\(anytime=([0-9.+-eE]+),\s*final=([0-9.+-eE]+)\))?",
    re.MULTILINE,
)
PARENT_RE = re.compile(r"^-\s*Parent:\s*(\S+)", re.MULTILINE)
STATUS_RE = re.compile(r"^-\s*Status:\s*(.+)$", re.MULTILINE)


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_tool_result(d: dict[str, Any]) -> tuple[str, bool]:
    message = d.get("message") if isinstance(d.get("message"), dict) else d
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return "", False
    texts: list[str] = []
    is_error = False
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "tool-result":
            continue
        is_error = is_error or bool(item.get("isError"))
        nested = item.get("content")
        if isinstance(nested, list):
            for part in nested:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
        data = item.get("data")
        if isinstance(data, dict) and isinstance(data.get("text"), str):
            texts.append(data["text"])
    return "\n".join(texts), is_error


def parse_trace(trace_path: Path) -> dict[str, Any]:
    calls: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    tool_counts: collections.Counter[str] = collections.Counter()

    proc = subprocess.Popen(
        ["zstdcat", str(trace_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    first_time_ms: int | None = None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        typ = obj.get("type")
        d = obj.get("data")
        if not isinstance(d, dict):
            continue
        if typ == "tool/call":
            name = str(d.get("name", ""))
            call_id = str(d.get("callId", ""))
            raw_args = d.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                args = {"_raw": raw_args}
            if not isinstance(args, dict):
                args = {"_raw": args}
            time_ms = obj.get("time")
            if isinstance(time_ms, int) and first_time_ms is None:
                first_time_ms = time_ms
            calls[call_id] = {
                "seq": obj.get("seq"),
                "time_ms": time_ms,
                "turn": d.get("turn"),
                "step": d.get("step"),
                "name": name,
                "args": args,
            }
            ordered_ids.append(call_id)
            tool_counts[name] += 1
        elif typ == "tool/result":
            message = d.get("message") if isinstance(d.get("message"), dict) else d
            source = message.get("source") if isinstance(message, dict) else None
            if not isinstance(source, dict):
                continue
            call_id = str(source.get("callId", ""))
            text, is_error = extract_tool_result(d)
            results[call_id] = {"text": text, "is_error": is_error}

    stderr = proc.stderr.read() if proc.stderr is not None else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"zstdcat failed with code {rc}: {stderr}")

    selfcheck_tool_calls = 0
    selfcheck_failed_tool_calls = 0
    scored_selfcheck_executions = 0
    selfcheck_records: list[dict[str, Any]] = []
    cordis_calls = 0
    sandbox_backend_failures = 0
    action_rows: list[dict[str, Any]] = []

    for index, call_id in enumerate(ordered_ids, start=1):
        call = calls[call_id]
        result = results.get(call_id, {"text": "", "is_error": False})
        name = call["name"]
        args = call["args"]
        result_text = str(result.get("text", "") or "")
        rendered_args = json.dumps(args, sort_keys=True, ensure_ascii=False)

        if (
            "no sandbox backend is usable" in result_text
            or 'sandbox escalation to "danger-full-access" requires approval' in result_text
        ):
            sandbox_backend_failures += 1
        if "cordis" in name.lower() or "cordis" in rendered_args.lower():
            cordis_calls += 1

        is_selfcheck = name == "bash" and "selfcheck" in rendered_args
        score_payloads: list[dict[str, Any]] = []
        if is_selfcheck:
            selfcheck_tool_calls += 1
            shell_failed = bool(result.get("is_error")) or bool(
                re.search(r"\[exit code:\s*[1-9][0-9]*\]", result_text)
            )
            if shell_failed:
                selfcheck_failed_tool_calls += 1
            for raw_line in result_text.splitlines():
                raw_line = raw_line.strip()
                if not raw_line.startswith("{"):
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and payload.get("metric") == METRIC_NAME:
                    score_payloads.append(payload)
            scored_selfcheck_executions += len(score_payloads)
            selfcheck_records.append(
                {
                    "call_id": call_id,
                    "turn": call.get("turn"),
                    "step": call.get("step"),
                    "description": args.get("description"),
                    "command": args.get("command"),
                    "tool_error": shell_failed,
                    "scores": score_payloads,
                }
            )

        action_rows.append(
            {
                "index": index,
                "call_id": call_id,
                "turn": call.get("turn"),
                "step": call.get("step"),
                "elapsed_sec": (
                    round((call["time_ms"] - first_time_ms) / 1000.0, 3)
                    if isinstance(call.get("time_ms"), int) and first_time_ms is not None
                    else None
                ),
                "name": name,
                "args": args,
                "is_error": (
                    bool(result.get("is_error"))
                    or bool(re.search(r"\[exit code:\s*[1-9][0-9]*\]", result_text))
                ),
                "selfcheck_scores": score_payloads,
            }
        )

    return {
        "trace_path": str(trace_path),
        "tool_counts": dict(sorted(tool_counts.items())),
        "selfcheck_tool_calls": selfcheck_tool_calls,
        "scored_selfcheck_executions": scored_selfcheck_executions,
        "failed_selfcheck_tool_calls": selfcheck_failed_tool_calls,
        "observed_selfcheck_executions": scored_selfcheck_executions + selfcheck_failed_tool_calls,
        "cordis_related_calls": cordis_calls,
        "sandbox_backend_failures": sandbox_backend_failures,
        "selfchecks": selfcheck_records,
        "actions": action_rows,
    }


def parse_experiment_log(path: Path, versions_dir: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(VERSION_HEADING_RE.finditer(text))
    out: list[dict[str, Any]] = []
    for i, match in enumerate(matches):
        version = match.group(1)
        title = (match.group(2) or "").strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end]
        score_match = SCORE_RE.search(section)
        parent_match = PARENT_RE.search(section)
        status_match = STATUS_RE.search(section)
        out.append(
            {
                "version": version,
                "title": title,
                "parent": parent_match.group(1) if parent_match else None,
                "score": safe_float(score_match.group(1)) if score_match else None,
                "score_anytime": safe_float(score_match.group(2)) if score_match and score_match.group(2) else None,
                "score_final": safe_float(score_match.group(3)) if score_match and score_match.group(3) else None,
                "status": status_match.group(1).strip() if status_match else None,
                "snapshot_exists": (versions_dir / version).is_dir(),
            }
        )
    return out


def concise_action(row: dict[str, Any]) -> str:
    name = row["name"]
    args = row["args"]
    if name == "read":
        return f"read `{args.get('file_path', '?')}`"
    if name == "write":
        content = args.get("content")
        n_bytes = len(content.encode("utf-8")) if isinstance(content, str) else None
        suffix = f" ({n_bytes} bytes)" if n_bytes is not None else ""
        return f"write `{args.get('file_path', '?')}`{suffix}"
    if name == "edit":
        return f"edit `{args.get('file_path', '?')}`"
    if name == "glob":
        return f"glob `{args.get('pattern', '?')}`"
    if name == "grep":
        pattern = args.get("pattern", "?")
        include = args.get("include")
        return f"grep `{pattern}`" + (f" in `{include}`" if include else "")
    if name == "bash":
        description = args.get("description")
        command = str(args.get("command", "")).replace("\n", " ; ")
        if len(command) > 180:
            command = command[:177] + "..."
        prefix = f"{description}: " if description else ""
        return f"bash {prefix}`{command}`"
    raw = json.dumps(args, ensure_ascii=False, sort_keys=True)
    if len(raw) > 180:
        raw = raw[:177] + "..."
    return f"{name} `{raw}`"


def write_agent_actions(path: Path, trace: dict[str, Any]) -> None:
    lines = [
        "# Agent Action Timeline",
        "",
        "This is a human-readable projection of the structured DSH tool trajectory. "
        "It records observable tool actions/results; it does not reconstruct private chain-of-thought.",
        "",
        "| # | t+sec | turn/step | action | result |",
        "|---:|---:|---|---|---|",
    ]
    for row in trace["actions"]:
        elapsed = "" if row["elapsed_sec"] is None else f"{row['elapsed_sec']:.3f}"
        turn_step = f"{row.get('turn', '')}/{row.get('step', '')}"
        action = concise_action(row).replace("|", "\\|")
        if row["selfcheck_scores"]:
            scores = ", ".join(
                f"score={float(x.get('score', 0.0)):.6f}"
                for x in row["selfcheck_scores"]
            )
            result = ("ERROR; " if row["is_error"] else "") + scores
        else:
            result = "ERROR" if row["is_error"] else "ok"
        lines.append(
            f"| {row['index']} | {elapsed} | {turn_step} | {action} | {result} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_version_history_csv(path: Path, versions: list[dict[str, Any]]) -> None:
    fields = [
        "version",
        "title",
        "parent",
        "score",
        "score_anytime",
        "score_final",
        "status",
        "snapshot_exists",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in versions:
            writer.writerow({k: row.get(k) for k in fields})


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_if_nonempty(src: Path, dst: Path) -> bool:
    if not src.is_file() or src.stat().st_size == 0:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--run-number", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    trial = Path(args.trial).resolve()
    trace_path = Path(args.trace).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise SystemExit(f"review directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)

    experiment_log = trial / "artifacts/app/methods/experiment_log.md"
    versions_dir = trial / "artifacts/app/methods/versions"
    final_solver = trial / "artifacts/app/methods/main/solver.py"
    bookkeeping_path = trial / "agent/autoresearch-audit.json"
    score_details_path = trial / "verifier/score_details.json"
    grade_debug_path = trial / "verifier/grade_debug.json"
    result_path = trial / "result.json"

    trace = parse_trace(trace_path)
    versions = parse_experiment_log(experiment_log, versions_dir)
    bookkeeping = load_json(bookkeeping_path, {}) or {}
    score_details = load_json(score_details_path, {}) or {}
    grade_debug = load_json(grade_debug_path, {}) or {}
    trial_result = load_json(result_path, {}) or {}

    runtime = grade_debug.get("runtime") if isinstance(grade_debug.get("runtime"), dict) else {}
    elapsed = safe_float(runtime.get("elapsed_sec"))
    budget = safe_float(runtime.get("time_budget_sec"))
    margin = (budget - elapsed) if elapsed is not None and budget is not None else None
    utilization = (elapsed / budget) if elapsed is not None and budget not in (None, 0.0) else None
    truncated = bool(runtime.get("truncated"))
    partial_runs = int(runtime.get("partial_runs") or 0)
    floor_filled_runs = int(runtime.get("floor_filled_runs") or 0)
    if truncated or partial_runs or floor_filled_runs:
        runtime_status = "FAIL"
    elif utilization is not None and (utilization >= 0.85 or (margin is not None and margin < 20.0)):
        runtime_status = "WARN"
    else:
        runtime_status = "PASS"

    scorer = grade_debug.get("scorer") if isinstance(grade_debug.get("scorer"), dict) else {}
    reward = scorer.get("score")
    if reward is None:
        rewards = (
            trial_result.get("verifier_result", {}).get("rewards", {})
            if isinstance(trial_result, dict)
            else {}
        )
        reward = rewards.get("reward") if isinstance(rewards, dict) else None

    candidate_versions = [x for x in versions if x["version"] != "v0"]
    scored_versions = [x for x in candidate_versions if x.get("score") is not None]
    best_visible = max(scored_versions, key=lambda x: x["score"]) if scored_versions else None

    summary = {
        "job_name": args.job_name,
        "condition": args.condition,
        "run_number": int(args.run_number),
        "reward": reward,
        "raw_score": scorer.get("raw_score"),
        "score_anytime": scorer.get("score_anytime"),
        "score_final": scorer.get("score_final"),
        "kpi": scorer.get("kpi"),
        "correctness": grade_debug.get("correctness"),
        "num_evals": grade_debug.get("num_evals"),
        "trace_shape": grade_debug.get("trace_shape"),
        "runtime": {
            "status": runtime_status,
            "elapsed_sec": elapsed,
            "time_budget_sec": budget,
            "margin_sec": margin,
            "utilization": utilization,
            "completed_runs": runtime.get("completed_runs"),
            "partial_runs": partial_runs,
            "floor_filled_runs": floor_filled_runs,
            "truncated": truncated,
            "timeout_reason": runtime.get("timeout_reason"),
        },
        "research": {
            "tool_counts": trace["tool_counts"],
            "selfcheck_tool_calls": trace["selfcheck_tool_calls"],
            "scored_selfcheck_executions": trace["scored_selfcheck_executions"],
            "failed_selfcheck_tool_calls": trace["failed_selfcheck_tool_calls"],
            "observed_selfcheck_executions": trace["observed_selfcheck_executions"],
            "cordis_related_calls": trace["cordis_related_calls"],
            "sandbox_backend_failures": trace["sandbox_backend_failures"],
            "logged_version_count": len(candidate_versions),
            "snapshot_version_count": bookkeeping.get("snapshot_version_count", bookkeeping.get("version_count")),
            "missing_snapshot_versions": bookkeeping.get("missing_snapshot_versions", []),
            "noncanonical_snapshot_dirs": bookkeeping.get("noncanonical_snapshot_dirs", []),
            "snapshot_complete": bookkeeping.get("snapshot_complete"),
            "best_visible_version": best_visible,
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "trace-audit.json").write_text(
        json.dumps({k: v for k, v in trace.items() if k != "actions"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_agent_actions(out_dir / "agent-actions.md", trace)
    write_version_history_csv(out_dir / "version-history.csv", versions)

    runtime_audit = summary["runtime"]
    (out_dir / "runtime-audit.json").write_text(
        json.dumps(runtime_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "bookkeeping-audit.json").write_text(
        json.dumps(bookkeeping, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary_md = [
        f"# Review — {args.job_name}",
        "",
        "## Formal score",
        "",
        f"- reward: `{reward}`",
        f"- raw_score: `{scorer.get('raw_score')}`",
        f"- score_anytime: `{scorer.get('score_anytime')}`",
        f"- score_final: `{scorer.get('score_final')}`",
        f"- KPI: `{scorer.get('kpi')}`",
        f"- correctness: `{grade_debug.get('correctness')}`",
        f"- num_evals: `{grade_debug.get('num_evals')}`",
        f"- trace_shape: `{grade_debug.get('trace_shape')}`",
        "",
        "## Runtime",
        "",
        f"- status: **{runtime_status}**",
        f"- elapsed_sec: `{elapsed}`",
        f"- budget_sec: `{budget}`",
        f"- margin_sec: `{margin}`",
        f"- utilization: `{None if utilization is None else round(utilization * 100.0, 2)}%`",
        f"- truncated: `{truncated}`",
        f"- partial_runs: `{partial_runs}`",
        f"- floor_filled_runs: `{floor_filled_runs}`",
        "",
        "## Agent research",
        "",
        f"- tool counts: `{trace['tool_counts']}`",
        f"- selfcheck tool calls: `{trace['selfcheck_tool_calls']}`",
        f"- scored selfcheck executions: `{trace['scored_selfcheck_executions']}`",
        f"- failed selfcheck tool calls: `{trace['failed_selfcheck_tool_calls']}`",
        f"- observed selfcheck executions: `{trace['observed_selfcheck_executions']}`",
        f"- Cordis-related calls: `{trace['cordis_related_calls']}`",
        f"- sandbox backend failures: `{trace['sandbox_backend_failures']}`",
        f"- logged candidate versions: `{len(candidate_versions)}`",
        f"- snapshot directories: `{bookkeeping.get('snapshot_version_count', bookkeeping.get('version_count'))}`",
        f"- snapshot complete: `{bookkeeping.get('snapshot_complete')}`",
        f"- missing snapshots: `{bookkeeping.get('missing_snapshot_versions', [])}`",
        f"- noncanonical snapshot dirs: `{bookkeeping.get('noncanonical_snapshot_dirs', [])}`",
        "",
        "See `agent-actions.md` for the observable action path and `version-history.csv` for the logged version history.",
    ]
    if best_visible:
        summary_md.extend(
            [
                "",
                "## Best logged visible version",
                "",
                f"- version: `{best_visible.get('version')}`",
                f"- title: `{best_visible.get('title')}`",
                f"- score: `{best_visible.get('score')}`",
                f"- status: `{best_visible.get('status')}`",
            ]
        )
    (out_dir / "README.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    # Human-review essentials. Keep the raw Harbor tree untouched; review/ is the
    # curated surface. Large trace uses a hard link when possible to avoid extra disk.
    hardlink_or_copy(trace_path, out_dir / "agent-trace.jsonl.zstd")
    copy_if_nonempty(experiment_log, out_dir / "experiment_log.md")
    copy_if_nonempty(final_solver, out_dir / "final_solver.py")
    copy_if_nonempty(score_details_path, out_dir / "verifier-score-details.json")
    copy_if_nonempty(trial / "formal_protocol.txt", out_dir / "meta/formal_protocol.txt")
    copy_if_nonempty(trial / "agent/effective-config.yml", out_dir / "meta/effective-config.yml")
    copy_if_nonempty(trial / "agent/prompt-source.txt", out_dir / "meta/prompt-source.txt")
    copy_if_nonempty(trial / "agent/dsh.stdout.log", out_dir / "meta/dsh.stdout.log")
    copy_if_nonempty(trial / "agent/dsh.stderr.log", out_dir / "meta/dsh.stderr.log")

    if versions_dir.is_dir():
        for version_dir in sorted((p for p in versions_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            solver = version_dir / "solver.py"
            if solver.is_file():
                copy_if_nonempty(solver, out_dir / "versions" / version_dir.name / "solver.py")

    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
