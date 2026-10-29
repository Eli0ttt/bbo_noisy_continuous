#!/usr/bin/env python3
"""BBO evaluator adapter using EvoMAS's original evolutionary operators."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import yaml

ADAPTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ADAPTER_DIR))

from native_bbo_runtime import (CandidateExtractionError, MASExecutionFailure, canonical_model_id,
                                configure_allowed_models, execute as execute_mas,
                                parse_model_pool, validate as validate_mas)


def validated_config(config_text, tool_mode):
    """Reject malformed evolved YAML before spending agent API calls."""
    from src.mas.spec import MasSpec
    config = yaml.safe_load(config_text)
    if not isinstance(config, dict):
        raise ValueError("MAS YAML must be a mapping")
    validate_mas(MasSpec(**config), tool_mode)
    return config_text


def ensure_session_history_tool(config_text):
    """Keep public within-run history accessible while other tools can evolve."""
    config = yaml.safe_load(config_text)
    if not isinstance(config, dict) or not isinstance(config.get("agents"), dict):
        return config_text
    changed = False
    for agent in config["agents"].values():
        if isinstance(agent, dict):
            assigned = agent.get("tools")
            if assigned is None:
                agent["tools"] = ["bbo_read_history"]
                changed = True
                continue
            if isinstance(assigned, list) and "bbo_read_history" not in assigned:
                assigned.append("bbo_read_history")
                changed = True
    return yaml.safe_dump(config, sort_keys=False) if changed else config_text


class DryRunMetaModel:
    """Deterministic no-API stand-in used only for pipeline wiring tests."""

    memory_evolution = False

    def select(self, task_query, task_description, pool_dir, k=2, mas_index=None):
        return [str(p) for p in sorted(Path(pool_dir).glob("*.yaml"))[:k]]

    def generate(self, mas_config_path, task_samples, task_description, model_list=None):
        return Path(mas_config_path).read_text(encoding="utf-8")

    def mutate(self, mas_config, execution_logs, observations, model_list=None):
        config = yaml.safe_load(mas_config)
        config.setdefault("meta", {})["fake_evolution_step"] = True
        return yaml.safe_dump(config, sort_keys=False)

    def crossover(self, mas_config_1, mas_config_2, logs_1, logs_2, model_list=None):
        return self.mutate(mas_config_1, logs_1, "dry-run crossover")

    def update_memory(self, **kwargs):
        return None


def restore_basic_seed_tools(config_text: str) -> str:
    """Give every seed agent the same generation tools; EvoMAS may later evolve them."""
    config = yaml.safe_load(config_text)
    agents = config["agents"]
    reports = config["topology"]["reports_to"]
    if any(a not in agents or any(target not in agents for target in targets)
           for a, targets in reports.items()):
        raise ValueError("Generated topology references unknown agents")
    terminal = {aid for aid in agents if not reports.get(aid)}
    if len(terminal) != 1:
        raise ValueError("Basic tool seed requires one terminal candidate agent")
    initial_tools = ["bbo_read_context", "bbo_read_public_data", "bbo_read_prior_feedback",
                     "bbo_read_prior_trace", "bbo_read_history", "bbo_write_candidate",
                     "bbo_preflight_candidate", "bbo_probe_candidate"]
    for agent in agents.values():
        agent["tools"] = list(initial_tools)
    return yaml.safe_dump(config, sort_keys=False)


def should_restore_basic_tools(seed_policy, original_seed_names, parent_path):
    """Never overwrite tools on an evolved configuration loaded from a prior pool."""
    return seed_policy == "basic" and Path(parent_path).name in original_seed_names


def normalize_basic_pool_seeds(working_pool, seed_policy, original_seed_names):
    """Normalize only copied original seeds before selection and generation."""
    if seed_policy != "basic":
        return
    for name in original_seed_names:
        path = working_pool / name
        if path.is_file():
            path.write_text(restore_basic_seed_tools(path.read_text(encoding="utf-8")),
                            encoding="utf-8")


def retarget_unavailable_models(working_pool, model_pool):
    """Retain valid MAS topology/tools when a past model is unavailable today."""
    changed = []
    for path in sorted(working_pool.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue  # sanitize_working_pool will quarantine this file.
        if not isinstance(data, dict) or not isinstance(data.get("agents"), dict):
            continue
        replaced = {}
        for aid, agent in data["agents"].items():
            if isinstance(agent, dict) and agent.get("model_id") not in model_pool:
                replaced[aid] = agent.get("model_id")
                agent["model_id"] = model_pool[0]
        if replaced:
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            changed.append({"config": path.name, "from": replaced,
                            "to": model_pool[0]})
    return changed


def metrics_from_bbo(result, runtime_summary, elapsed):
    return {
        "accuracy": float(result["score"]) * 100.0 if result.get("contract_valid", True) else -100.0,
        "total_tokens": int(
            runtime_summary.get("metadata", {}).get("total_tokens", 0)
        ),
        "total_time": elapsed,
        "bbo_score": float(result["score"]),
        "contract_valid": bool(result.get("contract_valid", True)),
        "selfcheck_passed": bool(result.get("selfcheck_passed", True)),
        "errors": result.get("contract_issue", ""),
        "behavior_probe": result.get("behavior_probe", {}),
        "score_anytime": float(result["score_anytime"]),
        "score_final": float(result["score_final"]),
        "mean_final_latent_objective": float(
            result["mean_final_latent_objective"]
        ),
    }


def build_research_feedback(evaluation_dir, candidates, metrics, best_index):
    """Public research feedback for the next MAS, never a verifier tool."""
    best = candidates[best_index]
    best_dir = evaluation_dir / best["label"]
    source = (best_dir / "candidate_solver.py").read_text(encoding="utf-8")
    if len(source) > 100_000:
        raise ValueError("Prior solver is too long for feedback")
    assessments = []
    for candidate, item in zip(candidates, metrics):
        score = (f"{candidate['visible_score']:.6f}"
                 if candidate["visible_score"] is not None else "invalid")
        probe = item.get("behavior_probe") or {}
        warnings = "; ".join(probe.get("warnings", []))
        uniqueness = ", ".join(
            f"ask({case.get('requests')}): {case.get('unique')}/{case.get('observed')} unique"
            for case in probe.get("cases", []) if case.get("observed"))
        assessments.append(
            f"{candidate['label']}: visible={score}, valid={candidate.get('contract_valid')}, "
            f"anytime={item['score_anytime']:.6f}, final={item['score_final']:.6f}; "
            f"mean_final_latent={item['mean_final_latent_objective']:.4f}; "
            f"behavior={warnings or probe.get('error', 'no warning')}; "
            f"query_diversity={uniqueness or 'unknown'}; "
            f"error={item.get('errors', '')[-450:]}"
        )
    agents = []
    for path in sorted(best_dir.glob("agent_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        content = str(data.get("content") or "").strip()
        agents.append(f"{data.get('agent_id')}: success={data.get('success')}; "
                      f"report-start={content[:280]!r}; report-end={content[-280:]!r}")
    tool_events = []
    audit_path = best_dir / "tool_audit.jsonl"
    if audit_path.exists():
        from collections import Counter
        counts = Counter((event["agent"], event["tool"], event["success"])
                         for line in audit_path.read_text(encoding="utf-8").splitlines()
                         if line.strip() for event in [json.loads(line)]
                         if event.get("event") == "called")
        tool_events = [f"{agent}.{tool}: {'ok' if success else 'failed'}={count}"
                       for (agent, tool, success), count in sorted(counts.items())]
    valid = best.get("contract_valid", True) and best.get("visible_score") is not None
    instruction = ("The incumbent is contract-valid. Preserve its working algorithm; "
                   "identify one concrete performance weakness and test one targeted "
                   "solver change in the new complete candidate. "
                   if valid else
                   "The incumbent is invalid. Diagnose and repair its contract failure "
                   "in the new complete candidate. ")
    summary = ("All completed and failed candidate sources in THIS session "
               "can be inspected with bbo_read_history(item='index') and "
               "bbo_read_history(item='<listed label>'). Compare a failed "
               "version with earlier valid ones before revising. "
               "Prior generated optimizer and PUBLIC visible selfcheck feedback. "
               + instruction + "Seek better anytime and final quality without sealed "
               "information. Repeated identical points cannot improve the latent "
               "best-so-far at that query but can improve noisy decisions later; "
               "evaluate their net value rather than removing them blindly. A static preflight "
               "does not measure optimizer quality.\n"
               + "\n".join(assessments) + "\nPrior MAS agent trajectory:\n"
               + "\n".join(agents[:24]) + "\nPrior tool calls:\n"
               + "\n".join(tool_events))
    trace_path = best_dir / "public_trace.json"
    if trace_path.is_file():
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        summary += ("\nPUBLIC visible diagnostic traces for incumbent (sampled "
                    "instances and seeds; checkpoints 20, 60, 120): "
                    + json.dumps(trace, separators=(",", ":"))[:1700])
    rejected_configs = [
        f"{path.name}: {path.read_text(encoding='utf-8')[:350]}"
        for path in sorted(evaluation_dir.glob("*.fallback_reason.txt"))
    ]
    if rejected_configs:
        summary += ("\nInvalid generated MAS configs were preserved and "
                    "replaced with their validated parent topology: "
                    + "; ".join(rejected_configs[-8:]))
    history = []
    for candidate in candidates:
        path = evaluation_dir / candidate["label"] / "candidate_solver.py"
        history.append({
            "label": candidate["label"], "sha256": candidate["sha256"],
            "visible_score": candidate["visible_score"],
            "contract_valid": candidate["contract_valid"],
            "error": candidate.get("contract_issue", "")[-900:],
            "solver_source": path.read_text(encoding="utf-8"),
        })
    summary += ("\nAll versions in THIS run (including invalid candidates) "
                "are available by label via bbo_read_history(item='index') "
                "and bbo_read_history(item='<label>'). "
                "Inspect earlier versions when a recent change fails.")
    return {"source_label": best["label"], "source_sha256": best["sha256"],
            "source_contract_valid": bool(valid),
            "summary": summary[:11500], "solver_source": source,
            "history": history,
            "public_trace": json.loads(trace_path.read_text(encoding="utf-8"))
                if trace_path.is_file() else None}


def load_prior_memory(run_dir):
    """Continue EvoMAS experience only; never read an earlier solver or its score."""
    run_dir = run_dir.resolve()
    if not (run_dir / "summary.json").is_file():
        raise FileNotFoundError("Continuation requires a completed run summary")
    memory = run_dir / "memory.json"
    return memory if memory.is_file() else None


def sanitize_working_pool(working_pool, tool_mode):
    """Keep only runnable configurations, and repair copied index paths."""
    rejected = []
    quarantine = working_pool.parent / "rejected_pool_configs"
    for path in sorted(working_pool.glob("*.yaml")):
        try:
            validated_config(path.read_text(encoding="utf-8"), tool_mode)
        except (ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
            quarantine.mkdir(exist_ok=True)
            shutil.move(str(path), str(quarantine / path.name))
            rejected.append({"name": path.name, "reason": str(exc)[:1500]})
    index_path = working_pool / "mas_index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if not isinstance(index, dict):
                raise ValueError("MAS index must be a JSON mapping")
        except (json.JSONDecodeError, ValueError) as exc:
            quarantine.mkdir(exist_ok=True)
            shutil.move(str(index_path), str(quarantine / "mas_index.invalid.json"))
            rejected.append({"name": "mas_index.json", "reason": str(exc)[:1000]})
            index = {}
        index = {name: dict(entry, path=str(working_pool / (name + ".yaml")))
                 for name, entry in index.items()
                 if (working_pool / (name + ".yaml")).is_file()}
        index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    if rejected:
        report = working_pool.parent / "rejected_pool_configs.json"
        previous = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else []
        report.write_text(json.dumps(previous + rejected, indent=2) + "\n",
                          encoding="utf-8")
    if not list(working_pool.glob("*.yaml")):
        raise ValueError("No valid MAS configurations remain in the working pool")
    return rejected


def copy_initial_memory(source, destination):
    """Migrate older BBO memory stored on a 0..100 rather than 0..1 scale."""
    entries = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("EvoMAS memory must be a JSON list")
    migrated = 0
    for entry in entries:
        old, new = float(entry["old_accuracy"]), float(entry["new_accuracy"])
        if max(abs(old), abs(new)) > 1.0:
            entry["old_accuracy"] = old / 100.0
            entry["new_accuracy"] = new / 100.0
            entry["analysis"] = (
                "Recorded BBO visible score changed from "
                f"{old / 100.0:.6f} to {new / 100.0:.6f}. "
                "Prior free-form analysis had a score-unit mismatch; "
                "the recorded configuration changes should be interpreted "
                "with these corrected numbers."
            )
            migrated += 1
    destination.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return migrated


def select_best_visible(candidates, expected_count, output_dir, dry_run):
    if len(candidates) != expected_count:
        raise RuntimeError("complete solver count differs from --solver-limit")
    if any(item["visible_score"] is not None and
           not 0.0 <= item["visible_score"] <= 1.0 for item in candidates):
        raise ValueError("non-finite or out-of-range visible score")
    eligible = [item for item in candidates
                if item.get("contract_valid", True) and item["visible_score"] is not None]
    if not eligible:
        raise RuntimeError("no BBO-contract-valid solver among completed versions")
    winner = max(eligible, key=lambda item: item["visible_score"])
    if not dry_run:
        source = Path(winner["candidate_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output_dir / "best_visible_solver.py"
        shutil.copyfile(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != winner["sha256"]:
            raise RuntimeError("visible winner copy failed integrity check")
    return winner


def evaluate_config(
    config_text,
    label,
    output_dir,
    instruction,
    container,
    dry_run,
    tool_mode,
    feedback=None,
):
    config_path = output_dir / f"{label}.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    if dry_run:
        config = yaml.safe_load(config_text)
        fake_step = bool(
            config.get("meta", {}).get("fake_evolution_step")
        )
        score = 0.60 if fake_step else 0.40
        return {
            "accuracy": score * 100.0,
            "total_tokens": 0,
            "total_time": 0.0,
            "bbo_score": score,
            "score_anytime": score,
            "score_final": score,
            "mean_final_latent_objective": 0.0,
        }

    if not container:
        raise ValueError("--container is required without --dry-run")

    from visible_selfcheck import CandidateSelfcheckError, evaluate_solver

    run_dir = output_dir / label
    started = time.monotonic()
    try:
        runtime_summary = execute_mas(
            config_path, instruction, run_dir, dry_run=False,
            tool_mode=tool_mode, feedback=feedback, container=container,
        )
    except (CandidateExtractionError, MASExecutionFailure) as exc:
        # Preserve the raw agent report and count this generated version as invalid.
        # A missing solver cannot receive an official BBO score.
        run_dir.mkdir(parents=True, exist_ok=True)
        # Preserve a terminal agent's partial authored file if it exists and
        # was actually written; otherwise keep a marked invalid placeholder.
        written = False
        audit = run_dir / "tool_audit.jsonl"
        if audit.is_file():
            written = any(json.loads(line).get("tool") == "bbo_write_candidate"
                          and json.loads(line).get("success") is True
                          for line in audit.read_text().splitlines() if line.strip())
        sources = list((run_dir / "tool_workspace").glob("*/candidate.py")) if written else []
        candidate = max(sources, key=lambda p: p.stat().st_mtime) if sources else None
        (run_dir / "candidate_solver.py").write_text(
            candidate.read_text(encoding="utf-8") if candidate else
            "# Invalid MAS output: no extractable Optimizer source.\n", encoding="utf-8")
        result = {"score": 0.0, "score_anytime": 0.0, "score_final": 0.0,
                  "mean_final_latent_objective": 0.0, "contract_valid": False,
                  "selfcheck_passed": False, "contract_issue": str(exc),
                  "behavior_probe": {"error": str(exc)}}
        (run_dir / "selfcheck.json").write_text(json.dumps(result, indent=2) + "\n")
        metrics = metrics_from_bbo(result, {"metadata": {}}, time.monotonic() - started)
        (run_dir / "evomas_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        return metrics
    mas_elapsed = time.monotonic() - started
    solver = (run_dir / "candidate_solver.py").read_text(
        encoding="utf-8"
    )
    try:
        result = evaluate_solver(solver, run_dir, container)
    except CandidateSelfcheckError as exc:
        # A generated solver can be invalid. It still consumes one complete MAS
        # generation and one official selfcheck attempt under the version cap.
        # Keep the exact candidate and trace, penalize its EvoMAS reward, and
        # pass the failure to the next mutation rather than aborting the run.
        result = {
            "score": 0.0,
            "score_anytime": 0.0,
            "score_final": 0.0,
            "mean_final_latent_objective": 0.0,
            "contract_valid": False,
            "selfcheck_passed": False,
            "contract_issue": str(exc)[-3000:],
            "behavior_probe": json.loads((run_dir / "behavior_probe.json").read_text())
                if (run_dir / "behavior_probe.json").exists() else {},
        }
        (run_dir / "selfcheck.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    metrics = metrics_from_bbo(
        result,
        runtime_summary,
        mas_elapsed,
    )
    (run_dir / "evomas_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def mark_interrupted_candidate(evaluation_dir, label, explanation):
    """Keep a partial agent attempt in this same session as an invalid version."""
    run_dir = evaluation_dir / label
    run_dir.mkdir(exist_ok=True)
    candidate = run_dir / "candidate_solver.py"
    completed_metrics = run_dir / "evomas_metrics.json"
    if completed_metrics.is_file() and candidate.is_file():
        return json.loads(completed_metrics.read_text(encoding="utf-8"))
    if not candidate.is_file():
        written = []
        audit = run_dir / "tool_audit.jsonl"
        if audit.is_file():
            written = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
        agents = {event["agent"] for event in written
                  if event.get("event") == "called"
                  and event.get("tool") == "bbo_write_candidate" and event.get("success")}
        sources = [p for name in agents
                   for p in (run_dir / "tool_workspace" / name).glob("candidate.py")]
        candidate.write_text(
            max(sources, key=lambda p: p.stat().st_mtime).read_text(encoding="utf-8")
            if sources else "# Interrupted MAS: no complete solver source.\n",
            encoding="utf-8")
    result = {"score": 0.0, "score_anytime": 0.0, "score_final": 0.0,
              "mean_final_latent_objective": 0.0, "contract_valid": False,
              "selfcheck_passed": False, "contract_issue": explanation,
              "behavior_probe": {"error": explanation}}
    (run_dir / "selfcheck.json").write_text(json.dumps(result, indent=2) + "\n")
    metrics = metrics_from_bbo(result, {"metadata": {}}, 0.0)
    (run_dir / "evomas_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evomas-root", type=Path, required=True)
    parser.add_argument("--seed-pool", type=Path, required=True)
    parser.add_argument("--instruction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-existing", action="store_true",
                        help="Continue this output directory's unfinished session, keeping its candidate history")
    parser.add_argument("--container")
    parser.add_argument(
        "--meta-model",
        default="deepseek-v4-flash-s1",
    )
    parser.add_argument("--model-pool", default="deepseek-v4-flash-s1,deepseek-v4-flash-s2",
                        help="Comma- or space-separated exact model IDs for MAS agents")
    parser.add_argument("--num-parents", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--skip-final-repeat", action="store_true",
                        help="Count only distinct candidate MAS executions; omits upstream final rerun")
    parser.add_argument("--solver-limit", "--max-solver-versions", dest="solver_limit", type=int, default=12)
    parser.add_argument("--research-time-limit", type=float, default=6750,
                        help="Soft wall-time target in seconds for MAS research; finish an in-flight version")
    parser.add_argument("--tool-mode", choices=("none", "bbo"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-memory", type=Path, help="Optional prior EvoMAS action memory JSON")
    parser.add_argument("--continue-from-run", type=Path,
                        help="Carry only EvoMAS memory from a completed run; pool is chosen separately")
    parser.add_argument(
        "--mutation-probability",
        type=float,
        default=0.8,
    )
    parser.add_argument("--memory-evolution", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--improvement-threshold", type=float, default=0.05)
    parser.add_argument("--cost-weight", choices=("tokens", "time", "both"), default="both")
    parser.add_argument("--beta", type=float, default=1e-6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hidden-task-root", type=Path,
                        help="Official Harbor task directory; enables automatic sealed verification")
    parser.add_argument("--hidden-jobs-dir", type=Path,
                        help="Harbor jobs output directory (requires --hidden-task-root)")
    args = parser.parse_args()
    run_started = time.monotonic()
    args.meta_model = canonical_model_id(args.meta_model)
    args.model_pool = parse_model_pool(args.model_pool)
    configure_allowed_models(args.model_pool)
    os.environ["EVOMAS_ALLOWED_MODELS"] = " ".join(args.model_pool)
    if bool(args.hidden_task_root) != bool(args.hidden_jobs_dir):
        raise ValueError("Provide both --hidden-task-root and --hidden-jobs-dir")
    if args.dry_run and args.hidden_task_root:
        raise ValueError("Cannot run sealed verification for --dry-run")
    if args.dry_run and args.continue_from_run:
        raise ValueError("Cannot continue a live run in dry-run mode")
    if args.initial_memory and args.continue_from_run:
        raise ValueError("Use either --initial-memory or --continue-from-run")

    repeat_count = 0 if args.skip_final_repeat else 1
    if args.max_steps is None:
        args.max_steps = args.solver_limit - args.num_parents - repeat_count
    if args.num_parents < 1 or args.max_steps < 0:
        raise ValueError("--num-parents must be positive; at least parents + final solver versions required")
    if args.solver_limit < 1 or args.num_parents + args.max_steps + repeat_count != args.solver_limit:
        raise ValueError("solver limit must equal parents + evolution steps + optional final execution")
    if not 0.0 <= args.mutation_probability <= 1.0:
        raise ValueError("--mutation-probability must be in [0, 1]")
    if args.research_time_limit <= 0:
        raise ValueError("--research-time-limit must be positive")
    if args.output_dir.exists() and not args.resume_existing:
        raise FileExistsError(
            f"Refusing to overwrite output directory: {args.output_dir}"
        )
    if args.resume_existing and (not args.output_dir.is_dir()
                                 or not (args.output_dir / "session_checkpoint.json").is_file()
                                 or (args.output_dir / "summary.json").is_file()):
        raise ValueError("Resume requires an unfinished directory with session_checkpoint.json")
    if not args.seed_pool.is_dir() or not args.instruction.is_file():
        raise FileNotFoundError("seed pool or instruction file is missing")
    if len(list(args.seed_pool.glob("*.yaml"))) < args.num_parents:
        raise ValueError("seed pool has fewer YAML configurations than requested parents")
    provenance_path = args.seed_pool / "provenance.json"
    provenance = (json.loads(provenance_path.read_text())
                  if provenance_path.exists() else {})
    seed_policy = provenance.get("seed_tool_policy", "empty")
    original_seed_names = set(provenance.get("configs", {}))
    if seed_policy == "basic" and args.tool_mode != "bbo":
        raise ValueError("Basic tool pool requires --tool-mode bbo")

    sys.path.insert(0, str(args.evomas_root))

    from src.meta_model.experience import consolidate_evolution_trace
    from src.meta_model.mas_index import MASIndex
    from src.meta_model.metamodel import MetaModel
    from src.meta_model.pool import ConfigurationPool, add_to_pool_if_better
    from src.meta_model.reward import compute_reward
    from src.tools.bbo_generation import CATALOG_TEXT

    if not args.resume_existing:
        args.output_dir.mkdir(parents=True)
    working_pool = args.output_dir / "working_pool"
    checkpoint_file = args.output_dir / "session_checkpoint.json"
    if args.resume_existing:
        state = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        signature = {"meta_model": args.meta_model, "model_pool": args.model_pool,
                     "solver_limit": args.solver_limit, "seed": args.seed,
                     "mutation_probability": args.mutation_probability}
        if state["signature"] != signature:
            raise ValueError("Resume parameters differ from the original session")
        run_started = time.monotonic() - max(0.0, time.time() - state["started_epoch"])
        retargeted_models = state["retargeted_models"]
        rejected_pool_configs = state["rejected_pool_configs"]
    else:
        shutil.copytree(args.seed_pool, working_pool)
        retargeted_models = retarget_unavailable_models(working_pool, args.model_pool)
        if args.tool_mode == "bbo":
            for path in working_pool.glob("*.yaml"):
                try:
                    path.write_text(ensure_session_history_tool(
                        path.read_text(encoding="utf-8")), encoding="utf-8")
                except yaml.YAMLError:
                    pass  # sanitize_working_pool quarantines malformed files.
        rejected_pool_configs = sanitize_working_pool(working_pool, args.tool_mode)
        normalize_basic_pool_seeds(working_pool, seed_policy, original_seed_names)
    if len(list(working_pool.glob("*.yaml"))) < args.num_parents:
        raise ValueError("Too few valid MAS configurations after pool validation")

    evaluation_dir = args.output_dir / "evaluations"
    evaluation_dir.mkdir(exist_ok=args.resume_existing)

    instruction_text = args.instruction.read_text(encoding="utf-8")
    task_query = "RSI-Exam Query-Budgeted Noisy Black-Box Optimization"
    task_description = instruction_text[:4000] + (
        "\nResearch focus for the EvoMAS configuration: 120 evaluations in 10D with "
        "noisy losses. Ask the agents to compare bounded covariance adaptation, "
        "local exploitation and noise-aware incumbent selection against global "
        "space filling. Preserve enough exploration and bounded per-query work "
        "for the aggregate sealed runtime. Select compact MAS topologies when "
        "they can produce comparable complete solvers, so the research budget "
        "supports more distinct public-feedback iterations. Inspect per-stage "
        "PUBLIC traces before making one controlled change: tell the solver only "
        "the noisy observations, separate observed and latent diagnostics, and "
        "compare 20/60/120-query checkpoints. Do not assume the public instances' "
        "optima or coefficients are available in sealed runs. Consider centered "
        "bounded designs, covariance adaptation, local trust regions or lightweight "
        "surrogate screening; do not forbid a family without a timed public test. "
        "These are hypotheses to evaluate with public selfcheck, not hardcoded output."
    )
    task_samples = [{"query": task_description}]

    memory_source = args.initial_memory
    if args.continue_from_run:
        memory_source = load_prior_memory(args.continue_from_run)
    if memory_source and not args.resume_existing:
        if not memory_source.is_file():
            raise FileNotFoundError(memory_source)
        migrated_memory_entries = copy_initial_memory(
            memory_source, args.output_dir / "memory.json")
    else:
        migrated_memory_entries = 0

    if args.dry_run:
        meta_model = DryRunMetaModel()
    else:
        meta_model = MetaModel(
            model_id=args.meta_model,
            memory_path=str(args.output_dir / "memory.json"),
            memory_evolution=args.memory_evolution,
            verbose=True,
        )
        meta_model.bbo_tool_policy = (
            "For the BBO no-tool baseline keep tools: [] on every agent."
            if args.tool_mode == "none" else CATALOG_TEXT
        )
        meta_model.bbo_tool_policy += (
            "\nIf you mutate prompts, put substantive BBO-specific natural-language "
            "instructions directly in each changed YAML prompt field. Merely "
            "renaming a template identifier does not express an actionable "
            "instruction to the agent. Preserve the other component types."
        )

    meta_error_file = args.output_dir / "meta_errors.json"
    meta_errors = (json.loads(meta_error_file.read_text(encoding="utf-8"))
                   if args.resume_existing and meta_error_file.is_file() else [])
    def recover_meta(stage, fallback, operation, *args, **kwargs):
        """A failed planning request consumes its slot without losing the run."""
        try:
            return operation(*args, **kwargs)
        except Exception as exc:
            record = {"stage": stage, "error_type": type(exc).__name__,
                      "message": str(exc)[-1200:]}
            meta_errors.append(record)
            meta_error_file.write_text(
                json.dumps(meta_errors, indent=2) + "\n", encoding="utf-8")
            return fallback()

    mas_index = MASIndex(str(working_pool))
    if args.resume_existing:
        parent_paths = state["parent_paths"]
    else:
        parent_paths = recover_meta("select",
            lambda: [str(p) for p in sorted(working_pool.glob("*.yaml"))[:args.num_parents]],
            meta_model.select,
            task_query, task_description, str(working_pool),
            k=args.num_parents, mas_index=mas_index,
        )
    if not parent_paths:
        raise RuntimeError("EvoMAS selection returned no parents")

    configs = []
    metrics = []
    candidates = []
    if args.resume_existing:
        manifest = args.output_dir / "candidate_manifest.json"
        candidates = json.loads(manifest.read_text()) if manifest.is_file() else []
        for item in candidates:
            label = item["label"]
            configs.append((evaluation_dir / f"{label}.yaml").read_text(encoding="utf-8"))
            saved_metrics = evaluation_dir / label / "evomas_metrics.json"
            if saved_metrics.is_file():
                metrics.append(json.loads(saved_metrics.read_text()))
            elif args.dry_run:
                score = item["visible_score"]
                metrics.append({"accuracy": score * 100, "bbo_score": score,
                                "score_anytime": score, "score_final": score,
                                "mean_final_latent_objective": 0.0,
                                "total_tokens": 0, "total_time": 0.0})
            else:
                raise FileNotFoundError(saved_metrics)
    else:
        state = {
            "signature": {"meta_model": args.meta_model, "model_pool": args.model_pool,
                          "solver_limit": args.solver_limit, "seed": args.seed,
                          "mutation_probability": args.mutation_probability},
            "started_epoch": time.time(), "parent_paths": parent_paths,
            "retargeted_models": retargeted_models,
            "rejected_pool_configs": rejected_pool_configs,
        }
        checkpoint_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    def record_candidate(label, item):
        record = {"label": label,
                  "visible_score": item["bbo_score"] if item.get("selfcheck_passed", True) else None,
                  "contract_valid": item.get("contract_valid", True),
                  "selfcheck_passed": item.get("selfcheck_passed", True)}
        if not record["contract_valid"]:
            record["contract_issue"] = item.get("errors", "")[-3000:]
        if not args.dry_run:
            candidate_path = evaluation_dir / label / "candidate_solver.py"
            record["candidate_path"] = str(candidate_path)
            record["sha256"] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
            config = yaml.safe_load((evaluation_dir / f"{label}.yaml").read_text())
            record["configured_tools"] = {
                aid: list(agent.get("tools", []))
                for aid, agent in config["agents"].items() if agent.get("tools")
            }
            audit = evaluation_dir / label / "tool_audit.jsonl"
            events = [json.loads(line) for line in audit.read_text().splitlines()] if audit.exists() else []
            record["tools_loaded"] = sum(e["event"] == "loaded" for e in events)
            record["tool_call_attempts"] = sum(e["event"] == "called" for e in events)
            record["tool_calls"] = sum(e["event"] == "called" and e["success"] for e in events)
            if record["configured_tools"] and record["tools_loaded"] == 0 and record["selfcheck_passed"]:
                raise RuntimeError(f"YAML tools were not instantiated for {label}")
        candidates.append(record)
        (args.output_dir / "candidate_manifest.json").write_text(
            json.dumps(candidates, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    for index, parent_path in enumerate(parent_paths):
        label = f"parent_{index:02d}"
        if any(item["label"] == label for item in candidates):
            continue
        if (evaluation_dir / f"{label}.yaml").is_file():
            generated = (evaluation_dir / f"{label}.yaml").read_text(encoding="utf-8")
            parent_metrics = mark_interrupted_candidate(
                evaluation_dir, label, "Process stopped during MAS evaluation")
            configs.append(generated)
            metrics.append(parent_metrics)
            record_candidate(label, parent_metrics)
            continue
        generated = recover_meta(f"parent_{index:02d}_generate",
            lambda: Path(parent_path).read_text(encoding="utf-8"),
            meta_model.generate,
            parent_path,
            task_samples,
            task_description,
            model_list=args.model_pool,
        )
        try:
            if args.tool_mode == "bbo":
                generated = ensure_session_history_tool(generated)
            validated_config(generated, args.tool_mode)
        except (ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
            rejected = evaluation_dir / f"parent_{index:02d}.rejected.yaml"
            rejected.write_text(str(generated), encoding="utf-8")
            (evaluation_dir / f"parent_{index:02d}.fallback_reason.txt").write_text(str(exc))
            generated = Path(parent_path).read_text(encoding="utf-8")
            generated = validated_config(generated, args.tool_mode)
        parent_metrics = evaluate_config(
            generated,
            label,
            evaluation_dir,
            args.instruction,
            args.container,
            args.dry_run,
            args.tool_mode,
        )
        configs.append(generated)
        metrics.append(parent_metrics)
        record_candidate(label, parent_metrics)

    rewards = [
        compute_reward(item, beta=args.beta, cost_weight=args.cost_weight)
        for item in metrics
    ]
    best_index = max(range(len(rewards)), key=rewards.__getitem__)
    operations = []
    if args.resume_existing:
        for idx in range(len(parent_paths), len(candidates)):
            label = candidates[idx]["label"]
            if not label.startswith("step_"):
                continue
            _, number, action = label.split("_", 2)
            operations.append({"step": int(number), "type": action,
                               "reward": rewards[idx],
                               "improved": rewards[idx] > max(rewards[:idx])})

    evolution_rng = random.Random(args.seed)
    stopped_for_time = False
    longest_version_seconds = max((item["total_time"] for item in metrics), default=0.0)
    for step in range(1, args.max_steps + 1):
        if any(item["label"].startswith(f"step_{step:02d}_") for item in candidates):
            if args.num_parents != 1:
                evolution_rng.random()  # advance the original operation schedule
            continue
        # Complete in-flight candidates and keep a hidden-verification reserve.
        # Agent latency can be much longer than the visible selfcheck itself.
        elapsed = time.monotonic() - run_started
        if elapsed + max(300.0, longest_version_seconds * 1.2) >= args.research_time_limit:
            stopped_for_time = True
            break
        old_index = best_index
        feedback = (None if args.dry_run else build_research_feedback(
            evaluation_dir, candidates, metrics, best_index))
        use_mutation = (
            args.num_parents == 1
            or evolution_rng.random() < args.mutation_probability
        )
        action = "mutation" if use_mutation else "crossover"
        label = f"step_{step:02d}_{action}"
        partial_config = evaluation_dir / f"{label}.yaml"
        if partial_config.is_file():
            offspring = partial_config.read_text(encoding="utf-8")
            child_metrics = mark_interrupted_candidate(
                evaluation_dir, label, "Process stopped during MAS evaluation")
            child_reward = compute_reward(
                child_metrics, beta=args.beta, cost_weight=args.cost_weight)
            configs.append(offspring)
            metrics.append(child_metrics)
            rewards.append(child_reward)
            record_candidate(label, child_metrics)
            best_index = max(range(len(rewards)), key=rewards.__getitem__)
            operations.append({"step": step, "type": action,
                               "reward": child_reward,
                               "improved": best_index == len(rewards) - 1})
            continue

        if use_mutation:
            offspring = recover_meta(f"step_{step:02d}_mutation",
                lambda: configs[best_index],
                meta_model.mutate,
                configs[best_index],
                metrics[best_index],
                feedback["summary"] if feedback is not None else "dry-run feedback",
                model_list=args.model_pool,
            )
            action = "mutation"
        else:
            ranked = sorted(
                range(len(rewards)),
                key=rewards.__getitem__,
                reverse=True,
            )
            offspring = recover_meta(f"step_{step:02d}_crossover",
                lambda: configs[ranked[0]],
                meta_model.crossover,
                configs[ranked[0]],
                configs[ranked[1]],
                metrics[ranked[0]],
                metrics[ranked[1]],
                model_list=args.model_pool,
            )
            action = "crossover"

        try:
            if args.tool_mode == "bbo":
                offspring = ensure_session_history_tool(offspring)
            validated_config(offspring, args.tool_mode)
        except (ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
            label = f"step_{step:02d}_{action}"
            (evaluation_dir / f"{label}.rejected.yaml").write_text(str(offspring), encoding="utf-8")
            (evaluation_dir / f"{label}.fallback_reason.txt").write_text(str(exc))
            offspring = validated_config(configs[old_index], args.tool_mode)
        child_metrics = evaluate_config(
            offspring,
            f"step_{step:02d}_{action}",
            evaluation_dir,
            args.instruction,
            args.container,
            args.dry_run,
            args.tool_mode,
            feedback=feedback,
        )
        child_reward = compute_reward(
            child_metrics,
            beta=args.beta, cost_weight=args.cost_weight,
        )

        configs.append(offspring)
        metrics.append(child_metrics)
        longest_version_seconds = max(longest_version_seconds, child_metrics["total_time"])
        rewards.append(child_reward)
        record_candidate(f"step_{step:02d}_{action}", child_metrics)
        best_index = max(range(len(rewards)), key=rewards.__getitem__)

        operations.append({
            "step": step,
            "type": action,
            "reward": child_reward,
            "improved": best_index == len(rewards) - 1,
        })

        recover_meta(f"step_{step:02d}_memory",
            lambda: None, meta_model.update_memory,
            action_type=action,
            query=task_query,
            old_config=configs[old_index],
            new_config=offspring,
            old_accuracy=metrics[old_index]["accuracy"] / 100.0,
            new_accuracy=child_metrics["accuracy"] / 100.0,
        )

    pool = ConfigurationPool(str(working_pool))
    validated_config(configs[best_index], args.tool_mode)
    accepted = add_to_pool_if_better(
        pool,
        configs[best_index],
        metrics[best_index],
        metrics[:len(parent_paths)],
        task_query,
        beta=args.beta, cost_weight=args.cost_weight,
        threshold=args.improvement_threshold,
    )
    if accepted:
        try:
            validated_config(Path(accepted).read_text(encoding="utf-8"), args.tool_mode)
        except (ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
            rejected = args.output_dir / "rejected_pool_configs"
            rejected.mkdir(exist_ok=True)
            shutil.move(str(accepted), str(rejected / Path(accepted).name))
            rejected_pool_configs.append({"name": Path(accepted).name,
                                          "reason": f"invalid pool write: {exc}"[:1000]})
            accepted = None
        else:
            mas_index.add_config(accepted)

    # Mirror upstream's final execution of the reward-best configuration.
    # This is one new MAS generation and one new visible self-check, not a copy.
    if args.skip_final_repeat:
        final_metrics = None
    else:
        final_metrics = evaluate_config(
            configs[best_index], "final_reward_best", evaluation_dir,
            args.instruction, args.container, args.dry_run, args.tool_mode,
        )
        record_candidate("final_reward_best", final_metrics)
    indexed_name = (
        Path(accepted).stem if accepted else
        Path(parent_paths[best_index]).stem if best_index < len(parent_paths) else None
    )
    if indexed_name:
        mas_index.record_query_result(
            config_name=indexed_name,
            task_id=args.seed,
            query_snippet=task_query,
            accuracy=metrics[best_index]["accuracy"],
            tokens=metrics[best_index]["total_tokens"],
            time_seconds=metrics[best_index]["total_time"],
            is_winner=True,
        )
    # Keep the output itself safe to inherit, even if an upstream pool operation
    # wrote a malformed YAML after the earlier import validation.
    rejected_pool_configs.extend(sanitize_working_pool(working_pool, args.tool_mode))
    visible_winner = select_best_visible(candidates, len(candidates),
                                         args.output_dir, args.dry_run)

    trace = consolidate_evolution_trace(
        operations,
        task_query,
        metrics[0]["accuracy"],
        metrics[best_index]["accuracy"],
        configs[best_index],
    )
    (args.output_dir / "evolution_trace.txt").write_text(
        trace + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "best_config.yaml").write_text(
        configs[best_index],
        encoding="utf-8",
    )

    summary = {
        "dry_run": args.dry_run,
        "bbo_feedback_adapter": "public_visible_previous_best_v1",
        "behavior_probe": "synthetic_ask_tell_no_objective_v1",
        "research_wall_seconds": time.monotonic() - run_started,
        "hidden_status": "not_run",
        "hidden_score": None,
        "seed_pool": str(args.seed_pool),
        "meta_model": args.meta_model,
        "model_pool": args.model_pool,
        "retargeted_seed_models": retargeted_models,
        "seed_tool_policy": seed_policy,
        "working_pool": str(working_pool),
        "num_parents": len(parent_paths),
        "max_steps": args.max_steps,
        "solver_limit": args.solver_limit,
        "completed_solver_versions": len(candidates),
        "stopped_for_time": stopped_for_time,
        "research_time_limit_seconds": args.research_time_limit,
        "final_repeat_executed": not args.skip_final_repeat,
        "tool_mode": args.tool_mode,
        "tool_use_observed": any(c.get("configured_tools") and c.get("tool_calls", 0) > 0
                                 for c in candidates),
        "candidates": candidates,
        "best_visible": visible_winner,
        "rejected_inherited_pool_configs": rejected_pool_configs,
        "meta_errors": meta_errors,
        "final_reward_best_metrics": final_metrics,
        "seed": args.seed,
        "operations": operations,
        "parent_metrics": metrics[:len(parent_paths)],
        "best_metrics": metrics[best_index],
        "best_reward": rewards[best_index],
        "best_config_sha256": hashlib.sha256(
            configs[best_index].encode()
        ).hexdigest(),
        "accepted_pool_path": accepted,
        "memory_evolution": args.memory_evolution,
        "initial_memory": str(args.initial_memory) if args.initial_memory else None,
        "continuation_from": str(args.continue_from_run) if args.continue_from_run else None,
        "migrated_memory_entries": migrated_memory_entries,
        "parameters": {
            "beta": args.beta,
            "cost_weight": args.cost_weight,
            "improvement_threshold": args.improvement_threshold,
            "mutation_probability": args.mutation_probability,
        },
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.hidden_task_root:
        from hidden_handoff import submit
        summary = submit(args.output_dir, args.hidden_task_root, args.hidden_jobs_dir,
                         Path.home()/".local/share/uv/tools/harbor/bin/harbor",
                         args.meta_model.removeprefix("openai:"))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
