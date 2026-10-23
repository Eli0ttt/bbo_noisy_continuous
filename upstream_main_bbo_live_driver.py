#!/usr/bin/env python3
"""Run EvoMAS's upstream evolutionary loop on the RSI BBO task.

This is an adapter, not a fork of EvoMAS: it imports the supplied upstream
``main.py`` and invokes its unmodified ``_run_single_batch``.  The only
replacement is ``interpret_mas``, because BBO evaluates generated optimizer
source using the official container selfcheck rather than an EvoMAS dataset
evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml


INVALID_ACCURACY = -1_000_000.0


class InvalidCandidateError(RuntimeError):
    """Generated MAS/solver is invalid under the declared BBO interface."""


class BBOTask:
    """The one BBO task description exposed to EvoMAS's meta-model."""

    def __init__(self, instruction: str):
        self.instruction = instruction

    def model_dump(self) -> dict[str, str]:
        return {"id": "rsi-bbo-visible", "query": self.instruction}


class BBODataset:
    def __init__(self, instruction: str):
        self.items = [BBOTask(instruction)]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> BBOTask:
        return self.items[index]


class DryMetaModel:
    """Deterministic zero-API stand-in used only by ``--live``-absent tests."""

    instances: list["DryMetaModel"] = []

    def __init__(self, *args: Any, **kwargs: Any):
        self.calls: list[str] = []
        type(self).instances.append(self)

    def select(self, task_query: str, task_description: str, pool_dir: str, k: int = 2, **_: Any) -> list[str]:
        self.calls.append("select")
        return [str(p) for p in sorted(Path(pool_dir).glob("*.yaml"))[:k]]

    def generate(self, mas_config_path: str, **_: Any) -> str:
        self.calls.append("generate")
        return Path(mas_config_path).read_text(encoding="utf-8")

    def mutate(self, mas_config: str, **_: Any) -> str:
        self.calls.append("mutate")
        spec = yaml.safe_load(mas_config)
        spec.setdefault("meta", {})["bbo_driver_dryrun"] = True
        return yaml.safe_dump(spec, sort_keys=False)

    def crossover(self, mas_config_1: str, **_: Any) -> str:
        self.calls.append("crossover")
        return self.mutate(mas_config_1)

    def update_memory(self, **_: Any) -> None:
        self.calls.append("update_memory")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")



def atomic_bytes(path: Path, data: bytes) -> None:
    """Replace a checkpoint atomically so a hard timeout cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def atomic_json_dump(path: Path, value: Any) -> None:
    """Write checkpoint metadata atomically alongside the solver bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_solver_checkpoint(output_dir: Path, solver: Path, score: dict[str, Any],
                           entry: dict[str, Any]) -> dict[str, Any]:
    """Keep latest valid solver and best official-score solver for interruption recovery."""
    solver_bytes = solver.read_bytes()
    digest = hashlib.sha256(solver_bytes).hexdigest()
    score_value = float(score["score"])
    metadata = {
        "official_score": score_value,
        "score_anytime": float(score["score_anytime"]),
        "score_final": float(score["score_final"]),
        "candidate_sha256": digest,
        "config_sha256": entry["config_sha256"],
        "evaluation_index": entry["index"],
        "source_config": entry["source_config"],
        "selection": "latest_valid_evaluation",
    }
    atomic_bytes(output_dir / "latest_solver.py", solver_bytes)
    atomic_json_dump(output_dir / "latest_solver.json", metadata)

    best_path = output_dir / "best_solver.json"
    old_best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else None
    if old_best is None or score_value > float(old_best["official_score"]):
        best_metadata = dict(metadata)
        best_metadata["selection"] = "highest_official_score_seen_in_this_run"
        atomic_bytes(output_dir / "best_solver.py", solver_bytes)
        atomic_json_dump(best_path, best_metadata)
    return metadata

def bbo_stats(score: dict[str, Any], runtime: dict[str, Any], elapsed: float) -> dict[str, Any]:
    """Map BBO's 0--1 score to EvoMAS's documented 0--100 accuracy scale."""
    metadata = runtime.get("metadata", {})
    return {
        "accuracy": float(score["score"]) * 100.0,
        "total_tokens": int(metadata.get("total_tokens", 0)),
        "total_time": float(elapsed),
        "bbo_score": float(score["score"]),
        "score_anytime": float(score["score_anytime"]),
        "score_final": float(score["score_final"]),
        "mean_final_latent_objective": float(score["mean_final_latent_objective"]),
        "official_score_available": True,
        "valid": True,
    }


def invalid_stats(error: str, elapsed: float) -> dict[str, Any]:
    """A non-score sentinel: invalid children can never enter the pool."""
    return {
        "accuracy": INVALID_ACCURACY,
        "total_tokens": 0,
        "total_time": float(elapsed),
        "bbo_score": None,
        "score_anytime": None,
        "score_final": None,
        "mean_final_latent_objective": None,
        "official_score_available": False,
        "valid": False,
        "failure": error,
    }


class BBOInterpreter:
    """Compatibility callable for upstream ``interpret_mas``."""

    def __init__(self, execute, evaluate_solver, instruction: Path, container: str, output_dir: Path, live: bool, max_agents: int = 0):
        self.execute = execute
        self.evaluate_solver = evaluate_solver
        self.instruction = instruction
        self.container = container
        self.output_dir = output_dir
        self.live = live
        self.max_agents = max_agents
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, config_path: str, save_outputs: bool, **_: Any) -> dict[str, Any]:
        index = len(self.calls)
        label = Path(config_path).stem
        run_dir = self.output_dir / "evaluations" / f"{index:02d}_{label}"
        run_dir.mkdir(parents=True, exist_ok=False)
        config = Path(config_path)
        shutil.copy2(config, run_dir / "mas_config.yaml")
        started = time.monotonic()
        entry: dict[str, Any] = {
            "index": index,
            "source_config": str(config),
            "config_sha256": sha256(config),
            "save_outputs": bool(save_outputs),
        }
        try:
            try:
                config_obj = yaml.safe_load(config.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise InvalidCandidateError(f"generated MAS config is invalid YAML: {exc}") from exc
            if not isinstance(config_obj, dict):
                raise InvalidCandidateError("generated MAS config root must be a mapping")
            agent_specs = config_obj.get("agents", {})
            if not isinstance(agent_specs, dict) or not agent_specs:
                raise InvalidCandidateError("generated MAS config has no agents mapping")
            model_ids = {v.get("model_id") for v in agent_specs.values() if isinstance(v, dict)}
            if not model_ids or model_ids != {"openai:deepseek-v4-flash-s2"}:
                raise InvalidCandidateError(f"BBO config is not s2-only: {sorted(str(x) for x in model_ids)}")
            if self.max_agents and len(agent_specs) > self.max_agents:
                raise InvalidCandidateError(
                    f"BBO pilot agent cap exceeded: {len(agent_specs)} > {self.max_agents}"
                )

            generation_started = time.monotonic()
            try:
                runtime = self.execute(config, self.instruction, run_dir / "mas", dry_run=not self.live)
            except ValueError as exc:
                raise InvalidCandidateError(f"BBO MAS/config/candidate validation failed: {exc}") from exc
            generation_elapsed = time.monotonic() - generation_started
            if not self.live:
                # This branch is used only for driver tests; no official score is claimed.
                score = {
                    "score": 0.40 + 0.01 * index,
                    "score_anytime": 0.40 + 0.01 * index,
                    "score_final": 0.40 + 0.01 * index,
                    "mean_final_latent_objective": 0.0,
                }
            else:
                solver = run_dir / "mas" / "candidate_solver.py"
                if not solver.is_file():
                    raise FileNotFoundError(f"native runtime did not save {solver}")
                try:
                    score = self.evaluate_solver(solver.read_text(encoding="utf-8"), run_dir / "official_selfcheck", self.container)
                except subprocess.CalledProcessError as exc:
                    cmd = " ".join(str(part) for part in (exc.cmd or []))
                    if "py_compile" not in cmd and "/app/selfcheck.py" not in cmd:
                        raise
                    raise InvalidCandidateError(
                        f"official candidate check failed: {cmd}; exit={exc.returncode}"
                    ) from exc
                except Exception as exc:
                    # CandidateSelfcheckError is raised by the BBO adapter
                    # only after the official evaluator rejects generated
                    # solver code. It is therefore a candidate-local
                    # invalid result, not a harness failure.
                    if (
                        type(exc).__name__ == "CandidateSelfcheckError"
                        and type(exc).__module__ == "evomas_bbo_step0"
                    ):
                        raise InvalidCandidateError(
                            f"official BBO selfcheck rejected generated solver: {exc}"
                        ) from exc
                    raise
            stats = bbo_stats(score, runtime, generation_elapsed)
            if not self.live:
                # Synthetic dry scores exercise EvoMAS control flow only.
                # They are intentionally never labelled as an official BBO result.
                stats["official_score_available"] = False
                stats["synthetic_dryrun_score"] = True
            entry.update({
                "valid": True,
                "candidate_sha256": runtime["candidate_sha256"],
                "official_score": score["score"] if self.live else None,
                "statistics": stats,
            })
            if self.live:
                entry["solver_checkpoint"] = save_solver_checkpoint(
                    self.output_dir, solver, score, entry
                )
        except InvalidCandidateError as exc:
            elapsed = time.monotonic() - started
            failure = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            stats = invalid_stats(failure, elapsed)
            entry.update({"valid": False, "error": failure, "statistics": stats})
            (run_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        except Exception as exc:
            failure = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            entry.update({"valid": None, "fatal_error": failure})
            (run_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            json_dump(run_dir / "evaluation.json", entry)
            self.calls.append(entry)
            raise
        json_dump(run_dir / "evaluation.json", entry)
        self.calls.append(entry)
        return {"statistics": stats}


def load_upstream(root: Path):
    main_file = root / "main.py"
    if not main_file.is_file() or not (root / "src").is_dir():
        raise FileNotFoundError("--evomas-root must contain main.py and src/")
    sys.path.insert(0, str(root.resolve()))
    spec = importlib.util.spec_from_file_location("evomas_upstream_main_bbo", main_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {main_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    required = (args.evomas_root / "main.py", args.adapter / "native_bbo_runtime.py",
                args.adapter / "evomas_bbo_step0.py", args.seed_pool, args.instruction)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.max_steps < 1 or args.num_parents < 1:
        raise ValueError("--max-steps and --num-parents must be positive")
    if args.model != "openai:deepseek-v4-flash-s2":
        raise ValueError("this experiment is deliberately restricted to openai:deepseek-v4-flash-s2")

    args.output_dir.mkdir(parents=True)
    working_pool = args.output_dir / "working_pool"
    shutil.copytree(args.seed_pool, working_pool)
    sys.path.insert(0, str(args.adapter.resolve()))
    from native_bbo_runtime import execute
    if args.live:
        from evomas_bbo_step0 import evaluate_solver
    else:
        # Keeping this import out of dry mode proves that no OpenAI client or
        # evaluator dependency is needed for the structural preflight.
        evaluate_solver = None

    upstream = load_upstream(args.evomas_root)
    instruction = args.instruction.read_text(encoding="utf-8")
    interpreter = BBOInterpreter(execute, evaluate_solver, args.instruction, args.container,
                                 args.output_dir, args.live, args.max_agents)
    upstream.load_dataset = lambda dataset_name, split="test": BBODataset(instruction)
    upstream.interpret_mas = interpreter
    if not args.live:
        # No model client is constructed in a dry run.  The unmodified
        # upstream loop still performs select/generate/mutate/update-memory.
        DryMetaModel.instances.clear()
        upstream.MetaModel = DryMetaModel
    # The upstream function samples the operation with module-global random.
    # Seeding it here makes the documented --seed fully reproducible.
    random.seed(args.seed)
    result = upstream._run_single_batch(
        dataset_name="rsi_bbo_noisy_continuous",
        pool_dir=str(working_pool),
        num_eval_tasks=1,
        max_steps=args.max_steps,
        num_parents=args.num_parents,
        seed=args.seed,
        output_dir=str(args.output_dir / "final"),
        meta_model_id=args.model,
        model_list=[args.model],
        llm_as_judge=None,
        task_ids=[0],
        memory_path=str(args.output_dir / "memory.json"),
        memory_evolution=not args.no_memory_evolution,
    )
    report = {
        "live": args.live,
        "upstream_commit": git_commit(args.evomas_root),
        "upstream_main_sha256": sha256(args.evomas_root / "main.py"),
        "model": args.model,
        "max_agents": args.max_agents,
        "max_steps": args.max_steps,
        "num_parents": args.num_parents,
        "seed": args.seed,
        "memory_evolution": not args.no_memory_evolution,
        "seed_pool": str(args.seed_pool),
        "seed_pool_files": sorted(p.name for p in args.seed_pool.glob("*.yaml")),
        "result": result,
        "evaluations": interpreter.calls,
        "valid_evaluations": sum(x["valid"] for x in interpreter.calls),
        "invalid_evaluations": sum(not x["valid"] for x in interpreter.calls),
    }
    best_checkpoint = args.output_dir / "best_solver.json"
    latest_checkpoint = args.output_dir / "latest_solver.json"
    report["solver_checkpoints"] = {
        "best": json.loads(best_checkpoint.read_text(encoding="utf-8")) if best_checkpoint.exists() else None,
        "latest": json.loads(latest_checkpoint.read_text(encoding="utf-8")) if latest_checkpoint.exists() else None,
        "best_solver_path": str(args.output_dir / "best_solver.py") if (args.output_dir / "best_solver.py").exists() else None,
        "latest_solver_path": str(args.output_dir / "latest_solver.py") if (args.output_dir / "latest_solver.py").exists() else None,
    }
    if not args.live:
        report["dry_meta_operators"] = DryMetaModel.instances[-1].calls
    json_dump(args.output_dir / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evomas-root", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--seed-pool", type=Path, required=True)
    ap.add_argument("--instruction", type=Path, required=True)
    ap.add_argument("--container", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=1)
    ap.add_argument("--num-parents", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="openai:deepseek-v4-flash-s2")
    ap.add_argument("--max-agents", type=int, default=0,
                     help="optional pilot-only cap; 0 disables the cap")
    ap.add_argument("--no-memory-evolution", action="store_true")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
