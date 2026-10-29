#!/usr/bin/env python3
"""Read-only audit of inherited EvoMAS topology configurations."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import yaml

ADAPTER = Path(__file__).resolve().parent
sys.path.insert(0, str(ADAPTER))
sys.path.insert(0, str(ADAPTER.parent.parent / "EvoMAS-openai"))
# Only CATALOG metadata is loaded; the read-only audit does not instantiate
# smolagents runners and remains usable in a light-weight local environment.
try:
    import smolagents  # noqa: F401
except ImportError:
    sys.modules["smolagents"] = types.SimpleNamespace(Tool=type("Tool", (), {}))
from native_bbo_runtime import configure_allowed_models, parse_model_pool, validate
from src.mas.spec import MasSpec


def audit(run_dir: Path, model_pool: str) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise ValueError("Only a completed run with summary.json can be inherited")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("hidden_status") not in ("completed", "not_run"):
        raise ValueError("The run has no completed final selection")
    configure_allowed_models(parse_model_pool(model_pool))
    entries = []
    for path in sorted((run_dir / "working_pool").glob("*.yaml")):
        try:
            spec = MasSpec(**yaml.safe_load(path.read_text(encoding="utf-8")))
            levels = validate(spec, "bbo")
            entry = {"name": path.name, "valid": True,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "agents": len(spec.agents), "levels": levels,
                     "models": {aid: agent.model_id for aid, agent in spec.agents.items()},
                     "tools": {aid: list(agent.tools) for aid, agent in spec.agents.items()}}
        except (ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
            entry = {"name": path.name, "valid": False, "reason": str(exc)[:1000]}
        entries.append(entry)
    return {"run_dir": str(run_dir), "configs": entries,
            "invalid_count": sum(not x["valid"] for x in entries),
            "retarget_note": "Unavailable models can be retargeted in the copied next-run pool before validation; this read-only audit does not alter the source."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model-pool", default="deepseek-v4-flash-s1,deepseek-v4-flash-s2")
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir, args.model_pool), indent=2))
