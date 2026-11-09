#!/usr/bin/env python3
"""Freeze the complete upstream BBEH MAS pool with one available model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--model", default="openai:deepseek-v4-flash-s2")
    parser.add_argument("--include", nargs="*", help="Upstream topology stems to retain")
    parser.add_argument("--seed-tool-policy", choices=("empty", "basic"), default="empty")
    args = parser.parse_args()
    source_files = sorted(args.source.glob("*.yaml"))
    if args.include:
        wanted = set(args.include)
        source_files = [p for p in source_files if p.stem in wanted]
        if {p.stem for p in source_files} != wanted:
            raise ValueError("--include includes a missing upstream topology")
    if not source_files:
        raise ValueError("no upstream YAML files")
    if args.destination.exists():
        raise FileExistsError(args.destination)
    from native_bbo_runtime import ROLE_TASKS, canonical_model_id
    args.model = canonical_model_id(args.model)

    args.destination.mkdir(parents=True)
    provenance = {"source": str(args.source), "model": args.model,
                  "seed_tool_policy": args.seed_tool_policy, "configs": {}}
    for path in source_files:
        original = yaml.safe_load(path.read_text(encoding="utf-8"))
        adapted = yaml.safe_load(path.read_text(encoding="utf-8"))
        before = {aid: agent["model_id"] for aid, agent in original["agents"].items()}
        for aid, agent in adapted["agents"].items():
            agent["model_id"] = args.model
            if agent["prompt"] not in ROLE_TASKS and agent["role"] not in ROLE_TASKS:
                raise ValueError(f"unmapped role in {path.name}: {agent['role']}")
            if agent.get("tools"):
                raise ValueError(f"upstream tools must be empty in {path.name}")
            if args.seed_tool_policy == "basic":
                agent["tools"] = ["bbo_read_context", "bbo_read_public_data",
                                  "bbo_read_prior_feedback", "bbo_read_prior_trace",
                                  "bbo_read_history",
                                  "bbo_write_candidate", "bbo_preflight_candidate",
                                  "bbo_probe_candidate"]
        back = yaml.safe_load(yaml.safe_dump(adapted, sort_keys=False))
        for aid, agent in back["agents"].items():
            agent["model_id"] = before[aid]
            if args.seed_tool_policy == "basic":
                agent["tools"] = original["agents"][aid].get("tools", [])
        if back != original:
            raise AssertionError(f"unintended adaptation in {path.name}")
        out = args.destination / path.name
        out.write_text(yaml.safe_dump(adapted, sort_keys=False), encoding="utf-8")
        provenance["configs"][path.name] = {
            "upstream_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "adapted_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "agent_count": len(adapted["agents"]),
        }
    (args.destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
