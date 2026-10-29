#!/usr/bin/env python3
"""No API or Docker: interrupted EvoMAS session keeps its candidate history."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

try:
    import dotenv
except ImportError:
    stub = types.ModuleType("dotenv")
    stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = stub
try:
    import smolagents
except ImportError:
    stub = types.ModuleType("smolagents")
    stub.Tool = type("Tool", (), {})
    sys.modules["smolagents"] = stub

import evomas_bbo_pipeline as pipeline

ROOT = Path(__file__).resolve().parents[4]
ADAPTER = Path(__file__).resolve().parent


def args(out: Path) -> list[str]:
    return [
        str(ADAPTER / "evomas_bbo_pipeline.py"),
        "--evomas-root", str(ROOT / "repro/work/EvoMAS-openai"),
        "--seed-pool", str(ADAPTER.parent / "pool/upstream_bbeh_s2_full_basic"),
        "--instruction", str(ROOT / "instruction.md"),
        "--output-dir", str(out), "--tool-mode", "bbo", "--dry-run",
        "--skip-final-repeat", "--solver-limit", "5", "--seed", "43",
        "--research-time-limit", "6750",
    ]


def run_one(out: Path, interrupt_label: str, partial: bool):
    original = pipeline.evaluate_config
    def interrupted(*values, **kwargs):
        if values[1].startswith(interrupt_label):
            if partial:
                (values[2] / (values[1] + ".yaml")).write_text(values[0])
            raise RuntimeError("simulated process loss")
        return original(*values, **kwargs)

    with patch.object(pipeline, "evaluate_config", side_effect=interrupted):
        with patch.object(sys, "argv", args(out)):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    pipeline.main()
            except RuntimeError as exc:
                assert "simulated process loss" in str(exc)
            else:
                raise AssertionError("run should have stopped")
    with patch.object(sys, "argv", args(out) + ["--resume-existing"]):
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.main()
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["candidates"]) == 5
    if partial:
        assert summary["candidates"][2]["contract_valid"] is False
        assert (out / "evaluations" / "step_01_mutation" /
                "candidate_solver.py").is_file()
    assert summary["best_visible"]["contract_valid"] is True


with tempfile.TemporaryDirectory() as temporary:
    base = Path(temporary)
    run_one(base / "after_parent", "parent_01", False)
    run_one(base / "during_child", "step_01", True)
print("PASS: same-session resume preserves completed and failed candidates")
