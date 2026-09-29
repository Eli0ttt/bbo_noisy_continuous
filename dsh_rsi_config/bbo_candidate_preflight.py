#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import inspect
import math
import sys
import time
from pathlib import Path

# The preflight must never contaminate the task tree with .pyc/__pycache__.
sys.dont_write_bytecode = True

import numpy as np


def fail(message: str) -> None:
    print(
        "CANDIDATE_INTERFACE_PREFLIGHT_FAIL " + message,
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(2)


def load_optimizer(path: Path):
    spec = importlib.util.spec_from_file_location(
        "bbo_candidate_preflight_solver",
        path,
    )
    if spec is None or spec.loader is None:
        fail(f"cannot import candidate from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    optimizer = getattr(module, "Optimizer", None)
    if optimizer is None:
        fail("solver.py does not define Optimizer")

    return optimizer


def safe_batch(opt, remaining: int) -> int:
    raw = getattr(opt, "batch", 1)
    if isinstance(raw, bool):
        fail("batch must not be bool")

    if isinstance(raw, (int, np.integer)):
        value = int(raw)
    elif isinstance(raw, float) and math.isfinite(raw) and raw.is_integer():
        value = int(raw)
    else:
        fail(f"batch must be an integer-like value, got {raw!r}")

    if value < 1:
        fail(f"batch must be positive, got {value}")

    return min(value, remaining)


def tell_once(opt, X: np.ndarray, y: np.ndarray) -> None:
    tell = opt.tell
    try:
        sig = inspect.signature(tell)
    except (TypeError, ValueError) as exc:
        fail(f"unable to inspect tell signature: {exc}")

    try:
        sig.bind(X, y, None)
    except TypeError:
        try:
            sig.bind(X, y)
        except TypeError as exc:
            fail(f"incompatible tell signature: {exc}")
        tell(X, y)
    else:
        tell(X, y, None)


def main() -> int:
    if len(sys.argv) != 2:
        fail("usage: bbo_candidate_preflight.py /path/to/solver.py")

    path = Path(sys.argv[1])
    if not path.is_file() or path.is_symlink():
        fail(f"candidate path is not a regular file: {path}")

    started = time.monotonic()
    Optimizer = load_optimizer(path)

    dim = 10
    budget = 120
    lower = np.full(dim, -5.0, dtype=float)
    upper = np.full(dim, 5.0, dtype=float)

    for run_index, seed in enumerate((20260921, 314159), start=1):
        rng = np.random.default_rng(seed)
        try:
            opt = Optimizer(
                dim=dim,
                lower=lower.copy(),
                upper=upper.copy(),
                budget=budget,
                seed=seed,
                rng=rng,
            )
        except Exception as exc:
            fail(
                f"run={run_index} constructor raised "
                f"{type(exc).__name__}: {exc}"
            )

        used = 0
        calls = 0

        while used < budget:
            requested = safe_batch(opt, budget - used)
            try:
                X = np.asarray(opt.ask(requested), dtype=float)
            except Exception as exc:
                fail(
                    f"run={run_index} ask#{calls + 1} raised "
                    f"{type(exc).__name__}: {exc}"
                )

            if X.ndim != 2:
                fail(
                    f"run={run_index} ask#{calls + 1} returned ndim={X.ndim}, expected 2"
                )

            rows = int(X.shape[0])
            if rows < 1 or rows > requested or X.shape[1] != dim:
                fail(
                    f"run={run_index} ask#{calls + 1} shape={X.shape}, "
                    f"expected [1..{requested}, {dim}]"
                )

            if not np.isfinite(X).all():
                fail(
                    f"run={run_index} ask#{calls + 1} returned non-finite values"
                )

            if np.any(X < lower) or np.any(X > upper):
                fail(
                    f"run={run_index} ask#{calls + 1} returned out-of-bounds values"
                )

            y = (
                np.sum(X * X, axis=1)
                + 0.05 * np.sin(np.sum(X, axis=1))
                + 0.001 * np.arange(rows, dtype=float)
            )

            try:
                tell_once(opt, X, y)
            except SystemExit:
                raise
            except Exception as exc:
                fail(
                    f"run={run_index} tell#{calls + 1} raised "
                    f"{type(exc).__name__}: {exc}"
                )

            used += rows
            calls += 1

            if calls > budget:
                fail(
                    f"run={run_index} exceeded {budget} ask/tell calls without consuming budget"
                )

    elapsed = time.monotonic() - started
    if elapsed > 5.0:
        fail(
            f"local interface preflight took {elapsed:.3f}s; "
            "candidate is too slow for a tiny smoke test"
        )

    print(
        "CANDIDATE_INTERFACE_PREFLIGHT_PASS "
        f"elapsed_sec={elapsed:.6f} runs=2 budget_per_run={budget}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
