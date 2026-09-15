from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "2.0"
VERSION_RE = re.compile(r"v[0-9]+")
FINAL_STATUSES = {"kept", "reverted", "submitted", "baseline"}
PENDING_STATUSES = {"evaluating", "evaluated", "selfcheck_failed"}


def app_root() -> Path:
    return Path(os.environ.get("BBO_APP_ROOT", "/app")).resolve()


def methods_root() -> Path:
    return Path(os.environ.get("BBO_METHODS_ROOT", str(app_root() / "methods"))).resolve()


def main_dir() -> Path:
    return methods_root() / "main"


def versions_dir() -> Path:
    return methods_root() / "versions"


def state_path() -> Path:
    return methods_root() / "version_checkpoints.json"


def log_path() -> Path:
    return methods_root() / "experiment_log.md"


def lock_path() -> Path:
    return methods_root() / ".version-checkpoint.lock"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def solver_sha(root: Path) -> str:
    solver = root / "solver.py"
    if not solver.is_file():
        raise RuntimeError(f"missing solver.py in {root}")
    return sha256_file(solver)


def version_number(value: str) -> int:
    if not VERSION_RE.fullmatch(value):
        raise ValueError(f"invalid version: {value!r}; expected v<N>")
    return int(value[1:])


def sanitize_cell(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value).replace("\r", " ").replace("\n", " ").replace("|", "¦").strip()
    return text or "-"


def format_score(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.12g}"
    return sanitize_cell(value)


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)


def load_state() -> dict[str, Any]:
    path = state_path()
    if not path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "canonical_version": None,
            "pending_version": None,
            "finalized": False,
            "final_version": None,
            "versions": {},
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("versions"), dict):
        raise RuntimeError(f"invalid checkpoint state: {path}")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"checkpoint state schema mismatch: {value.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )
    return value


def sorted_version_items(state: dict[str, Any]):
    return sorted(
        state.get("versions", {}).items(),
        key=lambda item: version_number(item[0]),
    )


def write_experiment_log(state: dict[str, Any]) -> None:
    lines = [
        "# Experiment Log",
        "",
        "This log is managed by the harness checkpoint state machine. Every recorded",
        "version is snapshotted before its official visible selfcheck. Intermediate",
        "versions must be explicitly kept or reverted before another versioned",
        "evaluation can begin; the final submitted label is determined from the exact",
        "solver artifact left in `/app/methods/main/solver.py` at handoff.",
        "",
        "| Version | Parent | Description | Score | Anytime | Final | Status | Decision note | Solver SHA256 |",
        "|---|---|---|---:|---:|---:|---|---|---|",
    ]
    for version, row in sorted_version_items(state):
        lines.append(
            "| "
            + " | ".join(
                [
                    version,
                    sanitize_cell(row.get("parent")),
                    sanitize_cell(row.get("description")),
                    format_score(row.get("score")),
                    format_score(row.get("score_anytime")),
                    format_score(row.get("score_final")),
                    sanitize_cell(row.get("status")),
                    sanitize_cell(row.get("decision_note")),
                    sanitize_cell(row.get("solver_sha256")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## State",
            "",
            f"- Canonical version: `{sanitize_cell(state.get('canonical_version'))}`",
            f"- Pending version: `{sanitize_cell(state.get('pending_version'))}`",
            f"- Finalized: `{bool(state.get('finalized'))}`",
            f"- Final version: `{sanitize_cell(state.get('final_version'))}`",
            "",
        ]
    )
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)


def save_state(state: dict[str, Any]) -> None:
    state["schema_version"] = SCHEMA_VERSION
    state["updated_at"] = utc_now()
    atomic_json_write(state_path(), state)
    write_experiment_log(state)


@contextlib.contextmanager
def locked_state():
    methods_root().mkdir(parents=True, exist_ok=True)
    with lock_path().open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = load_state()
        yield state
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def snapshot_main(version: str) -> tuple[Path, str]:
    version_number(version)
    src = main_dir()
    if not src.is_dir() or not (src / "solver.py").is_file():
        raise RuntimeError(f"invalid main method directory: {src}")
    dst = versions_dir() / version
    if dst.exists():
        raise RuntimeError(
            f"refusing to overwrite existing version snapshot {dst}; version ids are immutable"
        )
    versions_dir().mkdir(parents=True, exist_ok=True)
    tmp = versions_dir() / f".{version}.tmp-{uuid.uuid4().hex}"
    tmp.mkdir(parents=False, exist_ok=False)
    shutil.copy2(src / "solver.py", tmp / "solver.py")
    digest = solver_sha(tmp)
    os.replace(tmp, dst)
    return dst, digest


def restore_snapshot(version: str) -> str:
    version_number(version)
    src = versions_dir() / version / "solver.py"
    if not src.is_file():
        raise RuntimeError(f"snapshot does not exist: {src}")
    dst = main_dir()
    dst.mkdir(parents=True, exist_ok=True)
    fresh = dst / f".solver.restore-{uuid.uuid4().hex}.tmp"
    shutil.copy2(src, fresh)
    os.replace(fresh, dst / "solver.py")
    return solver_sha(dst)


def extract_metric_payload(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    best: dict[str, Any] | None = None
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "score" in value and "metric" in value:
            best = value
    return best


def ensure_not_finalized(state: dict[str, Any]) -> None:
    if state.get("finalized"):
        raise RuntimeError(
            f"research is already finalized as {state.get('final_version')}; no further version transitions are allowed"
        )


def init_baseline(_: argparse.Namespace) -> int:
    with locked_state() as state:
        versions = state.setdefault("versions", {})
        if "v0" not in versions:
            dst = versions_dir() / "v0"
            if dst.exists():
                if not (dst / "solver.py").is_file():
                    raise RuntimeError("existing v0 directory has no solver.py")
                digest = solver_sha(dst)
            else:
                _, digest = snapshot_main("v0")
            versions["v0"] = {
                "version": "v0",
                "parent": None,
                "description": "Shipped baseline",
                "status": "baseline",
                "score": None,
                "score_anytime": None,
                "score_final": None,
                "decision_note": None,
                "solver_sha256": digest,
                "created_at": utc_now(),
                "source": "checkpoint_state_machine",
            }
        state["canonical_version"] = state.get("canonical_version") or "v0"
        state["pending_version"] = state.get("pending_version")
        state["finalized"] = bool(state.get("finalized", False))
        state["final_version"] = state.get("final_version")
        save_state(state)
    print("VERSION_CHECKPOINT_INIT version=v0 canonical=v0")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    version = args.version
    version_number(version)
    if version == "v0":
        raise RuntimeError("v0 is reserved for the shipped baseline")

    with locked_state() as state:
        ensure_not_finalized(state)
        versions = state.setdefault("versions", {})
        pending = state.get("pending_version")
        if pending:
            raise RuntimeError(
                f"cannot evaluate {version}: {pending} is still pending; "
                f"run `keep --version {pending}` or `revert --version {pending}` first"
            )
        if version in versions:
            raise RuntimeError(f"version {version} already exists; version ids are immutable")

        existing_numbers = [version_number(v) for v in versions if VERSION_RE.fullmatch(v)]
        if existing_numbers and version_number(version) <= max(existing_numbers):
            raise RuntimeError(
                f"version ids must increase monotonically; {version} is not newer than "
                f"v{max(existing_numbers)}"
            )

        parent = state.get("canonical_version")
        if not isinstance(parent, str) or parent not in versions:
            raise RuntimeError(f"invalid canonical parent in checkpoint state: {parent!r}")

        _, digest = snapshot_main(version)
        versions[version] = {
            "version": version,
            "parent": parent,
            "description": args.description,
            "status": "evaluating",
            "score": None,
            "score_anytime": None,
            "score_final": None,
            "decision_note": None,
            "solver_sha256": digest,
            "created_at": utc_now(),
            "source": "checkpoint_state_machine",
            "selfcheck": {
                "command": [sys.executable, str(app_root() / "selfcheck.py"), "--json"],
                "timeout_sec": args.timeout,
            },
        }
        # The parent remains canonical until the agent explicitly keeps or
        # reverts this candidate.  The candidate itself is the sole pending
        # transition.
        state["pending_version"] = version
        save_state(state)

    command = [sys.executable, str(app_root() / "selfcheck.py"), "--json"]
    print(f"VERSION_SELFCHECK_START version={version}", flush=True)
    timed_out = False
    try:
        proc = subprocess.run(
            command,
            cwd=str(app_root()),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
            check=False,
        )
        rc = int(proc.returncode)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = 124
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stderr += f"\n[version-checkpoint] selfcheck timed out after {args.timeout} seconds\n"

    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")

    payload = extract_metric_payload(stdout)
    with locked_state() as state:
        row = state["versions"][version]
        row["evaluated_at"] = utc_now()
        row["selfcheck"]["return_code"] = rc
        row["selfcheck"]["timed_out"] = timed_out
        if rc == 0 and payload is not None:
            row["status"] = "evaluated"
            row["score"] = payload.get("score")
            row["score_anytime"] = payload.get("score_anytime")
            row["score_final"] = payload.get("score_final")
            row["selfcheck"]["metric"] = payload.get("metric")
        else:
            row["status"] = "selfcheck_failed"
            row["selfcheck"]["error"] = (
                "timeout" if timed_out else "nonzero_exit" if rc != 0 else "missing_score_payload"
            )
        save_state(state)
        score = row.get("score")
        status = row.get("status")
        digest = row.get("solver_sha256")
        parent = row.get("parent")

    print(
        f"VERSION_CHECKPOINT version={version} parent={parent} score={score} "
        f"status={status} solver_sha256={digest}"
    )
    print(
        f"VERSION_DECISION_REQUIRED version={version} choices=keep,revert "
        f"before_next_version=1"
    )
    if rc == 0 and payload is None:
        return 65
    return rc


def keep(args: argparse.Namespace) -> int:
    version_number(args.version)
    with locked_state() as state:
        ensure_not_finalized(state)
        pending = state.get("pending_version")
        if pending != args.version:
            raise RuntimeError(
                f"cannot keep {args.version}: pending version is {pending!r}"
            )
        row = state.get("versions", {}).get(args.version)
        if not isinstance(row, dict):
            raise RuntimeError(f"unknown version {args.version}")
        if row.get("status") not in {"evaluated", "selfcheck_failed"}:
            raise RuntimeError(
                f"cannot keep {args.version} from status {row.get('status')!r}"
            )
        row["status"] = "kept"
        row["decided_at"] = utc_now()
        row["decision_note"] = args.note or "kept by agent"
        state["canonical_version"] = args.version
        state["pending_version"] = None
        save_state(state)
    print(f"VERSION_DECISION version={args.version} status=kept canonical={args.version}")
    return 0


def revert(args: argparse.Namespace) -> int:
    version_number(args.version)
    with locked_state() as state:
        ensure_not_finalized(state)
        pending = state.get("pending_version")
        if pending != args.version:
            raise RuntimeError(
                f"cannot revert {args.version}: pending version is {pending!r}"
            )
        row = state.get("versions", {}).get(args.version)
        if not isinstance(row, dict):
            raise RuntimeError(f"unknown version {args.version}")
        target = args.to or row.get("parent")
        if not isinstance(target, str) or target not in state.get("versions", {}):
            raise RuntimeError(f"invalid revert target for {args.version}: {target!r}")
        restored_sha = restore_snapshot(target)
        row["status"] = "reverted"
        row["decided_at"] = utc_now()
        row["reverted_to"] = target
        row["decision_note"] = args.note or f"reverted by agent to {target}"
        state["canonical_version"] = target
        state["pending_version"] = None
        save_state(state)
    print(
        f"VERSION_DECISION version={args.version} status=reverted "
        f"canonical={target} solver_sha256={restored_sha}"
    )
    return 0


def checkout(args: argparse.Namespace) -> int:
    version_number(args.version)
    with locked_state() as state:
        ensure_not_finalized(state)
        pending = state.get("pending_version")
        if pending:
            raise RuntimeError(
                f"cannot checkout {args.version}: {pending} is pending; keep or revert it first"
            )
        if args.version not in state.get("versions", {}):
            raise RuntimeError(f"unknown version {args.version}")
        restored_sha = restore_snapshot(args.version)
        state["canonical_version"] = args.version
        state["last_checkout"] = {
            "version": args.version,
            "at": utc_now(),
        }
        save_state(state)
    print(
        f"VERSION_CHECKOUT version={args.version} canonical={args.version} "
        f"solver_sha256={restored_sha}"
    )
    return 0


def parse_logged_versions(path: Path) -> list[str]:
    if not path.is_file():
        return []
    result: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and VERSION_RE.fullmatch(cells[0]) and cells[0] not in result:
            result.append(cells[0])
    return result


def audit_payload() -> dict[str, Any]:
    state = load_state()
    state_versions = [v for v, _ in sorted_version_items(state)]
    state_candidates = [v for v in state_versions if v != "v0"]
    logged_versions = parse_logged_versions(log_path())
    logged_candidates = [v for v in logged_versions if v != "v0"]

    all_dirs = (
        sorted(
            [p.name for p in versions_dir().iterdir() if p.is_dir()],
            key=lambda x: version_number(x) if VERSION_RE.fullmatch(x) else 10**12,
        )
        if versions_dir().is_dir()
        else []
    )
    canonical_dirs = [name for name in all_dirs if VERSION_RE.fullmatch(name)]
    noncanonical_dirs = [name for name in all_dirs if name not in canonical_dirs]

    valid_snapshots: list[str] = []
    incomplete_snapshots: list[str] = []
    hash_mismatches: list[str] = []
    for name in canonical_dirs:
        solver = versions_dir() / name / "solver.py"
        if not solver.is_file():
            incomplete_snapshots.append(name)
            continue
        valid_snapshots.append(name)
        expected = state.get("versions", {}).get(name, {}).get("solver_sha256")
        if expected and sha256_file(solver) != expected:
            hash_mismatches.append(name)

    valid_candidates = [v for v in valid_snapshots if v != "v0"]
    missing_snapshot_versions = [v for v in logged_candidates if v not in valid_candidates]
    missing_manifest_versions = [v for v in logged_candidates if v not in state_candidates]
    extra_manifest_versions = [v for v in state_candidates if v not in logged_candidates]
    extra_snapshot_versions = [v for v in valid_candidates if v not in logged_candidates]

    undecided_versions = [
        v
        for v in state_candidates
        if state.get("versions", {}).get(v, {}).get("status") in PENDING_STATUSES
    ]
    missing_parent_versions = [
        v
        for v in state_candidates
        if (
            not isinstance(state["versions"].get(v, {}).get("parent"), str)
            or state["versions"][v]["parent"] not in state.get("versions", {})
        )
    ]

    final_solver = main_dir() / "solver.py"
    final_solver_sha256 = sha256_file(final_solver) if final_solver.is_file() else None
    matching_final_versions = (
        [
            v
            for v in state_versions
            if state.get("versions", {}).get(v, {}).get("solver_sha256") == final_solver_sha256
        ]
        if final_solver_sha256
        else []
    )
    submitted_versions = [
        v
        for v in state_versions
        if state.get("versions", {}).get(v, {}).get("status") == "submitted"
    ]
    submitted_matches_final = (
        len(submitted_versions) == 1 and submitted_versions[0] in matching_final_versions
    )

    snapshot_complete = (
        not missing_snapshot_versions
        and not missing_manifest_versions
        and not extra_manifest_versions
        and not extra_snapshot_versions
        and not hash_mismatches
        and not [v for v in incomplete_snapshots if v != "v0"]
        and not noncanonical_dirs
    )
    decision_complete = not undecided_versions and state.get("pending_version") is None
    lineage_complete = not missing_parent_versions

    guard_failure_reasons: list[str] = []
    if missing_snapshot_versions:
        guard_failure_reasons.append(
            "missing_snapshot_versions=" + ",".join(missing_snapshot_versions)
        )
    if missing_manifest_versions:
        guard_failure_reasons.append(
            "missing_manifest_versions=" + ",".join(missing_manifest_versions)
        )
    if extra_manifest_versions:
        guard_failure_reasons.append(
            "extra_manifest_versions=" + ",".join(extra_manifest_versions)
        )
    if extra_snapshot_versions:
        guard_failure_reasons.append(
            "extra_snapshot_versions=" + ",".join(extra_snapshot_versions)
        )
    if hash_mismatches:
        guard_failure_reasons.append(
            "snapshot_hash_mismatches=" + ",".join(hash_mismatches)
        )
    incomplete_candidates = [v for v in incomplete_snapshots if v != "v0"]
    if incomplete_candidates:
        guard_failure_reasons.append(
            "incomplete_snapshot_versions=" + ",".join(incomplete_candidates)
        )
    if noncanonical_dirs:
        guard_failure_reasons.append(
            "noncanonical_snapshot_dirs=" + ",".join(noncanonical_dirs)
        )
    if undecided_versions:
        guard_failure_reasons.append(
            "undecided_versions=" + ",".join(undecided_versions)
        )
    if state.get("pending_version") is not None:
        guard_failure_reasons.append(
            "pending_version=" + str(state.get("pending_version"))
        )
    if missing_parent_versions:
        guard_failure_reasons.append(
            "missing_parent_versions=" + ",".join(missing_parent_versions)
        )
    if not state.get("finalized"):
        guard_failure_reasons.append("state_not_finalized")
    if not submitted_matches_final:
        guard_failure_reasons.append(
            "submitted_final_mismatch="
            + f"submitted:{','.join(submitted_versions) or '<none>'};"
            + f"matching_final:{','.join(matching_final_versions) or '<none>'}"
        )

    checkpoint_guard_complete = (
        snapshot_complete
        and decision_complete
        and lineage_complete
        and bool(state.get("finalized"))
        and submitted_matches_final
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_guard_present": state_path().is_file(),
        "experiment_log_exists": log_path().is_file(),
        "experiment_log_nonempty": log_path().is_file() and log_path().stat().st_size > 0,
        "canonical_version": state.get("canonical_version"),
        "pending_version": state.get("pending_version"),
        "finalized": bool(state.get("finalized")),
        "final_version": state.get("final_version"),
        "logged_versions": logged_versions,
        "logged_candidate_versions": logged_candidates,
        "logged_version_count": len(logged_candidates),
        "manifest_versions": state_versions,
        "manifest_candidate_versions": state_candidates,
        "manifest_version_count": len(state_candidates),
        "snapshot_dirs": all_dirs,
        "valid_snapshot_versions": valid_snapshots,
        "candidate_snapshot_versions": valid_candidates,
        "snapshot_version_count": len(valid_candidates),
        "baseline_snapshot_present": "v0" in valid_snapshots,
        "incomplete_snapshot_versions": incomplete_snapshots,
        "noncanonical_snapshot_dirs": noncanonical_dirs,
        "missing_snapshot_versions": missing_snapshot_versions,
        "missing_manifest_versions": missing_manifest_versions,
        "extra_manifest_versions": extra_manifest_versions,
        "extra_snapshot_versions": extra_snapshot_versions,
        "snapshot_hash_mismatches": hash_mismatches,
        "missing_parent_versions": missing_parent_versions,
        "lineage_complete": lineage_complete,
        "undecided_versions": undecided_versions,
        "decision_complete": decision_complete,
        "final_solver_sha256": final_solver_sha256,
        "matching_final_versions": matching_final_versions,
        "submitted_versions": submitted_versions,
        "submitted_matches_final": submitted_matches_final,
        "snapshot_complete": snapshot_complete,
        "guard_failure_reasons": guard_failure_reasons,
        "checkpoint_guard_complete": checkpoint_guard_complete,
    }


def finalize(args: argparse.Namespace) -> int:
    error: str | None = None
    try:
        with locked_state() as state:
            versions = state.setdefault("versions", {})
            if state.get("finalized"):
                # Idempotent finalization: do not alter an already frozen run.
                pass
            else:
                final_solver = main_dir() / "solver.py"
                if not final_solver.is_file():
                    raise RuntimeError("final /app/methods/main/solver.py is missing")
                final_sha = sha256_file(final_solver)
                pending = state.get("pending_version")
                canonical = state.get("canonical_version")

                if pending:
                    pending_row = versions.get(pending)
                    canonical_row = versions.get(canonical) if isinstance(canonical, str) else None
                    if not isinstance(pending_row, dict):
                        raise RuntimeError(f"pending version {pending!r} is missing from manifest")
                    if final_sha == pending_row.get("solver_sha256"):
                        pending_row["previous_status"] = pending_row.get("status")
                        pending_row["status"] = "submitted"
                        pending_row["decided_at"] = utc_now()
                        pending_row["decision_note"] = (
                            "submitted deterministically at handoff because the final solver "
                            "exactly matched this pending evaluated checkpoint"
                        )
                        state["canonical_version"] = pending
                        state["pending_version"] = None
                        final_version = pending
                    elif isinstance(canonical_row, dict) and final_sha == canonical_row.get("solver_sha256"):
                        pending_row["previous_status"] = pending_row.get("status")
                        pending_row["status"] = "reverted"
                        pending_row["decided_at"] = utc_now()
                        pending_row["reverted_to"] = canonical
                        pending_row["decision_note"] = (
                            "reverted deterministically at handoff because the final solver "
                            "matched the canonical parent rather than this pending checkpoint"
                        )
                        state["pending_version"] = None
                        canonical_row["previous_status"] = canonical_row.get("status")
                        canonical_row["status"] = "submitted"
                        canonical_row["decided_at"] = utc_now()
                        canonical_row["decision_note"] = (
                            "submitted deterministically at handoff from final solver identity"
                        )
                        final_version = canonical
                    else:
                        raise RuntimeError(
                            "uncheckpointed final artifact: while "
                            f"{pending} was pending, final solver matched neither pending "
                            f"{pending} nor canonical parent {canonical}"
                        )
                else:
                    if not isinstance(canonical, str) or canonical not in versions:
                        raise RuntimeError(f"invalid canonical version at handoff: {canonical!r}")
                    canonical_row = versions[canonical]
                    if final_sha != canonical_row.get("solver_sha256"):
                        raise RuntimeError(
                            "uncheckpointed final artifact: final solver does not match "
                            f"canonical checkpoint {canonical}; evaluate the final candidate "
                            "before finishing"
                        )
                    canonical_row["previous_status"] = canonical_row.get("status")
                    canonical_row["status"] = "submitted"
                    canonical_row["decided_at"] = utc_now()
                    canonical_row["decision_note"] = (
                        "submitted deterministically at handoff from final solver identity"
                    )
                    final_version = canonical

                # There must never be multiple submitted labels.
                for version, row in versions.items():
                    if version != final_version and row.get("status") == "submitted":
                        row["status"] = "kept"
                        row["decision_note"] = (
                            "previous submitted label superseded before final handoff"
                        )
                state["finalized"] = True
                state["finalized_at"] = utc_now()
                state["final_version"] = final_version
                save_state(state)
    except Exception as exc:
        error = str(exc)

    payload = audit_payload()
    if error:
        payload["finalization_error"] = error
        reasons = list(payload.get("guard_failure_reasons") or [])
        reasons.insert(0, "finalization_error=" + error)
        payload["guard_failure_reasons"] = reasons
        payload["checkpoint_guard_complete"] = False

    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("checkpoint_guard_complete") else 2


def audit(args: argparse.Namespace) -> int:
    payload = audit_payload()
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["checkpoint_guard_complete"] else 2


def status(args: argparse.Namespace) -> int:
    state = load_state()
    payload = {
        "canonical_version": state.get("canonical_version"),
        "pending_version": state.get("pending_version"),
        "finalized": bool(state.get("finalized")),
        "final_version": state.get("final_version"),
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transactional version checkpoint state machine for RSI-Exam BBO"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser(
        "init", help="snapshot the shipped v0 baseline and initialize the state machine"
    )
    p_init.set_defaults(func=init_baseline)

    p_eval = sub.add_parser(
        "evaluate",
        help="snapshot one candidate, then run the unmodified official visible selfcheck",
    )
    p_eval.add_argument("--version", required=True)
    p_eval.add_argument("--description", required=True)
    p_eval.add_argument("--timeout", type=float, default=None)
    p_eval.set_defaults(func=evaluate)

    p_keep = sub.add_parser(
        "keep",
        help="accept the sole pending version as the canonical parent for future experiments",
    )
    p_keep.add_argument("--version", required=True)
    p_keep.add_argument("--note")
    p_keep.set_defaults(func=keep)

    p_revert = sub.add_parser(
        "revert",
        help="reject the sole pending version and atomically restore its parent or another checkpoint",
    )
    p_revert.add_argument("--version", required=True)
    p_revert.add_argument("--to")
    p_revert.add_argument("--note")
    p_revert.set_defaults(func=revert)

    p_checkout = sub.add_parser(
        "checkout",
        help="restore an existing checkpoint for a new branch; no pending version may exist",
    )
    p_checkout.add_argument("--version", required=True)
    p_checkout.set_defaults(func=checkout)

    p_finalize = sub.add_parser(
        "finalize",
        help="freeze the final artifact and assign submitted/reverted labels from exact file identity",
    )
    p_finalize.add_argument("--json", action="store_true")
    p_finalize.set_defaults(func=finalize)

    p_audit = sub.add_parser(
        "audit", help="verify log, manifest, decisions, immutable snapshots, and final artifact agree"
    )
    p_audit.add_argument("--json", action="store_true")
    p_audit.set_defaults(func=audit)

    p_status = sub.add_parser("status", help="show canonical/pending/final state")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=status)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"VERSION_CHECKPOINT_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
