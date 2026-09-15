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

SCHEMA_VERSION = "1.0"
VERSION_RE = re.compile(r"v[0-9]+")
ALLOWED_DECISIONS = {"kept", "reverted", "submitted"}


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


def load_state() -> dict[str, Any]:
    path = state_path()
    if not path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "versions": {},
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("versions"), dict):
        raise RuntimeError(f"invalid checkpoint state: {path}")
    return value


def version_number(value: str) -> int:
    if not VERSION_RE.fullmatch(value):
        raise ValueError(f"invalid version: {value!r}; expected v<N>")
    return int(value[1:])


def sorted_version_items(state: dict[str, Any]):
    items = list(state.get("versions", {}).items())
    return sorted(items, key=lambda item: version_number(item[0]))


def write_experiment_log(state: dict[str, Any]) -> None:
    lines = [
        "# Experiment Log",
        "",
        "This table is managed by the harness version-checkpoint guard so every recorded",
        "version has an immutable solver snapshot before the agent can continue editing.",
        "",
        "| Version | Parent | Description | Score | Anytime | Final | Status | Solver SHA256 |",
        "|---|---|---|---:|---:|---:|---|---|",
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
                    sanitize_cell(row.get("solver_sha256")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "Use `/opt/dsh-config/version_checkpoint.py decide` to update keep/revert/submitted status.",
            "Do not bulk-rewrite this version table manually.",
            "",
        ]
    )
    path = log_path()
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
                "solver_sha256": digest,
                "created_at": utc_now(),
                "source": "checkpoint_guard",
            }
        save_state(state)
    print("VERSION_CHECKPOINT_INIT version=v0")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    version = args.version
    version_number(version)
    if version == "v0":
        raise RuntimeError("v0 is reserved for the shipped baseline")
    parent = args.parent
    if parent is not None:
        version_number(parent)

    # Snapshot *before* launching selfcheck. The helper command is synchronous,
    # so the model cannot edit main/ between snapshot and evaluation.
    with locked_state() as state:
        versions = state.setdefault("versions", {})
        if version in versions:
            raise RuntimeError(f"version {version} already exists; version ids are immutable")
        if parent is not None and parent not in versions:
            raise RuntimeError(f"parent {parent} has not been checkpointed")
        _, digest = snapshot_main(version)
        versions[version] = {
            "version": version,
            "parent": parent,
            "description": args.description,
            "status": "evaluating",
            "score": None,
            "score_anytime": None,
            "score_final": None,
            "solver_sha256": digest,
            "created_at": utc_now(),
            "source": "checkpoint_guard",
            "selfcheck": {
                "command": [sys.executable, str(app_root() / "selfcheck.py"), "--json"],
                "timeout_sec": args.timeout,
            },
        }
        save_state(state)

    command = [sys.executable, str(app_root() / "selfcheck.py"), "--json"]
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

    print(
        f"VERSION_CHECKPOINT version={version} score={score} status={status} "
        f"solver_sha256={digest}"
    )
    if rc == 0 and payload is None:
        return 65
    return rc


def decide(args: argparse.Namespace) -> int:
    version_number(args.version)
    if args.status not in ALLOWED_DECISIONS:
        raise RuntimeError(f"status must be one of {sorted(ALLOWED_DECISIONS)}")
    with locked_state() as state:
        row = state.get("versions", {}).get(args.version)
        if not isinstance(row, dict):
            raise RuntimeError(f"unknown version {args.version}; evaluate it before deciding")
        if args.version == "v0" and args.status != "submitted":
            raise RuntimeError("v0 may only be selected as the final submitted baseline")
        if args.status == "submitted":
            for other_version, other_row in state.get("versions", {}).items():
                if other_version != args.version and isinstance(other_row, dict) and other_row.get("status") == "submitted":
                    other_row["status"] = "kept"
                    other_row["decision_note"] = "superseded by a later submitted version"
                    other_row["decided_at"] = utc_now()
        row["status"] = args.status
        row["decided_at"] = utc_now()
        if args.note:
            row["decision_note"] = args.note
        save_state(state)
    print(f"VERSION_DECISION version={args.version} status={args.status}")
    return 0


def restore(args: argparse.Namespace) -> int:
    version_number(args.version)
    src = versions_dir() / args.version
    if not (src / "solver.py").is_file():
        raise RuntimeError(f"snapshot does not exist: {src}")
    dst = main_dir()
    dst.mkdir(parents=True, exist_ok=True)
    fresh = dst / f".solver.restore-{uuid.uuid4().hex}.tmp"
    shutil.copy2(src / "solver.py", fresh)
    os.replace(fresh, dst / "solver.py")
    print(f"VERSION_RESTORE version={args.version} solver_sha256={solver_sha(dst)}")
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

    all_dirs = sorted(
        [p.name for p in versions_dir().iterdir() if p.is_dir()],
        key=lambda x: version_number(x) if VERSION_RE.fullmatch(x) else 10**12,
    ) if versions_dir().is_dir() else []
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
        v for v in state_candidates
        if state["versions"].get(v, {}).get("status") in {"evaluating", "evaluated"}
    ]
    final_solver = main_dir() / "solver.py"
    final_solver_sha256 = sha256_file(final_solver) if final_solver.is_file() else None
    matching_final_versions = [
        v for v in state_versions
        if state["versions"].get(v, {}).get("solver_sha256") == final_solver_sha256
    ] if final_solver_sha256 else []
    submitted_versions = [
        v for v in state_versions
        if state["versions"].get(v, {}).get("status") == "submitted"
    ]
    submitted_matches_final = (
        len(submitted_versions) == 1
        and submitted_versions[0] in matching_final_versions
    )
    guard_complete = (
        not missing_snapshot_versions
        and not missing_manifest_versions
        and not extra_manifest_versions
        and not extra_snapshot_versions
        and not hash_mismatches
        and not [v for v in incomplete_snapshots if v != "v0"]
        and not noncanonical_dirs
        and not undecided_versions
        and submitted_matches_final
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_guard_present": state_path().is_file(),
        "experiment_log_exists": log_path().is_file(),
        "experiment_log_nonempty": log_path().is_file() and log_path().stat().st_size > 0,
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
        "undecided_versions": undecided_versions,
        "decision_complete": not undecided_versions,
        "final_solver_sha256": final_solver_sha256,
        "matching_final_versions": matching_final_versions,
        "submitted_versions": submitted_versions,
        "submitted_matches_final": submitted_matches_final,
        "snapshot_complete": (
            not missing_snapshot_versions
            and not missing_manifest_versions
            and not extra_manifest_versions
            and not extra_snapshot_versions
            and not hash_mismatches
            and not [v for v in incomplete_snapshots if v != "v0"]
            and not noncanonical_dirs
        ),
        "checkpoint_guard_complete": guard_complete,
    }


def audit(args: argparse.Namespace) -> int:
    payload = audit_payload()
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["checkpoint_guard_complete"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Atomic version checkpoint guard for RSI-Exam BBO")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="snapshot the shipped v0 baseline and initialize the log")
    p_init.set_defaults(func=init_baseline)

    p_eval = sub.add_parser("evaluate", help="snapshot a candidate and run the official visible selfcheck")
    p_eval.add_argument("--version", required=True)
    p_eval.add_argument("--parent")
    p_eval.add_argument("--description", required=True)
    p_eval.add_argument("--timeout", type=float, default=None)
    p_eval.set_defaults(func=evaluate)

    p_decide = sub.add_parser("decide", help="record keep/revert/submitted decision")
    p_decide.add_argument("--version", required=True)
    p_decide.add_argument("--status", required=True, choices=sorted(ALLOWED_DECISIONS))
    p_decide.add_argument("--note")
    p_decide.set_defaults(func=decide)

    p_restore = sub.add_parser("restore", help="restore a previously snapshotted version to methods/main")
    p_restore.add_argument("--version", required=True)
    p_restore.set_defaults(func=restore)

    p_audit = sub.add_parser("audit", help="verify experiment log, manifest, and immutable snapshots agree")
    p_audit.add_argument("--json", action="store_true")
    p_audit.set_defaults(func=audit)
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
