#!/usr/bin/env python3
"""Stage obsolete EvoMAS BBO artifacts outside the git checkout; preview by default."""
import argparse
import shutil
from pathlib import Path

KEEP_RUN = "evomas_formal_s2_20261022_160042"
PATTERNS = ("*smoke*", "*patch*", "*fix*", "recover_*", "resume_*")
ROOT_OBSOLETE = ("cleanup_preview.py", "install_evomas_checkpoint_patch.py")


def stage(root: Path, archive: Path, apply: bool):
    root = root.resolve()
    base = root / "repro/work/EvoMAS-bbo"
    outputs, adapter = base / "outputs", base / "adapter"
    assert (root / ".git").exists(), "Not a git worktree root"
    assert outputs.is_dir() and not outputs.is_symlink()
    assert adapter.is_dir() and not adapter.is_symlink()
    assert (outputs / KEEP_RUN / "summary.json").is_file(), "Formal run summary missing"
    assert not (outputs / KEEP_RUN).is_symlink()

    archive = archive.resolve()
    assert not archive.is_relative_to(root), "Archive must be outside repo"
    assert not archive.exists(), "Archive path already exists"
    assert not any(p.is_symlink() for p in (root, base, outputs, adapter))

    targets = [
        p for p in outputs.iterdir()
        if p.name != KEEP_RUN and not p.name.startswith(KEEP_RUN + ".")
    ]
    targets.extend(
        p for p in adapter.iterdir()
        if any(p.match(pattern) for pattern in PATTERNS)
    )
    targets.extend(
        root / name for name in ROOT_OBSOLETE
        if (root / name).exists()
    )

    for p in sorted(targets):
        assert p.is_relative_to(root)
        assert not p.is_symlink(), f"Review symlink manually: {p}"
        rel = p.relative_to(root)
        print(f'{"MOVE" if apply else "WOULD MOVE"} {rel}')
        if apply:
            dst = archive / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst))
    print(f"archive: {archive}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    stage(args.root, args.archive, args.apply)