#!/usr/bin/env python3
"""Add a small BBO model-selection section to the existing private env file."""
from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

from native_bbo_runtime import canonical_model_id, parse_model_pool

START = "# BEGIN EVOMAS_BBO_MODEL_SELECTION"
END = "# END EVOMAS_BBO_MODEL_SELECTION"


def update_env(path: Path, meta: str, pool: str) -> bool:
    canonical_model_id(meta)
    parse_model_pool(pool)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected a regular private env file: {path}")
    original = path.read_text(encoding="utf-8")
    if original.count(START) != original.count(END) or original.count(START) > 1:
        raise ValueError("Malformed existing BBO model-selection block")
    if START in original:
        left, remainder = original.split(START, 1)
        _, right = remainder.split(END, 1)
        original_without_block = left + right
    else:
        original_without_block = original
    lines = [line for line in original_without_block.splitlines()
             if not re.match(r"\s*(?:export\s+)?BBO_(?:META_MODEL|MODEL_POOL)\s*=", line)]
    new_text = "\n".join(lines).rstrip("\n") + (
        "\n\n" + START + "\n"
        + f"export BBO_META_MODEL={meta}\n"
        + f"export BBO_MODEL_POOL={pool}\n"
        + END + "\n"
    )
    if new_text == path.read_text(encoding="utf-8"):
        return False
    fd, temporary = tempfile.mkstemp(prefix=".bbo-env-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(new_text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path("env/dsh_bbo_experiment.env"))
    parser.add_argument("--meta-model", default="deepseek-v4-flash-s1")
    parser.add_argument("--model-pool", default="deepseek-v4-flash-s1,deepseek-v4-flash-s2")
    args = parser.parse_args()
    changed = update_env(args.env_file, args.meta_model, args.model_pool)
    print("BBO model selection " + ("updated" if changed else "already current")
          + "; existing API keys preserved and not displayed")


if __name__ == "__main__":
    main()
