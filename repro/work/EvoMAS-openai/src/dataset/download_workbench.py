#!/usr/bin/env python3
"""WorkBench dataset downloader.

Clones the upstream WorkBench repository (olly-styles/WorkBench) into a
temp directory and materializes the files EvoMAS expects under
``dataset/workbench/<domain>/``:

- ``data.csv``       — copied from upstream ``data/processed/<source>.csv``
- ``test.json``      — derived from upstream
                       ``data/processed/queries_and_answers/<domain>_queries_and_answers.csv``

The per-row JSON layout matches the format shipped previously:

    {
        "id": <int>,
        "query": "...",
        "gt": "['domain.func(...)']",
        "tag": ["WorkBench-<CanonicalDomain>"],
        "source": "WorkBench",
        "domains": [...],
        "base_template": "...",
        "chosen_template": "..."
    }

Usage:
    python src/dataset/download_workbench.py --all
    python src/dataset/download_workbench.py --domain email
    python src/dataset/download_workbench.py --all --force
"""

import argparse
import ast
import csv
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

REPO_URL = "https://github.com/olly-styles/WorkBench.git"

# Upstream source-file name for each domain's data.csv. None = no data.csv
# (multi_domain has no dedicated data file; it composes the others).
DATA_CSV_SOURCE: Dict[str, str] = {
    "email": "emails.csv",
    "calendar": "calendar_events.csv",
    "analytics": "analytics_data.csv",
    "customer_relationship_manager": "customer_relationship_manager_data.csv",
    "project_management": "project_tasks.csv",
    "multi_domain": "",
}

# `tag` suffix per domain, matching the previously shipped test.json.
TAG_SUFFIX: Dict[str, str] = {
    "email": "Email",
    "calendar": "Calendar",
    "analytics": "Analytics",
    "customer_relationship_manager": "CRM",
    "project_management": "ProjectManagement",
    "multi_domain": "MultiDomain",
}

DOMAINS = list(DATA_CSV_SOURCE.keys())

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parse_domains_field(raw: str) -> List[str]:
    """Upstream stores the domains list as a Python-literal string, e.g.
    "['email']". Convert to a real list; fall back to [raw] on parse error."""
    try:
        val = ast.literal_eval(raw)
        return list(val) if isinstance(val, (list, tuple)) else [str(val)]
    except Exception:
        return [raw] if raw else []


def _build_test_json(queries_csv: Path, domain: str) -> List[Dict]:
    tag = f"WorkBench-{TAG_SUFFIX[domain]}"
    tasks = []
    with open(queries_csv, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            tasks.append({
                "id": i,
                "query": row["query"],
                "gt": row["answer"],
                "tag": [tag],
                "source": "WorkBench",
                "domains": _parse_domains_field(row.get("domains", "")),
                "base_template": row.get("base_template", ""),
                "chosen_template": row.get("chosen_template", ""),
            })
    return tasks


def _clone_upstream(dest: Path) -> None:
    logger.info(f"Cloning {REPO_URL} -> {dest}")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", REPO_URL, str(dest)],
        check=True,
    )


def _prepare_domain(upstream_root: Path, domain: str, out_dir: Path, force: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # data.csv
    src_name = DATA_CSV_SOURCE[domain]
    if src_name:
        src = upstream_root / "data" / "processed" / src_name
        dst = out_dir / "data.csv"
        if dst.exists() and not force:
            logger.info(f"  [skip] {dst} already present")
        elif src.exists():
            shutil.copy2(src, dst)
            logger.info(f"  [copy] {src.name} -> {dst}")
        else:
            logger.warning(f"  [miss] upstream data file not found: {src}")

    # test.json
    qsrc = upstream_root / "data" / "processed" / "queries_and_answers" / f"{domain}_queries_and_answers.csv"
    jdst = out_dir / "test.json"
    if jdst.exists() and not force:
        logger.info(f"  [skip] {jdst} already present")
        return
    if not qsrc.exists():
        logger.warning(f"  [miss] upstream queries file not found: {qsrc}")
        return
    tasks = _build_test_json(qsrc, domain)
    with open(jdst, "w") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    logger.info(f"  [write] {jdst} ({len(tasks)} tasks)")


def main():
    parser = argparse.ArgumentParser(description="Download and stage the WorkBench dataset.")
    parser.add_argument("--all", action="store_true", help="Prepare every domain")
    parser.add_argument("--domain", choices=DOMAINS, help="Prepare a single domain")
    parser.add_argument("--force", action="store_true", help="Re-download even if outputs exist")
    parser.add_argument("--output-root", default="dataset/workbench",
                        help="Destination root (default: dataset/workbench)")
    args = parser.parse_args()

    if not args.all and not args.domain:
        parser.error("must pass --all or --domain <name>")

    targets = DOMAINS if args.all else [args.domain]
    out_root = Path(args.output_root)

    # Reuse a single upstream clone for all targets
    with tempfile.TemporaryDirectory(prefix="workbench_upstream_") as tmp:
        upstream = Path(tmp) / "WorkBench"
        _clone_upstream(upstream)
        for domain in targets:
            logger.info(f"Domain: {domain}")
            _prepare_domain(upstream, domain, out_root / domain, args.force)

    logger.info(f"Done. Staged under {out_root}/")


if __name__ == "__main__":
    sys.exit(main() or 0)
