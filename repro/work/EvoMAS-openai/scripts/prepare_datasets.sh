#!/usr/bin/env bash
# One-click dataset preparation for EvoMAS. The published repo ships source
# code only; this script downloads and stages every dataset into the paths
# the pipeline expects:
#
#   dataset/bbeh/benchmark_tasks/bbeh_<subset>/test.json
#   dataset/workbench/<domain>/{data.csv,test.json}
#   dataset/swe_bench_{lite,verified}/test.json
#   dataset/repos/<repo_name>/   (cloned at pinned commits for SWE execution)
#
# Each step is idempotent — rerun to resume after a failure or to add what's
# missing. --force re-downloads everything.
#
# Usage:
#   scripts/prepare_datasets.sh                      # all four steps
#   scripts/prepare_datasets.sh --force              # wipe and re-fetch
#   scripts/prepare_datasets.sh --bbeh-only
#   scripts/prepare_datasets.sh --workbench-only
#   scripts/prepare_datasets.sh --swe-only           # SWE-bench task JSONs
#   scripts/prepare_datasets.sh --repos-only         # clone SWE source trees
#   scripts/prepare_datasets.sh --skip-repos         # everything except repo clones

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

FORCE=0
DO_REPOS=1
DO_BBEH=1
DO_SWE=1
DO_WB=1
for arg in "$@"; do
    case "$arg" in
        --force)            FORCE=1 ;;
        --repos-only)       DO_BBEH=0; DO_SWE=0; DO_WB=0 ;;
        --bbeh-only)        DO_REPOS=0; DO_SWE=0; DO_WB=0 ;;
        --swe-only)         DO_REPOS=0; DO_BBEH=0; DO_WB=0 ;;
        --workbench-only)   DO_REPOS=0; DO_BBEH=0; DO_SWE=0 ;;
        --skip-repos)       DO_REPOS=0 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown arg: $arg"; exit 2 ;;
    esac
done

step() { echo; echo "=============================="; echo "$1"; echo "=============================="; }

# ---------------------------------------------------------------------------
# 1. Clone SWE-bench source repositories at pinned commits
# ---------------------------------------------------------------------------
if [[ $DO_REPOS -eq 1 ]]; then
    step "Cloning SWE-bench source repositories into dataset/repos/"
    REPO_LIST="src/dataset/repos.txt"
    TARGET="dataset/repos"
    mkdir -p "$TARGET"

    if [[ ! -f "$REPO_LIST" ]]; then
        echo "ERROR: $REPO_LIST not found" >&2
        exit 1
    fi

    while read -r repo_url commit_hash; do
        [[ -z "$repo_url" || "$repo_url" =~ ^# ]] && continue
        repo_name="$(basename "$repo_url" .git)"
        repo_path="$TARGET/$repo_name"

        if [[ -d "$repo_path/.git" && $FORCE -eq 0 ]]; then
            echo "  [skip] $repo_name already cloned"
            continue
        fi

        if [[ $FORCE -eq 1 && -d "$repo_path" ]]; then
            echo "  [force] removing existing $repo_path"
            rm -rf "$repo_path"
        fi

        echo "  [clone] $repo_url -> $repo_path"
        git clone --quiet "$repo_url" "$repo_path"
        (
            cd "$repo_path"
            git checkout --quiet "$commit_hash"
        )
    done < "$REPO_LIST"
    echo "SWE-bench repos ready under $TARGET/"
fi

# ---------------------------------------------------------------------------
# 2. BBEH task JSONs — upstream google-deepmind/bbeh
# ---------------------------------------------------------------------------
if [[ $DO_BBEH -eq 1 ]]; then
    step "Downloading BBEH task JSONs into dataset/bbeh/benchmark_tasks/"
    BBEH_ROOT="dataset/bbeh/benchmark_tasks"
    need_download=0
    if [[ ! -d "$BBEH_ROOT" ]] || [[ -z "$(ls -A "$BBEH_ROOT" 2>/dev/null)" ]]; then
        need_download=1
    elif [[ $FORCE -eq 1 ]]; then
        need_download=1
    fi

    if [[ $need_download -eq 1 ]]; then
        extra=""; [[ $FORCE -eq 1 ]] && extra="--force"
        echo "  Running: python src/dataset/download_bbeh.py --all $extra"
        "${PY_RUN[@]}" src/dataset/download_bbeh.py --all $extra
    else
        n=$(find "$BBEH_ROOT" -maxdepth 1 -mindepth 1 -type d | wc -l)
        echo "  [ok] $n BBEH subsets present in $BBEH_ROOT (use --force to re-download)"
    fi

    # bbeh_mini is a custom aggregate not present in the upstream repo:
    # 20 stride-sampled tasks per subset (460 total across 23 subsets).
    # Rebuild it from the downloaded subsets so the paper-mini runs work.
    MINI_JSON="$BBEH_ROOT/bbeh_mini/test.json"
    if [[ ! -f "$MINI_JSON" || $FORCE -eq 1 ]]; then
        echo "  Building bbeh_mini from 23 subsets..."
        mkdir -p "$BBEH_ROOT/bbeh_mini"
        "${PY_RUN[@]}" - "$BBEH_ROOT" "$MINI_JSON" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
out  = Path(sys.argv[2])
K = 20  # tasks per subset
skipped = []
mini = []
gid = 0
for sub_dir in sorted(root.iterdir()):
    if sub_dir.name == "bbeh_mini": continue
    tf = sub_dir / "test.json"
    if not tf.exists():
        skipped.append(sub_dir.name); continue
    tasks = json.load(open(tf))
    n = len(tasks)
    if n < K:
        skipped.append(sub_dir.name); continue
    stride = n // K
    idxs = [(k + 1) * stride - 1 for k in range(K)]
    for i in idxs:
        t = dict(tasks[i])
        t["id"] = gid; gid += 1
        t["tag"] = ["BBEH-mini"]
        mini.append(t)
json.dump(mini, open(out, "w"), indent=2, ensure_ascii=False)
print(f"  [write] {out} ({len(mini)} tasks from {len(mini)//K} subsets)")
if skipped:
    print(f"  [warn] skipped: {skipped}")
PY
    else
        cnt=$(python -c "import json; print(len(json.load(open('$MINI_JSON'))))" 2>/dev/null || echo "?")
        echo "  [ok] $MINI_JSON ($cnt tasks)"
    fi
fi

# ---------------------------------------------------------------------------
# 3. WorkBench — upstream olly-styles/WorkBench (data.csv + test.json per domain)
# ---------------------------------------------------------------------------
if [[ $DO_WB -eq 1 ]]; then
    step "Downloading WorkBench data + queries into dataset/workbench/"
    need_wb=0
    for d in email calendar analytics customer_relationship_manager project_management multi_domain; do
        if [[ ! -f "dataset/workbench/$d/test.json" ]]; then need_wb=1; break; fi
    done
    if [[ $FORCE -eq 1 ]]; then need_wb=1; fi

    if [[ $need_wb -eq 1 ]]; then
        extra=""; [[ $FORCE -eq 1 ]] && extra="--force"
        echo "  Running: python src/dataset/download_workbench.py --all $extra"
        "${PY_RUN[@]}" src/dataset/download_workbench.py --all $extra
    else
        echo "  [ok] WorkBench data + test.json already in place (use --force to re-download)"
    fi
fi

# ---------------------------------------------------------------------------
# 4. SWE-bench Lite + Verified task JSONs — upstream princeton-nlp on HuggingFace
# ---------------------------------------------------------------------------
if [[ $DO_SWE -eq 1 ]]; then
    step "Downloading SWE-bench Lite + Verified task JSONs"
    for variant in swe_bench_lite swe_bench_verified; do
        json="dataset/${variant}/test.json"
        if [[ -f "$json" && $FORCE -eq 0 ]]; then
            cnt=$(python -c "import json; print(len(json.load(open('$json'))))" 2>/dev/null || echo "?")
            echo "  [ok] $json ($cnt tasks)"
        else
            hf_name="${variant//_/-}"   # swe_bench_lite -> swe-bench-lite (global sub)
            extra=""; [[ $FORCE -eq 1 ]] && extra="--force"
            echo "  Running: python src/dataset/swe_bench.py --dataset $hf_name --split test $extra"
            "${PY_RUN[@]}" src/dataset/swe_bench.py --dataset "$hf_name" --split test $extra
        fi
    done
fi

echo
echo "=============================="
echo "Dataset preparation complete."
echo "=============================="
echo "  BBEH         : dataset/bbeh/benchmark_tasks/"
echo "  WorkBench    : dataset/workbench/<domain>/{data.csv,test.json}"
echo "  SWE-bench    : dataset/swe_bench_{lite,verified}/test.json"
echo "  SWE sources  : dataset/repos/<repo>/"
