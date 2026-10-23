#!/usr/bin/env bash
# Shared environment + default hyperparameters for EvoMAS paper scripts.
# All run_*.sh files source this to pick up consistent defaults and the
# conda env. Any variable can be overridden by the caller:
#
#     NUM_EVAL_TASKS=1 scripts/run_bbeh_mini.sh
#     JUDGE_MODEL=bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0 scripts/run_workbench_email.sh

set -euo pipefail

# --- Project root (scripts live in <repo>/scripts/) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- Conda env ---
CONDA_ENV="${CONDA_ENV:-mas}"
PY_RUN=(conda run -n "$CONDA_ENV" --no-capture-output python -u)

# --- Model defaults ---
# Meta-model and judge share the same Sonnet 4.5 default so the reward
# signal and the evolutionary operator use the same reasoning backbone.
META_MODEL="${META_MODEL:-bedrock:global.anthropic.claude-sonnet-4-5-20250929-v1:0}"
JUDGE_MODEL="${JUDGE_MODEL:-$META_MODEL}"

# Worker model palette the meta-model can choose from when generating MAS
# configs. Override via `AGENT_MODELS="id1 id2 id3" ./script.sh`.
if [[ -n "${AGENT_MODELS:-}" ]]; then
    read -ra AGENT_MODELS <<< "$AGENT_MODELS"
else
    AGENT_MODELS=(
        "bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        "bedrock:qwen.qwen3-235b-a22b-2507-v1:0"
        "bedrock:qwen.qwen3-coder-480b-a35b-v1:0"
    )
fi

# --- Pipeline hyperparameters ---
# NUM_EVAL_TASKS is intentionally left unset so each per-dataset run script
# can auto-resolve it to the full task count of that dataset's test.json
# via count_tasks() below. Callers can still pin a specific count by
# exporting NUM_EVAL_TASKS before invoking the script.
MAX_STEPS="${MAX_STEPS:-2}"
NUM_PARENTS="${NUM_PARENTS:-2}"
SEED="${SEED:-42}"

# --- Batching / parallelism ---
BATCH_SIZE="${BATCH_SIZE:-1}"
WORKERS="${WORKERS:-16}"

# --- Memory ---
MEMORY_EVOLUTION="${MEMORY_EVOLUTION:-true}"
MEMORY_PATH="${MEMORY_PATH:-}"   # empty = default (dataset/<ds>/memory_<ts>.json)

# --- Output base (each script appends its own subdir) ---
OUTPUT_ROOT="${OUTPUT_ROOT:-output_paper}"

# Helper: prints a uniform banner so logs from different scripts are identifiable.
banner() {
    echo "==================================================================="
    echo "# $1"
    echo "==================================================================="
}

# Helper: build the memory-path arg conditionally
memory_args() {
    local args=(--memory-evolution "$MEMORY_EVOLUTION")
    if [[ -n "$MEMORY_PATH" ]]; then
        args+=(--memory-path "$MEMORY_PATH")
    fi
    printf '%s\n' "${args[@]}"
}

# Helper: count tasks in a dataset's test.json file. Returns 0 on missing/
# unreadable file. Per-dataset run scripts call this to set NUM_EVAL_TASKS
# to the full dataset size when the caller hasn't provided an override.
count_tasks() {
    local path="$1"
    if [[ -f "$path" ]]; then
        python -c "import json,sys; print(len(json.load(open('$path'))))" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

# Helper: resolve dataset name -> test.json path on disk. Mirrors the
# directory layout used for memory paths in main.py::run_evolution_pipeline.
test_json_for_dataset() {
    local ds="$1"
    local dsn
    dsn="$(echo "$ds" | tr '[:upper:]' '[:lower:]')"
    if [[ "$dsn" == bbeh_* ]]; then
        echo "dataset/bbeh/benchmark_tasks/${ds}/test.json"
    elif [[ "$dsn" == workbench_* ]]; then
        echo "dataset/workbench/${ds#workbench_}/test.json"
    elif [[ "$dsn" == swe_bench_* || "$dsn" == swebench* ]]; then
        echo "dataset/${ds}/test.json"
    else
        echo "dataset/${ds}/test.json"
    fi
}
