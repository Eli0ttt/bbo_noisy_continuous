#!/usr/bin/env bash
# Main EvoMAS run on a SWE-bench variant. Pass the variant as the first
# positional arg or via the VARIANT env var:
#
#     scripts/run_swebench.sh verified          # swe_bench_verified (default)
#     scripts/run_swebench.sh lite              # swe_bench_lite
#     VARIANT=lite scripts/run_swebench.sh
#
# Note: SWE-bench instances are substantially more expensive than bbeh/workbench
# tasks, so defaults below are smaller. Override via env vars to scale up.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# SWE-bench-specific smaller defaults for parallelism (each task is heavy).
BATCH_SIZE="${BATCH_SIZE:-5}"
WORKERS="${WORKERS:-2}"

VARIANT="${1:-${VARIANT:-verified}}"

VALID_VARIANTS=(lite verified)
if [[ ! " ${VALID_VARIANTS[*]} " =~ " ${VARIANT} " ]]; then
    echo "Unknown SWE-bench variant: '${VARIANT}'" >&2
    echo "Valid: ${VALID_VARIANTS[*]}" >&2
    exit 2
fi

DATASET="swe_bench_${VARIANT}"
NUM_EVAL_TASKS="${NUM_EVAL_TASKS:-$(count_tasks "dataset/${DATASET}/test.json")}"

OUT="${OUTPUT_ROOT}/${DATASET}_main"
mkdir -p "$OUT"

banner "EvoMAS main run — ${DATASET} (tasks=$NUM_EVAL_TASKS, batch=$BATCH_SIZE, workers=$WORKERS)"

mapfile -t MEM_ARGS < <(memory_args)

"${PY_RUN[@]}" main.py \
    --dataset "$DATASET" \
    --num-eval-tasks "$NUM_EVAL_TASKS" \
    --max-steps "$MAX_STEPS" \
    --num-parents "$NUM_PARENTS" \
    --seed "$SEED" \
    --output-dir "$OUT" \
    --meta-model-id "$META_MODEL" \
    --model-list "${AGENT_MODELS[@]}" \
    --llm-as-judge "$JUDGE_MODEL" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    "${MEM_ARGS[@]}" \
    2>&1 | tee "$OUT/run.log"
