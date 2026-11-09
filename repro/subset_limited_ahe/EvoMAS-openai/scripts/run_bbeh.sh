#!/usr/bin/env bash
# Main EvoMAS run on a BBEH subset. Pass the subset as the first positional
# arg or via the SUBSET env var (without the `bbeh_` prefix):
#
#     scripts/run_bbeh.sh mini                    # bbeh_mini
#     SUBSET=boolean_expressions scripts/run_bbeh.sh
#     scripts/run_bbeh.sh word_sorting
#
# The subset must match a directory under dataset/bbeh/benchmark_tasks/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

SUBSET="${1:-${SUBSET:-mini}}"
DATASET="bbeh_${SUBSET}"

TASK_DIR="dataset/bbeh/benchmark_tasks/${DATASET}"
if [[ ! -d "$TASK_DIR" ]]; then
    echo "Unknown BBEH subset: '${SUBSET}' (${TASK_DIR} not found)" >&2
    echo "Available subsets:" >&2
    ls dataset/bbeh/benchmark_tasks/ 2>/dev/null | sed 's/^bbeh_//' | sed 's/^/  /' >&2
    exit 2
fi

NUM_EVAL_TASKS="${NUM_EVAL_TASKS:-$(count_tasks "${TASK_DIR}/test.json")}"

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
