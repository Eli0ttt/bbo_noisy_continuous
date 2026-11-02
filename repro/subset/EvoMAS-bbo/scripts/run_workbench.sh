#!/usr/bin/env bash
# Main EvoMAS run on a WorkBench subdomain. Pass the subdomain as the first
# positional arg or via the SUBDOMAIN env var:
#
#     scripts/run_workbench.sh email
#     SUBDOMAIN=calendar scripts/run_workbench.sh
#     scripts/run_workbench.sh multi_domain
#
# Supported subdomains mirror mas_pools/workbench/*/:
#   analytics | calendar | customer_relationship_manager
#   email     | multi_domain | project_management

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

SUBDOMAIN="${1:-${SUBDOMAIN:-email}}"

VALID_SUBDOMAINS=(analytics calendar customer_relationship_manager email multi_domain project_management)
if [[ ! " ${VALID_SUBDOMAINS[*]} " =~ " ${SUBDOMAIN} " ]]; then
    echo "Unknown WorkBench subdomain: '${SUBDOMAIN}'" >&2
    echo "Valid: ${VALID_SUBDOMAINS[*]}" >&2
    exit 2
fi

DATASET="workbench_${SUBDOMAIN}"
NUM_EVAL_TASKS="${NUM_EVAL_TASKS:-$(count_tasks "dataset/workbench/${SUBDOMAIN}/test.json")}"

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
