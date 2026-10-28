#!/usr/bin/env bash
# Run a version-limited, tool-enabled EvoMAS BBO experiment and sealed handoff.
set -euo pipefail
cd "$HOME/experiments/bbo_noisy_continuous"

VERSIONS="${1:-3}"
if [[ ! "$VERSIONS" =~ ^[0-9]+$ ]] || (( VERSIONS < 3 )); then
  printf 'Usage: %s [solver_versions >= 3] [previous_completed_run]\n' "$0" >&2
  exit 2
fi
CONTINUE_ARGS=()
SEED_POOL="repro/work/EvoMAS-bbo/pool/upstream_bbeh_s2_full_basic"
if [[ -n "${2:-}" ]]; then
  test -f "$2/summary.json"
  test -d "$2/working_pool"
  CONTINUE_ARGS=(--continue-from-run "$2")
  SEED_POOL="$2/working_pool"
fi

set -a
source env/dsh_bbo_experiment.env
set +a
export OPENAI_API_KEY="${OPENAI_API_KEY:-${DEEPSEEK_API_KEY:-}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${DEEPSEEK_BASE_URL:-}}"
export EVOMAS_ALLOWED_MODELS="openai:deepseek-v4-flash-s2"
test -n "$OPENAI_API_KEY"
test -n "$OPENAI_BASE_URL"
test -d "$SEED_POOL"
CONTAINER="${BBO_EVAL_CONTAINER:-bbo-evomas-visible-wenxinyi}"
docker inspect -f '{{.State.Running}}' "$CONTAINER" | grep -qx true
HIDDEN_ARGS=()
if [[ "${BBO_RUN_HIDDEN:-0}" == 1 ]]; then
  test -f dsh_rsi_runs/bbo_noisy_continuous_prebuilt/task.toml
fi
python repro/work/EvoMAS-bbo/adapter/feedback_regression_smoke.py

OUT="$PWD/repro/work/EvoMAS-bbo/outputs/live_tools_feedback_$(date -u +%Y%m%d_%H%M%S)"
if [[ "${BBO_RUN_HIDDEN:-0}" == 1 ]]; then
  HIDDEN_ARGS=(--hidden-task-root "$PWD/dsh_rsi_runs/bbo_noisy_continuous_prebuilt"
               --hidden-jobs-dir "$OUT/harbor_jobs")
fi
printf 'RUN_DIR=%s\n' "$OUT"
python repro/work/EvoMAS-bbo/adapter/evomas_bbo_pipeline.py \
  --evomas-root repro/work/EvoMAS-openai \
  --seed-pool "$SEED_POOL" \
  --instruction instruction.md \
  --output-dir "$OUT" \
  --container "$CONTAINER" \
  --meta-model openai:deepseek-v4-flash-s2 \
  --num-parents 2 \
  --max-solver-versions "$VERSIONS" \
  --skip-final-repeat \
  --tool-mode bbo \
  --memory-evolution \
  --seed 43 \
  "${CONTINUE_ARGS[@]}" \
  "${HIDDEN_ARGS[@]}" \
  2>&1 | tee "${OUT}.log"

python - "$OUT/summary.json" <<'PY'
import json
import sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for name in ("best_visible", "operations", "tool_use_observed",
             "hidden_status", "hidden_score", "total_runtime_seconds"):
    print(name, s.get(name))
PY
