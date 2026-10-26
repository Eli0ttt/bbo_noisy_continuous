#!/usr/bin/env bash
# Run a version-limited, tool-enabled EvoMAS BBO experiment and sealed handoff.
set -euo pipefail
cd "$HOME/experiments/bbo_noisy_continuous"

VERSIONS="${1:-3}"
if [[ ! "$VERSIONS" =~ ^[0-9]+$ ]] || (( VERSIONS < 3 )); then
  printf 'Usage: %s [solver_versions >= 3]\n' "$0" >&2
  exit 2
fi

set -a
source env/dsh_bbo_experiment.env
set +a
export OPENAI_API_KEY="${OPENAI_API_KEY:-${DEEPSEEK_API_KEY:-}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${DEEPSEEK_BASE_URL:-}}"
export EVOMAS_ALLOWED_MODELS="openai:deepseek-v4-flash-s2"
test -n "$OPENAI_API_KEY"
test -n "$OPENAI_BASE_URL"
test -f dsh_rsi_runs/bbo_noisy_continuous_prebuilt/task.toml
test -d repro/work/EvoMAS-bbo/pool/upstream_bbeh_s2_full_basic
python repro/work/EvoMAS-bbo/adapter/feedback_regression_smoke.py

OUT="$PWD/repro/work/EvoMAS-bbo/outputs/live_tools_feedback_$(date -u +%Y%m%d_%H%M%S)"
printf 'RUN_DIR=%s\n' "$OUT"
python repro/work/EvoMAS-bbo/adapter/evomas_bbo_pipeline.py \
  --evomas-root repro/work/EvoMAS-openai \
  --seed-pool repro/work/EvoMAS-bbo/pool/upstream_bbeh_s2_full_basic \
  --instruction instruction.md \
  --output-dir "$OUT" \
  --container bbo_noisy_continuous_prebuilt__ootgwrk__env-main-1 \
  --meta-model openai:deepseek-v4-flash-s2 \
  --num-parents 2 \
  --max-solver-versions "$VERSIONS" \
  --skip-final-repeat \
  --tool-mode bbo \
  --memory-evolution \
  --seed 43 \
  --hidden-task-root "$PWD/dsh_rsi_runs/bbo_noisy_continuous_prebuilt" \
  --hidden-jobs-dir "$OUT/harbor_jobs" \
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
