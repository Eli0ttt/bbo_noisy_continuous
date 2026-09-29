#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -lt 1 || $# -gt 2 || ! $1 =~ ^[1-9][0-9]*$ ]]; then
  echo 'usage: ./collab-run.sh RUN_NUM [no-cordis|dynamic-cordis]' >&2; exit 2
fi
RUN_NUM="$1"; CONDITION="${2:-no-cordis}"
case "$CONDITION" in no-cordis|dynamic-cordis) ;; *) echo "invalid condition: $CONDITION" >&2; exit 2;; esac
ROOT="${BBO_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
DSH_ROOT="${DSH_ROOT:-/data/user/wenxinyi/experiments/deepseek-harness}"
NODE_ROOT="${NODE_ROOT:-$HOME/.nvm/versions/node/v22.23.2}"
HARBOR_BIN="${HARBOR_BIN:-$HOME/.local/share/uv/tools/harbor/bin/harbor}"
TASK_ROOT="${TASK_ROOT:-$ROOT/dsh_rsi_runs/bbo_noisy_continuous_prebuilt}"
JOBS_DIR="${JOBS_DIR:-$ROOT/dsh_rsi_jobs}"
ENV_FILE="${ENV_FILE:-$ROOT/env/dsh_bbo_experiment.env}"
test -f "$ENV_FILE"
set -a; source "$ENV_FILE"; set +a
: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY missing}"
: "${DEEPSEEK_BASE_URL:?DEEPSEEK_BASE_URL missing}"
export AGENT_B_BASE_URL="${AGENT_B_BASE_URL:-https://openrouter.ai/api/v1}"
export AGENT_B_MODEL="${AGENT_B_MODEL:-z-ai/glm-5.3-flash}"
export COLLAB_REVIEWERS="${COLLAB_REVIEWERS:-B}"
IFS=',' read -r -a REVIEWER_IDS <<< "$COLLAB_REVIEWERS"
declare -A REVIEWER_SEEN=()
for reviewer_id in "${REVIEWER_IDS[@]}"; do
  reviewer_id="${reviewer_id//[[:space:]]/}"
  [[ $reviewer_id =~ ^[A-Z]$ ]] || { echo "invalid reviewer id '$reviewer_id' in COLLAB_REVIEWERS" >&2; exit 2; }
  [[ -z "${REVIEWER_SEEN[$reviewer_id]+x}" ]] || { echo "duplicate reviewer id '$reviewer_id'" >&2; exit 2; }
  REVIEWER_SEEN[$reviewer_id]=1
  key_var="AGENT_${reviewer_id}_API_KEY"; url_var="AGENT_${reviewer_id}_BASE_URL"; model_var="AGENT_${reviewer_id}_MODEL"
  reviewer_key="${!key_var:-}"; reviewer_url="${!url_var:-}"; reviewer_model="${!model_var:-}"
  [[ -n $reviewer_key && -n $reviewer_url && -n $reviewer_model ]] || {
    echo "AGENT_${reviewer_id}_API_KEY, _BASE_URL and _MODEL are required" >&2; exit 2;
  }
done
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-${MODEL:-deepseek-v4-flash-s1}}"
export MODEL="$DEEPSEEK_MODEL"
export COLLAB_TOTAL_SEC="${COLLAB_TOTAL_SEC:-43200}"
if [[ ! $COLLAB_TOTAL_SEC =~ ^[1-9][0-9]*$ ]] || (( COLLAB_TOTAL_SEC > 43200 )); then echo 'COLLAB_TOTAL_SEC must be 1..43200' >&2; exit 2; fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
for f in "$DSH_ROOT/apps/cli/src/bin.ts" "$NODE_ROOT/bin/node" "$ROOT/dsh_rsi_agent/bbo_collab_agent.py" \
  "$ROOT/dsh_rsi_config/bbo-collab-common.yml" "$ROOT/dsh_rsi_config/bbo-collab-primary.yml" \
  "$ROOT/dsh_rsi_config/bbo-collab-reviewer.yml" "$ROOT/dsh_rsi_config/bbo-collab-guard.mjs" \
  "$ROOT/dsh_rsi_config/bbo-selfcheck-guard.mjs" "$ROOT/dsh_rsi_config/bbo-cordis-extra.yml" \
  "$ROOT/dsh_rsi_agent/audit_collab_job.py" \
  "$TASK_ROOT/task.toml"; do test -e "$f" || { echo "missing $f" >&2; exit 1; }; done
test -x "$HARBOR_BIN"
command -v zstd >/dev/null || { echo 'zstd is required on the Harbor host to audit collected DSH traces' >&2; exit 1; }
BUILD=$(python3 - "$ROOT/dsh_rsi_agent/bbo_collab_agent.py" <<'PY'
import ast,sys
m=ast.parse(open(sys.argv[1],encoding='utf-8').read())
c=next(x for x in m.body if isinstance(x,ast.ClassDef) and x.name=='BboCollabAgent')
print(next(x.value.value for x in c.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BUILD_ID' for t in x.targets)))
PY
)
test "$BUILD" = collab-20260929-r28-clean-audit || { echo "unexpected build $BUILD" >&2; exit 1; }
python3 - "$ROOT/dsh_rsi_config/bbo-no-cordis.yml" "$ROOT/dsh_rsi_config/bbo-collab-common.yml" <<'PYTOOLS'
from pathlib import Path
import sys
single, dual = (Path(x).read_text(encoding="utf-8") for x in sys.argv[1:])
assert dual.startswith(single), "dual DSH common tool configuration drifted from single-model no-Cordis config"
assert dual.count("bbo-collab-guard.mjs") == 1, "collaboration guard must be additive and unique"
print("SINGLE_DUAL_TOOLSET_BASE_PASS")
PYTOOLS
JOB_NAME="bbo-collab-$CONDITION-$RUN_NUM"
if [[ -e "$JOBS_DIR/$JOB_NAME" ]]; then echo "job exists: $JOBS_DIR/$JOB_NAME" >&2; exit 3; fi
mkdir -p "$JOBS_DIR"
MOUNTS=$(python3 - "$DSH_ROOT" "$NODE_ROOT" "$ROOT/dsh_rsi_config" <<'PY'
import json,sys
print(json.dumps([{'type':'bind','source':s,'target':t,'read_only':True} for s,t in zip(sys.argv[1:],['/opt/deepseek-harness','/opt/node','/opt/dsh-config'])]))
PY
)
set +e
"$HARBOR_BIN" run --yes --env docker --job-name "$JOB_NAME" --jobs-dir "$JOBS_DIR" \
  --n-concurrent 1 --n-attempts 1 --max-retries 0 --path "$TASK_ROOT" \
  --agent dsh_rsi_agent.bbo_collab_agent:BboCollabAgent --model "$DEEPSEEK_MODEL" \
  --agent-kwarg "condition=$CONDITION" --mounts "$MOUNTS" \
  --allow-agent-host "${DEEPSEEK_ALLOW_AGENT_HOST:-183.230.173.202}" --allow-agent-host openrouter.ai
STATUS=$?
python3 "$ROOT/dsh_rsi_agent/audit_collab_job.py" "$JOBS_DIR/$JOB_NAME"
AUDIT_STATUS=$?
set -e
find "$JOBS_DIR/$JOB_NAME" -name final-selection.json -print -exec cat {} \;
find "$JOBS_DIR/$JOB_NAME" -name audit.json -print
[[ $STATUS -eq 0 ]] || exit "$STATUS"
[[ $AUDIT_STATUS -eq 0 ]] || echo "WARNING: trace audit failed (Harbor task result is still authoritative)" >&2
exit 0
