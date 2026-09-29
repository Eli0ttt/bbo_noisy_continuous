#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -lt 1 || $# -gt 2 || ! $1 =~ ^[1-9][0-9]*$ ]]; then
  echo 'usage: ./collab-a-run.sh RUN_NUM [no-cordis|dynamic-cordis]' >&2
  exit 2
fi
RUN_NUM="$1"
CONDITION="${2:-no-cordis}"
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
: "${AGENT_B_API_KEY:?AGENT_B_API_KEY missing}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-${MODEL:-deepseek-v4-flash-s1}}"
export MODEL="$DEEPSEEK_MODEL"
export AGENT_B_BASE_URL="${AGENT_B_BASE_URL:-https://openrouter.ai/api/v1}"
export AGENT_B_MODEL="${AGENT_B_MODEL:-z-ai/glm-5.3-flash}"

# Only one research wall-clock limit. 30/60 min are quick experiments;
# 43200 sec is the official task's 12-hour agent limit.
export COLLAB_TOTAL_SEC="${COLLAB_TOTAL_SEC:-1800}"
export COLLAB_MIN_NEW_ROUND_SEC="${COLLAB_MIN_NEW_ROUND_SEC:-600}"
[[ "$COLLAB_TOTAL_SEC" =~ ^[1-9][0-9]*$ ]] || { echo "COLLAB_TOTAL_SEC must be a positive integer" >&2; exit 2; }
(( COLLAB_TOTAL_SEC <= 43200 )) || { echo "COLLAB_TOTAL_SEC must be <= 43200" >&2; exit 2; }

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
for f in \
  "$DSH_ROOT/apps/cli/src/bin.ts" "$NODE_ROOT/bin/node" \
  "$ROOT/dsh_rsi_agent/bbo_split_collab_agent.py" \
  "$ROOT/dsh_rsi_agent/audit_split_job.py" \
  "$ROOT/dsh_rsi_config/bbo-no-cordis.yml" \
  "$ROOT/dsh_rsi_config/bbo-cordis-extra.yml" \
  "$ROOT/dsh_rsi_config/bbo-split-agent-a.yml" \
  "$ROOT/dsh_rsi_config/bbo-split-agent-b.yml" \
  "$ROOT/dsh_rsi_config/bbo-split-guard.yml" \
  "$ROOT/dsh_rsi_config/bbo-split-guard.mjs" \
  "$ROOT/dsh_rsi_config/bbo-split-merge.py" \
  "$ROOT/dsh_rsi_config/bbo_candidate_preflight.py" \
  "$TASK_ROOT/task.toml"; do
  test -e "$f" || { echo "missing $f" >&2; exit 1; }
done
test -x "$HARBOR_BIN"
command -v zstd >/dev/null || { echo "zstd is required for post-run trace audit" >&2; exit 1; }

python3 -m py_compile \
  "$ROOT/dsh_rsi_agent/bbo_split_collab_agent.py" \
  "$ROOT/dsh_rsi_agent/audit_split_job.py" \
  "$ROOT/dsh_rsi_config/bbo-split-merge.py"

JOB_NAME="bbo-split-$CONDITION-$RUN_NUM"
[[ ! -e "$JOBS_DIR/$JOB_NAME" ]] || { echo "job exists: $JOBS_DIR/$JOB_NAME" >&2; exit 3; }
mkdir -p "$JOBS_DIR"

MOUNTS=$(python3 - "$DSH_ROOT" "$NODE_ROOT" "$ROOT/dsh_rsi_config" <<'PY'
import json,sys
print(json.dumps([
  {"type":"bind","source":sys.argv[1],"target":"/opt/deepseek-harness","read_only":True},
  {"type":"bind","source":sys.argv[2],"target":"/opt/node","read_only":True},
  {"type":"bind","source":sys.argv[3],"target":"/opt/dsh-config","read_only":True},
]))
PY
)

set +e
"$HARBOR_BIN" run \
  --yes --env docker \
  --job-name "$JOB_NAME" --jobs-dir "$JOBS_DIR" \
  --n-concurrent 1 --n-attempts 1 --max-retries 0 \
  --path "$TASK_ROOT" \
  --agent dsh_rsi_agent.bbo_split_collab_agent:BboSplitCollabAgent \
  --model "$DEEPSEEK_MODEL" \
  --agent-kwarg "condition=$CONDITION" \
  --mounts "$MOUNTS" \
  --allow-agent-host "${DEEPSEEK_ALLOW_AGENT_HOST:-183.230.173.202}" \
  --allow-agent-host openrouter.ai
STATUS=$?

python3 "$ROOT/dsh_rsi_agent/audit_split_job.py" "$JOBS_DIR/$JOB_NAME"
AUDIT_STATUS=$?
find "$JOBS_DIR/$JOB_NAME" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
find "$JOBS_DIR/$JOB_NAME" -type f -name '*.pyc' -delete 2>/dev/null || true

# Harbor needs these logs while the trial is running. Once Harbor has finished
# AND our canonical audit was written successfully, they are redundant for the
# retained experiment archive. Do not touch result/config/lock/verifier files.
if [[ $AUDIT_STATUS -eq 0 ]]; then
  rm -f "$JOBS_DIR/$JOB_NAME/job.log"
  find "$JOBS_DIR/$JOB_NAME" -mindepth 2 -maxdepth 2 -type f -name 'trial.log' -delete 2>/dev/null || true
fi
set -e

find "$JOBS_DIR/$JOB_NAME" -path '*/agent/audit.json' -print

[[ $STATUS -eq 0 ]] || exit "$STATUS"
[[ $AUDIT_STATUS -eq 0 ]] || echo "WARNING: trace audit failed; Harbor result remains authoritative" >&2
