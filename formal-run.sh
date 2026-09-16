#!/usr/bin/env bash
set -euo pipefail
umask 077

usage() {
  cat >&2 <<'EOF'
usage: ./formal-run.sh no-cordis|dynamic-cordis RUN_NUM

examples:
  ./formal-run.sh dynamic-cordis 1
  ./formal-run.sh no-cordis 1
EOF
}

if [ "$#" -ne 2 ]; then
  usage
  exit 2
fi

export CONDITION="$1"
export RUN_NUM="$2"
case "$CONDITION" in
  no-cordis|dynamic-cordis) ;;
  *) echo "invalid condition: $CONDITION" >&2; usage; exit 2 ;;
esac
if ! [[ "$RUN_NUM" =~ ^[1-9][0-9]*$ ]]; then
  echo "RUN_NUM must be a positive integer: $RUN_NUM" >&2
  exit 2
fi

export EXP_ROOT="${EXP_ROOT:-/data/user/wenxinyi/experiments}"
export BBO_ROOT="${BBO_ROOT:-$EXP_ROOT/bbo_noisy_continuous}"
export DSH_ROOT="${DSH_ROOT:-$EXP_ROOT/deepseek-harness}"
export TASK_ROOT="${TASK_ROOT:-$BBO_ROOT/dsh_rsi_runs/bbo_noisy_continuous_prebuilt}"
export OFFICIAL_TASK_ROOT="${OFFICIAL_TASK_ROOT:-$BBO_ROOT}"
export ENV_FILE="${ENV_FILE:-$BBO_ROOT/env/dsh_bbo_experiment.env}"
export JOBS_DIR="${JOBS_DIR:-$BBO_ROOT/dsh_rsi_jobs}"
export AGENT_FILE="$BBO_ROOT/dsh_rsi_agent/dsh_bbo_agent.py"
export AGENT_IMPORT="dsh_rsi_agent.dsh_bbo_agent:DshBboAgent"
export TRACE_REVIEW_PY="$BBO_ROOT/dsh_rsi_agent/trace_review.py"
export CONFIG_ROOT="$BBO_ROOT/dsh_rsi_config"
export SELFCHECK_GUARD="$CONFIG_ROOT/bbo-selfcheck-guard.mjs"
export CHECKPOINT_HELPER="$CONFIG_ROOT/version_checkpoint.py"
export OFFICIAL_ROOT="$BBO_ROOT/official_rsi"
export PROMPT_ROOT="$OFFICIAL_ROOT/infra/prompts"
export ARB_PROMPT_TEMPLATE="$PROMPT_ROOT/autoresearch.j2"
export ARB_PROMPT_PROGRAM="$PROMPT_ROOT/autoresearch.md"
export ARB_MOUNT_FILE="$PROMPT_ROOT/mount.yaml"
export ARB_BUDGET_PY="$PROMPT_ROOT/budget.py"
export DEEPSEEK_ALLOW_AGENT_HOST="${DEEPSEEK_ALLOW_AGENT_HOST:-183.230.173.202}"
export HARBOR_BIN="${HARBOR_BIN:-$HOME/.local/share/uv/tools/harbor/bin/harbor}"
export HARBOR_PY="${HARBOR_PY:-$HOME/.local/share/uv/tools/harbor/bin/python}"
export NODE_ROOT="${NODE_ROOT:-$HOME/.nvm/versions/node/v22.23.2}"
export EXPECTED_AGENT_VERSION="0.8.2-official-autoresearch-complete-visible-ledger"

if [ ! -f "$ENV_FILE" ]; then
  echo "missing environment file: $ENV_FILE" >&2
  echo "create it before running this script" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

export PYTHONPATH="$BBO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is required in $ENV_FILE}"
: "${DEEPSEEK_BASE_URL:?DEEPSEEK_BASE_URL is required in $ENV_FILE}"
export MODEL="${MODEL:-${DEEPSEEK_MODEL:-deepseek-v4-flash-s1}}"

# Keep the official 12h outer-rollout ceiling.  These are caps, not targets.
export ARB_PROGRAM="$ARB_PROMPT_PROGRAM"
export ARB_BUDGET_PY="$ARB_BUDGET_PY"
export ARB_AGENT_TIMEOUT_SEC=43200
export ARB_OUTPUT_TOKEN_LIMIT=500000

for path in "$HARBOR_BIN" "$HARBOR_PY" "$NODE_ROOT/bin/node" "$NODE_ROOT/bin/pnpm"; do
  test -x "$path" || { echo "missing executable: $path" >&2; exit 1; }
done
for dir in "$DSH_ROOT" "$TASK_ROOT" "$OFFICIAL_TASK_ROOT"; do
  test -d "$dir" || { echo "missing directory: $dir" >&2; exit 1; }
done
for file in \
  "$TASK_ROOT/task.toml" \
  "$TASK_ROOT/instruction.md" \
  "$OFFICIAL_TASK_ROOT/task.toml" \
  "$OFFICIAL_TASK_ROOT/instruction.md" \
  "$AGENT_FILE" \
  "$TRACE_REVIEW_PY" \
  "$CONFIG_ROOT/bbo-no-cordis.yml" \
  "$SELFCHECK_GUARD" \
  "$CONFIG_ROOT/bbo-cordis-extra.yml" \
  "$CHECKPOINT_HELPER" \
  "$ARB_PROMPT_TEMPLATE" \
  "$ARB_PROGRAM" \
  "$ARB_MOUNT_FILE" \
  "$ARB_BUDGET_PY"; do
  test -f "$file" || { echo "missing file: $file" >&2; exit 1; }
done

command -v docker >/dev/null
command -v sha256sum >/dev/null
command -v zstd >/dev/null
command -v zstdcat >/dev/null
mkdir -p "$JOBS_DIR"

JOB_NAME="bbo-${CONDITION}-dsh-${RUN_NUM}"
JOB_ROOT="$JOBS_DIR/$JOB_NAME"
PROTOCOL_FILE="$JOBS_DIR/.${JOB_NAME}.formal-protocol.tmp"
if [ -e "$JOB_ROOT" ] || [ -e "$PROTOCOL_FILE" ]; then
  echo "refusing to overwrite existing run: $JOB_NAME" >&2
  echo "choose the next RUN_NUM or remove the old run intentionally" >&2
  exit 3
fi

cleanup_tmp_protocol() {
  if [ -f "$PROTOCOL_FILE" ]; then
    rm -f -- "$PROTOCOL_FILE"
  fi
}
trap cleanup_tmp_protocol EXIT

printf '%s\n' '===== official task equivalence ====='
"$HARBOR_PY" - "$OFFICIAL_TASK_ROOT/task.toml" "$TASK_ROOT/task.toml" <<'PY'
import copy
import sys
import tomllib
from pathlib import Path

def load(path: str):
    value = copy.deepcopy(tomllib.loads(Path(path).read_text(encoding="utf-8")))
    value.get("environment", {}).pop("docker_image", None)
    value.get("verifier", {}).get("environment", {}).pop("docker_image", None)
    return value

assert load(sys.argv[1]) == load(sys.argv[2]), "task.toml differs beyond docker_image selectors"
assert (
    Path(sys.argv[1]).with_name("instruction.md").read_bytes()
    == Path(sys.argv[2]).with_name("instruction.md").read_bytes()
), "instruction.md differs"
print("OFFICIAL_TASK_EQUIVALENCE_PASS")
PY
for required_dir in environment tests; do
  diff -qr -- "$OFFICIAL_TASK_ROOT/$required_dir" "$TASK_ROOT/$required_dir"
done
printf '%s\n' 'OFFICIAL_TASK_FILES_PASS'

EXPECTED_AGENT_IMAGE='sha256:270b8113ef2ba8ada5e6c4eb09c6d6ef2b16eb0b6e49751e392b9556da171c7c'
EXPECTED_VERIFIER_IMAGE='sha256:4d7d5f9224e5fd3931e52bf09295f48c3c9f3f6ca3ffabc836279e32d2440ef2'
test "$(docker image inspect rsi-bbo-noisy:public --format '{{.Id}}')" = "$EXPECTED_AGENT_IMAGE"
test "$(docker image inspect rsi-bbo-noisy-verifier:public --format '{{.Id}}')" = "$EXPECTED_VERIFIER_IMAGE"

"$HARBOR_PY" - <<'PY'
from harbor.environments.docker.docker import DockerEnvironment
assert DockerEnvironment._egress_control_kernel_support() is True
print("HARBOR_EGRESS_CONTROL_PASS")
PY

if git -C "$DSH_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  HARNESS_HEAD=$(git -C "$DSH_ROOT" rev-parse HEAD)
  HARNESS_DIFF_SHA256=$(git -C "$DSH_ROOT" diff --no-ext-diff --binary HEAD | sha256sum | awk '{print $1}')
else
  HARNESS_HEAD=unknown
  HARNESS_DIFF_SHA256=unknown
fi

if git -C "$OFFICIAL_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  OFFICIAL_RSI_HEAD=$(git -C "$OFFICIAL_ROOT" rev-parse HEAD)
else
  OFFICIAL_RSI_HEAD=unknown
fi

export MOUNTS_JSON='[
  {
    "type": "bind",
    "source": "'"$DSH_ROOT"'",
    "target": "/opt/deepseek-harness",
    "read_only": true
  },
  {
    "type": "bind",
    "source": "'"$NODE_ROOT"'",
    "target": "/opt/node",
    "read_only": true
  },
  {
    "type": "bind",
    "source": "'"$CONFIG_ROOT"'",
    "target": "/opt/dsh-config",
    "read_only": true
  }
]'

BASE_URL_SHA256=$(printf '%s' "$DEEPSEEK_BASE_URL" | sha256sum | awk '{print $1}')
{
  echo "job_name=$JOB_NAME"
  echo "condition=$CONDITION"
  echo "run_number=$RUN_NUM"
  echo "model=$MODEL"
  echo "agent_version=$EXPECTED_AGENT_VERSION"
  echo "official_rsi_head=$OFFICIAL_RSI_HEAD"
  echo "harness_head=$HARNESS_HEAD"
  echo "harness_tracked_diff_sha256=$HARNESS_DIFF_SHA256"
  echo "agent_image=$EXPECTED_AGENT_IMAGE"
  echo "verifier_image=$EXPECTED_VERIFIER_IMAGE"
  echo "agent_timeout_sec=43200"
  echo "arb_output_token_limit=500000"
  echo "base_url_sha256=$BASE_URL_SHA256"
  echo "official_task_root=$OFFICIAL_TASK_ROOT"
  echo "task_root=$TASK_ROOT"
  echo "prompt_template_sha256=$(sha256sum "$ARB_PROMPT_TEMPLATE" | awk '{print $1}')"
  echo "prompt_program_sha256=$(sha256sum "$ARB_PROGRAM" | awk '{print $1}')"
  echo "mount_yaml_sha256=$(sha256sum "$ARB_MOUNT_FILE" | awk '{print $1}')"
  echo "budget_py_sha256=$(sha256sum "$ARB_BUDGET_PY" | awk '{print $1}')"
  echo "agent_sha256=$(sha256sum "$AGENT_FILE" | awk '{print $1}')"
  echo "trace_review_sha256=$(sha256sum "$TRACE_REVIEW_PY" | awk '{print $1}')"
  echo "version_checkpoint_helper_sha256=$(sha256sum "$CHECKPOINT_HELPER" | awk '{print $1}')"
  echo "selfcheck_guard_sha256=$(sha256sum "$SELFCHECK_GUARD" | awk '{print $1}')"
  echo "visible_selfcheck_protocol=checkpoint-helper-only"
  echo "bookkeeping_checkpoint_guard=1"
  echo "version_checkpoint_protocol=transactional-complete-visible-ledger"
  echo "version_decision_protocol=resolve-before-next-version"
  echo "final_submission_protocol=outer-harness-last-explicitly-committed-canonical"
  echo "finalization_owner=outer_harness"
  echo "no_cordis_config_sha256=$(sha256sum "$CONFIG_ROOT/bbo-no-cordis.yml" | awk '{print $1}')"
  echo "cordis_extra_config_sha256=$(sha256sum "$CONFIG_ROOT/bbo-cordis-extra.yml" | awk '{print $1}')"
} > "$PROTOCOL_FILE"

printf 'START_FORMAL=%s\n' "$JOB_NAME"
"$HARBOR_BIN" run \
  --yes \
  --env docker \
  --job-name "$JOB_NAME" \
  --jobs-dir "$JOBS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --max-retries 0 \
  --path "$TASK_ROOT" \
  --agent "$AGENT_IMPORT" \
  --model "$MODEL" \
  --agent-kwarg "condition=$CONDITION" \
  --ak "prompt_template_path=$ARB_PROMPT_TEMPLATE" \
  --extra-docker-compose "$ARB_MOUNT_FILE" \
  --mounts "$MOUNTS_JSON" \
  --allow-agent-host "$DEEPSEEK_ALLOW_AGENT_HOST"

test -f "$JOB_ROOT/result.json"
mapfile -t TRIALS < <(find "$JOB_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'bbo_noisy*' -print | sort)
test "${#TRIALS[@]}" -eq 1
TRIAL="${TRIALS[0]}"

test -f "$TRIAL/result.json"
test -f "$TRIAL/artifacts/app/methods/main/solver.py"

set +e
"$HARBOR_PY" - "$JOB_ROOT/result.json" "$TRIAL/result.json" "$CONDITION" "$EXPECTED_AGENT_VERSION" <<'PY'
import json
import sys
from pathlib import Path

job = json.loads(Path(sys.argv[1]).read_text())
trial = json.loads(Path(sys.argv[2]).read_text())
condition = sys.argv[3]
expected_version = sys.argv[4]

def fail(message: str, code: int = 20) -> None:
    print("FORMAL_TRIAL_FAIL " + message, file=sys.stderr)
    raise SystemExit(code)

if job.get("n_total_trials") != 1:
    fail(f"unexpected_total_trials={job.get('n_total_trials')}")
stats = job.get("stats") or {}
if stats.get("n_errored_trials", 0) != 0 or trial.get("exception_info") is not None:
    fail("trial_exception=" + json.dumps(trial.get("exception_info"), ensure_ascii=False, sort_keys=True), 21)
if stats.get("n_completed_trials") != 1:
    fail(f"unexpected_completed_trials={stats.get('n_completed_trials')}", 22)

agent_info = trial.get("agent_info") or {}
if agent_info.get("version") != expected_version:
    fail(f"agent_version={agent_info.get('version')} expected={expected_version}", 23)

metadata = ((trial.get("agent_result") or {}).get("metadata") or {})
required = {
    "condition": condition,
    "official_prompt_protocol": True,
    "rendered_autoresearch_prompt": True,
    "single_persistent_session": True,
    "bookkeeping_checkpoint_guard": True,
    "version_checkpoint_protocol": "transactional-complete-visible-ledger",
    "visible_selfcheck_protocol": "checkpoint-helper-only",
    "direct_selfcheck_guard": True,
    "version_decision_protocol": "resolve-before-next-version",
    "final_submission_protocol": "outer-harness-last-explicitly-committed-canonical",
    "checkpoint_guard_complete": True,
    "decision_complete": True,
    "lineage_complete": True,
    "finalized": True,
    "dsh_permission_mode": "danger-full-access",
    "isolation_boundary": "harbor-docker-task-container",
}
for key, expected in required.items():
    if metadata.get(key) != expected:
        fail(f"metadata_{key}={metadata.get(key)!r} expected={expected!r}", 24)

verifier = trial.get("verifier_result")
if not verifier:
    fail("verifier_result=null", 25)
reward = ((verifier.get("rewards") or {}).get("reward"))
if not isinstance(reward, (int, float)):
    fail(f"invalid_reward={reward!r}", 26)
print(f"FORMAL_RESULT_PASS condition={condition} reward={reward}")
PY
RESULT_CHECK_RC=$?
set -e

mv "$PROTOCOL_FILE" "$TRIAL/formal_protocol.txt"
trap - EXIT

if [ "$RESULT_CHECK_RC" -ne 0 ]; then
  if [ -f "$TRIAL/agent/autoresearch-audit.json" ]; then
    "$HARBOR_PY" - "$TRIAL/agent/autoresearch-audit.json" <<'PY' >&2
import json, sys
x=json.load(open(sys.argv[1]))
print(
    "FORMAL_CHECKPOINT_DIAGNOSTIC "
    f"checkpoint_guard_complete={int(bool(x.get('checkpoint_guard_complete')))} "
    f"decision_complete={int(bool(x.get('decision_complete')))} "
    f"lineage_complete={int(bool(x.get('lineage_complete')))} "
    f"finalized={int(bool(x.get('finalized')))}"
)
reasons=x.get("guard_failure_reasons") or []
if reasons:
    print("FORMAL_CHECKPOINT_HARD_FAILURES " + " | ".join(map(str, reasons)))
if x.get("finalization_error"):
    print("FORMAL_CHECKPOINT_FINALIZATION_ERROR " + str(x.get("finalization_error")))
PY
  fi
  exit "$RESULT_CHECK_RC"
fi

AGENT_AUTORESEARCH_AUDIT="$TRIAL/agent/autoresearch-audit.json"
EXPERIMENT_LOG="$TRIAL/artifacts/app/methods/experiment_log.md"
VERSIONS_DIR="$TRIAL/artifacts/app/methods/versions"
VERSION_CHECKPOINT_STATE="$TRIAL/artifacts/app/methods/version_checkpoints.json"
VERIFIER_SCORE_DETAILS="$TRIAL/verifier/score_details.json"
VERIFIER_GRADE_DEBUG="$TRIAL/verifier/grade_debug.json"
EFFECTIVE_CONFIG="$TRIAL/agent/effective-config.yml"

for required in \
  "$EFFECTIVE_CONFIG" \
  "$AGENT_AUTORESEARCH_AUDIT" \
  "$VERSION_CHECKPOINT_STATE" \
  "$VERIFIER_SCORE_DETAILS" \
  "$VERIFIER_GRADE_DEBUG"; do
  test -f "$required" || { echo "missing required formal artifact: $required" >&2; exit 1; }
done

mapfile -t TRACE_FILES < <(
  find "$TRIAL/artifacts/logs/artifacts/dsh-home/sessions" \
    -type f -name 'session.jsonl.zstd' -print 2>/dev/null | sort
)
if [ "${#TRACE_FILES[@]}" -ne 1 ]; then
  echo "expected exactly one structured DSH session trace, found ${#TRACE_FILES[@]}" >&2
  exit 1
fi
AGENT_TRACE_FILE="${TRACE_FILES[0]}"

# Curated human-review surface. The raw Harbor tree remains intact for
# reproducibility/upload; review/ contains only the evidence normally needed
# for manual inspection.
REVIEW_DIR="$JOB_ROOT/review"
"$HARBOR_PY" "$TRACE_REVIEW_PY" \
  --trial "$TRIAL" \
  --job-name "$JOB_NAME" \
  --condition "$CONDITION" \
  --run-number "$RUN_NUM" \
  --trace "$AGENT_TRACE_FILE" \
  --out-dir "$REVIEW_DIR"

REVIEW_SUMMARY="$REVIEW_DIR/summary.json"
AGENT_ACTIONS="$REVIEW_DIR/agent-actions.md"
VERSION_HISTORY="$REVIEW_DIR/version-history.csv"
TRACE_AUDIT="$REVIEW_DIR/trace-audit.json"
BOOKKEEPING_AUDIT="$REVIEW_DIR/bookkeeping-audit.json"
RUNTIME_AUDIT="$REVIEW_DIR/runtime-audit.json"

for required in \
  "$REVIEW_SUMMARY" \
  "$AGENT_ACTIONS" \
  "$VERSION_HISTORY" \
  "$TRACE_AUDIT" \
  "$BOOKKEEPING_AUDIT" \
  "$RUNTIME_AUDIT" \
  "$REVIEW_DIR/final_solver.py" \
  "$REVIEW_DIR/version_checkpoints.json" \
  "$REVIEW_DIR/agent-trace.jsonl.zstd"; do
  test -f "$required" || { echo "missing review artifact: $required" >&2; exit 1; }
done

# Trace/runtime diagnostics are descriptive. Version checkpoint fidelity is a
# hard pre-verifier contract in agent v0.8.0 and is rechecked here post-hoc.
read -r TRACE_SELFCHECKS TRACE_FAILED TRACE_DIRECT_ATTEMPTS TRACE_DIRECT_BLOCKED TRACE_DIRECT_UNBLOCKED TRACE_LLM_RETRIES TRACE_TOOL_ERRORS TRACE_BASH_NONZERO TRACE_CORDIS TRACE_SANDBOX < <(
  "$HARBOR_PY" - "$TRACE_AUDIT" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
print(
    x.get('scored_selfcheck_executions', 0),
    x.get('failed_selfcheck_tool_calls', 0),
    x.get('direct_selfcheck_attempts', 0),
    x.get('blocked_direct_selfcheck_attempts', 0),
    x.get('unblocked_direct_selfcheck_executions', 0),
    x.get('llm_retry_events', 0),
    x.get('tool_api_errors', 0),
    x.get('bash_nonzero_calls', 0),
    x.get('cordis_related_calls', 0),
    x.get('sandbox_backend_failures', 0),
)
PY
)

if [ "$TRACE_DIRECT_UNBLOCKED" -ne 0 ]; then
  echo "FORMAL_DIRECT_SELFCHECK_ROUTE_FAIL unblocked_direct_selfchecks=$TRACE_DIRECT_UNBLOCKED" >&2
  exit 1
fi
if [ "$TRACE_SANDBOX" -ne 0 ]; then
  echo "FORMAL_INFRASTRUCTURE_AUDIT_FAIL sandbox_backend_failures=$TRACE_SANDBOX" >&2
  exit 1
fi
if [ "$CONDITION" = "no-cordis" ] && [ "$TRACE_CORDIS" -ne 0 ]; then
  echo "FORMAL_TREATMENT_AUDIT_FAIL unexpected_cordis_calls=$TRACE_CORDIS" >&2
  exit 1
fi

read -r BOOKKEEPING_STATUS LOGGED_VERSIONS SNAPSHOT_VERSIONS SNAPSHOT_COMPLETE CHECKPOINT_GUARD DECISION_COMPLETE LINEAGE_COMPLETE CANONICAL_ELIGIBLE INVALID_COMMITTED FINALIZED FINALIZATION_OWNER_COMPLETE MISSING_SNAPSHOTS HASH_MISMATCHES < <(
  "$HARBOR_PY" - "$BOOKKEEPING_AUDIT" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
print(
    x.get('status', 'WARN'),
    x.get('logged_version_count', 0),
    x.get('snapshot_version_count', 0),
    int(bool(x.get('snapshot_complete'))),
    int(bool(x.get('checkpoint_guard_complete'))),
    int(bool(x.get('decision_complete'))),
    int(bool(x.get('lineage_complete'))),
    int(bool(x.get('canonical_eligible'))),
    len(x.get('invalid_committed_versions', [])),
    int(bool(x.get('finalized'))),
    int(bool(x.get('finalization_owner_complete'))),
    x.get('missing_snapshot_count', len(x.get('missing_snapshot_versions', []))),
    len(x.get('snapshot_hash_mismatches', [])),
)
PY
)

read -r RUNTIME_STATUS RUNTIME_ELAPSED RUNTIME_BUDGET RUNTIME_MARGIN RUNTIME_UTIL < <(
  "$HARBOR_PY" - "$RUNTIME_AUDIT" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
u=x.get('utilization')
print(
    x.get('status', 'UNKNOWN'),
    x.get('elapsed_sec'),
    x.get('time_budget_sec'),
    x.get('margin_sec'),
    'NA' if u is None else f'{100*u:.2f}',
)
PY
)

printf 'FORMAL_AUTORESEARCH_AUDIT status=%s logged_versions=%s snapshot_versions=%s snapshot_complete=%s checkpoint_guard_complete=%s decision_complete=%s lineage_complete=%s canonical_eligible=%s invalid_committed=%s finalized=%s finalization_owner_complete=%s missing_snapshots=%s hash_mismatches=%s\n' \
  "$BOOKKEEPING_STATUS" "$LOGGED_VERSIONS" "$SNAPSHOT_VERSIONS" "$SNAPSHOT_COMPLETE" "$CHECKPOINT_GUARD" "$DECISION_COMPLETE" "$LINEAGE_COMPLETE" "$CANONICAL_ELIGIBLE" "$INVALID_COMMITTED" "$FINALIZED" "$FINALIZATION_OWNER_COMPLETE" "$MISSING_SNAPSHOTS" "$HASH_MISMATCHES"
printf 'FORMAL_TRACE_AUDIT scored_selfchecks=%s failed_selfcheck_tool_calls=%s direct_selfcheck_attempts=%s blocked_direct_selfchecks=%s unblocked_direct_selfchecks=%s llm_retries=%s tool_api_errors=%s bash_nonzero_calls=%s cordis_calls=%s\n' \
  "$TRACE_SELFCHECKS" "$TRACE_FAILED" "$TRACE_DIRECT_ATTEMPTS" "$TRACE_DIRECT_BLOCKED" "$TRACE_DIRECT_UNBLOCKED" "$TRACE_LLM_RETRIES" "$TRACE_TOOL_ERRORS" "$TRACE_BASH_NONZERO" "$TRACE_CORDIS"
printf 'FORMAL_RUNTIME_AUDIT status=%s elapsed_sec=%s budget_sec=%s margin_sec=%s utilization_pct=%s\n' \
  "$RUNTIME_STATUS" "$RUNTIME_ELAPSED" "$RUNTIME_BUDGET" "$RUNTIME_MARGIN" "$RUNTIME_UTIL"

# v0.8.2 keeps every scored visible evaluation on the helper ledger and makes finalization outer-harness-owned. Every
# intermediate version must be explicitly kept/reverted before the next
# versioned evaluation, while final submission is derived deterministically
# from the exact final artifact. By the time hidden verification completes all
# bookkeeping invariants must therefore be true.
if [ "$SNAPSHOT_COMPLETE" -ne 1 ] || [ "$CHECKPOINT_GUARD" -ne 1 ] || [ "$DECISION_COMPLETE" -ne 1 ] || [ "$LINEAGE_COMPLETE" -ne 1 ] || [ "$CANONICAL_ELIGIBLE" -ne 1 ] || [ "$INVALID_COMMITTED" -ne 0 ] || [ "$FINALIZED" -ne 1 ] || [ "$FINALIZATION_OWNER_COMPLETE" -ne 1 ] || [ "$HASH_MISMATCHES" -ne 0 ]; then
  echo "FORMAL_BOOKKEEPING_AUDIT_FAIL snapshot_complete=$SNAPSHOT_COMPLETE checkpoint_guard_complete=$CHECKPOINT_GUARD decision_complete=$DECISION_COMPLETE lineage_complete=$LINEAGE_COMPLETE canonical_eligible=$CANONICAL_ELIGIBLE invalid_committed=$INVALID_COMMITTED finalized=$FINALIZED finalization_owner_complete=$FINALIZATION_OWNER_COMPLETE missing_snapshots=$MISSING_SNAPSHOTS hash_mismatches=$HASH_MISMATCHES" >&2
  exit 1
fi

# This warning is post-hoc telemetry from the official verifier. It is not fed
# back to the research agent, which avoids hidden-runtime feedback becoming an
# outer-loop tuning signal.
if [ "$RUNTIME_STATUS" = "WARN" ]; then
  echo "FORMAL_RUNTIME_MARGIN_WARN: verifier completed, but runtime headroom is thin" >&2
fi

REVIEW_ARCHIVE="$JOB_ROOT/${JOB_NAME}.review.tar.zst"
tar --zstd -C "$JOB_ROOT" -cf "$REVIEW_ARCHIVE" review
REVIEW_ARCHIVE_SHA256=$(sha256sum "$REVIEW_ARCHIVE" | awk '{print $1}')

printf 'FORMAL_INFRASTRUCTURE_AUDIT_PASS scored_selfchecks=%s logged_versions=%s\n' \
  "$TRACE_SELFCHECKS" "$LOGGED_VERSIONS"
printf 'FORMAL_RUN_PASS=%s\n' "$JOB_NAME"
printf 'RAW_JOB_DIR=%s\n' "$JOB_ROOT"
printf 'RAW_TRIAL_DIR=%s\n' "$TRIAL"
printf 'REVIEW_DIR=%s\n' "$REVIEW_DIR"
printf 'REVIEW_README=%s\n' "$REVIEW_DIR/README.md"
printf 'REVIEW_SUMMARY=%s\n' "$REVIEW_SUMMARY"
printf 'AGENT_ACTIONS=%s\n' "$AGENT_ACTIONS"
printf 'VERSION_HISTORY=%s\n' "$VERSION_HISTORY"
printf 'EXPERIMENT_LOG=%s\n' "$REVIEW_DIR/experiment_log.md"
printf 'FINAL_SOLVER=%s\n' "$REVIEW_DIR/final_solver.py"
printf 'VERSIONS_DIR=%s\n' "$REVIEW_DIR/versions"
printf 'VERSION_CHECKPOINTS=%s\n' "$REVIEW_DIR/version_checkpoints.json"
printf 'AGENT_TRACE_FILE=%s\n' "$REVIEW_DIR/agent-trace.jsonl.zstd"
printf 'TRACE_AUDIT=%s\n' "$TRACE_AUDIT"
printf 'BOOKKEEPING_AUDIT=%s\n' "$BOOKKEEPING_AUDIT"
printf 'RUNTIME_AUDIT=%s\n' "$RUNTIME_AUDIT"
printf 'VERIFIER_SCORE_DETAILS=%s\n' "$REVIEW_DIR/verifier-score-details.json"
printf 'REVIEW_ARCHIVE=%s\n' "$REVIEW_ARCHIVE"
printf 'REVIEW_ARCHIVE_SHA256=%s\n' "$REVIEW_ARCHIVE_SHA256"
df -h "$BBO_ROOT" /tmp
