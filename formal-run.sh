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
export CONFIG_ROOT="$BBO_ROOT/dsh_rsi_config"
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
export EXPECTED_AGENT_VERSION="0.4.0-official-autoresearch-shellfix"

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
  "$CONFIG_ROOT/bbo-no-cordis.yml" \
  "$CONFIG_ROOT/bbo-cordis-extra.yml" \
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

"$HARBOR_PY" - "$JOB_ROOT/result.json" "$TRIAL/result.json" "$CONDITION" "$EXPECTED_AGENT_VERSION" <<'PY'
import json
import sys
from pathlib import Path

job = json.loads(Path(sys.argv[1]).read_text())
trial = json.loads(Path(sys.argv[2]).read_text())
condition = sys.argv[3]
expected_version = sys.argv[4]

assert job["n_total_trials"] == 1
assert job["stats"]["n_completed_trials"] == 1
assert job["stats"]["n_errored_trials"] == 0
assert trial["exception_info"] is None
assert trial["agent_info"]["version"] == expected_version
metadata = trial["agent_result"]["metadata"]
assert metadata["condition"] == condition
assert metadata["official_prompt_protocol"] is True
assert metadata["rendered_autoresearch_prompt"] is True
assert metadata["single_persistent_session"] is True
assert metadata["dsh_permission_mode"] == "danger-full-access"
assert metadata["isolation_boundary"] == "harbor-docker-task-container"
reward = trial["verifier_result"]["rewards"]["reward"]
assert isinstance(reward, (int, float))
print(f"FORMAL_RESULT_PASS condition={condition} reward={reward}")
PY

mv "$PROTOCOL_FILE" "$TRIAL/formal_protocol.txt"
trap - EXIT

AGENT_STDOUT="$TRIAL/agent/dsh.stdout.log"
AGENT_STDERR="$TRIAL/agent/dsh.stderr.log"
EFFECTIVE_CONFIG="$TRIAL/agent/effective-config.yml"
AGENT_AUTORESEARCH_AUDIT="$TRIAL/agent/autoresearch-audit.json"
EXPERIMENT_LOG="$TRIAL/artifacts/app/methods/experiment_log.md"
VERSIONS_DIR="$TRIAL/artifacts/app/methods/versions"
VERIFIER_SCORE_DETAILS="$TRIAL/verifier/score_details.json"
VERIFIER_GRADE_DEBUG="$TRIAL/verifier/grade_debug.json"

for required in \
  "$AGENT_STDOUT" \
  "$AGENT_STDERR" \
  "$EFFECTIVE_CONFIG" \
  "$AGENT_AUTORESEARCH_AUDIT" \
  "$VERIFIER_SCORE_DETAILS"; do
  test -f "$required" || { echo "missing required formal artifact: $required" >&2; exit 1; }
done

# The official prompt asks the model to use selfcheck/log/versioning, but those
# are agent research decisions, not extra hard benchmark validity gates.  A
# model that stops early or fails to improve should remain an observed result,
# not be silently rerun until it behaves better.  Record bookkeeping instead
# of requiring an arbitrary minimum version count.
VERSION_DIRS=()
if [ -d "$VERSIONS_DIR" ]; then
  mapfile -t VERSION_DIRS < <(find "$VERSIONS_DIR" -mindepth 1 -maxdepth 1 -type d -name 'v*' -print | sort -V)
fi

mapfile -t TRACE_FILES < <(
  find "$TRIAL/artifacts/logs/artifacts/dsh-home/sessions" \
    -type f -name 'session.jsonl.zstd' -print 2>/dev/null | sort
)
if [ "${#TRACE_FILES[@]}" -lt 1 ]; then
  echo "no structured DSH session trace found" >&2
  exit 1
fi
AGENT_TRACE_FILE="${TRACE_FILES[0]}"
TRACE_AUDIT="$TRIAL/agent/trace-audit.json"

# Audit the structured DSH trajectory.  Do not impose a minimum number of
# selfchecks/versions: those are agent decisions under the official protocol.
# Fail only on the known infrastructure bug where DSH's nested sandbox blocks
# bash before the requested command can execute.
"$HARBOR_PY" - "$AGENT_TRACE_FILE" "$TRACE_AUDIT" <<'PY'
import collections
import json
import subprocess
import sys
from pathlib import Path

trace_path = sys.argv[1]
out_path = Path(sys.argv[2])

calls = {}
results = {}
tool_counts = collections.Counter()

proc = subprocess.Popen(
    ["zstdcat", trace_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
assert proc.stdout is not None
for line in proc.stdout:
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    if not isinstance(obj, dict):
        continue
    typ = obj.get("type")
    d = obj.get("data")
    if not isinstance(d, dict):
        continue
    if typ == "tool/call":
        name = str(d.get("name", ""))
        call_id = str(d.get("callId", ""))
        raw_args = d.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            args = {"_raw": raw_args}
        if not isinstance(args, dict):
            args = {"_raw": args}
        calls[call_id] = {"name": name, "args": args}
        tool_counts[name] += 1
    elif typ == "tool/result":
        # DSH session JSONL stores tool results below data.message.
        # Keep a fallback for older/alternate encodings as well.
        message = d.get("message") if isinstance(d.get("message"), dict) else d
        source = message.get("source")
        content = message.get("content")
        if not isinstance(source, dict) or not isinstance(content, list):
            continue
        call_id = str(source.get("callId", ""))
        is_error = False
        texts = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "tool-result":
                continue
            is_error = is_error or bool(item.get("isError"))
            nested = item.get("content")
            if isinstance(nested, list):
                for part in nested:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        texts.append(part["text"])
            data = item.get("data")
            if isinstance(data, dict) and isinstance(data.get("text"), str):
                texts.append(data["text"])
        results[call_id] = {"is_error": is_error, "text": "\n".join(texts)}

stderr = proc.stderr.read() if proc.stderr is not None else ""
rc = proc.wait()
if rc != 0:
    raise SystemExit(f"zstdcat failed with code {rc}: {stderr}")

selfchecks = []
cordis_calls = []
sandbox_backend_failures = []
for call_id, call in calls.items():
    name = call["name"]
    args = call["args"]
    rendered_args = json.dumps(args, sort_keys=True, ensure_ascii=False)
    result = results.get(call_id, {"is_error": True, "text": ""})
    text = result.get("text", "") or ""
    if name == "bash" and "selfcheck" in rendered_args:
        selfchecks.append(
            {
                "call_id": call_id,
                "is_error": bool(result.get("is_error", True)),
                "score_signal": (
                    "oracle_normalized_auc70_final30" in text
                    or "visible oracle-normalized score" in text
                ),
            }
        )
    if (
        "no sandbox backend is usable" in text
        or "sandbox escalation to \"danger-full-access\" requires approval" in text
    ):
        sandbox_backend_failures.append(call_id)
    if "cordis" in name.lower() or "cordis" in rendered_args.lower():
        cordis_calls.append(call_id)

successful_selfchecks = sum(1 for x in selfchecks if not x["is_error"])
score_signal_selfchecks = sum(1 for x in selfchecks if x["score_signal"])
audit = {
    "trace_path": trace_path,
    "tool_counts": dict(sorted(tool_counts.items())),
    "selfcheck_calls": len(selfchecks),
    "successful_selfcheck_calls": successful_selfchecks,
    "selfchecks_with_score_signal": score_signal_selfchecks,
    "cordis_related_calls": len(set(cordis_calls)),
    "sandbox_backend_failures": len(set(sandbox_backend_failures)),
}
out_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(audit, sort_keys=True))
if sandbox_backend_failures:
    raise SystemExit(
        "AUTORESEARCH_TRACE_AUDIT_FAIL: DSH nested-sandbox infrastructure "
        "blocked tool execution"
    )
PY

# Cross-check the agent-side bookkeeping audit without inventing a minimum
# number of versions/selfchecks beyond the official protocol.
"$HARBOR_PY" - "$AGENT_AUTORESEARCH_AUDIT" <<'PY'
import json
import sys
from pathlib import Path
x = json.loads(Path(sys.argv[1]).read_text())
assert x["final_solver_exists"] is True
print(
    "FORMAL_AUTORESEARCH_BOOKKEEPING_AUDIT "
    f"experiment_log={int(bool(x.get('experiment_log_nonempty')))} "
    f"versions={int(x.get('version_count', 0))}"
)
PY

TRACE_ARCHIVE="$JOB_ROOT/${JOB_NAME}.agent-trace.tar.zst"
TRACE_MANIFEST="$JOB_ROOT/${JOB_NAME}.agent-trace.manifest.txt"
ARCHIVE_INPUTS=(
  "agent/dsh.stdout.log"
  "agent/dsh.stderr.log"
  "agent/effective-config.yml"
  "agent/rendered-prompt.sha256"
  "agent/prompt-source.txt"
  "agent/autoresearch-audit.json"
  "agent/trace-audit.json"
  "artifacts/app/methods"
  "result.json"
  "formal_protocol.txt"
  "verifier/reward.json"
  "verifier/reward.txt"
  "verifier/score_details.json"
)
if [ -f "$TRIAL/verifier/grade_debug.json" ]; then
  ARCHIVE_INPUTS+=("verifier/grade_debug.json")
fi
if [ -f "$TRIAL/verifier/test-stdout.txt" ]; then
  ARCHIVE_INPUTS+=("verifier/test-stdout.txt")
fi
for trace in "${TRACE_FILES[@]}"; do
  ARCHIVE_INPUTS+=("${trace#"$TRIAL/"}")
done

tar --zstd -C "$TRIAL" -cf "$TRACE_ARCHIVE" "${ARCHIVE_INPUTS[@]}"
TRACE_ARCHIVE_SHA256=$(sha256sum "$TRACE_ARCHIVE" | awk '{print $1}')
{
  echo "job_name=$JOB_NAME"
  echo "condition=$CONDITION"
  echo "run_number=$RUN_NUM"
  echo "agent_version=$EXPECTED_AGENT_VERSION"
  echo "archive_sha256=$TRACE_ARCHIVE_SHA256"
  echo "trace_count=${#TRACE_FILES[@]}"
  echo "version_count=${#VERSION_DIRS[@]}"
  echo
  echo "===== archive contents ====="
  tar --zstd -tf "$TRACE_ARCHIVE"
} > "$TRACE_MANIFEST"

TRACE_SELFCHECKS=$("$HARBOR_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["successful_selfcheck_calls"])' "$TRACE_AUDIT")
printf 'FORMAL_INFRASTRUCTURE_AUDIT_PASS successful_selfchecks=%s versions=%s\n' "$TRACE_SELFCHECKS" "${#VERSION_DIRS[@]}"
printf 'FORMAL_RUN_PASS=%s\n' "$JOB_NAME"
printf 'JOB_RESULT=%s\n' "$JOB_ROOT/result.json"
printf 'TRIAL_RESULT=%s\n' "$TRIAL/result.json"
printf 'FINAL_SOLVER=%s\n' "$TRIAL/artifacts/app/methods/main/solver.py"
printf 'EXPERIMENT_LOG=%s\n' "$EXPERIMENT_LOG"
printf 'VERSIONS_DIR=%s\n' "$VERSIONS_DIR"
printf 'VERIFIER_SCORE_DETAILS=%s\n' "$VERIFIER_SCORE_DETAILS"
printf 'AGENT_TRACE_FILE=%s\n' "$AGENT_TRACE_FILE"
printf 'AGENT_TRACE_ARCHIVE=%s\n' "$TRACE_ARCHIVE"
printf 'AGENT_TRACE_ARCHIVE_SHA256=%s\n' "$TRACE_ARCHIVE_SHA256"
printf 'AGENT_TRACE_MANIFEST=%s\n' "$TRACE_MANIFEST"
printf 'AGENT_TRACE_AUDIT=%s\n' "$TRACE_AUDIT"
printf 'FULL_DSH_STDOUT=%s\n' "$AGENT_STDOUT"
printf 'FULL_DSH_STDERR=%s\n' "$AGENT_STDERR"
printf 'EFFECTIVE_DSH_CONFIG=%s\n' "$EFFECTIVE_CONFIG"
df -h "$BBO_ROOT" /tmp
