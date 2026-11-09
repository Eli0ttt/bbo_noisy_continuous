#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "${1:?Usage: run.sh ROOT STATE TASK MODE SEED}" && pwd)"
STATE="$(realpath "${2:?STATE}")"; TASK="${3:?TASK}"; MODE="${4:?MODE}"; SEED="${5:?SEED}"
[[ "$TASK" =~ ^(bbo|inventory|graph|qec)$ ]] || exit 2
[[ "$MODE" == original ]] || exit 2
[[ "$SEED" =~ ^[0-9]+$ ]] || exit 2
VERSIONS="${SUBSET_MAX_VERSIONS:-6}"
[[ "$VERSIONS" == 4 || "$VERSIONS" == 6 ]] || { echo "SUBSET_MAX_VERSIONS must be 4 or 6" >&2; exit 2; }
cd "$ROOT"
set -a
source "$ROOT/env/dsh_bbo_experiment.env"
set +a
export OPENAI_API_KEY="${OPENAI_API_KEY:-${DEEPSEEK_API_KEY:-}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${DEEPSEEK_BASE_URL:-}}"
test -n "$OPENAI_API_KEY"
[[ "$OPENAI_BASE_URL" == 'http://183.230.173.202:36789/v1' ]] || exit 1
FORK="$ROOT/repro/subset_limited_ahe"
test -f "$STATE/manifest.json";test -x "$HOME/.local/share/uv/tools/harbor/bin/harbor"
OUT="$FORK/outputs/${TASK}_${MODE}_seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
INPUT="$OUT.input"
python "$FORK/code/state.py" export --root "$ROOT" --state "$STATE" --task "$TASK" --dest "$INPUT"
printf 'RUN_DIR=%s\nINPUT_SNAPSHOT=%s\n' "$OUT" "$INPUT"
if [[ "$TASK" == bbo ]];then
  CONTAINER="${BBO_EVAL_CONTAINER:-bbo-evomas-visible-wenxinyi}"
  [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" == true ]]
  python "$FORK/task_preflight.py" bbo "$ROOT"
  PYTHONPATH="$FORK/adapter${PYTHONPATH:+:$PYTHONPATH}" \
    python "$FORK/adapter/evomas_bbo_pipeline.py" \
    --evomas-root "$FORK/EvoMAS-bbo" --seed-pool "$INPUT/working_pool" \
    --instruction "$ROOT/instruction.md" --output-dir "$OUT" --container "$CONTAINER" \
    --meta-model deepseek-v4-flash-s1 --model-pool deepseek-v4-flash-s1,deepseek-v4-flash-s2,GLM-5.3-W4A8,GLM-5.3-Flash \
    --num-parents 2 --max-solver-versions "$VERSIONS" --skip-final-repeat \
    --mutation-probability 0.7 --mutation-mode "$MODE" --tool-mode bbo \
    --seed "$SEED" --pool-warm-start \
    --knowledge-scope cross-run --initial-memory "$INPUT/memory.json" \
    --no-memory-evolution 2>&1 | tee "$OUT.log"
else
  case "$TASK" in
   inventory) NAME=bbo_simopt_inventory;;
   graph) NAME=johnson1991_leighton_graph_coloring;;
   qec) NAME=qec_decoder_arena;;
  esac
  TASK_ROOT="$ROOT/rsi_exam_tasks/$NAME"
  test -f "$TASK_ROOT/task.toml"
  docker image inspect "rsi-subset-$NAME:public" >/dev/null
  python "$FORK/code/run_subset_task.py" "$TASK" \
    --project-root "$ROOT" --task-root "$TASK_ROOT" --seed-pool "$INPUT/working_pool" \
    --initial-memory "$INPUT/memory.json" --output-dir "$OUT" \
    --mutation-mode "$MODE" --seed "$SEED" --max-versions "$VERSIONS" \
    --mutation-probability 0.7 2>&1 | tee "$OUT.log"
fi
cp "$INPUT/provenance.json" "$OUT/input_provenance.json"
python "$FORK/code/submit_hidden.py" --root "$ROOT" --run "$OUT" --task "$TASK" \
  2>&1 | tee "$OUT.hidden_handoff.log"
python "$FORK/code/state.py" commit --root "$ROOT" --state "$STATE" --run "$OUT" \
  2>&1 | tee "$OUT.memory_commit.log"
