#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "${1:?Usage: run.sh ROOT STATE TASK original SEED}" && pwd)"
STATE="$(realpath "${2:?STATE}")"
TASK="${3:?TASK}"; MODE="${4:?MODE}"; SEED="${5:?SEED}"
[[ "$TASK" =~ ^(bbo|inventory|graph|qec)$ && "$MODE" == original && "$SEED" =~ ^[0-9]+$ ]] || exit 2
set -a
source "$ROOT/env/dsh_bbo_experiment.env"
set +a
[[ "$BBO_ROOT" == "$ROOT" && "$OPENAI_BASE_URL" == 'http://183.230.173.202:36789/v1' ]] || exit 2
[[ "$META_MODEL" == 'deepseek-v4-flash-s1' && "$MODEL_POOL" == 'deepseek-v4-flash-s1,deepseek-v4-flash-s2' ]] || exit 2
[[ "$SUBSET_MAX_VERSIONS" == 6 && "$MUTATION_PROBABILITY" == 0.7 ]] || exit 2
[[ -n "$OPENAI_API_KEY" && -f "$STATE/manifest.json" && -x "$HARBOR_BIN" ]] || exit 2
FORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$FORK/outputs/${TASK}_${MODE}_seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
INPUT="$OUT.input"
python "$FORK/code/state.py" export --root "$ROOT" --state "$STATE" --task "$TASK" --dest "$INPUT"
printf 'RUN_DIR=%s\nINPUT_SNAPSHOT=%s\n' "$OUT" "$INPUT"
if [[ "$TASK" == bbo ]]; then
  [[ "$(docker inspect -f '{{.State.Running}}' "$BBO_EVAL_CONTAINER")" == true ]]
  python "$FORK/task_preflight.py" bbo "$ROOT"
  PYTHONPATH="$FORK/adapter${PYTHONPATH:+:$PYTHONPATH}" \
    python "$FORK/adapter/evomas_bbo_pipeline.py" \
      --evomas-root "$FORK/EvoMAS-bbo" --seed-pool "$INPUT/working_pool" \
      --instruction "$ROOT/instruction.md" --output-dir "$OUT" --container "$BBO_EVAL_CONTAINER" \
      --meta-model "$META_MODEL" --model-pool "$MODEL_POOL" \
      --num-parents 2 --max-solver-versions "$SUBSET_MAX_VERSIONS" --skip-final-repeat \
      --mutation-probability "$MUTATION_PROBABILITY" --mutation-mode "$MODE" --tool-mode bbo \
      --seed "$SEED" --pool-warm-start --knowledge-scope cross-run \
      --initial-memory "$INPUT/memory.json" --knowledge-snapshot "$INPUT" \
      --no-memory-evolution 2>&1 | tee "$OUT.log"
else
  case "$TASK" in
    inventory) NAME=bbo_simopt_inventory;;
    graph) NAME=johnson1991_leighton_graph_coloring;;
    qec) NAME=qec_decoder_arena;;
  esac
  TASK_ROOT="$RSI_TASKS_ROOT/$NAME"
  [[ -f "$TASK_ROOT/task.toml" ]] || exit 2
  docker image inspect "rsi-subset-$NAME:public" >/dev/null
  python "$FORK/code/run_subset_task.py" "$TASK" \
    --project-root "$ROOT" --task-root "$TASK_ROOT" --seed-pool "$INPUT/working_pool" \
    --initial-memory "$INPUT/memory.json" --output-dir "$OUT" \
    --mutation-mode "$MODE" --seed "$SEED" --max-versions "$SUBSET_MAX_VERSIONS" \
    --mutation-probability "$MUTATION_PROBABILITY" 2>&1 | tee "$OUT.log"
fi
cp "$INPUT/provenance.json" "$OUT/input_provenance.json"
python "$FORK/code/submit_hidden.py" --root "$ROOT" --run "$OUT" --task "$TASK" \
  2>&1 | tee "$OUT.hidden_handoff.log"
python "$FORK/code/state.py" commit --root "$ROOT" --state "$STATE" --run "$OUT" \
  2>&1 | tee "$OUT.memory_commit.log"
