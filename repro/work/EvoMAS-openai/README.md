# EvoMAS - Evolutionary Generation of Multi-Agent Systems

This repository implements the following paper:

> **EvoMAS: Evolutionary Generation of Multi-Agent Systems**
> Yuntong Hu, Yuting Zhang, Matthew Trager, Yi Zhang, Shuo Yang, Wei Xia, Stefano Soatto
> *ICML 2026*
> [[arXiv]](https://arxiv.org/abs/2602.06511)

This project is released under the [CC BY-NC 4.0](LICENSE) license. You are free to share and adapt the material for non-commercial purposes, with appropriate credit to the authors. Commercial use is not permitted.

---

EvoMAS uses an LLM meta-model as an evolutionary operator to select, mutate,
and cross over MAS configurations from a pool, evaluating them on target
benchmarks. This README covers the short path: install, then run one of the
benchmark scripts.

## Setup

```bash
conda create -n mas python=3.11 -y
conda activate mas
pip install -r requirements.txt

# API keys — only fill in the providers you'll use
cp .env.example .env
# then edit .env
```

AWS Bedrock models use the standard credential chain (`aws configure`, env
vars, or an EC2 instance role). No `.env` entry is needed when AWS CLI
credentials are already set up.

**Note on model availability:** Some model IDs (e.g., on Bedrock) referenced
in the default configuration and MAS pool files may be deprecated or retired
over time. If you encounter model-not-found errors, update the model IDs to
currently available versions. You can override model IDs without editing source
code via `META_MODEL`, `JUDGE_MODEL`, and `AGENT_MODELS` environment variables
(see [Editing parameters](#editing-parameters)), or pass `--meta-model-id` /
`--model-list` flags to `main.py` directly.

## Download datasets

Task data is not shipped with the repo. Run the one-click preparation script
to download every benchmark into the paths the pipeline expects:

```bash
scripts/prepare_datasets.sh
```

This performs four idempotent steps:

1. **SWE-bench source repos** — clones 15 upstream projects into
   `dataset/repos/` at the pinned commits listed in `src/dataset/repos.txt`.
2. **BBEH** — downloads all 23 subsets from `google-deepmind/bbeh` into
   `dataset/bbeh/benchmark_tasks/` and rebuilds `bbeh_mini` (20 stride-sampled
   tasks per subset = 460 total).
3. **WorkBench** — clones `olly-styles/WorkBench` and stages per-domain
   `data.csv` + `test.json` under `dataset/workbench/<domain>/`.
4. **SWE-bench task metadata** — downloads Lite (300 tasks) and Verified
   (500 tasks) from HuggingFace into `dataset/swe_bench_{lite,verified}/`.

Useful flags:

```bash
scripts/prepare_datasets.sh --force         # re-fetch everything
scripts/prepare_datasets.sh --skip-repos    # everything except the ~GB repo clones
scripts/prepare_datasets.sh --bbeh-only     # target a single dataset family
scripts/prepare_datasets.sh --workbench-only
scripts/prepare_datasets.sh --swe-only
scripts/prepare_datasets.sh --repos-only
```

## Running a benchmark

Three benchmark scripts live in `scripts/`. Each accepts a subset/variant
either as a positional arg or as an env var, and writes outputs under
`$OUTPUT_ROOT/<dataset>_main/` (default `output_paper/`).

```bash
# BBEH — 24 subsets available under dataset/bbeh/benchmark_tasks/
scripts/run_bbeh.sh mini
scripts/run_bbeh.sh boolean_expressions
SUBSET=word_sorting scripts/run_bbeh.sh

# WorkBench — 6 subdomains
scripts/run_workbench.sh email                # default
scripts/run_workbench.sh calendar
SUBDOMAIN=multi_domain scripts/run_workbench.sh

# SWE-bench — verified (default) or lite
scripts/run_swebench.sh verified
scripts/run_swebench.sh lite
VARIANT=lite scripts/run_swebench.sh
```

Each script auto-resolves `NUM_EVAL_TASKS` to the full task count of the
chosen subset / domain / variant (e.g. 460 for `bbeh_mini`, 500 for
`swe_bench_verified`). To run on a smaller sample, override via env var
(see below).

There is also a batch-size ablation script:

```bash
DATASET=bbeh_mini scripts/run_ablation_batch_size.sh     # sweep BS ∈ {1, 10, 460}
```

## Editing parameters

All tunable parameters are env vars with defaults in `scripts/common.sh`.
Override any of them before invoking a script:

| Variable | Default | Description |
|---|---|---|
| `NUM_EVAL_TASKS` | (full dataset) | Number of tasks to run |
| `MAX_STEPS` | `2` | Evolutionary iterations per batch |
| `NUM_PARENTS` | `2` | Parent configs the meta-model selects per batch |
| `SEED` | `42` | Random seed |
| `BATCH_SIZE` | `1` | Tasks per evolution batch (1 = per-query; `N` = one shared trajectory over N tasks) |
| `WORKERS` | `16` | Parallel batches (ThreadPoolExecutor) |
| `META_MODEL` | `bedrock:global.anthropic.claude-sonnet-4-5-...` | Evolutionary-operator LLM |
| `JUDGE_MODEL` | same as `META_MODEL` | LLM-as-judge for reward |
| `AGENT_MODELS` | `bedrock:us.anthropic.claude-3-5-sonnet-... bedrock:qwen.qwen3-235b-... bedrock:qwen.qwen3-coder-480b-...` | Space-separated worker model palette |
| `MEMORY_EVOLUTION` | `true` | Persist meta-model memory updates |
| `MEMORY_PATH` | (auto) | Explicit memory JSON path; empty = `dataset/<subset-path>/memory_<ts>.json` |
| `OUTPUT_ROOT` | `output_paper` | Root directory for run outputs |
| `CONDA_ENV` | `mas` | conda environment name |
| `EVOMAS_REPOS_DIR` | `dataset/repos` | Directory containing cloned SWE-bench source repos |

Examples:

```bash
# Small smoke test
NUM_EVAL_TASKS=1 MAX_STEPS=1 WORKERS=1 scripts/run_bbeh.sh mini

# Larger parallel run on workbench_email
WORKERS=32 scripts/run_workbench.sh email

# Swap the worker palette to a single model
AGENT_MODELS="bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0" \
    scripts/run_bbeh.sh boolean_expressions

# Continue a prior memory file instead of starting fresh
MEMORY_PATH=dataset/bbeh/benchmark_tasks/bbeh_mini/memory_20260423_101510.json \
    scripts/run_bbeh.sh mini
```

## Direct `main.py` invocation

Under the hood, each script runs `python main.py` with the resolved
arguments. For ad-hoc configurations not covered by the scripts, call
`main.py` directly:

```bash
python main.py --dataset bbeh_boolean_expressions \
    --num-eval-tasks 50 \
    --batch-size 1 \
    --workers 8 \
    --max-steps 2 \
    --meta-model-id bedrock:global.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --llm-as-judge bedrock:global.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --model-list bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0 \
                 bedrock:qwen.qwen3-235b-a22b-2507-v1:0 \
                 bedrock:qwen.qwen3-coder-480b-a35b-v1:0

python main.py --help
```

CLI flags mirror the env-var names (`--num-eval-tasks`, `--batch-size`,
`--workers`, `--memory-path`, `--memory-evolution`, `--task-ids`, etc.).
`--task-ids` accepts int indices (BBEH / WorkBench) or string instance IDs
(SWE-bench, e.g. `astropy__astropy-12907`).
