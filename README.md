# BBO Noisy Continuous Research Workflows

This repository contains three research workflows for `bbo_noisy_continuous_v1`:

1. **Single Model** — one autonomous model researches and edits the complete solver.
2. **Scheme B: Reviewer Collaboration** — a primary model writes the complete solver while a second model reviews evaluated experiments and guides the next iteration.
3. **Scheme A: Split Collaboration** — two models modify different owned parts of the solver in parallel, followed by deterministic AST-based merging.

## Repository layout

```text
bbo_noisy_continuous/
├── dsh_rsi_agent/
│   ├── dsh_bbo_agent.py              # Single-model workflow
│   ├── bbo_collab_agent.py           # Scheme B coordinator
│   ├── bbo_split_collab_agent.py     # Scheme A coordinator
│   ├── audit_collab_job.py           # Scheme B audit
│   └── audit_split_job.py            # Scheme A audit
│
├── dsh_rsi_config/
│   ├── bbo-collab-*                   # Scheme B configuration
│   ├── bbo-split-*                    # Scheme A configuration / merger
│   └── bbo_candidate_preflight.py     # Candidate validation
│
├── formal-run.sh                      # Single-model entry point
├── collab-run.sh                      # Scheme B entry point
├── split-collab-run.sh                # Scheme A entry point
├── instruction.md
└── task.toml
```

## 1. Single Model

A single persistent research model controls the complete optimizer. It reads the current solver and previous experimental evidence, proposes a hypothesis, edits the solver, evaluates the candidate through the controlled selfcheck workflow, and then keeps or reverts the experiment before continuing.

```text
solver → hypothesis → full-solver edit → selfcheck → keep/revert → next experiment
```

### Run

No-Cordis:

```bash
./formal-run.sh no-cordis
```

Dynamic-Cordis:

```bash
./formal-run.sh dynamic-cordis
```

## 2. Scheme A — Split Collaboration

Scheme A lets two models write different parts of the optimizer in parallel.

- **Model A** owns state, learning, initialization, and update logic such as `__init__` and `tell`.
- **Model B** owns proposal generation and query scheduling such as `ask` and `batch`.

Both models receive the same parent solver and ownership contract. Their candidates are mechanically checked, and valid role-owned changes are combined by a deterministic AST merger.

There is no reviewer or LLM-based semantic merge.

```text
                 parent + contract
                  /             \
                 ↓               ↓
        Model A: state      Model B: search
                 \               /
                  ↓             ↓
                ownership validation
                         ↓
               deterministic AST merge
                         ↓
                      selfcheck
                         ↓
              next parent + champion
```

### Run

No-Cordis:

```bash
COLLAB_TOTAL_SEC=43200 \
COLLAB_MIN_NEW_ROUND_SEC=600 \
./split-collab-run.sh <run-id> no-cordis
```

Dynamic-Cordis:

```bash
COLLAB_TOTAL_SEC=43200 \
COLLAB_MIN_NEW_ROUND_SEC=600 \
./split-collab-run.sh <run-id> dynamic-cordis
```

For shorter experiments, change `COLLAB_TOTAL_SEC`, for example `3600` for one hour.


## 3. Scheme B — Reviewer Collaboration

Scheme B separates **implementation** from **post-evaluation review**.

Primary A writes the complete solver and proposes one research mechanism at a time. After the candidate is evaluated, Reviewer B receives the candidate, diff, score, and recent experiment history. B analyzes the evidence and writes a review that guides A's next iteration.

The reviewer does not edit the solver and does not choose the final version.

```text
parent
  ↓
Primary A: hypothesis + full solver
  ↓
selfcheck
  ↓
Reviewer B: evidence-based review
  ↓
next Primary A iteration
```

### Run

No-Cordis:

```bash
COLLAB_TOTAL_SEC=43200 \
./collab-run.sh <run-id> no-cordis
```

Dynamic-Cordis:

```bash
COLLAB_TOTAL_SEC=43200 \
./collab-run.sh <run-id> dynamic-cordis
```

For shorter experiments, change `COLLAB_TOTAL_SEC`, for example `3600` for one hour.

