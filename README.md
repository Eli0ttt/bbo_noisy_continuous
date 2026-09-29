# BBO Noisy Continuous — Autonomous Research Workflows

This repository contains three experimental workflows for the RSI-Exam task
[`bbo_noisy_continuous_v1`](https://rsi-exam.ai/tasks/bbo_noisy_continuous_v1.html):

1. **Single-model autoresearch** — one persistent research model edits the full solver.
2. **Dual-model reviewer collaboration (Scheme B)** — a primary model writes the full solver and a second model performs post-evaluation review.
3. **Dual-model split collaboration (Scheme A)** — two models modify different owned parts of the solver in parallel and their changes are merged deterministically.

The goal of this repository is to study **research orchestration**, not to change the benchmark itself. All three workflows target the same optimizer interface and ultimately submit a self-contained `/app/methods/main/solver.py` for Harbor's sealed evaluation.

---

## 1. Repository layout

```text
bbo_noisy_continuous/
├── dsh_rsi_agent/
│   ├── dsh_bbo_agent.py              # single-model agent
│   ├── bbo_collab_agent.py            # Scheme B coordinator
│   ├── bbo_split_collab_agent.py      # Scheme A coordinator
│   ├── audit_collab_job.py            # Scheme B audit
│   └── audit_split_job.py             # Scheme A audit
│
├── dsh_rsi_config/
│   ├── bbo-collab-common.yml          # Scheme B shared config
│   ├── bbo-collab-primary.yml         # Scheme B primary model
│   ├── bbo-collab-reviewer.yml        # Scheme B reviewer
│   ├── bbo-collab-guard.mjs           # Scheme B tool/role guard
│   │
│   ├── bbo-split-agent-a.yml          # Scheme A model A
│   ├── bbo-split-agent-b.yml          # Scheme A model B
│   ├── bbo-split-guard.yml            # Scheme A guard config
│   ├── bbo-split-guard.mjs            # Scheme A tool/role guard
│   ├── bbo-split-merge.py             # deterministic AST merger
│   └── bbo_candidate_preflight.py     # candidate structural/API checks
│
├── formal-run.sh                       # single-model entry point
├── collab-run.sh                       # Scheme B entry point
├── split-collab-run.sh                 # Scheme A entry point
├── instruction.md
├── task.toml
└── .gitignore
```

Generated Harbor/DSH outputs such as `dsh_rsi_jobs/` and `dsh_rsi_runs/` are intentionally not tracked by Git.

---

## 2. Common experimental structure

At a high level, all three workflows use the same outer evaluation structure:

```text
Harbor
  │
  ├── creates the task/container environment
  │
  ├── launches the selected research agent
  │
  ▼
Research workflow
  │
  ├── reads the official task instruction
  ├── modifies candidate solver(s)
  ├── performs allowed structural/smoke checks
  ├── uses coordinator-controlled visible selfcheck
  └── selects a final solver
  │
  ▼
/app/methods/main/solver.py
  │
  ▼
Harbor sealed verifier
  │
  ▼
Hidden reward
```

### Responsibility boundaries

**Research model(s)** propose algorithms and write solver code.

**Harness/coordinator** controls the research protocol: prompts, candidate delivery, version bookkeeping, role restrictions, validation, visible selfcheck, selection, and audit artifacts.

**Harbor** owns the outer trial lifecycle and the final independent sealed verification.

Visible selfcheck is treated as development evidence. The final Harbor reward is produced separately by the sealed verifier.

---

# Workflow 1 — Single-model autoresearch

## 3. Idea

The single-model workflow uses one persistent autonomous research model. The model can modify the complete optimizer and repeatedly perform:

```text
current canonical solver
        │
        ▼
research hypothesis
        │
        ▼
edit full solver
        │
        ▼
checkpoint + visible selfcheck
        │
        ▼
inspect evidence
   ┌────┴────┐
   ▼         ▼
 KEEP      REVERT
   │         │
   └────┬────┘
        ▼
next experiment
```

The same model therefore performs both algorithm design and implementation.

## 4. Prompt / research protocol

The single model receives the official autoresearch-style task context and is asked to work as an autonomous researcher:

```text
read the current solver and previous evidence
        ↓
form a hypothesis
        ↓
implement one experiment
        ↓
measure it through the controlled checkpoint/selfcheck path
        ↓
record the result
        ↓
keep or revert
        ↓
continue research
```

The model is also reminded that the visible score is only a development proxy and that the final evaluation is sealed.

## 5. Version control inside a run

The harness maintains a checkpoint ledger so that every measured score is tied to a concrete solver snapshot.

Typical recorded information includes:

```text
version
parent version
solver snapshot / SHA256
experiment description
visible overall score
visible anytime score
visible final score
keep / revert / submit decision
```

Rejected experiments remain available as research history instead of disappearing.

The important distinction from Scheme A is that **the single research model may explicitly choose the final canonical checkpoint based on its interpretation of the development evidence; the harness verifies that the submitted solver matches that committed checkpoint.**

## 6. Entry point

```bash
./formal-run.sh
```

The formal research budget is defined by the task configuration. Shorter runs can be used during workflow testing when appropriate.

---

# Workflow 2 — Dual-model reviewer collaboration (Scheme B)

## 7. Idea

Scheme B keeps one model responsible for the **entire solver implementation**, while a second model acts as an **evidence-based post-evaluation reviewer**.

```text
canonical parent
      │
      ▼
Primary A
one research hypothesis
+ full solver candidate
      │
      ▼
Coordinator
structural/API checks
      │
      ▼
visible selfcheck
      │
      ▼
measured candidate + diff + history
      │
      ▼
Reviewer B
post-evaluation review
      │
      ▼
review.md
      │
      ▼
next Primary A iteration
```

The reviewer does **not** write the solver and does **not** decide the winner.

## 8. Primary A

Primary A receives the current parent solver, recent measured outcomes, task constraints, and the previous review.

Its research discipline is:

> **one iteration = one falsifiable mechanism whenever possible**

The goal is to avoid changing many unrelated optimizer mechanisms at once, because broad changes make score attribution difficult.

Primary A produces a complete candidate solver.

## 9. Reviewer B

Reviewer B runs **after the candidate has been evaluated**.

It receives evidence such as:

```text
current parent
current candidate
candidate diff
candidate visible score
recent experiment index/history
primary delivery summary
```

The reviewer is expected to distinguish:

- **Observed evidence** — directly supported by code/score artifacts.
- **Inference** — a plausible interpretation of the observation.
- **Hypothesis** — the next mechanism worth testing.

A useful review describes what changed, what the score evidence says, what should be preserved, one next falsifiable experiment, implementation guidance, query-budget implications, and relevant guardrails.

The reviewer does not vote on keep/revert and does not run the coordinator-owned visible selfcheck.

## 10. Why post-evaluation review?

The reviewer is intended to answer:

```text
What did A actually change?
What happened to the measured score?
Which explanation is evidence vs inference?
Which mechanism should be preserved?
Which mechanism should be tested next?
Has this experiment already failed historically?
What implementation/process checks should A perform?
```

This turns the second model into an **experiment interpreter and next-experiment designer**, rather than a second independent solver author.

## 11. Selection

Candidate measurement remains objective:

```text
candidate
   ↓
visible selfcheck
   ↓
measured history
```

At the research deadline, the workflow submits the highest visible-scored structurally/API-valid solver according to the coordinator's selection policy. Reviewer advice influences future research, but the reviewer has no direct vote over final selection.

## 12. Entry point

Example one-hour no-Cordis run:

```bash
COLLAB_TOTAL_SEC=3600 \
./collab-run.sh <run-id> no-cordis
```

Formal 12-hour research:

```bash
COLLAB_TOTAL_SEC=43200 \
./collab-run.sh <run-id> no-cordis
```

The current implementation also supports reviewer/model configuration through the corresponding environment variables and YAML configuration files.

---

# Workflow 3 — Dual-model split collaboration (Scheme A)

## 13. Idea

Scheme A tests a different collaboration hypothesis: instead of using the second model as a reviewer, **both models write code in parallel**, but each owns a different part of the optimizer.

```text
                    parent solver P
                          │
                 machine contract
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
          Model A                 Model B
       state / update          proposal / search
              │                       │
       A/candidate.py          B/candidate.py
       A/handoff.json          B/handoff.json
              │                       │
              └───────────┬───────────┘
                          ▼
                 ownership validation
                          │
                          ▼
              deterministic AST merger
                          │
       M = P + owned ΔA + owned ΔB
                          │
                          ▼
                   merged solver
                          │
                          ▼
                      preflight
                          │
                          ▼
                 visible selfcheck
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        next development          champion
            parent             best visible solver
```

There is **no reviewer** in this workflow.

## 14. Model A ownership

Model A is the **state / learning / update owner**.

Typical ownership:

```text
Optimizer.__init__
Optimizer.tell
A-owned helpers
new _a_* helpers
imports
```

Its prompt asks it to improve state representation, learning/update logic, observation handling, noise adaptation, and related mechanisms without modifying B-owned proposal logic.

## 15. Model B ownership

Model B is the **proposal / query-schedule owner**.

Typical ownership:

```text
Optimizer.ask
batch
B-owned helpers
new _b_* helpers
```

Its prompt asks it to improve candidate generation, exploration/exploitation, proposal distributions, and query scheduling without modifying A-owned update logic.

## 16. Machine-checkable contract

Both models receive the same parent solver and a machine-generated ownership contract.

The contract separates:

```text
A-owned symbols
B-owned symbols
shared/frozen symbols
role-prefixed helpers
peer-visible state/interface constraints
```

The models do not rely only on natural-language instructions. Their candidates are mechanically checked against the ownership rules.

## 17. Deterministic merge

The merger parses:

```text
parent
A candidate
B candidate
```

and validates that each model changed only its owned AST region.

Conceptually:

```text
Merged = Parent
       + A-owned changes
       + B-owned changes
```

The merge is deterministic; a third LLM is not used to reconcile the two patches.

Invalid ownership changes fail closed or go through the limited mechanical-repair path before merge.

## 18. Cross-role communication

A and B do not depend on uncommitted same-round state created by the other model.

Each role can write a `handoff.json` describing useful requests or observations for the peer. Once a change is successfully merged into the next parent, both models can safely consume it in the following round.

This intentionally introduces a one-round delay for new cross-role interfaces in exchange for deterministic integration.

## 19. Development lineage vs champion

Scheme A keeps two concepts separate:

**Development lineage**

```text
latest structurally valid merged solver
        ↓
next round parent
```

Research can therefore continue from the latest merged idea even if its visible score regresses.

**Champion**

```text
best visible-scored valid merged solver
        ↓
final submission
```

This preserves continuous research while avoiding intentional submission of a visibly worse historical version.

## 20. Entry point

Recommended one-hour engineering/research run:

```bash
COLLAB_TOTAL_SEC=3600 \
COLLAB_MIN_NEW_ROUND_SEC=600 \
./split-collab-run.sh <run-id> no-cordis
```

Formal 12-hour research:

```bash
COLLAB_TOTAL_SEC=43200 \
COLLAB_MIN_NEW_ROUND_SEC=600 \
./split-collab-run.sh <run-id> no-cordis
```

`COLLAB_MIN_NEW_ROUND_SEC` only prevents starting a new round when too little global time remains; it is not a per-role thinking-time cap.

---

# 21. Tool conditions

The experiments support two tool conditions.

## no-cordis

The base research environment provides the coding/research tools needed by the workflow, including filesystem operations and shell execution. Coordinator-controlled evaluation remains separate from model-owned algorithm editing.

## dynamic-cordis

`dynamic-cordis` starts from the same base tool configuration and overlays the Cordis integration.

This distinction should be interpreted carefully: **making Cordis available does not imply that a particular research trajectory actually called it.** Tool traces/audits should be checked before attributing a result to Cordis usage.

---

# 22. Comparison of the three workflows

| Property | Single model | Scheme B: Reviewer | Scheme A: Split |
|---|---|---|---|
| Research models | 1 | 2+ | 2 |
| Full solver author | Single model | Primary A | Neither role alone |
| Second-model role | — | Post-evaluation reviewer | Parallel code author |
| Code ownership split | No | No | Yes |
| Reviewer | No | Yes | No |
| Merge required | No | No | Deterministic AST merge |
| Visible evaluation | Harness-controlled | Coordinator-controlled | Coordinator-controlled |
| Research feedback | Same persistent model | Score + reviewer feedback | Score + cross-round handoff |
| Final selection | Committed canonical checkpoint | Best valid visible-scored solver | Best valid visible-scored champion |
| Main entry point | `formal-run.sh` | `collab-run.sh` | `split-collab-run.sh` |

---

# 23. Representative experiment records

These are **individual experimental runs**, not claims that one workflow is universally better than another. Research trajectories are stochastic, so comparisons should use replicated runs.

## Single-model records

| Condition | Candidate iterations | Best visible version | Best visible overall | Submitted version | Submitted visible | Harbor hidden reward |
|---|---:|---:|---:|---:|---:|---:|
| no-cordis | 22 | v15 | 0.215684 | v15 | 0.215684 | 0.200485 |
| dynamic-cordis | 16 | v10 | 0.230607 | v16 | 0.229242 | 0.211615 |

In the recorded dynamic-Cordis trajectory, Cordis was available but was not actually called, so the difference from no-Cordis should not be interpreted as a causal Cordis effect.

## Scheme B representative R28 records

A 30-minute no-Cordis run attempted 7 versions and selected **v6**:

```text
best visible score : 0.277414
Harbor hidden reward: 0.265984
```

A separate one-hour no-Cordis R28 run attempted 14 versions and selected **v10**:

```text
best visible score : 0.198084
Harbor hidden reward: 0.209129
```

The difference between these independent trajectories is an important reminder that research-path variance can be substantial.

## Scheme A frozen engineering checkpoint

The one-hour no-Cordis run 17 completed without Harbor exceptions:

```text
versions recorded  : 3
visible champion   : v2
champion score     : 0.046430
Harbor hidden reward: 0.044086
```

The run demonstrated the intended chain:

```text
parallel A/B candidates
→ ownership validation
→ deterministic merge
→ visible scoring
→ next parent / champion tracking
→ sealed Harbor verification
```

Earlier Scheme A runs have produced different scores; run 17 is recorded here primarily as the current frozen workflow checkpoint.

---

# 24. Research artifacts and audit

The collaboration workflows keep research evidence separate from generated Harbor runtime output.

For Scheme A, the useful per-version evidence includes:

```text
versions/vN/
├── A/
│   ├── candidate.py
│   └── handoff.json
├── B/
│   ├── candidate.py
│   └── handoff.json
├── merged/
│   └── solver.py
└── merge-report.json
```

A canonical `audit.json` summarizes the run and compressed traces preserve lower-level model/tool activity when deeper debugging is required.

For Scheme B, the important research evidence includes the candidate solver/diff, experiment metadata, reviewer `review.md`, measured history, final selection, and audit information.

Large generated job directories are excluded from Git and should be archived separately when needed for experiment analysis.

---

# 25. Reproducing runs

Environment credentials/configuration are intentionally not committed.

After configuring the required environment variables, choose one of the three entry points:

```bash
# Single model
./formal-run.sh

# Scheme B — reviewer collaboration
COLLAB_TOTAL_SEC=3600 \
./collab-run.sh <run-id> no-cordis

# Scheme A — split collaboration
COLLAB_TOTAL_SEC=3600 \
COLLAB_MIN_NEW_ROUND_SEC=600 \
./split-collab-run.sh <run-id> no-cordis
```

Use a fresh run ID for each experiment so that results remain independent and auditable.

---

## Notes

- The official benchmark/task constraints remain authoritative.
- Development visible scores and Harbor sealed rewards are different signals and should be reported separately.
- A single run is not sufficient for a causal comparison between orchestration strategies or tool conditions.
- The two dual-model schemes are intentionally kept as separate implementations so that Scheme A does not overwrite or modify Scheme B.
