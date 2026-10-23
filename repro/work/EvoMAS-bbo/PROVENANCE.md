# EvoMAS-BBO adapter workspace

Upstream EvoMAS:
- path: repro/upstream/EvoMAS
- commit: 93fd9d6766b093f6bbdeeffc35896910cbece6e2

OpenAI-compatible EvoMAS smoke fork:
- path: repro/work/EvoMAS-openai

BBO task:
- official task instruction copied from the existing project
- evaluator: official /app/selfcheck.py
- query budget: 120 per run
- visible instances: 12
- deterministic seeds: 20
- submission contract: one self-contained solver.py

Floor baseline:
- copied from repro/baseline_floor/solver.py
- visible score copied from repro/baseline_floor/visible_selfcheck.json
