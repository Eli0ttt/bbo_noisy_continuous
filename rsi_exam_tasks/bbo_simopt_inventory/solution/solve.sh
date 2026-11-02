#!/usr/bin/env bash
set -euo pipefail

REF="${SOLUTION_DIR:-/solution}/reference_optimizer.py"
cp "$REF" /app/methods/main/solver.py
echo "[oracle] installed reference_optimizer.py -> /app/methods/main/solver.py"
