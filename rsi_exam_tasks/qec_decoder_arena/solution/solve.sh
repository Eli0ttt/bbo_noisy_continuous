#!/usr/bin/env bash
# Oracle solver: install the selected anchor decoder as the agent deliverable.
#   ANCHOR=baseline|sota  bash solve.sh
# NB: the sota anchor uses chromobius + clorco via CHROMOBIUS_SRC and so only
# runs author-side; baseline is pure stim+pymatching and runs in the agent image.
set -euo pipefail
ANCHOR="${ANCHOR:-baseline}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p /app/methods/main
cp -r "$HERE/methods/$ANCHOR/." /app/methods/main/
echo "installed anchor '$ANCHOR' -> /app/methods/main/"
