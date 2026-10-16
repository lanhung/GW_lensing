#!/usr/bin/env bash
# Write self-contained GW+EM event inputs (JSON) for every "<scenario> <family> <event_id>" line
# of a task list (e.g. results/pe/tasks_all.txt from scripts/select_pe_events.py).
# Usage: scripts/prepare_gwem_inputs.sh TASKS OUT_DIR [PAR]
set -euo pipefail
TASKS=$1; OUT=$2; PAR=${3:-8}
PY=${PY:-python}
mkdir -p "$OUT"
xargs -P "$PAR" -L 1 bash -c '
  [ -f "'"$OUT"'/$0_$2.json" ] && exit 0
  '"$PY"' -m gwlens.joint_gwem prepare --catalogue data/pe/$1.parquet --scenario $0 --event-ids $2 --out-dir "'"$OUT"'" 2>/dev/null
' < "$TASKS"
