#!/usr/bin/env bash
# Run gwlens.joint_gwem full for every event JSON listed in a file, PAR events at a time.
# Usage: scripts/run_gwem_batch.sh LIST OUT_DIR [PAR] [NPOOL] [extra joint_gwem arguments...]
set -euo pipefail
LIST=$1; OUT=$2; PAR=${3:-2}; NPOOL=${4:-16}; shift 4 || shift $#
PY=${PY:-python}
mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH:-.}
EXTRA="$*"
xargs -P "$PAR" -L 1 bash -c '
  ev=$0; base=$(basename "$ev" .json)
  ls "'"$OUT"'"/${base}_full_*_summary.json >/dev/null 2>&1 && exit 0
  '"$PY"' -m gwlens.joint_gwem full --event "$ev" --out-dir "'"$OUT"'" --npool '"$NPOOL"' '"$EXTRA"' \
    > "'"$OUT"'/logs/${base}.log" 2>&1 || echo "FAILED $base"
' < "$LIST"
touch "$OUT/batch.done"
