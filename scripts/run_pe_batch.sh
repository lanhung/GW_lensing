#!/usr/bin/env bash
# Run joint PE for every line "<scenario> <family> <event_id>" of a task list, PAR events at a time.
# Usage: scripts/run_pe_batch.sh TASKS OUTDIR [PAR] [NPOOL] [NLIVE] [NACCEPT]
# KNOWN_SKY=1 fixes the sky position to the truth (identified host); outputs get the suffix _knownsky.
set -euo pipefail
TASKS=$1; OUT=$2; PAR=${3:-7}; NPOOL=${4:-8}; NLIVE=${5:-150}; NACCEPT=${6:-30}
PY=${PY:-python}
SUF=""; FLAG=""; [ "${KNOWN_SKY:-0}" = 1 ] && SUF=_knownsky && FLAG=--known-sky
mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
xargs -P "$PAR" -L 1 bash -c '
  scen=$0; fam=$1; eid=$2
  [ -f "'"$OUT"'/$scen/${scen}_${eid}'"$SUF"'_summary.json" ] && exit 0
  nice -n 10 '"$PY"' -m gwlens.joint_pe --catalogue data/pe/$fam.parquet --scenario $scen --event-ids $eid \
    --outdir "'"$OUT"'/$scen" --nlive '"$NLIVE"' --naccept '"$NACCEPT"' --npool '"$NPOOL"' '"$FLAG"' \
    > "'"$OUT"'/logs/${scen}_${eid}'"$SUF"'.log" 2>&1 || echo "FAILED $scen $eid"
' < "$TASKS"
touch "$OUT/batch.done"
