#!/usr/bin/env bash
# Run joint PE for every line "<scenario> <family> <event_id>" of a task list, PAR events at a time.
# Usage: scripts/run_pe_batch.sh TASKS OUTDIR [PAR] [NPOOL] [NLIVE] [NACCEPT]
set -euo pipefail
TASKS=$1; OUT=$2; PAR=${3:-7}; NPOOL=${4:-8}; NLIVE=${5:-150}; NACCEPT=${6:-30}
PY=${PY:-python}
mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
xargs -P "$PAR" -L 1 bash -c '
  scen=$0; fam=$1; eid=$2
  [ -f "'"$OUT"'/$scen/${scen}_${eid}_summary.json" ] && exit 0
  nice -n 10 '"$PY"' -m gwlens.joint_pe --catalogue data/pe/$fam.parquet --scenario $scen --event-ids $eid \
    --outdir "'"$OUT"'/$scen" --nlive '"$NLIVE"' --naccept '"$NACCEPT"' --npool '"$NPOOL"' \
    > "'"$OUT"'/logs/${scen}_${eid}.log" 2>&1 || echo "FAILED $scen $eid"
' < "$TASKS"
touch "$OUT/batch.done"
