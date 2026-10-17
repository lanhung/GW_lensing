#!/usr/bin/env bash
# Run joint PE for every line "<scenario> <family> <event_id> <known_sky 0|1>" of a task list, while
# keeping at most SLOTS joint_pe events running on the host (counting other batches as well).
# Usage: scripts/run_pe_queue.sh TASKS OUTDIR [SLOTS] [NPOOL] [NLIVE] [NACCEPT]
set -euo pipefail
TASKS=$1; OUT=$2; SLOTS=${3:-2}; NPOOL=${4:-8}; NLIVE=${5:-150}; NACCEPT=${6:-30}
PY=${PY:-python}
mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OUT SLOTS NPOOL NLIVE NACCEPT PY
running() {  # distinct joint_pe events (a known-sky run counts separately from a free-sky one)
  pgrep -fa 'gwlens\.joint_pe --catalogue' | grep -E '^[0-9]+ [^ ]*python' | grep -o -- '--event-ids .*' | sort -u | wc -l
}
export -f running
xargs -P "$SLOTS" -L 1 bash -c '
  scen=$0; fam=$1; eid=$2; ks=$3
  suf=""; flag=""; [ "$ks" = 1 ] && suf=_knownsky && flag=--known-sky
  [ -f "$OUT/$scen/${scen}_${eid}${suf}_summary.json" ] && exit 0
  exec 9> "$OUT/.slot.lock"; flock 9
  while [ "$(running)" -ge "$SLOTS" ]; do sleep 60; done
  nice -n 10 $PY -m gwlens.joint_pe --catalogue data/pe/$fam.parquet --scenario $scen --event-ids $eid \
    --outdir "$OUT/$scen" --nlive $NLIVE --naccept $NACCEPT --npool $NPOOL $flag \
    > "$OUT/logs/${scen}_${eid}${suf}.log" 2>&1 9>&- &
  pid=$!; sleep 30; flock -u 9; exec 9>&-
  wait $pid || echo "FAILED $scen $eid$suf"
' < "$TASKS"
touch "$OUT/queue.done"
