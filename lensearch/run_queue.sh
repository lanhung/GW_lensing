#!/usr/bin/env bash
# Run lensearch.realpe tasks from a list with a fixed number of concurrent slots.
#   lensearch/run_queue.sh TASKS SLOTS NPOOL QUEUE_DIR
# Each line of TASKS holds the realpe arguments of one run; options on the line override the defaults below, e.g.
#   joint GW230707_124047 GW230708_230935 --outdir results/stage2
#   single X --prior-events X Y --injections results/inject/placed_s1.csv --noise-dir data/inject_noise_s1 --nlive 500 --outdir results/inject_pe
# A task whose marker QUEUE_DIR/done/<line> exists is skipped, so the queue can be restarted at any time.
# Environment: PY (python), PE_DIR (data/pe), CACHE (data/strain), NLIVE (1000), NACCEPT (60), APPROX (IMRPhenomXPHM).
set -u
TASKS=$1; SLOTS=$2; NPOOL=$3; Q=$4
PY=${PY:-python}
PE_DIR=${PE_DIR:-data/pe}; CACHE=${CACHE:-data/strain}
NLIVE=${NLIVE:-1000}; NACCEPT=${NACCEPT:-60}; APPROX=${APPROX:-IMRPhenomXPHM}
mkdir -p "$Q/logs" "$Q/done"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
n=0
while read -r line; do
  [ -z "$line" ] && continue
  n=$((n + 1))
  key=$(echo "$line" | cut -d' ' -f1-2 | tr ' /' '__')_$(echo "$line" | md5sum | cut -c1-10)
  [ -e "$Q/done/$key" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do sleep 20; done
  (
    # shellcheck disable=SC2086
    if $PY -m lensearch.realpe --pe-dir "$PE_DIR" --cache "$CACHE" --outdir "$Q/out" --approximant "$APPROX" \
         --nlive "$NLIVE" --naccept "$NACCEPT" --npool "$NPOOL" $line > "$Q/logs/$key.log" 2>&1; then
      touch "$Q/done/$key"; echo "$(date +%F_%T) done $n $line"
    else
      echo "$(date +%F_%T) FAILED $n $line"
    fi
  ) &
  sleep 2
done < "$TASKS"
wait
echo "$(date +%F_%T) queue finished"
