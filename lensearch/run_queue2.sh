#!/usr/bin/env bash
# Run lensearch.realpe tasks from a list, keeping a fixed number of realpe runs alive ON THE WHOLE MACHINE.
#   lensearch/run_queue2.sh TASKS SLOTS NPOOL QUEUE_DIR
# Same task-line format and done markers as run_queue.sh (a task whose QUEUE_DIR/done/<key> exists is skipped), but
# the slot count includes runs started by an earlier queue that is still draining, so the two can overlap safely and
# SLOTS is the total concurrency of the machine. Set SLOTS above the core quota / NPOOL: the dynesty pool idles
# while it waits for the slowest walk of each batch, so a modest oversubscription fills the quota.
# Environment: PY (python), PE_DIR (data/pe), CACHE (data/strain), NLIVE (1000), NACCEPT (60), APPROX (IMRPhenomXPHM).
set -u
TASKS=$1; SLOTS=$2; NPOOL=$3; Q=$4
PY=${PY:-python}
PE_DIR=${PE_DIR:-data/pe}; CACHE=${CACHE:-data/strain}
NLIVE=${NLIVE:-1000}; NACCEPT=${NACCEPT:-60}; APPROX=${APPROX:-IMRPhenomXPHM}
mkdir -p "$Q/logs" "$Q/done"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# number of realpe runs on the machine = python processes running lensearch.realpe whose parent is not one of them
# (the dynesty pool workers are forked from the run's main process)
nrun() {
  ps -eo pid=,ppid=,args= | awk '/python[0-9.]* -m lensearch\.realpe/ && !/awk/ {p[$1]=1; pp[$1]=$2}
                                  END {n=0; for (i in p) if (!(pp[i] in p)) n++; print n}'
}

n=0
while read -r line; do
  [ -z "$line" ] && continue
  n=$((n + 1))
  key=$(echo "$line" | cut -d' ' -f1-2 | tr ' /' '__')_$(echo "$line" | md5sum | cut -c1-10)
  [ -e "$Q/done/$key" ] && continue
  # a task that a run of an earlier queue is still working on (its done marker appears only when it finishes) must
  # not be started again: two runs of the same task share the checkpoint file and one of them crashes on it
  if ps -eo args= | grep -F -- "-m lensearch.realpe" | grep -v grep | grep -qF -- " $(echo "$line" | tr -s ' ')"; then
    echo "$(date +%F_%T) running already, skipped $n $line"; continue
  fi
  while [ "$(nrun)" -ge "$SLOTS" ]; do sleep 30; done
  (
    # shellcheck disable=SC2086
    if $PY -m lensearch.realpe --pe-dir "$PE_DIR" --cache "$CACHE" --outdir "$Q/out" --approximant "$APPROX" \
         --nlive "$NLIVE" --naccept "$NACCEPT" --npool "$NPOOL" $line > "$Q/logs/$key.log" 2>&1; then
      touch "$Q/done/$key"; echo "$(date +%F_%T) done $n $line"
    else
      echo "$(date +%F_%T) FAILED $n $line"
    fi
  ) &
  sleep 45   # let the new run reach the sampler before the next slot check
done < "$TASKS"
wait
echo "$(date +%F_%T) queue finished"
