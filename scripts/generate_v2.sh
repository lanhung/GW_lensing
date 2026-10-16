#!/bin/bash
# Generate the v2 catalogues. Usage: scripts/generate_v2.sh <n_systems> <family> [<family> ...]
# Waits for any running catalogue job before starting.
set -e
PY=${PY:-python}
N=$1; shift
while pgrep -f "gwlens[.]catalogue" > /dev/null; do sleep 20; done
for f in "$@"; do
  $PY -m gwlens.catalogue --family "$f" --n-systems "$N" --seed 20261016 --chunk 250 --processes ${PROCS:-62} \
    --out data/v2/"$f".parquet > logs/generate_"$f".log 2>&1
done
