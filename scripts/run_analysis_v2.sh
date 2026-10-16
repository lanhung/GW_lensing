#!/bin/bash
# Run the full v2 analysis: HistGB/CQR benchmark and conditional calibration (CPU),
# neural posterior estimation for several input sets (GPU) and the figures.
set -e
PY=${PY:-python}
OUT=${OUT:-results/v2}
NPE=${OUT}_npe_conf
mkdir -p "$OUT" "$NPE" logs
npe() { CUDA_VISIBLE_DEVICES=$1 $PY analysis/npe.py --out-dir "$NPE" --scenario $2 --input-set "$3" --epochs $4 > "logs/npe_$2_$5.log" 2>&1; }
(npe 0 ET "Geometry+GW(rel)+FU" 40 geogwfu; npe 0 ET GW 40 gw; npe 0 ET Full 40 full) &
(npe 1 ET Geometry 40 geo; npe 1 O5 "Geometry+GW(rel)+FU" 150 geogwfu; npe 1 O5 GW 150 gw) &
(npe 2 O5 Geometry 150 geo; npe 2 O5 Full 150 full) &
$PY analysis/benchmark.py --out-dir "$OUT" --jobs ${JOBS:-15} --threads ${THREADS:-4} > logs/benchmark.log 2>&1
$PY analysis/conditional_calibration.py --out-dir "${OUT}_conditional" > logs/conditional.log 2>&1
wait
$PY analysis/make_figures.py --results "$OUT" --npe "$NPE/npe_ET_Geometry_GWrel_FU.csv" --npe-dir "$NPE" --out-dir figures/v2 > logs/figures.log 2>&1
echo ANALYSIS_DONE >> logs/benchmark.log
