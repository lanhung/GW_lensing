#!/usr/bin/env bash
# Two-stage GW+EM inference for every event with a finished joint_pe posterior, in several
# configurations. Usage: scripts/run_gwem_posthoc.sh INPUT_DIR PE_DIR OUT_DIR [PAR]
# CONFIGS entries are cosmology:sigma_kext:lens_model[:sigma_img_gw]; lens_model 'same' = true family.
set -euo pipefail
IN=$1; PE=$2; OUT=$3; PAR=${4:-8}
PY=${PY:-python}
CONFIGS=${CONFIGS:-"free:0:same free:0:EPL_shear lcdm:0.025:EPL_shear free:0.025:EPL_shear"}
mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for post in "$PE"/*/*_posterior.parquet; do
  base=$(basename "$post" _posterior.parquet)        # e.g. ET_SIS-7777-0019701
  [ -f "$IN/$base.json" ] || continue
  for c in $CONFIGS; do echo "$base $post $c"; done
done | xargs -P "$PAR" -L 1 bash -c '
  base=$0; post=$1; IFS=: read cosmo sk model ast <<< "$2"
  fam=$(echo $base | cut -d_ -f2- | sed "s/-[0-9]*-[0-9]*$//")
  [ "$model" = same ] && model=$fam
  ast=${ast:-0}; tag=""; [ "$ast" != 0 ] && tag=_ast$ast
  label=${base}_posthoc_${cosmo}_${model}${tag}
  out="'"$OUT"'/${cosmo}_k${sk}_${model}${tag}"
  [ -f "$out/${label}_summary.json" ] && exit 0
  nice -n 10 '"$PY"' -m gwlens.joint_gwem posthoc --event "'"$IN"'/$base.json" --gw-posterior "$post" --out-dir "$out" \
    --cosmology $cosmo --sigma-kext $sk --lens-model $model --sigma-img-gw $ast > "'"$OUT"'/logs/${label}_k${sk}.log" 2>&1 || echo "FAILED $label"
'
