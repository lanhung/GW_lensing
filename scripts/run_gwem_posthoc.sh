#!/usr/bin/env bash
# Two-stage GW+EM inference for every event with a finished joint_pe posterior, in several
# configurations. Usage: scripts/run_gwem_posthoc.sh INPUT_DIR PE_DIR OUT_DIR [PAR]
# CONFIGS entries are cosmology:sigma_kext:lens_model[:sigma_img_gw]; lens_model 'same' = true family.
# Known-sky GW posteriors (*_knownsky_posterior.parquet) go to output directories with the suffix _ks.
# SHARD="r1,r2,.../N" keeps only the tasks whose checksum modulo N is one of the listed residues,
# so that several hosts can share the task list without overlap.
set -euo pipefail
IN=$1; PE=$2; OUT=$3; PAR=${4:-8}
PY=${PY:-python}
CONFIGS=${CONFIGS:-"free:0:same free:0:EPL_shear lcdm:0.025:EPL_shear free:0.025:EPL_shear"}
mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for post in "$PE"/*/*_posterior.parquet; do
  base=$(basename "$post" _posterior.parquet)        # e.g. ET_SIS-7777-0019701 or ET_SIS-7777-0019701_knownsky
  ks=0; [ "${base%_knownsky}" != "$base" ] && ks=1 && base=${base%_knownsky}
  [ -f "$IN/$base.json" ] || continue
  for c in $CONFIGS; do
    if [ -n "${SHARD:-}" ]; then
      h=$(printf '%s' "$base $c $ks" | cksum | cut -d' ' -f1)
      [[ ",${SHARD%/*}," == *",$((h % ${SHARD#*/})),"* ]] || continue
    fi
    echo "$base $post $c $ks"
  done
done | xargs -P "$PAR" -L 1 bash -c '
  base=$0; post=$1; IFS=: read cosmo sk model ast <<< "$2"; ks=$3
  fam=$(echo $base | cut -d_ -f2- | sed "s/-[0-9]*-[0-9]*$//")
  [ "$model" = same ] && model=$fam
  ast=${ast:-0}; tag=""; [ "$ast" != 0 ] && tag=_ast$ast
  label=${base}_posthoc_${cosmo}_${model}${tag}
  [ "$ks" = 1 ] && tag=${tag}_ks
  out="'"$OUT"'/${cosmo}_k${sk}_${model}${tag}"
  [ -f "$out/${label}_summary.json" ] && exit 0
  nice -n 10 '"$PY"' -m gwlens.joint_gwem posthoc --event "'"$IN"'/$base.json" --gw-posterior "$post" --out-dir "$out" \
    --cosmology $cosmo --sigma-kext $sk --lens-model $model --sigma-img-gw $ast > "'"$OUT"'/logs/${label}_k${sk}${tag}.log" 2>&1 || echo "FAILED $label"
'
