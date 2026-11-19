#!/usr/bin/env bash
# Reruns every current two-stage fit of O5 EPL_shear-7777-0017024 with the image labels of the GW injection
# (input em.gw_a/gw_b swapped to match the joint-PE labels). Outputs under fix0017024/results/<batch>.
cd /root/autodl-tmp/h0paper
F=fix0017024
cp results/pe/O5/O5_EPL_shear-7777-0017024_posterior.parquet $F/pe/O5/
export PY=/root/autodl-tmp/gwlens_env/bin/python PYTHONPATH=/root/autodl-tmp/h0paper
run() { CONFIGS="$1" EXTRA_ARGS="$2" TAG="$3" bash scripts/run_gwem_posthoc.sh $F/inputs/gwem $F/pe $F/results/$4 8 > $F/$4.log 2>&1; echo "done $4" >> $F/done.txt; }
run 'free:0:same lcdm:0.025:EPL_shear free:0.025:EPL_shear free:0:same:0.03 free:0:same:0.1' '--source-norm' _sn gwem_posthoc_sn &
run 'free:0:same free:0:same:0.03 free:0:same:0.1' '--exact-positions --source-norm' _xpos_sn gwem_posthoc_xpos_sn &
run 'free:0:same lcdm:0.025:EPL_shear' '--assignment marginalise --source-norm' _sn gwem_posthoc_asg_sn &
run 'lcdm:0.025:same' '--h0-free --source-norm' _h0_sn gwem_posthoc_h0_sn &
run 'lcdm:0.025:same' '--h0-free --source-norm --inject-kext 0.025' _h0_ki_sn h0_ki &
run 'lcdm:0.025:same' '--h0-free --source-norm --no-gw-distance' _h0_nod_sn h0_nod &
run 'lcdm:0.025:EPL_shear' '--h0-free' _h0 gwem_posthoc_h0 &
wait
echo all done >> $F/done.txt
