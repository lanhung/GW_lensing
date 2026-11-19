#!/usr/bin/env bash
# Copy every finished run (summary + posterior, no hdf5) from all compute nodes to the GPU server:
#   lensearch/collect_all.sh            -> GPU /root/autodl-tmp/lensearch/results/collected/<dir>/
# Runs locally; a temporary ssh-agent forwards the local key so that each node can push to the GPU server.
set -u
NODES=${NODES:-"55094 57190 57998 53555 53273 55011"}
DIRS=${DIRS:-"stage2 stage2_marginal inject_pe bg_pe bg_pe_n1000 bg_null"}
GPU="-p 2338 root@gpu.chzmark.com"
eval "$(ssh-agent -s)" > /dev/null
trap 'ssh-agent -k > /dev/null' EXIT
ssh-add -q ~/.ssh/id_ed25519
for p in $NODES; do
  for d in $DIRS; do
    ssh -A -p "$p" -o ConnectTimeout=30 root@u3ssh.casdao.com "cd /root/lensearch/results/$d 2>/dev/null || exit 0
      find . -maxdepth 1 \( -name '*_summary.json' -o -name '*_posterior.parquet' \) -print0 | tar cf - --null -T - \
      | ssh -o StrictHostKeyChecking=accept-new $GPU 'mkdir -p /root/autodl-tmp/lensearch/results/collected/$d && cd /root/autodl-tmp/lensearch/results/collected/$d && tar xf - && echo \"node $p $d: \$(ls *_summary.json 2>/dev/null | wc -l) summaries total\"'"
  done
done
# the GPU's own results
ssh $GPU 'cd /root/autodl-tmp/lensearch/results && for d in '"$DIRS"' pilot; do [ -d $d ] || continue; mkdir -p collected/$d; cp -u $d/*_summary.json $d/*_posterior.parquet collected/$d/ 2>/dev/null; done; for d in collected/*; do echo "$d: $(ls $d/*_summary.json 2>/dev/null | wc -l) summaries"; done'
