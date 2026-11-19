#!/usr/bin/env bash
# Copy finished run outputs (summaries and posteriors only) from a compute node to the GPU server.
#   lensearch/node_collect.sh HOST PORT NODE_RESULTS_DIR GPU_DEST_DIR
set -euo pipefail
H=$1; P=$2; SRC=$3; DEST=$4
eval "$(ssh-agent -s)" > /dev/null
trap 'ssh-agent -k > /dev/null' EXIT
ssh-add -q ~/.ssh/id_ed25519
ssh -A -p "$P" "root@$H" "cd $SRC && find . -maxdepth 1 \( -name '*_summary.json' -o -name '*_posterior.parquet' -o -name '*_result.hdf5' \) -print0 \
  | tar cf - --null -T - | ssh -p 2338 root@gpu.chzmark.com 'mkdir -p $DEST && cd $DEST && tar xf - && ls | wc -l'"
