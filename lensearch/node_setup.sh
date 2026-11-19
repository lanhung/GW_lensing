#!/usr/bin/env bash
# Prepare a compute node: pinned Python environment, code and data pulled directly from the GPU server.
#   lensearch/node_setup.sh HOST PORT
# Runs locally; a temporary ssh-agent forwards the local key so that the node can reach the GPU server
# (the agent is killed on exit). Data: PE extractions, strain segments, PSDs, injection noise and tables (~1.5 GB).
set -euo pipefail
H=$1; P=$2
GPU="-p 2338 root@gpu.chzmark.com"
eval "$(ssh-agent -s)" > /dev/null
trap 'ssh-agent -k > /dev/null' EXIT
ssh-add -q ~/.ssh/id_ed25519
ssh -A -o StrictHostKeyChecking=accept-new -p "$P" "root@$H" "set -e
  mkdir -p /root/lensearch && cd /root/lensearch
  ssh -o StrictHostKeyChecking=accept-new $GPU 'cd /root/autodl-tmp/lensearch && tar cf - code env_freeze.txt data/pe data/strain data/psd data/inject_noise_s1 results/inject' | tar xf -
  PYBIN=\$(command -v python3.10 || command -v python3)
  [ -x /root/lensearch_env/bin/python ] || \$PYBIN -m venv /root/lensearch_env
  /root/lensearch_env/bin/pip install -q --upgrade pip
  /root/lensearch_env/bin/pip install -q -r env_freeze.txt
  PYTHONPATH=/root/lensearch/code /root/lensearch_env/bin/python -c 'import bilby, lal, healpy, ligo.skymap, lensearch.realpe; print(\"env ok\", bilby.__version__)'
  nproc; free -g | head -2; df -h /root | tail -1"
