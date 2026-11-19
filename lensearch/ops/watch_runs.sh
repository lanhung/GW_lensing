#!/usr/bin/env bash
# Wait until no realpe run writing to results/{inject_pe,bg_null,bg_pe,bg_pe_n1000} is alive on any machine (two
# consecutive checks), or MAXH hours. Prints one line per check. Usage: bash lensearch/ops/watch_runs.sh [MAXH]
MAXH=${1:-96}; end=$((SECONDS + MAXH * 3600)); zero=0
cnt='ps -eo pid=,ppid=,args= | awk "/python[0-9.]* -m lensearch\\.realpe/ && /--outdir results\\/(inject_pe|bg_null|bg_pe|bg_pe_n1000)( |$)/ && !/awk/ {p[\$1]=1; pp[\$1]=\$2} END {n=0; for (i in p) if (!(pp[i] in p)) n++; print n}"'
while [ $SECONDS -lt $end ]; do
  tot=0; miss=""
  for p in 55094 57190 57998 53555 53273 55011; do
    n=""; for t in 1 2 3; do n=$(timeout 90 ssh -p $p -o ConnectTimeout=30 root@u3ssh.casdao.com "$cnt" < /dev/null 2>/dev/null) && [ -n "$n" ] && break; sleep 20; done
    [ -n "$n" ] || { miss="$miss $p"; n=1; }
    tot=$((tot + n))
  done
  n=$(timeout 90 ssh -p 2338 -o ConnectTimeout=30 root@gpu.chzmark.com "$cnt" < /dev/null 2>/dev/null); [ -n "$n" ] || { miss="$miss gpu"; n=1; }
  tot=$((tot + n))
  echo "$(date -u +%F_%T) runs alive $tot${miss:+ (unreachable:$miss)}"
  if [ "$tot" -eq 0 ]; then zero=$((zero + 1)); [ $zero -ge 2 ] && { echo ALL_RUNS_DONE; exit 0; }; else zero=0; fi
  sleep 3600
done
echo TIMEOUT
