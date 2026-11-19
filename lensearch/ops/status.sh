#!/usr/bin/env bash
# Status of the stage-2 compute: per machine the running realpe tasks (main processes, hours), loop count, and
# done/total per task list. Usage: bash lensearch/ops/status.sh
CMD='cd $D; echo "  loops: $(ps -eo ppid=,args= | grep run_queue2.sh | grep -v grep | awk "\$1==1 {print \$NF\"/\"\$(NF-3)}" | sort | uniq -c | tr -s " " | tr "\n" ";")"
ps -eo pid=,ppid=,etimes=,args= | awk "/python[0-9.]* -m lensearch\\.realpe/ && !/awk/ {p[\$1]=\$0; pp[\$1]=\$2} END {for (i in p) if (!(pp[i] in p)) print p[i]}" | sed -E "s/^ *([0-9]+) +([0-9]+) +([0-9]+) .* (single|joint) /\\3 \\4 /; s/ --(pe-dir|prior-events|injections|cache|outdir|approximant|nlive|naccept|npool).*//" | awk "{printf \"  %5.1fh %s %s %s\n\", \$1/3600, \$2, \$3, \$4}" | sort -k2 -r
for L in $(ps -eo args= | grep run_queue2.sh | grep -v grep | awk "{print \$3}" | sort -u); do n=0; d=0; while read -r line; do [ -z "$line" ] && continue; n=$((n+1)); key=$(echo "$line" | cut -d" " -f1-2 | tr " /" "__")_$(echo "$line" | md5sum | cut -c1-10); [ -e queue/done/$key ] && d=$((d+1)); done < $L; echo "  $L done $d/$n"; done'
for p in 55094 57190 57998 53555 53273 55011; do
  echo "== node $p"
  for t in 1 2 3; do ssh -p $p -o ConnectTimeout=30 root@u3ssh.casdao.com "D=/root/lensearch; $CMD" 2>/dev/null && break; sleep 10; done
done
echo "== GPU"
ssh -p 2338 -o ConnectTimeout=30 root@gpu.chzmark.com "D=/root/autodl-tmp/lensearch; $CMD" 2>/dev/null
