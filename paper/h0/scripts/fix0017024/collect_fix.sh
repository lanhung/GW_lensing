#!/usr/bin/env bash
# Pulls the corrected runs of O5 EPL_shear-7777-0017024 from the GPU server into p2/fix (see README.md).
set -euo pipefail
F=$(cd "$(dirname "$0")" && pwd)
ssh -p 2338 root@gpu.chzmark.com 'cd /root/autodl-tmp/h0paper/fix0017024 && /root/autodl-tmp/gwlens_env/bin/python - <<PY
import glob, json, os
import numpy as np, pandas as pd
# The runs used an input whose truth.mu_a/mu_b were still in the old order (they enter only the
# truth-dependent summary fields); swap them and recompute those fields from the posterior.
for f in sorted(glob.glob("results/*/*/*_summary.json")):
    s = json.load(open(f))
    s["true_mu_a"], s["true_mu_b"] = s["true_mu_b"], s["true_mu_a"]
    x = pd.read_parquet(f.replace("_summary.json", "_posterior.parquet"), columns=["ln_mu_a"]).ln_mu_a.dropna().to_numpy()
    y = np.log(s["true_mu_a"])
    s.update(pit=float(np.mean(x < y)), rel_err=float(abs(np.exp(np.median(x)) - s["true_mu_a"]) / s["true_mu_a"]),
             covered90=bool(s["ln_mu_a_q05"] <= y <= s["ln_mu_a_q95"]))
    o = "summaries/" + f[len("results/"):]
    os.makedirs(os.path.dirname(o), exist_ok=True)
    json.dump(s, open(o, "w"), indent=1)
out = {}
for d in ("gwem_posthoc_h0_sn", "h0_ki", "h0_nod", "gwem_posthoc_h0"):
    for f in glob.glob(f"results/{d}/*/*_posterior.parquet"):
        key = os.path.basename(os.path.dirname(f)) + "|" + os.path.basename(f).split("_posthoc")[0]
        out[key + "|h"] = pd.read_parquet(f, columns=["h_rel"]).h_rel.values.astype(np.float32)
        s = json.load(open(f.replace("_posterior.parquet", "_summary.json")))
        out[key + "|k"] = np.float32(s.get("kext_injected", np.nan))
np.savez_compressed("h_fix.npz", **out)
print(sorted(out))
PY
tar czf fix_summaries.tgz -C summaries .'
scp -P 2338 -q root@gpu.chzmark.com:/root/autodl-tmp/h0paper/fix0017024/{h_fix.npz,fix_summaries.tgz} "$F/"
rm -rf "${F:?}/posthoc" && mkdir -p "$F/posthoc" && tar xzf "$F/fix_summaries.tgz" -C "$F/posthoc"
python3 - "$F" <<'PY'
import glob, json, os, sys
F = sys.argv[1]
with open(f'{F}/all_fix.jsonl', 'w') as o:
    for f in sorted(f for b in ('gwem_posthoc_h0_sn', 'h0_ki', 'h0_nod', 'gwem_posthoc_h0')
                    for f in glob.glob(f'{F}/posthoc/{b}/*/*_summary.json')):
        r = json.load(open(f)); r['dir'] = os.path.basename(os.path.dirname(f)); r['file'] = os.path.basename(f)
        o.write(json.dumps(r) + '\n')
print(open(f'{F}/all_fix.jsonl').read().count('\n'), 'summaries')
PY
