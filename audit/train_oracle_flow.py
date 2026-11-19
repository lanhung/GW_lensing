"""Audit task D1: neural posterior estimators trained with the TRUE relative magnification as input.

Same data, split, architecture and training schedule as analysis/npe.py for the scale-free input set
Geometry+GW(ratio)+FU, except that the measured ln(mu_B/mu_A) is replaced by the simulation truth.
This is the oracle upper bound of the downstream chain (configuration D1 of the audit). The
checkpoint has the same layout as those of analysis/npe.py, so analysis/pe_analysis.py-style
evaluation can load it; the test-set metrics per family are written next to it.

    python audit/train_oracle_flow.py --scenario ET --epochs 40 --out-dir out/oracle_flows
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.benchmark import metrics, split  # noqa: E402
from analysis.features import ALL_SETS, FAMILIES, check_inputs, load  # noqa: E402
from analysis.npe import posterior_samples, standardise, train_flow  # noqa: E402

INPUT_SET = 'Geometry+GW(ratio)+FU'


def conformal(v):
    v = np.sort(v)
    k = int(np.ceil((len(v) + 1) * 0.9))
    return float(v[min(k, len(v)) - 1]) if len(v) else np.inf


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, default=Path('data/v2'))
    p.add_argument('--out-dir', type=Path, default=Path('out/oracle_flows'))
    p.add_argument('--scenario', default='ET')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cols = check_inputs(ALL_SETS[INPUT_SET])
    data = load(a.data_dir, a.scenario, max_per_family=60000, seed=1)
    data['gw_ln_mu_rel'] = np.log(data.mu_b / data.mu)          # oracle: exact relative magnification
    lab = split(data, a.seed)
    trall, ca, te = data[lab == 'train'], data[lab == 'cal'], data[lab == 'test']
    vmask = np.random.default_rng(a.seed + 1).random(len(trall)) < 0.1
    tr, va = trall[~vmask], trall[vmask]
    xtr, m, s = standardise(tr[cols].to_numpy(float))
    xva, _, _ = standardise(va[cols].to_numpy(float), m, s)
    xte, _, _ = standardise(te[cols].to_numpy(float), m, s)
    xca, _, _ = standardise(ca[cols].to_numpy(float), m, s)
    t0 = time.time()
    flow, val_nll = train_flow(xtr, np.log(tr.mu.to_numpy()), xva, np.log(va.mu.to_numpy()), a.device, a.epochs, a.seed)
    samples = posterior_samples(flow, xte, a.device)
    sca = posterior_samples(flow, xca, a.device)
    y, yca = np.log(te.mu.to_numpy()), np.log(ca.mu.to_numpy())
    sc = np.maximum(np.quantile(sca, 0.05, 1) - yca, yca - np.quantile(sca, 0.95, 1))
    qconf = {f: conformal(sc[ca.family.to_numpy() == f]) for f in FAMILIES}
    qglob = conformal(sc)
    torch.save(dict(state=flow.state_dict(), cols=cols, mean=m, std=s, input_set=INPUT_SET + ' [oracle true ratio]',
                    scenario=a.scenario, arch=dict(features=1, context=len(cols), transforms=4, hidden_features=[256, 256, 256], bins=12),
                    qconf=qconf, qgroup={'family': qconf, 'none': {'0': qglob}}),
               a.out_dir / f'flow_{a.scenario}_oracle_Geometry_GWratio_FU.pt')
    med = np.exp(np.median(samples, 1))
    rows = []
    for fam in FAMILIES + ['all']:
        mk = np.ones(len(te), bool) if fam == 'all' else (te.family.to_numpy() == fam)
        qv = np.array([qconf[f] for f in te.family.to_numpy()[mk]])
        lo, hi = np.quantile(samples[mk], 0.05, 1), np.quantile(samples[mk], 0.95, 1)
        r = dict(scenario=a.scenario, input_set=INPUT_SET, gw_ratio='true (oracle)', family=fam, method='NPE (NSF)',
                 **metrics(te.mu.to_numpy()[mk], med[mk], np.exp(lo), np.exp(hi), np.random.default_rng(0)))
        r['coverage90_conformal'] = float(np.mean((y[mk] >= lo - qv) & (y[mk] <= hi + qv)))
        r['width_ln_conformal'] = float(np.mean(hi - lo + 2 * qv))
        q16, q84 = np.quantile(samples[mk], 0.16, 1), np.quantile(samples[mk], 0.84, 1)
        r['coverage_68'] = float(np.mean((y[mk] >= q16) & (y[mk] <= q84)))
        r['p90_rel'] = float(np.quantile(np.abs(med[mk] - te.mu.to_numpy()[mk]) / te.mu.to_numpy()[mk], 0.9))
        rows.append(r)
    pd.DataFrame(rows).to_csv(a.out_dir / f'npe_{a.scenario}_oracle_Geometry_GWratio_FU.csv', index=False)
    pd.DataFrame(dict(event_id=te.event_id, family=te.family, mu=te.mu, pred=med, lo=np.exp(np.quantile(samples, 0.05, 1)),
                      hi=np.exp(np.quantile(samples, 0.95, 1)), pit=(samples < y[:, None]).mean(1))).to_parquet(
        a.out_dir / f'npe_predictions_{a.scenario}_oracle_Geometry_GWratio_FU.parquet', index=False)
    print(json.dumps(dict(scenario=a.scenario, val_nll=val_nll, seconds=time.time() - t0, n_train=len(tr), qconf=qconf)))
    print(pd.DataFrame(rows)[['family', 'median_rel', 'p90_rel', 'coverage90', 'coverage90_conformal', 'coverage_68']].round(3).to_string())


if __name__ == '__main__':
    main()
