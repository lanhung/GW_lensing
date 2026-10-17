"""Response of trained flows to a distance scale different from the simulated cosmology.

Applies distance_scale_transform (all distances / h at fixed redshifts) to the test systems of each
saved NPE flow and reports the median shift of the predicted ln mu and the coverage of the global
conformal 90 per cent interval. The true magnifications are unchanged, so any shift is a bias.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.benchmark import split  # noqa: E402
from analysis.features import FAMILIES, distance_scale_transform, load  # noqa: E402
from analysis.npe import load_flow, posterior_samples, standardise  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data/v2'))
    p.add_argument('--flow-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--h', nargs='+', type=float, default=[0.9, 0.95, 1.0, 1.05, 1.1])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    rows = []
    for ck in sorted(a.flow_dir.glob('flow_*.pt')):
        flow, meta = load_flow(ck, a.device)
        data = load(a.data_dir, meta['scenario'], max_per_family=60000, seed=1)
        te = data[split(data, 0) == 'test']
        q = meta['qgroup']['none']['0']
        y = np.log(te.mu.to_numpy())
        base = None
        for h in a.h:
            x, _, _ = standardise(distance_scale_transform(te, h)[meta['cols']].to_numpy(float), meta['mean'], meta['std'])
            s = posterior_samples(flow, x, a.device, n=400)
            med = np.median(s, 1)
            base = med if h == 1.0 else base
            cov = (y >= np.quantile(s, 0.05, 1) - q) & (y <= np.quantile(s, 0.95, 1) + q)
            for fam in FAMILIES + ['all']:
                m = np.ones(len(te), bool) if fam == 'all' else te.family.to_numpy() == fam
                rows.append(dict(scenario=meta['scenario'], input_set=meta['input_set'], family=fam, h=h, med=med[m], cov=cov[m]))
        for r in rows:
            if r['scenario'] == meta['scenario'] and r['input_set'] == meta['input_set'] and 'median_dln_pred' not in r:
                m = np.ones(len(te), bool) if r['family'] == 'all' else te.family.to_numpy() == r['family']
                r['median_dln_pred'] = float(np.median(r.pop('med') - base[m]))
                r['coverage90_conformal_global'] = float(r.pop('cov').mean())
        print(ck.name, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df[df.family == 'all'].pivot_table(index=['scenario', 'input_set'], columns='h', values='median_dln_pred').round(3).to_string())


if __name__ == '__main__':
    main()
