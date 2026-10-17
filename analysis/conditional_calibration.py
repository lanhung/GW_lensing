"""Coverage near caustics: marginal versus magnification-conditional conformal calibration.

Split-conformal CQR guarantees coverage only on average over the test population. Because
highly magnified images are rare, intervals can under-cover at large mu while over-covering
elsewhere. Here the conformal correction is computed separately in bins of the *predicted*
magnification (a Mondrian partition that uses only observed quantities), per lens family, and
coverage is reported in bins of the true magnification.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.benchmark import ALPHA, CQR, split  # noqa: E402
from analysis.features import ALL_SETS, FAMILIES, load  # noqa: E402

TRUE_BINS = [0, 2, 3, 5, 10, 20, np.inf]


def conformal_q(scores):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - ALPHA)))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data/v2'))
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--input-set', default='Geometry+GW(ratio)')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--pred-bins', type=int, default=5)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scen in ['ET', 'O5']:
        data = load(a.data_dir, scen, max_per_family=60000, seed=1)
        for fam in FAMILIES:
            d = data[data.family == fam].reset_index(drop=True)
            for seed in a.seeds:
                lab = split(d, seed)
                tr, ca, te = d[lab == 'train'], d[lab == 'cal'], d[lab == 'test']
                m = CQR(ALL_SETS[a.input_set], seed).fit(tr)
                lo_c, med_c, hi_c = m.raw(ca)
                y_c = np.log(ca.mu.to_numpy())
                s_c = np.maximum(lo_c - y_c, y_c - hi_c)
                lo_t, med_t, hi_t = m.raw(te)
                y_t = np.log(te.mu.to_numpy())
                edges = np.quantile(med_c, np.linspace(0, 1, a.pred_bins + 1))
                edges[0], edges[-1] = -np.inf, np.inf
                bc, bt = np.digitize(med_c, edges[1:-1]), np.digitize(med_t, edges[1:-1])
                q_marg = conformal_q(s_c)
                q_cond = np.array([conformal_q(s_c[bc == b]) for b in range(a.pred_bins)])
                for mode, q in (('marginal', np.full(len(te), q_marg)), ('conditional', q_cond[bt])):
                    cov = (y_t >= lo_t - q) & (y_t <= hi_t + q)
                    width = (hi_t + q) - (lo_t - q)
                    tb = pd.cut(te.mu.to_numpy(), TRUE_BINS)
                    for b, idx in pd.Series(np.arange(len(te))).groupby(tb, observed=True):
                        rows.append(dict(scenario=scen, family=fam, seed=seed, calibration=mode, mu_bin=str(b), n=len(idx),
                                         coverage90=float(cov[idx].mean()), width_ln=float(width[idx].mean())))
                    rows.append(dict(scenario=scen, family=fam, seed=seed, calibration=mode, mu_bin='all', n=len(te),
                                     coverage90=float(cov.mean()), width_ln=float(width.mean())))
        print(scen, 'done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out_dir / 'conditional_calibration.csv', index=False)
    print(df.groupby(['scenario', 'calibration', 'mu_bin']).coverage90.mean().unstack(1).round(3).to_string())


if __name__ == '__main__':
    main()
