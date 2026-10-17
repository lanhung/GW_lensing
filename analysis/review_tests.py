"""Information-content, mass-sheet and calibration tests requested in review.

ablation  in-family HistGB-CQR for input sets that separate the dimensionless GW information
          (relative magnification, arrival order) from the clock |dt| and the absolute SNRs, both
          of which carry distance-scale information; plus a prior-only baseline that ignores the
          event data. The seed-0 models are also applied to the test systems after an external
          mass-sheet transformation (lam = 1 - kappa_ext), which shows how each estimator responds.
mixed     one model for all families with conformal calibration that is global, by true family
          (diagnostic only), by observed image multiplicity, or by quintile of the predicted mu.

Coverage is reported with Wilson 68 per cent binomial intervals.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.benchmark import ALPHA, CQR, metrics, split  # noqa: E402
from analysis.features import ALL_SETS, FAMILIES, load, mass_sheet_transform  # noqa: E402

ABLATION = ['Geometry', 'Geometry+FU', 'Geometry+GW(ratio)', 'Geometry+GW(rel)', 'Geometry+GW(ratio)+FU',
            'Geometry+GW(ratio+clock)+FU', 'Geometry+GW(ratio+SNR)+FU', 'Geometry+GW(rel)+FU', 'Full']
MIXED = ['Geometry+GW(ratio)', 'Geometry+GW(ratio)+FU', 'Geometry+GW(rel)+FU']
LAMBDAS = (0.9, 0.95, 1.0, 1.05, 1.1)


def wilson(k, n, z=1.0):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def conformal_q(s):
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - ALPHA)))
    return np.inf if k > n or n == 0 else float(np.sort(s)[k - 1])


def coverage_row(mu, lo, hi):
    cov = (mu >= lo) & (mu <= hi)
    lo68, hi68 = wilson(int(cov.sum()), len(cov))
    return dict(coverage90=float(cov.mean()), coverage90_lo68=lo68, coverage90_hi68=hi68,
                width_ln=float(np.mean(np.log(hi / lo))))


def prior_only(tr, ca, te):
    """Constant predictor: training-set quantiles of ln mu, conformally calibrated."""
    y = np.log(tr.mu.to_numpy())
    lo, med, hi = np.quantile(y, [ALPHA / 2, 0.5, 1 - ALPHA / 2])
    yc = np.log(ca.mu.to_numpy())
    q = conformal_q(np.maximum(lo - yc, yc - hi))
    n = len(te)
    return np.full(n, np.exp(med)), np.full(n, np.exp(lo - q)), np.full(n, np.exp(hi + q))


def ablation_fit(tr, ca, te, name, seed, tag, do_msd):
    rows = []
    if name == 'Prior only':
        pred, lo, hi = prior_only(tr, ca, te)
        rows.append(dict(tag, input_set=name, **metrics(te.mu.to_numpy(), pred),
                         **coverage_row(te.mu.to_numpy(), lo, hi)))
        return rows, []
    m = CQR(ALL_SETS[name], seed).fit(tr).calibrate(ca, by_family=False)
    pred, lo, hi, _, _ = m.predict(te)
    mu = te.mu.to_numpy()
    rows.append(dict(tag, input_set=name, **metrics(mu, pred), **coverage_row(mu, lo, hi)))
    msd = []
    if do_msd:
        for lam in LAMBDAS:
            tt = mass_sheet_transform(te, lam)
            p2, l2, h2, _, _ = m.predict(tt)
            mu2 = tt.mu.to_numpy()
            msd.append(dict(tag, input_set=name, method='HistGB-CQR', lam=lam,
                            median_dln_pred=float(np.median(np.log(p2 / pred))),
                            median_ln_bias=float(np.median(np.log(p2 / mu2))),
                            **coverage_row(mu2, l2, h2)))
    return rows, msd


def mixed_fit(tr, ca, te, name, seed, tag, pred_bins=5):
    m = CQR(ALL_SETS[name], seed).fit(tr)
    lo_c, med_c, hi_c = m.raw(ca)
    y_c = np.log(ca.mu.to_numpy())
    s_c = np.maximum(lo_c - y_c, y_c - hi_c)
    lo_t, med_t, hi_t = m.raw(te)
    edges = np.quantile(med_c, np.linspace(0, 1, pred_bins + 1))[1:-1]
    g_c = dict(global_=np.zeros(len(ca), int), true_family=ca.family.to_numpy(),
               n_images=np.minimum(ca.geo_n_images.to_numpy(), 4).astype(int), pred_mu=np.digitize(med_c, edges))
    g_t = dict(global_=np.zeros(len(te), int), true_family=te.family.to_numpy(),
               n_images=np.minimum(te.geo_n_images.to_numpy(), 4).astype(int), pred_mu=np.digitize(med_t, edges))
    rows = []
    for g in g_c:
        qd = {k: conformal_q(s_c[g_c[g] == k]) for k in np.unique(g_c[g])}
        q_all = conformal_q(s_c)
        q = np.array([qd.get(k, q_all) for k in g_t[g]])
        lo, hi = np.exp(lo_t - q), np.exp(hi_t + q)
        for fam in FAMILIES + ['all']:
            mask = np.ones(len(te), bool) if fam == 'all' else (te.family.to_numpy() == fam)
            mu = te.mu.to_numpy()[mask]
            rows.append(dict(tag, input_set=name, calibration=g.rstrip('_'), test_family=fam,
                             median_rel=float(np.median(np.abs(np.exp(med_t[mask]) - mu) / mu)),
                             **coverage_row(mu, lo[mask], hi[mask])))
        # Coverage in bins of the true magnification (the near-caustic failure mode).
        mu = te.mu.to_numpy()
        for b, idx in pd.Series(np.arange(len(te))).groupby(pd.cut(mu, [0, 3, 5, 10, 20, np.inf]), observed=True):
            rows.append(dict(tag, input_set=name, calibration=g.rstrip('_'), test_family=f'mu_true {b}', median_rel=np.nan,
                             **coverage_row(mu[idx], lo[idx], hi[idx])))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, default=Path('data/v2'))
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--scenarios', nargs='+', default=['ET', 'O5'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--max-per-family', type=int, default=60000)
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    os.environ['OMP_NUM_THREADS'] = str(a.threads)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    abl, msd, mix = [], [], []
    for scen in a.scenarios:
        data = load(a.data_dir, scen, max_per_family=a.max_per_family, seed=1)
        jobs = []
        for fam in FAMILIES:
            d = data[data.family == fam].reset_index(drop=True)
            for seed in a.seeds:
                lab = split(d, seed)
                tr, ca, te = d[lab == 'train'], d[lab == 'cal'], d[lab == 'test']
                tag = dict(experiment='ablation', scenario=scen, family=fam, split_seed=seed, n_train=len(tr))
                for name in ['Prior only'] + ABLATION:
                    jobs.append(delayed(ablation_fit)(tr, ca, te, name, seed, tag, seed == a.seeds[0]))
        res = Parallel(n_jobs=a.jobs, verbose=2)(jobs)
        abl += [r for rows, _ in res for r in rows]
        msd += [r for _, m in res for r in m]
        jobs = []
        for seed in a.seeds:
            lab = split(data, seed)
            tr, ca, te = data[lab == 'train'], data[lab == 'cal'], data[lab == 'test']
            for name in MIXED:
                jobs.append(delayed(mixed_fit)(tr, ca, te, name, seed, dict(experiment='mixed', scenario=scen, split_seed=seed)))
        mix += [r for rows in Parallel(n_jobs=a.jobs, verbose=2)(jobs) for r in rows]
        for df, f in ((abl, 'ablation.csv'), (msd, 'msd.csv'), (mix, 'mixed_calibration.csv')):
            pd.DataFrame(df).to_csv(a.out_dir / f, index=False)
        print(scen, 'done', flush=True)
    t = pd.DataFrame(abl).groupby(['scenario', 'input_set', 'family'], sort=False).median_rel.mean().unstack()
    print((100 * t).round(1).to_string())
    m = pd.DataFrame(msd)
    print(m[m.lam.isin([0.9, 1.1])].groupby(['scenario', 'input_set', 'lam']).median_dln_pred.mean().unstack().round(3).to_string())


if __name__ == '__main__':
    main()
