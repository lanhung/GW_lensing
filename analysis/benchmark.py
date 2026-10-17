"""Magnification-inference benchmark on the v2 lensed-BBH catalogues.

Experiments
-----------
inputs     in-family comparison of observed input sets and analytic baselines
mixed      one model trained on all families; global versus per-family (Mondrian) conformal
transfer   train on one family, test on another; leave-one-family-out
sky        GW relative magnification with estimated versus known sky/orientation
learning   learning curve (training size) for the geometry and full input sets

The point estimator is the conditional median of ln(mu) from histogram gradient boosting. Ninety
per cent intervals use conformalized quantile regression (CQR; Romano et al. 2019) with 5 and 95
per cent quantile models calibrated on a held-out calibration set.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.features import ALL_SETS, FAMILIES, INPUT_SETS, check_inputs, load  # noqa: E402

# Main input set for the mixed-family, transfer and sky experiments: geometry plus the scale-free GW
# observables (relative magnification and arrival order), invariant under the mass-sheet transformation.
MAIN = 'Geometry+GW(ratio)'
BENCH_SETS = {**INPUT_SETS, MAIN: ALL_SETS[MAIN]}

ALPHA = 0.10
HGB = dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.1,
           early_stopping=True, validation_fraction=0.1, n_iter_no_change=30)


def split(df, seed, frac=(0.6, 0.2, 0.2)):
    rng = np.random.default_rng(seed)
    lab = np.empty(len(df), object)
    for fam, idx in df.groupby('family').indices.items():
        idx = rng.permutation(idx)
        a, b = int(frac[0] * len(idx)), int((frac[0] + frac[1]) * len(idx))
        lab[idx[:a]], lab[idx[a:b]], lab[idx[b:]] = 'train', 'cal', 'test'
    return lab


class CQR:
    """Median and 5/95 per cent quantile HistGB models on ln(mu) with split-conformal calibration."""

    def __init__(self, cols, seed):
        self.cols = check_inputs(list(cols))
        self.seed = seed
        self.models = {}

    def fit(self, train):
        y = np.log(train.mu.to_numpy())
        for q in (ALPHA / 2, 0.5, 1 - ALPHA / 2):
            m = HistGradientBoostingRegressor(loss='quantile', quantile=q, random_state=self.seed, **HGB)
            self.models[q] = m.fit(train[self.cols], y)
        return self

    def raw(self, d):
        p = {q: m.predict(d[self.cols]) for q, m in self.models.items()}
        lo, med, hi = p[ALPHA / 2], p[0.5], p[1 - ALPHA / 2]
        return np.minimum(lo, med), med, np.maximum(hi, med)

    def calibrate(self, cal, by_family):
        lo, _, hi = self.raw(cal)
        y = np.log(cal.mu.to_numpy())
        score = np.maximum(lo - y, y - hi)

        def q(s):
            n = len(s)
            k = int(np.ceil((n + 1) * (1 - ALPHA)))
            return np.inf if k > n else float(np.sort(s)[k - 1])
        self.q_global = q(score)
        self.q_family = {f: q(score[cal.family.to_numpy() == f]) for f in cal.family.unique()} if by_family else {}
        return self

    def predict(self, d, mondrian=False):
        lo, med, hi = self.raw(d)
        if mondrian:
            qv = np.array([self.q_family.get(f, self.q_global) for f in d.family])
        else:
            qv = self.q_global
        return np.exp(med), np.exp(lo - qv), np.exp(hi + qv), np.exp(lo), np.exp(hi)


def metrics(mu, pred, lo=None, hi=None, rng=None, nboot=200):
    err = pred - mu
    rel = np.abs(err) / mu
    out = dict(n=len(mu), mae=float(np.mean(np.abs(err))), median_ae=float(np.median(np.abs(err))),
               median_rel=float(np.median(rel)), mae_ln=float(np.mean(np.abs(np.log(pred / mu)))),
               frac_rel_lt10=float(np.mean(rel < 0.1)), bias=float(np.mean(err)))
    if rng is not None and len(mu) > 20:
        b = [np.mean(np.abs(err[ix])) for ix in (rng.integers(0, len(mu), len(mu)) for _ in range(nboot))]
        out['mae_lo95'], out['mae_hi95'] = float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))
    if lo is not None:
        out['coverage90'] = float(np.mean((mu >= lo) & (mu <= hi)))
        out['width_ln'] = float(np.mean(np.log(hi / lo)))
        out['width_mu_median'] = float(np.median(hi - lo))
    return out


def analytic_baselines(d):
    """Closed-form SIS inversions and the cosmology (known-redshift) estimator."""
    y_geo = np.clip(np.abs(d.geo_asym.to_numpy()), 1e-3, 0.999)
    outer = d.geo_r_a.to_numpy() >= d.geo_r_b.to_numpy()
    mu_geo = np.where(outer, 1 + 1 / y_geo, np.maximum(1 / y_geo - 1, 1e-3))
    r = np.exp(d.gw_ln_mu_rel.to_numpy())  # mu_b / mu_a
    y_gw = np.clip(np.abs(1 - r) / (1 + r), 1e-3, 0.999)
    mu_gw = np.where(r <= 1, 1 + 1 / y_gw, np.maximum(1 / y_gw - 1, 1e-3))
    mu_cos = np.exp(d.gw_ln_mu_cosmo.to_numpy())
    return {'SIS analytic (geometry)': mu_geo, 'SIS analytic (GW relative)': mu_gw, 'Known z_s + cosmology': mu_cos}


def run_fit(train, cal, test, cols, seed, tag, mondrian=False, keep_pred=False):
    t0 = time.time()
    m = CQR(cols, seed).fit(train).calibrate(cal, by_family=mondrian)
    rows, preds = [], []
    rng = np.random.default_rng(seed)
    for fam, g in test.groupby('family'):
        mu = g.mu.to_numpy()
        for mode in ([False, True] if mondrian else [False]):
            pred, lo, hi, rlo, rhi = m.predict(g, mondrian=mode)
            r = dict(tag, test_family=fam, calibration='mondrian' if mode else 'global', **metrics(mu, pred, lo, hi, rng))
            r['raw_coverage90'] = float(np.mean((mu >= rlo) & (mu <= rhi)))
            r['fit_seconds'] = time.time() - t0
            rows.append(r)
            if keep_pred and not mode:
                preds.append(pd.DataFrame(dict(event_id=g.event_id, family=fam, mu=mu, mu_b=g.mu_b, y_true=g.y_true, pred=pred, lo=lo, hi=hi,
                                               dl_true=g.dl_true, ln_dapp_a=g.gw_ln_dapp_a, z_s_true=g.z_s_true,
                                               n_images=g.n_images_true, **{k: v for k, v in tag.items() if isinstance(v, (str, int))})))
    return rows, preds


def experiment_inputs(data, scenario, seeds, max_train):
    jobs, base_rows = [], []
    for fam in FAMILIES:
        d = data[data.family == fam].reset_index(drop=True)
        for seed in seeds:
            lab = split(d, seed)
            tr, ca, te = d[lab == 'train'], d[lab == 'cal'], d[lab == 'test']
            if max_train:
                tr = tr.iloc[:max_train]
            rng = np.random.default_rng(seed)
            for name, mu_hat in analytic_baselines(te).items():
                base_rows.append(dict(experiment='inputs', scenario=scenario, train_family=fam, test_family=fam, input_set=name,
                                      method='analytic', split_seed=seed, calibration='none', n_train=0,
                                      **metrics(te.mu.to_numpy(), mu_hat, rng=rng)))
            for name, cols in BENCH_SETS.items():
                tag = dict(experiment='inputs', scenario=scenario, train_family=fam, input_set=name, method='HistGB-CQR',
                           split_seed=seed, n_train=len(tr))
                jobs.append((tr, ca, te, cols, seed, tag, False, seed == seeds[0]))
    return jobs, base_rows


def experiment_mixed(data, scenario, seeds, max_train):
    jobs = []
    for seed in seeds:
        lab = split(data, seed)
        tr, ca, te = data[lab == 'train'], data[lab == 'cal'], data[lab == 'test']
        if max_train:
            tr = tr.groupby('family').head(max_train)
        for name in ('Geometry', MAIN, 'Full'):
            tag = dict(experiment='mixed', scenario=scenario, train_family='all', input_set=name, method='HistGB-CQR', split_seed=seed, n_train=len(tr))
            jobs.append((tr, ca, te, BENCH_SETS[name], seed, tag, True, seed == seeds[0]))
        for omit in FAMILIES:
            keep = tr.family != omit
            tag = dict(experiment='leave_one_out', scenario=scenario, train_family=f'all-{omit}', input_set=MAIN,
                       method='HistGB-CQR', split_seed=seed, n_train=int(keep.sum()))
            jobs.append((tr[keep], ca[ca.family != omit], te[te.family == omit], BENCH_SETS[MAIN], seed, tag, False, False))
    return jobs


def experiment_transfer(data, scenario, seed, max_train):
    jobs = []
    lab = split(data, seed)
    for ftr in FAMILIES:
        tr = data[(lab == 'train') & (data.family == ftr)]
        ca = data[(lab == 'cal') & (data.family == ftr)]
        if max_train:
            tr = tr.iloc[:max_train]
        te = data[lab == 'test']
        for name in ('Geometry', MAIN):
            tag = dict(experiment='transfer', scenario=scenario, train_family=ftr, input_set=name, method='HistGB-CQR', split_seed=seed, n_train=len(tr))
            jobs.append((tr, ca, te, BENCH_SETS[name], seed, tag, False, False))
    return jobs


def experiment_learning(data, scenario, seed, sizes):
    jobs = []
    for fam in ('SIS', 'SIE_shear'):
        d = data[data.family == fam].reset_index(drop=True)
        lab = split(d, seed)
        tr, ca, te = d[lab == 'train'], d[lab == 'cal'], d[lab == 'test']
        for n in sizes:
            if n > len(tr):
                continue
            for name in ('Geometry', 'Full'):
                tag = dict(experiment='learning', scenario=scenario, train_family=fam, input_set=name, method='HistGB-CQR', split_seed=seed, n_train=n)
                jobs.append((tr.iloc[:n], ca, te, INPUT_SETS[name], seed, tag, False, False))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, default=Path('data/v2'))
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--scenarios', nargs='+', default=['ET', 'O5'])
    p.add_argument('--experiments', nargs='+', default=['inputs', 'mixed', 'transfer', 'sky', 'learning'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--max-per-family', type=int, default=60000)
    p.add_argument('--max-train', type=int, default=0)
    p.add_argument('--jobs', type=int, default=16)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    os.environ['OMP_NUM_THREADS'] = str(a.threads)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, all_preds = [], []
    for scen in a.scenarios:
        data = load(a.data_dir, scen, max_per_family=a.max_per_family, seed=1)
        counts = data.family.value_counts().to_dict()
        print(scen, counts, flush=True)
        jobs, base = [], []
        if 'inputs' in a.experiments:
            j, base = experiment_inputs(data, scen, a.seeds, a.max_train)
            jobs += j
        if 'mixed' in a.experiments:
            jobs += experiment_mixed(data, scen, a.seeds[:3], a.max_train)
        if 'transfer' in a.experiments:
            jobs += experiment_transfer(data, scen, a.seeds[0], a.max_train)
        if 'learning' in a.experiments and scen == 'ET':
            jobs += experiment_learning(data, scen, a.seeds[0], [500, 1000, 3000, 10000, 30000])
        if 'sky' in a.experiments and scen == 'ET':
            known = load(a.data_dir, scen, max_per_family=a.max_per_family, seed=1, sky='known')
            for fam in FAMILIES:
                d = known[known.family == fam].reset_index(drop=True)
                lab = split(d, a.seeds[0])
                for name in ('GW', MAIN):
                    tag = dict(experiment='sky_known', scenario=scen, train_family=fam, input_set=name, method='HistGB-CQR', split_seed=a.seeds[0], n_train=int((lab == 'train').sum()))
                    jobs.append((d[lab == 'train'], d[lab == 'cal'], d[lab == 'test'], BENCH_SETS[name], a.seeds[0], tag, False, False))
                rng = np.random.default_rng(0)
                te = d[lab == 'test']
                base.append(dict(experiment='sky_known', scenario=scen, train_family=fam, test_family=fam, input_set='SIS analytic (GW relative)',
                                 method='analytic', split_seed=a.seeds[0], calibration='none', n_train=0,
                                 **metrics(te.mu.to_numpy(), analytic_baselines(te)['SIS analytic (GW relative)'], rng=rng)))
        print(f'{scen}: {len(jobs)} fits', flush=True)
        res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(run_fit)(*j) for j in jobs)
        all_rows += base + [r for rows, _ in res for r in rows]
        all_preds += [p for _, preds in res for p in preds]
        pd.DataFrame(all_rows).to_csv(a.out_dir / 'metrics.csv', index=False)
        pd.concat(all_preds).to_parquet(a.out_dir / 'predictions.parquet', index=False)
        (a.out_dir / f'counts_{scen}.json').write_text(json.dumps({k: int(v) for k, v in counts.items()}))
    print('done', flush=True)


if __name__ == '__main__':
    main()
