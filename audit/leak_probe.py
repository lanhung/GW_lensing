"""Audit task A: feature provenance, split integrity and leakage probes for the v2 benchmark.

1. integrity.json   unique event identifiers, duplicate feature rows, disjoint train/cal/test splits
                    (analysis.benchmark.split, seed 0), and the input gate (features.check_inputs) applied
                    to every declared input set and to truth columns.
2. correlations.csv Spearman rank correlation of every candidate input with ln mu_A, ln mu_B and the
                    source offset y, per family (a near-unit correlation would flag a leak).
3. single_feature.csv  HistGB median-of-ln-mu regressors trained on ONE input at a time (and on the prior
                    only) per family, with the median relative error of mu_A on the held-out test split;
                    plus the declared input sets and, as a demonstration of what leakage looks like, the
                    forbidden truth columns (mu_B, y, mu_A itself) fed deliberately past the gate.
4. published.csv    the paper's HistGB metrics (results/v2b/metrics.csv) averaged over splits as in the
                    paper's Table, and the rows whose metrics round to 0.0 at one decimal.

    python audit/leak_probe.py --data-dir data/v2 --published published_results/v2b --out-dir out/leak
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.benchmark import HGB, split  # noqa: E402
from analysis.features import ALL_SETS, COSMO, FAMILIES, FOLLOWUP, GEO, GW, check_inputs, load  # noqa: E402

CANDIDATES = GEO + GW + FOLLOWUP + COSMO
TRUTH = ['mu_b', 'y_true', 'dl_true', 'z_s_true', 'n_images_true', 'morse_a', 'mu']


def fit_predict(tr, te, cols, seed):
    m = HistGradientBoostingRegressor(loss='quantile', quantile=0.5, random_state=seed, **HGB)
    m.fit(tr[cols], np.log(tr.mu.to_numpy()))
    pred = np.exp(m.predict(te[cols]))
    mu = te.mu.to_numpy()
    rel = np.abs(pred - mu) / mu
    return dict(median_rel=float(np.median(rel)), mae_ln=float(np.mean(np.abs(np.log(pred / mu)))),
                frac_rel_lt1pct=float(np.mean(rel < 0.01)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, default=Path('data/v2'))
    p.add_argument('--published', type=Path, default=Path('published_results/v2b'))
    p.add_argument('--out-dir', type=Path, default=Path('out/leak'))
    p.add_argument('--scenario', default='ET')
    p.add_argument('--n-train', type=int, default=20000)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    os.environ['OMP_NUM_THREADS'] = str(a.threads)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    data = load(a.data_dir, a.scenario, max_per_family=60000, seed=1)   # exactly as analysis/benchmark.py
    lab = split(data, 0)

    # 1. integrity
    feat_hash = pd.util.hash_pandas_object(data[CANDIDATES].round(12), index=False)
    integ = dict(scenario=a.scenario, n_events=int(len(data)), n_unique_event_id=int(data.event_id.nunique()),
                 n_duplicate_feature_rows=int(feat_hash.duplicated().sum()),
                 split_sizes={k: int((lab == k).sum()) for k in ('train', 'cal', 'test')},
                 split_disjoint=bool(len(set(data.event_id[lab == 'train']) & set(data.event_id[lab == 'test'])) == 0
                                     and len(set(data.event_id[lab == 'cal']) & set(data.event_id[lab == 'test'])) == 0),
                 gate_accepts_declared_sets={k: check_inputs(v) == v for k, v in ALL_SETS.items()})
    rejected = {}
    for c in TRUTH + ['event_id', 'family']:
        try:
            check_inputs(GEO + [c])
            rejected[c] = 'ACCEPTED'
        except ValueError as e:
            rejected[c] = 'rejected: ' + str(e)[:60]
    integ['gate_on_truth_columns'] = rejected
    # Columns of the raw catalogue that feed the features, by prefix (provenance)
    raw = pd.read_parquet(a.data_dir / 'SIS.parquet').columns
    integ['raw_catalogue_columns'] = {pre: sorted(c for c in raw if c.startswith(pre)) for pre in ('true_', 'obs_', f'{a.scenario}__')}
    (a.out_dir / 'integrity.json').write_text(json.dumps(integ, indent=1))
    print(json.dumps({k: v for k, v in integ.items() if k != 'raw_catalogue_columns'}, indent=1), flush=True)

    # 2. correlations
    rows = []
    for fam, g in data.groupby('family'):
        for c in CANDIDATES:
            x = np.nan_to_num(g[c].to_numpy(float), nan=-1.0)
            rows.append(dict(family=fam, feature=c,
                             spearman_ln_mu_a=float(stats.spearmanr(x, np.log(g.mu)).statistic),
                             spearman_ln_mu_b=float(stats.spearmanr(x, np.log(g.mu_b)).statistic),
                             spearman_y=float(stats.spearmanr(x, g.y_true).statistic)))
    corr = pd.DataFrame(rows)
    corr.to_csv(a.out_dir / 'correlations.csv', index=False)
    print(corr.sort_values('spearman_ln_mu_a', key=np.abs, ascending=False).head(12).round(3).to_string(), flush=True)

    # 3. single-feature and set probes
    rows = []
    for fam in FAMILIES:
        d = data[data.family == fam]
        tr, te = d[lab[data.family == fam] == 'train'].iloc[:a.n_train], d[lab[data.family == fam] == 'test']
        y = np.log(tr.mu.to_numpy())
        mu = te.mu.to_numpy()
        med = np.exp(np.median(y))
        rows.append(dict(family=fam, probe='prior only (training median)', kind='baseline', n_inputs=0,
                         median_rel=float(np.median(np.abs(med - mu) / mu)), mae_ln=float(np.mean(np.abs(np.log(med / mu)))),
                         frac_rel_lt1pct=float(np.mean(np.abs(med - mu) / mu < 0.01))))
        t0 = time.time()
        for c in CANDIDATES:
            rows.append(dict(family=fam, probe=c, kind='single observable', n_inputs=1, **fit_predict(tr, te, [c], 0)))
        for name, cols in ALL_SETS.items():
            rows.append(dict(family=fam, probe=name, kind='declared input set', n_inputs=len(cols), **fit_predict(tr, te, cols, 0)))
        for c in ('mu_b', 'y_true', 'mu'):
            rows.append(dict(family=fam, probe=f'Geometry + {c} (TRUTH, forbidden)', kind='leak demonstration',
                             n_inputs=len(GEO) + 1, **fit_predict(tr, te, GEO + [c], 0)))
        rows.append(dict(family=fam, probe='Geometry + mu_b + gw_ln_mu_rel (TRUTH, forbidden)', kind='leak demonstration',
                         n_inputs=len(GEO) + 2, **fit_predict(tr, te, GEO + ['mu_b', 'gw_ln_mu_rel'], 0)))
        print(fam, f'{time.time() - t0:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(a.out_dir / 'single_feature.csv', index=False)
    sf = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print(sf.pivot_table(index=['kind', 'probe'], columns='family', values='median_rel').round(3).to_string(), flush=True)

    # 4. published metrics
    if (a.published / 'metrics.csv').exists():
        m = pd.read_csv(a.published / 'metrics.csv')
        tab = (m[(m.experiment == 'inputs') & (m.scenario == a.scenario) & (m.calibration != 'mondrian')]
               .groupby(['input_set', 'method', 'test_family'])[['median_rel', 'coverage90', 'mae_ln', 'bias', 'mae']]
               .mean().reset_index())
        tab.to_csv(a.out_dir / 'published.csv', index=False)
        print(tab.pivot_table(index=['method', 'input_set'], columns='test_family', values='median_rel').round(3).to_string())
        z = m[['experiment', 'scenario', 'input_set', 'test_family', 'method', 'mae', 'median_ae', 'median_rel', 'mae_ln', 'bias']].copy()
        near0 = {c: int((z[c].round(1) == 0).sum()) for c in ('mae', 'median_ae', 'median_rel', 'mae_ln', 'bias')}
        print('metric values that round to 0.0 at one decimal:', near0)
        z[(z.median_rel.round(1) == 0) | (z.mae_ln.round(1) == 0) | (z.bias.round(1) == 0)].to_csv(a.out_dir / 'published_rounds_to_zero.csv', index=False)
        pred_path = a.published / 'predictions.parquet'
        if pred_path.exists():
            pr = pd.read_parquet(pred_path)
            rep = pr.groupby(['scenario', 'experiment', 'input_set', 'train_family', 'family']).apply(
                lambda g: pd.Series(dict(n=len(g), median_rel_recomputed=float(np.median(np.abs(g.pred - g.mu) / g.mu)),
                                         coverage90_recomputed=float(np.mean((g.mu >= g.lo) & (g.mu <= g.hi)))))).reset_index()
            rep.to_csv(a.out_dir / 'published_recomputed_from_predictions.csv', index=False)
    print('done')


if __name__ == '__main__':
    main()
