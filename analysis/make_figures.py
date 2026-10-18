"""Figures and LaTeX tables for the v2 magnification benchmark.

Reads the outputs of analysis/benchmark.py, analysis/npe.py and the real-data scripts and writes
PDF figures and LaTeX table fragments into ``--out-dir``.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.features import FAMILIES, load  # noqa: E402

FAM_LABEL = {'SIS': 'SIS', 'SIE': 'SIE', 'SIE_shear': 'SIE+shear', 'EPL_shear': 'EPL+shear'}
MAIN = 'Geometry+GW(ratio)'  # scale-free main input set
# Names shown in figures: the observables each set contains ("mu ratio" = GW magnification ratio and arrival order).
DISPLAY = {'GW': 'GW only', 'GW+z': 'GW + $z$', 'Geometry': 'Geometry', 'Geometry+FU': r'Geometry + $z,\sigma_v$',
           'Geometry+GW(ratio)': r'Geometry + $\mu$ ratio', 'Geometry+GW(ratio)+FU': r'Geometry + $\mu$ ratio + $z,\sigma_v$',
           'Geometry+GW(rel)': r'Geometry + $\mu$ ratio + $\Delta t$, SNR',
           'Geometry+GW(rel)+FU': r'Geometry + $\mu$ ratio + $\Delta t$, SNR + $z,\sigma_v$', 'Full': 'All observables',
           'SIS analytic (geometry)': 'SIS analytic (geometry)', 'SIS analytic (GW relative)': r'SIS analytic ($\mu$ ratio)',
           'Known z_s + cosmology': r'$z_s$ + cosmology'}


def show(name):
    return DISPLAY.get(name, name)
SET_ORDER = ['GW', 'GW+z', 'Geometry', 'Geometry+FU', MAIN, 'Geometry+GW(rel)', 'Full']
BASE_ORDER = ['SIS analytic (GW relative)', 'SIS analytic (geometry)', 'Known z_s + cosmology']
COLORS = {'SIS': '#2a6f97', 'SIE': '#c8553d', 'SIE_shear': '#588157', 'EPL_shear': '#8d6a9f'}
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'figure.dpi': 150, 'savefig.bbox': 'tight', 'font.family': 'serif'})
COL = 3.32   # MNRAS column width (inches)
PAGE = 6.97  # MNRAS text width (inches)


def label(axes):
    for k, ax in enumerate(np.atleast_1d(axes)):
        ax.text(-0.02, 1.02, f'({chr(97 + k)})', transform=ax.transAxes, va='bottom', ha='right', fontsize=8, fontweight='bold')


def fmt(m, s=None, d=2):
    return f'{m:.{d}f}' if s is None or not np.isfinite(s) else f'{m:.{d}f}$\\pm${s:.{d}f}'


NPE_SETS = ['GW', 'Geometry', MAIN, 'Geometry+GW(rel)+FU', 'Full']


def load_npe(npe_dir):
    if npe_dir is None or not npe_dir.exists():
        return None
    parts = [pd.read_csv(f) for f in sorted(npe_dir.glob('npe_[EO]*.csv'))]
    return pd.concat(parts, ignore_index=True) if parts else None


def table_inputs(met, scen, out, npe_all=None):
    """Per-family median relative error and coverage for each input set (in-family training)."""
    g = met[(met.experiment == 'inputs') & (met.scenario == scen)]
    agg = g.groupby(['input_set', 'test_family']).agg(mrel=('median_rel', 'mean'), mrel_sd=('median_rel', 'std'),
                                                      mae=('mae', 'mean'), cov=('coverage90', 'mean'), frac=('frac_rel_lt10', 'mean')).reset_index()
    lines = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l' + 'c' * len(FAMILIES) + '@{}}', r'\toprule',
             'Input set & ' + ' & '.join(FAM_LABEL[f] for f in FAMILIES) + r' \\', r'\midrule']
    for name in BASE_ORDER + SET_ORDER:
        cells = []
        for f in FAMILIES:
            r = agg[(agg.input_set == name) & (agg.test_family == f)]
            if r.empty:
                cells.append('--')
                continue
            r = r.iloc[0]
            c = f'{100 * r.mrel:.1f}'
            if np.isfinite(r['cov']):
                c += f' ({r["cov"]:.2f})'
            cells.append(c)
        if name == SET_ORDER[0]:
            lines.append(r'\midrule')
        lines.append(name.replace('_', r'\_').replace('z_s', r'$z_s$') + ' & ' + ' & '.join(cells) + r' \\')
    if npe_all is not None:
        q = npe_all[npe_all.scenario == scen]
        first = True
        for name in NPE_SETS:
            r = q[q.input_set == name].set_index('family')
            if r.empty:
                continue
            if first:
                lines.append(r'\midrule')
                first = False
            cells = [f'{100 * r.loc[f, "median_rel"]:.1f} ({r.loc[f, "coverage90_conformal"]:.2f})' if f in r.index else '--' for f in FAMILIES]
            lines.append('NPE: ' + name + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}']
    (out / f'table_inputs_{scen}.tex').write_text('\n'.join(lines))
    agg.to_csv(out / f'table_inputs_{scen}.csv', index=False)


def fig_inputs(met, out):
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.5), sharey=True)
    for ax, scen in zip(axes, ['ET', 'O5']):
        g = met[(met.experiment == 'inputs') & (met.scenario == scen)]
        names = BASE_ORDER + SET_ORDER
        x = np.arange(len(names))
        w = 0.2
        for k, f in enumerate(FAMILIES):
            v = g[g.test_family == f].groupby('input_set').median_rel.agg(['mean', 'std']).reindex(names)
            ax.bar(x + (k - 1.5) * w, 100 * v['mean'], w, yerr=100 * v['std'].fillna(0), color=COLORS[f], label=FAM_LABEL[f], capsize=1, lw=0)
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([show(n) for n in names], rotation=40, ha='right', fontsize=6)
        ax.axvline(len(BASE_ORDER) - 0.5, color='k', lw=0.5, ls=':')
        ax.set_title(f'{scen} scenario')
        ax.grid(axis='y', alpha=0.3, which='both')
    axes[0].set_ylabel(r'median $|\hat\mu-\mu|/\mu$ (%)')
    axes[0].legend(ncol=2, frameon=False)
    label(axes)
    fig.savefig(out / 'fig_inputs.pdf')
    plt.close(fig)


def fig_tail(pred, out, npe_dir=None):
    fig, axes = plt.subplots(1, 4, figsize=(PAGE, 2.3), sharey=True)
    bins = np.geomspace(1, 100, 16)
    for ax, name in zip(axes, ['Geometry', MAIN, 'NPE: ' + MAIN, 'NPE: Geometry+GW(rel)+FU']):
        if name.startswith('NPE'):
            tag = name[5:].replace('+', '_').replace('(', '').replace(')', '')
            npe_path = npe_dir / f'npe_predictions_ET_{tag}.parquet' if npe_dir else None
            if npe_path is None or not npe_path.exists():
                continue
            p = pd.read_parquet(npe_path)
        else:
            p = pred[(pred.experiment == 'inputs') & (pred.scenario == 'ET') & (pred.input_set == name)]
        for f in FAMILIES:
            q = p[p.family == f]
            idx = np.digitize(q.mu, bins)
            rel = np.abs(q.pred - q.mu) / q.mu
            c = [(np.sqrt(bins[i - 1] * bins[i]), np.median(rel[idx == i]), (idx == i).sum()) for i in range(1, len(bins)) if (idx == i).sum() >= 20]
            if c:
                c = np.array(c)
                ax.plot(c[:, 0], 100 * c[:, 1], 'o-', ms=2.5, color=COLORS[f], label=FAM_LABEL[f])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'true $\mu_A$')
        t = show(name[5:]) if name.startswith('NPE') else show(name)
        t = t.replace(' + $\\Delta t$', '\n+ $\\Delta t$')  # long names on two lines
        ax.set_title(('NPE: ' if name.startswith('NPE') else 'HistGB: ') + t, fontsize=6)
        ax.grid(alpha=0.3, which='both')
    axes[0].set_ylabel('median relative error (%)')
    axes[0].legend(frameon=False)
    label(axes)
    fig.savefig(out / 'fig_tail.pdf')
    plt.close(fig)


def fig_coverage(met, npe, cond, out, mixed=None):
    fig, axes = plt.subplots(1, 3, figsize=(PAGE, 2.4), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(FAMILIES))
    if mixed is not None:  # analysis/review_tests.py: calibration groups without and with the true family
        g = mixed[(mixed.scenario == 'ET') & (mixed.input_set == MAIN)]
        cals = [('global', 'CQR, global', '#999999'), ('pred_mu', r'CQR, by predicted $\mu$', '#b0a24a'),
                ('epl_only', 'CQR, EPL+shear cal.', '#588157'), ('true_family', 'CQR, true family', '#2a6f97')]
        for k, (cal, lab, c) in enumerate(cals):
            v = g[g.calibration == cal].groupby('test_family').coverage90.agg(['mean', 'std']).reindex(FAMILIES)
            ax.errorbar(x + (k - 2) * 0.13, v['mean'], v['std'].fillna(0), fmt='o', color=c, label=lab, ms=2.5, capsize=1.5)
    else:
        g = met[(met.experiment == 'mixed') & (met.scenario == 'ET') & (met.input_set == MAIN)]
        for k, (cal, lab, c) in enumerate([('global', 'CQR, global', '#999999'), ('mondrian', 'CQR, per family', '#2a6f97')]):
            v = g[g.calibration == cal].groupby('test_family').coverage90.agg(['mean', 'std']).reindex(FAMILIES)
            ax.errorbar(x + (k - 1) * 0.18, v['mean'], v['std'].fillna(0), fmt='o', color=c, label=lab, ms=3, capsize=2)
    if npe is not None:
        v = npe.set_index('family').reindex(FAMILIES)
        ax.plot(x + 0.26, v.coverage90, 's', mfc='none', color='#c8553d', ms=3, label='NPE, uncorrected')
        if 'coverage90_conformal' in v:
            ax.plot(x + 0.39, v.coverage90_conformal, 's', color='#c8553d', ms=3, label='NPE, true family')
    ax.axhline(0.9, color='k', lw=0.6, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels([FAM_LABEL[f] for f in FAMILIES], rotation=20)
    ax.set_ylabel('coverage of 90% intervals')
    ax.legend(frameon=False, loc='lower left', fontsize=6)
    ax.set_ylim(0.6, 1.0)
    ax.set_title('mixed-family model', fontsize=8)
    ax = axes[1]
    if cond is not None:
        c = cond[(cond.scenario == 'ET') & (cond.mu_bin != 'all')]
        order = [b for b in c.mu_bin.unique()]
        order = sorted(order, key=lambda b: float(b.split(',')[0].strip('(')))
        for mode, mk, col in (('marginal', 'o-', '#999999'), ('conditional', 's-', '#2a6f97')):
            v = c[c.calibration == mode].groupby('mu_bin').coverage90.mean().reindex(order)
            ax.plot(np.arange(len(order)), v.to_numpy(), mk, color=col, ms=3, label=f'{mode} calibration')
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([b.replace('inf', r'$\infty$') for b in order], rotation=30, fontsize=6)
        ax.set_xlabel(r'true $\mu_A$ bin')
        ax.axhline(0.9, color='k', lw=0.6, ls='--')
        ax.set_ylim(0.4, 1.0)
        ax.legend(frameon=False, fontsize=6, loc='lower left')
    ax = axes[2]
    if npe is not None:
        for f in FAMILIES:
            h = json.loads(npe.set_index('family').loc[f, 'pit_hist'])
            ax.step(np.linspace(0.05, 0.95, 10), np.array(h) * 10, where='mid', color=COLORS[f], label=FAM_LABEL[f])
        ax.axhline(1, color='k', lw=0.6, ls='--')
        ax.set_xlabel(r'PIT of true $\ln\mu_A$ (NPE)')
        ax.set_ylabel('density')
        ax.set_ylim(0, 2)
        ax.legend(frameon=False, ncol=2, fontsize=6)
    label(axes)
    fig.savefig(out / 'fig_coverage.pdf')
    plt.close(fig)


def fig_transfer(met, out):
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.9), constrained_layout=True)
    for ax, name in zip(axes, ['Geometry', MAIN]):
        g = met[(met.experiment == 'transfer') & (met.scenario == 'ET') & (met.input_set == name)]
        m = g.pivot_table(index='train_family', columns='test_family', values='median_rel').reindex(index=FAMILIES, columns=FAMILIES)
        c = g.pivot_table(index='train_family', columns='test_family', values='coverage90').reindex(index=FAMILIES, columns=FAMILIES)
        im = ax.imshow(100 * m.to_numpy(), cmap='magma_r', vmin=0, vmax=max(40, np.nanmax(100 * m.to_numpy())))
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{100 * m.iloc[i, j]:.1f}\n[{c.iloc[i, j]:.2f}]', ha='center', va='center', fontsize=6.5,
                        color='w' if 100 * m.iloc[i, j] > 25 else 'k')
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels([FAM_LABEL[f] for f in FAMILIES], rotation=30)
        ax.set_yticklabels([FAM_LABEL[f] for f in FAMILIES])
        ax.set_xlabel('test family')
        ax.set_title(name, fontsize=8)
        plt.colorbar(im, ax=ax, label='median rel. error (%)', fraction=0.046, pad=0.02)
    axes[0].set_ylabel('training family')
    label(axes)
    fig.savefig(out / 'fig_transfer.pdf')
    plt.close(fig)


def fig_distance(pred, out):
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    p = pred[(pred.experiment == 'inputs') & (pred.scenario == 'ET')]
    for name, ls in [(MAIN, '-'), ('Full', '--'), ('GW', ':')]:
        q = p[p.input_set == name]
        dhat = np.exp(q.ln_dapp_a) * np.sqrt(q.pred)
        e = np.log(dhat / q.dl_true)
        ax.hist(e, bins=np.linspace(-1.2, 1.2, 80), histtype='step', density=True, ls=ls, label=show(name), color='k' if name != 'GW' else '#c8553d')
    q = p[p.input_set == 'Full']
    e0 = np.log(np.exp(q.ln_dapp_a) * np.sqrt(q.mu) / q.dl_true)
    ax.hist(e0, bins=np.linspace(-1.2, 1.2, 80), histtype='stepfilled', density=True, alpha=0.25, color='#2a6f97', label='true $\\mu$ (GW amplitude error only)')
    ax.set_xlabel(r'$\ln(\hat d_L/d_L)$')
    ax.set_ylabel('density')
    ax.legend(frameon=False, fontsize=6)
    fig.savefig(out / 'fig_distance.pdf')
    plt.close(fig)


def fig_learning(met, out):
    g = met[met.experiment == 'learning']
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for (fam, name), q in g.groupby(['train_family', 'input_set']):
        q = q.groupby('n_train').median_rel.mean()
        ax.plot(q.index, 100 * q.values, 'o-' if name == 'Geometry' else 's--', ms=3, color=COLORS[fam], label=f'{FAM_LABEL[fam]}, {show(name)}')
    ax.set_xscale('log')
    ax.set_xlabel('training events')
    ax.set_ylabel('median rel. error (%)')
    ax.legend(frameon=False, fontsize=6)
    ax.grid(alpha=0.3)
    fig.savefig(out / 'fig_learning.pdf')
    plt.close(fig)


def fig_population(data_dir, out):
    fig, axes = plt.subplots(1, 4, figsize=(PAGE, 1.9))
    for scen, ls in [('ET', '-'), ('O5', '--')]:
        d = load(data_dir, scen, max_per_family=20000, seed=3)
        for f in FAMILIES:
            q = d[d.family == f]
            kw = dict(histtype='step', density=True, color=COLORS[f], ls=ls, label=f'{FAM_LABEL[f]} {scen}' if f == 'SIS' or scen == 'ET' else None)
            axes[0].hist(q.z_s_true, bins=np.linspace(0, 8, 40), **kw)
            axes[1].hist(np.log10(q.mu), bins=np.linspace(-0.3, 2, 40), **kw)
            axes[2].hist(np.log10(np.exp(q.gw_ln_abs_dt) / 86400), bins=np.linspace(-3, 2.6, 40), **kw)
            axes[3].hist(q.geo_theta_proxy, bins=np.linspace(0, 3, 40), **kw)
    for ax, lab in zip(axes, ['$z_s$', r'$\log_{10}\mu_A$', r'$\log_{10}(|\Delta t|/{\rm d})$', r'$\hat\theta_E$ (arcsec)']):
        ax.set_xlabel(lab)
        ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=5)
    label(axes)
    fig.savefig(out / 'fig_population.pdf')
    plt.close(fig)


def fig_realdata(rd, out):
    ev = pd.read_csv(rd / 'gwtc_bbh_events.csv')
    cal = json.loads((rd / 'distance_error_calibration.json').read_text())
    inj_path = rd / 'o4a_injections.csv'
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.3))
    ax = axes[0]
    for cat, m in zip(sorted(ev.catalog.unique()), 'osd^'):
        q = ev[ev.catalog == cat]
        ax.plot(q.network_matched_filter_snr, q.sigma_lnD, m, ms=2.5, alpha=0.7, label=cat)
    r = np.geomspace(6, 100, 100)
    ax.plot(r, np.sqrt(cal['a'] ** 2 + (cal['b'] / r) ** 2), 'k-', lw=1, label=rf'fit: $a={cal["a"]:.2f}$, $b={cal["b"]:.2f}$')
    ax.plot(r, 1 / r, 'k:', lw=0.8, label=r'$1/\rho$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'network matched-filter SNR $\rho$')
    ax.set_ylabel(r'$\sigma_{\ln d_L}$')
    ax.legend(frameon=False, fontsize=5.5)
    ax = axes[1]
    if inj_path.exists():
        inj = pd.read_csv(inj_path)
        inj = inj[inj.rho_b > 8]
        bins = np.linspace(-5, 5, 41)
        for mode, c in [('o4a', '#c8553d'), ('gaussian', '#2a6f97')]:
            q = inj[inj.noise == mode]
            ax.hist(q.z, bins=bins, density=True, histtype='step', color=c, label=f'{"O4a H1+L1 noise" if mode == "o4a" else "Gaussian noise"} (N={len(q)})')
        x = np.linspace(-5, 5, 200)
        ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), 'k--', lw=0.8, label='N(0,1) model')
        ax.set_xlabel(r'standardised error of $\ln(\hat\mu_B/\hat\mu_A)$')
        ax.set_ylabel('density')
        ax.legend(frameon=False, fontsize=6)
    label(axes)
    fig.savefig(out / 'fig_realdata.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--npe', type=Path, default=None, help='NPE summary CSV used for the coverage figure')
    p.add_argument('--npe-dir', type=Path, default=None, help='Directory with all NPE outputs')
    p.add_argument('--review-dir', type=Path, default=None, help='Output of analysis/review_tests.py (calibration groups)')
    p.add_argument('--realdata', type=Path, default=Path('realdata/outputs'))
    p.add_argument('--data-dir', type=Path, default=Path('data/v2'))
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    met = pd.read_csv(a.results / 'metrics.csv')
    pred = pd.read_parquet(a.results / 'predictions.parquet')
    npe = pd.read_csv(a.npe) if a.npe and a.npe.exists() else None
    cpath = a.results.parent / (a.results.name + '_conditional') / 'conditional_calibration.csv'
    cond = pd.read_csv(cpath) if cpath.exists() else None
    npe_all = load_npe(a.npe_dir)
    for scen in met.scenario.unique():
        table_inputs(met, scen, a.out_dir, npe_all)
    fig_inputs(met, a.out_dir)
    fig_tail(pred, a.out_dir, a.npe_dir)
    mixed = pd.read_csv(a.review_dir / 'mixed_calibration.csv') if a.review_dir else None
    fig_coverage(met, npe, cond, a.out_dir, mixed)
    fig_transfer(met, a.out_dir)
    fig_distance(pred, a.out_dir)
    fig_learning(met, a.out_dir)
    fig_population(a.data_dir, a.out_dir)
    fig_realdata(a.realdata, a.out_dir)
    print('figures written to', a.out_dir)


if __name__ == '__main__':
    main()
