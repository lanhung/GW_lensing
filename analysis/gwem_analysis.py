"""Summarise the Bayesian GW+EM lens reconstructions (gwlens.joint_gwem) and compare them with NPE.

Reads the full (strain + EM) and two-stage (joint-PE summary x EM) summaries, and the per-event
NPE intervals written by analysis/pe_analysis.py, and reports

* accuracy and 90 per cent coverage of ln mu_A for every analysis configuration, per scenario and
  family, with the GW posterior from the free-sky or the known-sky joint PE;
* the two-stage approximation against the full analysis on the same events and data
  (shift of the median in units of the full posterior width, ratio of widths);
* Bayesian two-stage against NPE with the same (scale-free) information, event by event.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Early free-sky ET joint-PE runs whose noise realisation predates the per-event seed, so that their
# GW data differ from those of the full GW+EM analysis; excluded from the matched comparison.
UNMATCHED_NOISE = {('ET', 'SIE-7777-0023296'), ('ET', 'EPL_shear-7777-0028423'), ('ET', 'SIS-7777-0007321'),
                   ('ET', 'SIE-7777-0020513'), ('ET', 'EPL_shear-7777-0005239'), ('ET', 'SIE_shear-7777-0017305')}
FAMILIES = ['SIS', 'SIE', 'SIE_shear', 'EPL_shear']
# the GW posterior each scenario relies on: the host fixes the sky for ET, whose single-site
# free-sky posteriors are multimodal; O5 is analysed with a free sky
PRIMARY_SKY = {'ET': 'known', 'O5': 'free'}


def load(full_dir, posthoc_dirs):
    rows = []
    for f in Path(full_dir).glob('*_summary.json'):
        rows.append(dict(json.loads(f.read_text()), sky='free', config='full'))
    for f in (f for d in posthoc_dirs for f in Path(d).glob('*/*_summary.json')):
        # directory = <cosmology>_k<sigma_kext>_<lens model>[_ast<sigma>][_h0][_xpos][_sn][_ks] (the tags of
        # scripts/run_gwem_posthoc.sh); the EPL+shear directories also hold the EPL+shear family analysed with
        # its own model. Free-H0 runs are analysed separately.
        m = re.fullmatch(r'(free|lcdm)_k([0-9.]+)_(\w+?)(?:_ast([0-9.]+))?(_h0)?(_xpos)?(?:_sn)?(_ks)?', f.parent.name)
        cosmo, kext, model, ast, h0, xpos, ks = m.groups()
        if h0:
            continue
        s = json.loads(f.read_text())
        if xpos:
            cfgs = ['two-stage exact' + (f' ast{float(ast):g}' if ast else '')]
        elif ast:
            cfgs = [f'two-stage ast{float(ast):g}']
        elif float(kext) or cosmo == 'lcdm':
            cfgs = [f"two-stage EPL_shear kext{float(kext):g}" + (' lcdm' if cosmo == 'lcdm' else '')]
        elif model == 'EPL_shear':
            cfgs = ['two-stage EPL_shear'] + (['two-stage'] if s['family'] == 'EPL_shear' else [])
        else:
            cfgs = ['two-stage']
        for c in cfgs:
            rows.append(dict(s, sky='known' if ks else 'free', config=c))
    d = pd.DataFrame(rows)
    d['err_ln'] = d.ln_mu_a_median - np.log(d.true_mu_a)
    d['width90'] = d.ln_mu_a_q95 - d.ln_mu_a_q05
    return d


def npe_rows(path):
    n = pd.read_csv(path)
    n['sky'] = np.where(n.known_sky, 'known', 'free')
    n['config'] = 'NPE ' + n.gw_input
    n['true_mu_a'] = np.exp(n.true_ln_mu_a)
    n['err_ln'] = n.ln_mu_a_median - n.true_ln_mu_a
    n['width90'] = n.ln_mu_a_q95 - n.ln_mu_a_q05
    n['rel_err'] = np.abs(np.exp(n.ln_mu_a_median) - n.true_mu_a) / n.true_mu_a
    n['covered90'] = (n.true_ln_mu_a >= n.ln_mu_a_q05) & (n.true_ln_mu_a <= n.ln_mu_a_q95)
    return n[['scenario', 'event_id', 'family', 'sky', 'config', 'true_mu_a', 'ln_mu_a_median', 'ln_mu_a_q05',
              'ln_mu_a_q95', 'err_ln', 'width90', 'rel_err', 'covered90']]


def accuracy(d):
    g = d.groupby(['scenario', 'sky', 'config'])
    out = g.agg(n=('rel_err', 'size'), median_rel=('rel_err', 'median'), coverage90=('covered90', 'mean'),
                median_width90=('width90', 'median')).reset_index()
    fam = d.groupby(['scenario', 'sky', 'config', 'family']).agg(median_rel=('rel_err', 'median'),
                                                                 coverage90=('covered90', 'mean'), n=('rel_err', 'size'))
    return out, fam.reset_index()


def full_vs_twostage(d):
    """Two-stage (free-sky GW posterior, same data) versus full analysis, event by event."""
    full = d[d.config == 'full'].set_index(['scenario', 'event_id'])
    two = d[(d.config == 'two-stage') & (d.sky == 'free')].set_index(['scenario', 'event_id'])
    j = full.join(two, rsuffix='_2s', how='inner')
    j = j[[k not in UNMATCHED_NOISE for k in j.index]]
    sd = j.width90 / 3.29
    j['shift'] = (j.ln_mu_a_median_2s - j.ln_mu_a_median) / sd
    j['width_ratio'] = j.width90_2s / j.width90
    rows = []
    for scen, g in list(j.groupby(level=0)) + [('all', j)]:
        rows.append(dict(scenario=scen, n=len(g), median_abs_shift=float(np.median(np.abs(g['shift']))),
                         frac_shift_gt1=float(np.mean(np.abs(g['shift']) > 1)),
                         median_width_ratio=float(np.median(g.width_ratio)),
                         coverage_full=float(g.covered90.mean()), coverage_2s=float(g.covered90_2s.mean()),
                         median_rel_full=float(g.rel_err.median()), median_rel_2s=float(g.rel_err_2s.median())))
    return pd.DataFrame(rows), j.reset_index()


def primary(d):
    return d[d.sky == d.scenario.map(PRIMARY_SKY)]


def table(acc):
    """LaTeX table: median relative error (per cent) and coverage per scenario for each analysis."""
    rows = [('full', 'free', 'Full strain + EM fit, free sky'),
            ('two-stage', 'free', 'Two-stage, free-sky GW PE'),
            ('two-stage', 'known', 'Two-stage, known-sky GW PE'),
            ('two-stage EPL_shear', None, r'\quad EPL+shear model'),
            ('two-stage EPL_shear kext0.025', None, r'\quad EPL+shear, $\sigma_{\kappa}=0.025$'),
            ('two-stage EPL_shear kext0.025 lcdm', None, r'\quad EPL+shear, $\sigma_{\kappa}=0.025$, $\Lambda$CDM'),
            ('two-stage ast0.03', None, r'\quad GW-image astrometry 0.03 arcsec'),
            ('two-stage ast0.1', None, r'\quad GW-image astrometry 0.1 arcsec'),
            ('two-stage exact', None, r'\quad exact image-position likelihood'),
            ('two-stage exact ast0.03', None, r'\quad GW-image astrometry 0.03 arcsec, exact likelihood'),
            ('two-stage exact ast0.1', None, r'\quad GW-image astrometry 0.1 arcsec, exact likelihood'),
            ('NPE SNR model', None, r'NPE, SNR-model $\mu_B/\mu_A$'),
            ('NPE joint PE', None, r'NPE, joint-PE $\mu_B/\mu_A$')]
    a = acc.set_index(['scenario', 'sky', 'config'])
    lines = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccccc@{}}', r'\toprule',
             r' & \multicolumn{3}{c}{ET} & \multicolumn{3}{c}{O5} \\',
             r'Analysis & $N$ & err. (\%) & cov. & $N$ & err. (\%) & cov. \\', r'\midrule']
    for cfg, sky, label in rows:
        cells = []
        for scen in ('ET', 'O5'):
            key = (scen, sky or PRIMARY_SKY[scen], cfg)
            if key in a.index:
                r = a.loc[key]
                cells += [f'{int(r.n)}', f'{100 * r.median_rel:.1f}', f'{r.coverage90:.2f}']
            else:
                cells += ['--'] * 3
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}']
    return '\n'.join(lines)


def figure(d, j, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.4))
    marker = {'ET': 'o', 'O5': 's'}
    for scen, g in j.groupby('scenario'):
        ax[0].errorbar(g.err_ln, g.err_ln_2s, xerr=g.width90 / 3.29, yerr=g.width90_2s / 3.29, fmt=marker[scen],
                       ms=3, lw=0.5, alpha=0.7, label=scen)
    lim = 1.1 * np.nanmax(np.abs(np.r_[j.err_ln, j.err_ln_2s]))
    ax[0].plot([-lim, lim], [-lim, lim], color='grey', lw=0.8)
    ax[0].set_xlim(-lim, lim)
    ax[0].set_ylim(-lim, lim)
    ax[0].set_xlabel(r'full: $\ln\hat\mu_A-\ln\mu_A$')
    ax[0].set_ylabel(r'two-stage: $\ln\hat\mu_A-\ln\mu_A$')
    ax[0].legend(fontsize=6, frameon=False)
    p = primary(d)
    two = p[p.config == 'two-stage'].set_index(['scenario', 'event_id'])
    npe = p[p.config == 'NPE joint PE'].set_index(['scenario', 'event_id'])
    m = two.join(npe, rsuffix='_npe', how='inner')
    for i, fam in enumerate(FAMILIES):
        for scen in ('ET', 'O5'):
            g = m[(m.family == fam) & (m.index.get_level_values(0) == scen)]
            ax[1].scatter(g.rel_err_npe, g.rel_err, s=8, marker=marker[scen], color=f'C{i}',
                          label=fam.replace('_', '+') if scen == 'ET' else None)
    ax[1].plot([1e-4, 3], [1e-4, 3], color='grey', lw=0.8)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlim(1e-4, 3)
    ax[1].set_ylim(1e-4, 3)
    ax[1].set_xlabel('NPE relative error')
    ax[1].set_ylabel('two-stage relative error')
    ax[1].legend(fontsize=6, frameon=False, loc='upper left')
    pit = lambda g: np.sort(np.clip((np.log(g.true_mu_a) - g.ln_mu_a_q05) / g.width90, -0.5, 1.5))  # noqa: E731
    for cfg, ls, label in (('full', '-', 'full'), ('two-stage', '--', 'two-stage'), ('NPE joint PE', ':', 'NPE')):
        g = d[d.config == cfg] if cfg == 'full' else p[p.config == cfg]
        x = pit(g)
        ax[2].plot(x, (np.arange(len(x)) + 1) / len(x), ls=ls, color='k', drawstyle='steps-post', label=label)
    for v in (0, 1):
        ax[2].axvline(v, color='grey', lw=0.6)
    ax[2].axhline(0.05, color='grey', lw=0.6, ls=':')
    ax[2].axhline(0.95, color='grey', lw=0.6, ls=':')
    ax[2].set_xlabel(r'position of $\ln\mu_A$ in the 90% interval')
    ax[2].set_ylabel('CDF')
    ax[2].legend(fontsize=6, frameon=False, loc='upper left')
    for a, l in zip(ax, 'abc'):
        a.text(-0.22, 1.04, f'({l})', transform=a.transAxes, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--full-dir', type=Path, default=Path('results/gwem_full'))
    p.add_argument('--posthoc-dir', type=Path, nargs='+', default=[Path('results/gwem_posthoc')],
                   help='two-stage output directories, e.g. the standard and the exact-position runs')
    p.add_argument('--npe-events', type=Path, default=Path('results/pe_analysis/pe_end_to_end_events.csv'))
    p.add_argument('--out-dir', type=Path, default=Path('results/gwem_analysis'))
    p.add_argument('--figure', type=Path, default=None)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    d = load(a.full_dir, a.posthoc_dir)
    if a.npe_events.exists():
        d = pd.concat([d, npe_rows(a.npe_events)], ignore_index=True)
    d.to_csv(a.out_dir / 'gwem_events.csv', index=False)
    acc, fam = accuracy(d)
    acc.to_csv(a.out_dir / 'gwem_accuracy.csv', index=False)
    fam.to_csv(a.out_dir / 'gwem_accuracy_family.csv', index=False)
    cmp, j = full_vs_twostage(d)
    cmp.to_csv(a.out_dir / 'gwem_full_vs_twostage.csv', index=False)
    (a.out_dir / 'table_gwem.tex').write_text(table(acc))
    pd.set_option('display.width', 250)
    print(acc.round(3).to_string())
    print(cmp.round(3).to_string())
    print(fam[fam.config.isin(['full', 'two-stage', 'NPE joint PE'])].pivot_table(
        index=['scenario', 'sky', 'config'], columns='family', values='median_rel').round(3).to_string())
    if a.figure:
        figure(d, j, a.figure)


if __name__ == '__main__':
    main()
