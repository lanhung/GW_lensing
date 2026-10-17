"""LaTeX tables for the information-content, distance-scale and calibration tests.

Reads the outputs of analysis/review_tests.py (HistGB), analysis/npe.py and
analysis/review_distance_scale.py (NPE) and the analytic baselines of analysis/benchmark.py.
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = ['SIS', 'SIE', 'SIE_shear', 'EPL_shear']
HEAD = 'SIS & SIE & SIE+shear & EPL+shear'
HGB_ROWS = ['Prior only', 'GW', 'GW+z', 'Geometry', 'Geometry+FU', 'Geometry+GW(ratio)', 'Geometry+GW(ratio)+FU',
            'Geometry+GW(rel)', 'Geometry+GW(rel)+FU', 'Full']
NPE_ROWS = ['GW', 'Geometry', 'Geometry+GW(ratio)', 'Geometry+GW(ratio)+FU', 'Geometry+GW(rel)', 'Geometry+GW(rel)+FU', 'Full']
SCALE_FREE = {'Prior only', 'Geometry', 'Geometry+FU', 'Geometry+GW(ratio)', 'Geometry+GW(ratio)+FU'}


def tex(name):
    return name.replace('GW+z', r'GW+$z$')


def accuracy_table(abl, npe, analytic, scen):
    """Median relative error (per cent) and coverage of the 90 per cent interval."""
    lines = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccc@{}}', r'\toprule', f'Input set & {HEAD} \\\\', r'\midrule']
    if analytic is not None:
        a = analytic[(analytic.scenario == scen) & (analytic.method == 'analytic') & (analytic.experiment == 'inputs')]
        for name, g in a.groupby('input_set', sort=False):
            m = g.groupby('test_family').median_rel.mean()
            lines.append(name.replace('z_s', r'$z_s$') + ' & ' + ' & '.join(f'{100 * m[f]:.1f}' for f in FAMILIES) + r' \\')
        lines.append(r'\midrule')
    h = abl[abl.scenario == scen].groupby(['input_set', 'family'])[['median_rel', 'coverage90']].mean()
    for name in HGB_ROWS:
        if name not in h.index.get_level_values(0):
            continue
        cells = [f"{100 * h.loc[(name, f), 'median_rel']:.1f} ({h.loc[(name, f), 'coverage90']:.2f})" for f in FAMILIES]
        mark = r'$^\dagger$' if name in SCALE_FREE else ''
        lines.append(f'{tex(name)}{mark} & ' + ' & '.join(cells) + r' \\')
    lines.append(r'\midrule')
    n = npe[npe.scenario == scen].set_index(['input_set', 'family'])
    for name in NPE_ROWS:
        if name not in n.index.get_level_values(0):
            continue
        cells = [f"{100 * n.loc[(name, f), 'median_rel']:.1f} ({n.loc[(name, f), 'coverage90_conformal']:.2f})" for f in FAMILIES]
        mark = r'$^\dagger$' if name in SCALE_FREE else ''
        lines.append(f'NPE: {tex(name)}{mark} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}']
    return '\n'.join(lines)


def response_table(msd_h, msd_n, dsc_n):
    """d ln(mu_hat) / d ln(lambda) and d ln(mu_hat) / d ln(h), from the +-10 per cent transforms."""
    def slope(df, key):
        g = df.set_index(key).median_dln_pred
        return (g.loc[1.1] - g.loc[0.9]) / (np.log(1.1) - np.log(0.9))
    rows = []
    for scen in ('ET', 'O5'):
        for name in ['Geometry+GW(ratio)+FU', 'Geometry+GW(rel)', 'Geometry+GW(ratio+clock)+FU', 'Geometry+GW(ratio+SNR)+FU',
                     'Geometry+GW(rel)+FU', 'Full']:
            r = dict(scenario=scen, input_set=name)
            for tr, col in (('mass_sheet', 'hgb_lam'), ('distance_scale', 'hgb_h')):
                d = msd_h[(msd_h.scenario == scen) & (msd_h.input_set == name) & (msd_h['transform'] == tr)]
                if len(d):
                    r[col] = slope(d.groupby('lam').median_dln_pred.mean().reset_index(), 'lam')
            d = msd_n[(msd_n.scenario == scen) & (msd_n.input_set == name) & (msd_n.family == 'all')]
            if len(d):
                r['npe_lam'] = slope(d, 'lam')
            d = dsc_n[(dsc_n.scenario == scen) & (dsc_n.input_set == name) & (dsc_n.family == 'all')]
            if len(d):
                r['npe_h'] = slope(d, 'h')
            rows.append(r)
    df = pd.DataFrame(rows)
    lines = [r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llcccc@{}}', r'\toprule',
             r' & & \multicolumn{2}{c}{${\rm d}\ln\hat\mu/{\rm d}\ln\lambda$} & \multicolumn{2}{c}{${\rm d}\ln\hat\mu/{\rm d}\ln h$} \\',
             r'Scen. & Input set & HistGB & NPE & HistGB & NPE \\', r'\midrule']
    fmt = lambda v: '--' if pd.isna(v) else f'{v:+.2f}'  # noqa: E731
    for _, r in df.iterrows():
        lines.append(f"{r.scenario} & {tex(r.input_set)} & {fmt(r.get('hgb_lam'))} & {fmt(r.get('npe_lam'))} & "
                     f"{fmt(r.get('hgb_h'))} & {fmt(r.get('npe_h'))} \\\\")
    lines += [r'\bottomrule', r'\end{tabular*}']
    return '\n'.join(lines), df


def calibration_table(mix, npe, scen):
    """Coverage of the 90 per cent intervals per family for one mixed-family model and several calibrations."""
    cal = [('global', 'global', '_global'), ('n_images', 'image multiplicity', '_nimg'), ('pred_mu', r'predicted $\mu$', '_predmu'),
           ('epl_only', 'EPL+shear only', '_epl'), ('true_family', 'true family (oracle)', '')]
    lines = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccccc@{}}', r'\toprule',
             f'Estimator & Calibration & {HEAD} & width \\\\', r'\midrule']
    m = mix[(mix.scenario == scen) & (mix.input_set == 'Geometry+GW(ratio)')]
    g = m.groupby(['calibration', 'test_family'])[['coverage90', 'coverage90_lo68', 'coverage90_hi68', 'width_ln']].mean()
    for key, label, _ in cal:
        if key not in g.index.get_level_values(0):
            continue
        cells = [f"{g.loc[(key, f), 'coverage90']:.2f}" for f in FAMILIES]
        lines.append(f'HistGB & {label} & ' + ' & '.join(cells) + f" & {g.loc[(key, 'all'), 'width_ln']:.2f} \\\\")
    lines.append(r'\midrule')
    n = npe[(npe.scenario == scen) & (npe.input_set == 'Geometry+GW(ratio)')].set_index('family')
    for _, label, suf in cal:
        col = f'coverage90_conformal{suf}'
        if col not in n:
            continue
        cells = [f'{n.loc[f, col]:.2f}' for f in FAMILIES]
        lines.append(f'NPE & {label} & ' + ' & '.join(cells) + f" & {n.loc['all', f'width_ln_conformal{suf}']:.2f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular*}']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hgb-dir', type=Path, required=True)
    p.add_argument('--npe-dir', type=Path, required=True)
    p.add_argument('--benchmark', type=Path, default=None, help='metrics.csv of analysis/benchmark.py (analytic rows)')
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    abl = pd.read_csv(a.hgb_dir / 'ablation.csv')
    msd_h = pd.read_csv(a.hgb_dir / 'msd.csv')
    if 'transform' not in msd_h:
        msd_h['transform'] = 'mass_sheet'
    mix = pd.read_csv(a.hgb_dir / 'mixed_calibration.csv')
    npe = pd.concat([pd.read_csv(f) for f in glob.glob(str(a.npe_dir / 'npe_[EO]*.csv'))])
    msd_n = pd.concat([pd.read_csv(f) for f in glob.glob(str(a.npe_dir / 'npe_msd_*.csv'))])
    dsc_n = pd.read_csv(a.npe_dir / 'distance_scale.csv')
    analytic = pd.read_csv(a.benchmark) if a.benchmark else None
    for scen in ('ET', 'O5'):
        (a.out_dir / f'table_ablation_{scen}.tex').write_text(accuracy_table(abl, npe, analytic, scen))
        (a.out_dir / f'table_calibration_{scen}.tex').write_text(calibration_table(mix, npe, scen))
    t, df = response_table(msd_h, msd_n, dsc_n)
    (a.out_dir / 'table_response.tex').write_text(t)
    df.to_csv(a.out_dir / 'table_response.csv', index=False)
    print(df.round(3).to_string())
    for f in sorted(a.out_dir.glob('table_*.tex')):
        print(f'--- {f.name}\n{f.read_text()}')


if __name__ == '__main__':
    main()
