"""Fig.: the analysed systems against the simulated populations (parent = all lensed systems of the catalogs,
detected = those passing the detection criteria), and the sample-summary table. usage: fig_sample.py OUT_DIR"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1]
FAMS = ['SIS', 'SIE', 'SIE_shear', 'EPL_shear']
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 1.1, 'mathtext.fontset': 'cm'})
d = pd.read_csv(f'{S}/p2/sample.csv')
d['rho_min'] = np.minimum(d.rho_a, d.rho_b)
d['mu_max'] = np.maximum(d.mu_a, d.mu_b)
cat = np.load(f'{S}/p2/catalog_dists.npz')


def pooled(pop, var, fams=FAMS):
    """Catalog values and weights giving each family equal total weight (as in the analysed sample)."""
    v, w = [], []
    for f in fams:
        x = cat[f'{f}|{pop}|{var}']
        v.append(x); w.append(np.full(len(x), 1 / len(x)))
    return np.concatenate(v), np.concatenate(w)


panels = [('z_l', r'lens redshift $z_l$', np.linspace(0, 3, 16), False, FAMS),
          ('z_s', r'source redshift $z_s$', np.linspace(0, 8, 17), False, FAMS),
          ('theta_E', r'Einstein radius $\theta_E$ (arcsec)', np.linspace(0, 2.1, 15), False, FAMS),
          ('gamma', r"radial slope $\gamma'$ (EPL+shear)", np.linspace(1.6, 2.6, 21), False, ['EPL_shear']),
          ('dt_days', r'time delay $|\Delta t|$ (days)', np.logspace(-2.3, 2.6, 15), True, FAMS),
          ('mu_max', r'$|\mu|$ of the brighter image', np.logspace(0, 1.8, 13), True, FAMS),
          ('rho_min', r'SNR of the fainter image', np.logspace(np.log10(5), np.log10(100), 14), True, FAMS),
          ('sep', r'image separation ($^{\prime\prime}$)', np.logspace(-1, np.log10(5), 14), True, None)]
fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.3), constrained_layout=True)
for ax, (c, lab, bins, log, fams) in zip(axes.flat, panels):
    if c in ('z_l', 'z_s', 'theta_E', 'gamma'):
        v, w = pooled('parent', c, fams)
        ax.hist(v, bins=bins, weights=w, density=True, color='0.88', lw=0, label='all lensed systems')
    for sc, ls, col in (('ET', '-', 'k'), ('O5', '--', '#d95f02')):
        if fams is not None:
            v, w = pooled(sc, c, fams)
            ax.hist(v, bins=bins, weights=w, density=True, histtype='step', color=col, ls=ls, lw=0.6, alpha=0.8,
                    label=f'detected, {sc}')
        q = d[d.scenario == sc]
        if c == 'gamma':
            q = q[q.family == 'EPL_shear']
        ax.hist(q[c], bins=bins, density=True, histtype='step', color=col, ls=ls, lw=1.4, label=f'analyzed, {sc}')
    if log:
        ax.set_xscale('log')
    ax.set_xlabel(lab)
    ax.set_yticks([])
axes.flat[4].axvline(365.25, color='0.6', lw=0.6)
axes.flat[6].axvline(8, color='0.6', lw=0.6)
axes.flat[7].axvline(0.1, color='0.6', lw=0.6)
axes.flat[6].set_xticks([5, 10, 20, 50, 100]); axes.flat[6].set_xticklabels(['5', '10', '20', '50', '100']); axes.flat[6].minorticks_off()
axes.flat[7].set_xticks([0.1, 0.3, 1, 3]); axes.flat[7].set_xticklabels(['0.1', '0.3', '1', '3']); axes.flat[7].minorticks_off()
axes.flat[5].set_xticks([1, 3, 10, 30]); axes.flat[5].set_xticklabels(['1', '3', '10', '30']); axes.flat[5].minorticks_off()
axes.flat[4].set_xticks([0.01, 0.1, 1, 10, 100]); axes.flat[4].set_xticklabels(['0.01', '0.1', '1', '10', '100']); axes.flat[4].minorticks_off()
for ax, t in zip(axes.flat, 'abcdefgh'):
    ax.text(0.97, 0.90, f'({t})', transform=ax.transAxes, ha='right')
axes.flat[0].set_ylabel('density'); axes.flat[4].set_ylabel('density')
h_, l_ = axes.flat[1].get_legend_handles_labels()
axes.flat[4].legend(h_, l_, frameon=False, loc='upper right', bbox_to_anchor=(1.02, 0.9), handlelength=1.6, labelspacing=0.25)
fig.savefig(f'{OUT}/fig_sample.pdf'); plt.close(fig)

# summary table of the analysed sample
N = {}
rows = [('z_l', r'Lens redshift $z_l$', '{:.2f}'), ('z_s', r'Source redshift $z_s$', '{:.2f}'),
        ('theta_E', r'Einstein radius $\theta_E$ (arcsec)', '{:.2f}'), ('sep', r'Image separation (arcsec)', '{:.2f}'),
        ('dt_days', r'Time delay $|\Delta t|$ (days)', '{:.1f}'), ('mu_max', r'$|\mu|$ of the brighter image', '{:.1f}'),
        ('rho_a', r'Network SNR, image $A$', '{:.1f}'), ('rho_b', r'Network SNR, image $B$', '{:.1f}')]
t = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcc@{}}', r'\toprule', r' & ET (60 systems) & O5 (59 systems)\\', r'\midrule']
for c, lab, f in rows:
    cells = []
    for sc in ('ET', 'O5'):
        v = d[d.scenario == sc][c].to_numpy()
        cells.append(f'{f.format(np.median(v))} ({f.format(np.quantile(v, .1))}--{f.format(np.quantile(v, .9))})')
        N[f'{c}_{sc}'] = [float(np.median(v)), float(np.quantile(v, .1)), float(np.quantile(v, .9))]
    t.append(f'{lab} & ' + ' & '.join(cells) + r'\\')
for lab, fn in (('Four-image systems', lambda q: int((q.n_images == 4).sum())),
                (r'Systems with $|\mu|\ge5$', lambda q: int((q.mu_max >= 5).sum())),
                (r'Image $A$ at a saddle point', lambda q: int((q.morse_a == 0.5).sum()))):
    cells = [str(fn(d[d.scenario == sc])) for sc in ('ET', 'O5')]
    t.append(f'{lab} & ' + ' & '.join(cells) + r'\\')
    N[lab] = cells
t += [r'\bottomrule', r'\end{tabular*}']
open(f'{OUT}/table_sample.tex', 'w').write('\n'.join(t) + '\n')
# catalog-level numbers for the text
for sc in ('ET', 'O5'):
    for f_ in FAMS:
        n_img = cat[f'{f_}|{sc}|n_images']; mu = cat[f'{f_}|{sc}|mu_max']; dt = cat[f'{f_}|{sc}|dt_days']
        N[f'cat_{sc}_{f_}'] = dict(n=int(len(mu)), frac=float(len(mu) / len(cat[f'{f_}|parent|z_s'])), quads=float(np.mean(n_img == 4)),
                                   mu5=float(np.mean(mu >= 5)), dt_med=float(np.median(dt)), zs_med=float(np.median(cat[f'{f_}|{sc}|z_s'])),
                                   gamma_mean=float(np.mean(cat[f'{f_}|{sc}|gamma'])))
    N[f'cat_parent_gamma_mean'] = float(np.mean(cat['EPL_shear|parent|gamma']))
json.dump(N, open(f'{OUT}/numbers_sample.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in N.items() if k.startswith('cat')}, indent=None))
print('written', OUT)
