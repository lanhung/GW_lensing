"""Fig.: per-system posteriors of h (median, 90% interval, with the selection term) for all systems, model of the
simulated family, and the combined 68% interval. usage: fig_persystem.py OUT_DIR"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2lib import S, FAMS, FLAB, FC, SUM, keys, post, quantiles, joint, fam_of, grid  # noqa: E402

OUT = sys.argv[1]
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 1.1, 'mathtext.fontset': 'cm'})
smp = pd.read_csv(f'{S}/p2/sample.csv').set_index(['scenario', 'event_id'])
fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.4), constrained_layout=True, sharex=False)
for ax, sc in zip(axes, ('ET', 'O5')):
    ks = keys(sc, '_h0_sn', 'own')
    rows = []
    for k in ks:
        b = k.split('|')[1]
        eid = b.split('_', 1)[1]
        mu = max(SUM[k]['true_mu_a'], SUM[k]['true_mu_b'])
        lo, m, hi = quantiles(post(k))
        rows.append((fam_of(b), mu, lo, m, hi, smp.loc[(sc, eid), 'dt_days']))
    rows.sort(key=lambda r: (FAMS.index(r[0]), r[1]))
    x0 = 0
    for fam in FAMS:
        rr = [r for r in rows if r[0] == fam]
        x = np.arange(x0, x0 + len(rr))
        for xi, (f, mu, lo, m, hi, dt) in zip(x, rr):
            miss = not (lo <= 1 <= hi)
            ax.plot([xi, xi], [lo, hi], color=FC[fam], lw=0.9, alpha=0.9)
            ax.plot(xi, m, marker='^' if mu >= 5 else 'o', ms=3.2 if mu >= 5 else 2.8, color=FC[fam],
                    mfc='white' if miss else FC[fam], mew=0.8)
        ax.text(x0 + len(rr) / 2 - 0.5, 0.52, FLAB[fam], ha='center', va='bottom', color=FC[fam], fontsize=7)
        x0 += len(rr) + 1.5
        if fam != FAMS[-1]:
            ax.axvline(x0 - 1.25, color='0.85', lw=0.5)
    m, lo, hi, _ = joint(ks)
    ax.axhspan(lo, hi, color='0.75', alpha=0.5, lw=0)
    ax.axhline(1, color='k', lw=0.6)
    ax.set_xlim(-1, x0 - 1.5); ax.set_ylim(0.5, 1.5)
    ax.set_xticks([])
    ax.set_ylabel(r'$h=H_0/H_0^{\rm true}$')
    ax.text(0.01, 0.95, f'({"ab"[sc == "O5"]}) {sc}: {len(ks)} systems, combined $h={m:.3f}^{{+{hi - m:.3f}}}_{{-{m - lo:.3f}}}$',
            transform=ax.transAxes, va='top', bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
ax = axes[0]
ax.plot([], [], 'o', color='0.4', ms=2.8, label=r'$|\mu|<5$')
ax.plot([], [], '^', color='0.4', ms=3.2, label=r'$|\mu|\geq5$')
ax.plot([], [], 'o', color='0.4', mfc='white', ms=2.8, label='interval misses $h=1$')
ax.legend(frameon=False, loc='lower right', ncol=3, handletextpad=0.3, columnspacing=1.0, bbox_to_anchor=(1.0, 1.0), borderaxespad=0.1)
axes[1].set_xlabel('systems, ordered by lens family and magnification of the brighter image')
fig.savefig(f'{OUT}/fig_persystem.pdf'); plt.close(fig)
print('written', OUT)
