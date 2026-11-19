"""Fig. (appendix): calibration of the per-system posteriors of h: PIT histograms and 90% coverage versus the
magnification of the brighter image, model of the simulated family, with the selection term. usage: fig_pit.py OUT_DIR"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2lib import SUM, keys, post, quantiles, grid  # noqa: E402

OUT = sys.argv[1]
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 1.1, 'mathtext.fontset': 'cm'})
fig, ax = plt.subplots(1, 2, figsize=(3.4, 2.0), constrained_layout=True)
N = {}
edges = np.array([1, 2, 3.5, 6, 12, 50])
for sc, col, ls in (('ET', 'k', '-'), ('O5', '#d95f02', '--')):
    ks = keys(sc, '_h0_sn', 'own')
    pit, cov, mu = [], [], []
    for k in ks:
        p = post(k)
        c = np.cumsum(p); c /= c[-1]
        pit.append(float(np.interp(1.0, grid, c)))
        lo, m, hi = quantiles(p)
        cov.append(lo <= 1 <= hi)
        mu.append(max(SUM[k]['true_mu_a'], SUM[k]['true_mu_b']))
    pit, cov, mu = map(np.array, (pit, cov, mu))
    ax[0].hist(pit, bins=np.linspace(0, 1, 6), histtype='step', color=col, ls=ls, lw=1.0, label=sc)
    # Kolmogorov-Smirnov distance of the PIT values from uniform
    s = np.sort(pit); n = len(s)
    ks_d = float(max(np.max(np.arange(1, n + 1) / n - s), np.max(s - np.arange(n) / n)))
    N[f'pit_{sc}'] = dict(n=int(n), mean=float(pit.mean()), ks=ks_d, frac_lt_half=float(np.mean(pit < 0.5)))
    xc, yc, ye, nn = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s_ = (mu >= a) & (mu < b)
        if s_.sum() < 3:
            continue
        f = cov[s_].mean()
        xc.append(np.sqrt(a * b)); yc.append(f); ye.append(np.sqrt(f * (1 - f) / s_.sum()) if 0 < f < 1 else 1 / s_.sum()); nn.append(int(s_.sum()))
    ax[1].errorbar(xc, yc, yerr=ye, fmt='o' if sc == 'ET' else 's', ms=3, color=col, ls=ls, lw=0.8, elinewidth=0.7, capsize=0, label=sc)
    N[f'cov_bins_{sc}'] = dict(mu=[float(x) for x in xc], cov=[float(y) for y in yc], n=nn)
ax[0].axhline(len(keys('ET', '_h0_sn', 'own')) / 5, color='k', lw=0.5, ls=':')
ax[0].axhline(len(keys('O5', '_h0_sn', 'own')) / 5, color='#d95f02', lw=0.5, ls=':')
ax[0].set_xlabel('posterior probability below $h=1$'); ax[0].set_ylabel('systems'); ax[0].set_xlim(0, 1)
ax[0].legend(frameon=False, loc='lower center', ncol=2, columnspacing=1.0)
ax[1].axhline(0.9, color='0.5', lw=0.6)
ax[1].set_xscale('log'); ax[1].set_xlabel(r'$|\mu|$, brighter image'); ax[1].set_ylabel('coverage, 90% intervals')
ax[1].set_ylim(0.4, 1.05); ax[1].set_xticks([1, 2, 5, 10, 20]); ax[1].set_xticklabels(['1', '2', '5', '10', '20']); ax[1].minorticks_off()
ax[0].text(0.04, 0.93, '(a)', transform=ax[0].transAxes)
ax[1].text(0.97, 0.06, '(b)', transform=ax[1].transAxes, ha='right')
fig.savefig(f'{OUT}/fig_pit.pdf'); plt.close(fig)
json.dump(N, open(f'{OUT}/numbers_pit.json', 'w'), indent=1)
print(json.dumps(N, indent=None))
