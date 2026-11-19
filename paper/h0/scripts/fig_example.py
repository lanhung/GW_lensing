"""Fig.: one system worked through (ET, EPL+shear family, event EPL_shear-7777-0006095).
(a) image plane; (b) joint-PE posterior of the GW observables and its Gaussian summary; (c) posterior of h from the
delay only, with the GW apparent distance, and divided by the selection fraction; (d) slope-H0 degeneracy.
usage: fig_example.py OUT_DIR"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2lib import S, H, post, logpost, grid, lalpha  # noqa: E402

OUT = sys.argv[1]
EID = 'EPL_shear-7777-0006095'
E = f'{S}/p2/example'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 1.1, 'mathtext.fontset': 'cm'})
ev = json.load(open(f'{S}/p2/inputs/inputs/gwem/ET_{EID}.json'))
t, em = ev['truth'], ev['em']
ia, ib = em['gw_a'], em['gw_b']
pe = pd.read_parquet(f'{E}/ET_{EID}_knownsky_posterior.parquet')
base = pd.read_parquet(f'{E}/ET_{EID}_posthoc_lcdm_EPL_shear_h0_sn_posterior.parquet')
nod = pd.read_parquet(f'{E}/ET_{EID}_posthoc_lcdm_EPL_shear_h0_nod_sn_posterior.parquet')
key = f'lcdm_k0.025_EPL_shear_h0_sn_ks|ET_{EID}'
N = {}

fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.95), constrained_layout=True)

# (a) image plane: critical curve, caustic, images, source
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
lm = LensModel(['EPL', 'SHEAR'])
kw = [dict(theta_E=t['theta_E'], gamma=t['gamma'], e1=t['e1'], e2=t['e2'], center_x=t['center_x'], center_y=t['center_y']),
      dict(gamma1=t['gamma1'], gamma2=t['gamma2'], ra_0=t['center_x'], dec_0=t['center_y'])]
rc, dc, rca, dca = LensModelExtensions(lm).critical_curve_caustics(kw, compute_window=5 * t['theta_E'], grid_scale=0.004,
                                                                    center_x=t['center_x'], center_y=t['center_y'])
a = ax[0]
for x, y in zip(rc, dc):
    a.plot(x, y, 'k-', lw=0.8)
for x, y in zip(rca, dca):
    a.plot(x, y, '--', color='0.45', lw=0.8)
a.plot(t['center_x'], t['center_y'], '+', color='k', ms=6, mew=0.8)
a.plot(t['beta_x'], t['beta_y'], 'o', color='#377eb8', ms=3.5, mec='k', mew=0.4)
for i, lab in ((ia, 'A'), (ib, 'B')):
    a.plot(em['x'][i], em['y'][i], '*', color='#e41a1c', ms=8, mec='k', mew=0.4)
    a.annotate(f"{lab}: $\\mu={t['mu'][i]:+.2f}$", (em['x'][i], em['y'][i]), xytext=(4, 6) if lab == 'A' else (6, -11),
               textcoords='offset points', fontsize=6.5, ha='left')
r = 2.1 * t['theta_E']
a.set_xlim(-r, r); a.set_ylim(-r, r); a.set_aspect('equal')
a.set_xlabel('arcsec'); a.set_ylabel('arcsec')
a.text(0.03, 0.93, '(a)', transform=a.transAxes)
N['system'] = dict(z_l=em['z_l'], z_s=em['z_s'], theta_E=t['theta_E'], gamma=t['gamma'], dt_days=abs(t['t'][ib] - t['t'][ia]) / 86400,
                   mu=[t['mu'][ia], t['mu'][ib]], morse=[t['morse'][ia], t['morse'][ib]], rho=[ev['row']['ET__true_rho_a'], ev['row']['ET__true_rho_b']],
                   sigma_v=em['sigma_v'], q=t['q'])

# (b) GW observables: joint-PE posterior divided by the PE prior (weights D_A^-3) and its Gaussian summary
x = np.column_stack([pe.ln_mu_rel.to_numpy(), np.log(pe.luminosity_distance.to_numpy())])
w = np.exp(-3 * (x[:, 1] - x[:, 1].mean())); w /= w.sum()
mean = w @ x; d = x - mean; cov = (w[:, None] * d).T @ d / (1 - np.sum(w ** 2))
rng = np.random.default_rng(1)
idx = rng.choice(len(x), 1500, p=w)
a = ax[1]
a.scatter(x[idx, 0], np.exp(x[idx, 1]) / 1e3, s=2, color='0.6', lw=0, alpha=0.6, rasterized=True)
vals, vecs = np.linalg.eigh(cov)
ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
for nsig, ls in ((1, '-'), (2, '--')):
    # ellipse in (ln mu_rel, ln D); drawn in (ln mu_rel, D/Gpc) by sampling the boundary
    th = np.linspace(0, 2 * np.pi, 200)
    pts = mean[:, None] + nsig * (vecs @ (np.sqrt(vals)[:, None] * np.array([np.cos(th), np.sin(th)])))
    a.plot(pts[0], np.exp(pts[1]) / 1e3, 'k', ls=ls, lw=0.8)
tr = (np.log(abs(t['mu'][ib]) / abs(t['mu'][ia])), t['d_l'] / np.sqrt(abs(t['mu'][ia])) / 1e3)
a.plot(tr[0], tr[1], 'x', color='#e41a1c', ms=6, mew=1.2)
a.set_yscale('log')
a.set_xlabel(r'$\ln(\mu_B/\mu_A)$'); a.set_ylabel(r'$D_A=d_L/\sqrt{|\mu_A|}$ (Gpc)')
a.set_xlim(mean[0] - 4 * np.sqrt(cov[0, 0]), mean[0] + 4 * np.sqrt(cov[0, 0]))
a.set_ylim(np.exp(mean[1] - 4 * np.sqrt(cov[1, 1])) / 1e3, np.exp(mean[1] + 4 * np.sqrt(cov[1, 1])) / 1e3)
yt = [7, 8, 9, 10, 12]
a.set_yticks(yt); a.set_yticklabels([str(v) for v in yt]); a.minorticks_off()
a.text(0.03, 0.93, '(b)', transform=a.transAxes)
fr = {}
n = len(pe)
for na in (0.0, 0.5, 1.0):
    for nb in (0.0, 0.5, 1.0):
        k = np.sum(w[(pe.morse_n_a.to_numpy() == na) & (pe.morse_n_b.to_numpy() == nb)])
        fr[(na, nb)] = (k * n + 0.5) / (n + 4.5)
top = sorted(fr.items(), key=lambda kv: -kv[1])[:2]
a.text(0.97, 0.04, '\n'.join(f"$P(n_A,n_B\\!=\\!{na:g},{nb:g})={p:.2f}$" for (na, nb), p in top), transform=a.transAxes,
       ha='right', va='bottom', fontsize=6)
N['gw'] = dict(mean=mean.tolist(), sd=np.sqrt(np.diag(cov)).tolist(), corr=float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
               truth=[float(tr[0]), float(np.log(tr[1] * 1e3))], morse=[(k, float(v)) for k, v in top])

# (c) posterior of h
a = ax[2]
lp_nod = logpost(nod.h_rel.to_numpy())
p_nod = np.exp(lp_nod - lp_nod.max()); p_nod /= np.trapezoid(p_nod, grid)
p_base = post(key, sel=False)
p_sel = post(key, sel=True)
a.plot(grid, p_nod, ':', color='0.4', label='delay + lens model')
a.plot(grid, p_base, '--', color='#377eb8', label='+ GW apparent distance')
a.plot(grid, p_sel, '-', color='k', label=r'$\div\,\alpha(h)$')
a.axvline(1, color='#e41a1c', lw=0.7)
a.set_xlim(0.7, 1.5); a.set_yticks([]); a.set_ylim(0, a.get_ylim()[1] * 1.7)
a.set_xlabel(r'$h=H_0/H_0^{\rm true}$'); a.set_ylabel('posterior')
a.legend(frameon=False, loc='upper left', bbox_to_anchor=(-0.02, 0.93), handlelength=1.8, labelspacing=0.3)
a.text(0.03, 0.93, '(c)', transform=a.transAxes)


def q(p):
    c = np.cumsum(p); c /= c[-1]
    return [float(grid[np.searchsorted(c, v)]) for v in (0.05, 0.5, 0.95)]


N['h'] = dict(nod=q(p_nod), base=q(p_base), sel=q(p_sel), sd_nod=float(np.std(nod.h_rel)), sd_base=float(np.std(H[key])))

# (d) slope-H0 degeneracy in the baseline posterior
a = ax[3]
g, h = base.gamma.to_numpy(), base.h_rel.to_numpy()
idx = rng.choice(len(g), min(1500, len(g)), replace=False)
a.scatter(g[idx], h[idx], s=2, color='0.6', lw=0, alpha=0.6, rasterized=True)
g0, h0 = np.median(g), np.median(h)
gg = np.linspace(1.7, 2.4, 50)
a.plot(gg, h0 * np.exp((gg - g0) / (g0 - 1)), 'k-', lw=0.8, label='Eq. (3)')
a.plot(t['gamma'], 1.0, 'x', color='#e41a1c', ms=6, mew=1.2, label='truth')
a.set_xlim(1.7, 2.4); a.set_ylim(0.7, 1.4)
a.set_xlabel(r"radial slope $\gamma'$"); a.set_ylabel(r'$h$')
a.legend(frameon=False, loc='upper left', bbox_to_anchor=(-0.02, 0.93), handlelength=1.5, labelspacing=0.3)
a.text(0.03, 0.93, '(d)', transform=a.transAxes)
N['slope'] = dict(corr=float(np.corrcoef(g, np.log(h))[0, 1]), fit=float(np.polyfit(g, np.log(h), 1)[0]),
                  gamma_med=float(g0), gamma_sd=float(np.std(g)), h_med=float(h0), h_sd=float(np.std(h)))
fig.savefig(f'{OUT}/fig_example.pdf', dpi=300); plt.close(fig)
json.dump(N, open(f'{OUT}/numbers_example.json', 'w'), indent=1)
print(json.dumps(N, indent=None))
