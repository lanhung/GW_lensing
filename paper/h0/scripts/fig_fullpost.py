"""Full posteriors of one system: (1) corner plot of the joint parameter estimation of both GW images, (2) corner plot of
the GW+EM posterior of the lens model, h and kappa_ext, (3) a table of the true values, medians and 90% intervals of every
parameter. Default: the worked example of Paper II (ET EPL_shear-7777-0006095); the Supplemental Material uses
O5 EPL_shear-7777-0025278 (free sky). usage: fig_fullpost.py OUT_DIR [SCENARIO EVENT_ID FILE_SUFFIX]"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import corner

S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1]
SC, EID, SUF = sys.argv[2:5] if len(sys.argv) > 2 else ('ET', 'EPL_shear-7777-0006095', '')
E = f'{S}/p2/example'
plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'axes.labelsize': 7.5, 'xtick.labelsize': 5.5,
                     'ytick.labelsize': 5.5, 'axes.linewidth': 0.5, 'mathtext.fontset': 'cm', 'lines.linewidth': 0.8})
ev = json.load(open(f'{S}/p2/inputs/inputs/gwem/{SC}_{EID}.json'))
t, row = ev['truth'], ev['row']
ks = '_knownsky' if SC == 'ET' else ''
pe = pd.read_parquet(f'{E}/{SC}_{EID}{ks}_posterior.parquet')
st = json.load(open(f'{S}/p2/pe/results/pe/{SC}/{SC}_{EID}{ks}_summary.json'))  # true values of the GW injection
gx = pd.read_parquet(f'{E}/{SC}_{EID}_posthoc_lcdm_EPL_shear_h0_sn_posterior.parquet')

# ---------------------------------------------------------------- true values (as injected by gwlens/joint_pe.py)
mu_a = (t['d_l'] / st['true_luminosity_distance']) ** 2
mu_b = mu_a * np.exp(st['true_ln_mu_rel'])
e1, e2 = t['e1'], t['e2']
phi_q_true = (0.5 * np.arctan2(e2, e1)) % np.pi
g_ext_true = np.hypot(t['gamma1'], t['gamma2'])
phi_ext_true = (0.5 * np.arctan2(t['gamma2'], t['gamma1'])) % np.pi

# ---------------------------------------------------------------- (1) GW joint parameter estimation
GW = [  # (label for the figure, label for the table, samples, truth, format)
    (r'$\mathcal{M}$', r'Chirp mass $\mathcal M$ ($M_\odot$)', pe.chirp_mass, st['true_chirp_mass'], '{:.3f}'),
    (r'$m_2/m_1$', r'Mass ratio $m_2/m_1$', pe.mass_ratio, st['true_mass_ratio'], '{:.2f}'),
    (r'$\chi_1$', r'Spin $\chi_1$', pe.chi_1, st['true_chi_1'], '{:.2f}'),
    (r'$\chi_2$', r'Spin $\chi_2$', pe.chi_2, st['true_chi_2'], '{:.2f}'),
    (r'$\theta_{JN}$', r'Inclination $\theta_{JN}$', pe.theta_jn, st['true_theta_jn'], '{:.2f}'),
    (r'$\psi$', r'Polarization $\psi$', pe.psi, st['true_psi'], '{:.2f}'),
    (r'$\phi_c$', r'Phase $\phi_c$', pe.phase, st['true_phase'], '{:.2f}'),
]
if not st['known_sky']:
    GW += [(r'$\alpha$', r'Right ascension $\alpha$', pe.ra, st['true_ra'], '{:.3f}'),
           (r'$\delta$', r'Declination $\delta$', pe.dec, st['true_dec'], '{:.3f}')]
GW += [
    (r'$D_A$', r'$D_A=d_L/\sqrt{|\mu_A|}$ (Gpc)', pe.luminosity_distance / 1e3, st['true_luminosity_distance'] / 1e3, '{:.2f}'),
    (r'$\ln|\mu_B/\mu_A|$', r'$\ln|\mu_B/\mu_A|$', pe.ln_mu_rel, st['true_ln_mu_rel'], '{:.2f}'),
    (r'$\delta t_A$ (ms)', r'$\delta t_A=t_A-t_A^{\rm true}$ (ms)', (pe.geocent_time - st['true_geocent_time']) * 1e3, 0.0, '{:.3f}'),
    (r'$\delta\Delta t$ (ms)', r'$\delta\Delta t=\Delta t-\Delta t^{\rm true}$ (ms)', (pe.delta_t - st['true_delta_t']) * 1e3, 0.0,
     '{:.3f}'),
]
LENS = [
    (r'$\theta_E$', r'$\theta_E$ (arcsec)', gx.theta_e, t['theta_E'], '{:.3f}'),
    (r"$\gamma'$", r"Radial slope $\gamma'$", gx.gamma, t['gamma'], '{:.2f}'),
    (r'$q$', r'Axis ratio $q$', gx.q, t['q'], '{:.2f}'),
    (r'$\varphi$', r'Orientation $\varphi$', gx.phi_q, phi_q_true, '{:.2f}'),
    (r'$\gamma_{\rm ext}$', r'Shear $\gamma_{\rm ext}$', np.exp(gx.ln_gamma_ext), g_ext_true, '{:.3f}'),
    (r'$\varphi_{\rm ext}$', r'Shear orientation $\varphi_{\rm ext}$', gx.phi_ext, phi_ext_true, '{:.2f}'),
    (r'$x_c$', r'Center $x_c$ (arcsec)', gx.center_x, t['center_x'], '{:.3f}'),
    (r'$y_c$', r'Center $y_c$ (arcsec)', gx.center_y, t['center_y'], '{:.3f}'),
    (r'$\sigma_v$', r'$\sigma_v$ (km\,s$^{-1}$)', gx.sigma_v, row['true_sigma_v'], '{:.0f}'),
    (r'$\kappa_{\rm ext}$', r'$\kappa_{\rm ext}$', gx.kappa_ext_derived, t['kappa_ext'], '{:.3f}'),
    (r'$\mu_A$', r'$\mu_A$', gx.mu_a, mu_a, '{:.2f}'),
    (r'$|\mu_B|$', r'$|\mu_B|$', gx.mu_b, mu_b, '{:.2f}'),
    (r'$h$', r'$h=H_0/H_0^{\rm true}$', gx.h_rel, 1.0, '{:.2f}'),
]


def corner_plot(spec, fname, size):
    X = np.column_stack([np.asarray(s, float) for _, _, s, _, _ in spec])
    tr = [v for _, _, _, v, _ in spec]
    lo = np.minimum(np.quantile(X, 0.002, axis=0), tr)
    hi = np.maximum(np.quantile(X, 0.998, axis=0), tr)
    pad = 0.06 * (hi - lo)
    fig = corner.corner(X, labels=[l for l, _, _, _, _ in spec],
                        range=list(zip(lo - pad, hi + pad)), bins=25, smooth=1.0, smooth1d=1.0,
                        quantiles=[0.05, 0.5, 0.95], levels=(0.5, 0.9), plot_datapoints=False, fill_contours=True,
                        color='#2166ac', max_n_ticks=3, labelpad=0.2,
                        hist_kwargs=dict(lw=0.8), contour_kwargs=dict(linewidths=0.6))
    corner.overplot_lines(fig, tr, color='#e66101', lw=0.6)
    corner.overplot_points(fig, [tr], marker='s', ms=2.2, color='#e66101')
    fig.set_size_inches(size, size)
    for ax in fig.get_axes():
        ax.tick_params(width=0.4, length=2, pad=1)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_rotation(45)
    fig.subplots_adjust(left=0.075, bottom=0.075, right=0.99, top=0.99, wspace=0.1, hspace=0.1)
    fig.savefig(f'{OUT}/{fname}')
    plt.close(fig)


corner_plot(GW, f'fig_corner_gw{SUF}.pdf', 7.0)
corner_plot(LENS, f'fig_corner_lens{SUF}.pdf', 7.0)

# ---------------------------------------------------------------- (3) table of every parameter
N = {}


def rows(spec):
    out = []
    for _, lab, s, tv, f in spec:
        s = np.asarray(s, float)
        q05, q50, q95 = np.quantile(s, [0.05, 0.5, 0.95])
        m = lambda x: '$' + f.format(x) + '$'
        out.append(f'{lab} & {m(tv)} & {m(q50)} & $[{f.format(q05)},\\ {f.format(q95)}]$\\\\')
        N[lab] = dict(truth=float(tv), median=float(q50), q05=float(q05), q95=float(q95),
                      z=float((q50 - tv) / np.std(s)))
    return out


pm = pd.crosstab(pe.morse_n_a, pe.morse_n_b, normalize=True)
morse = {f'({a:g},{b:g})': float(pm.loc[a, b]) for a in pm.index for b in pm.columns if pm.loc[a, b] > 0}
N['morse_probs'] = morse
N['n_samples'] = dict(gw=len(pe), gwem=len(gx))
fmt_m = lambda k: k.replace('0.5', r'\tfrac12').replace('(', '$(').replace(')', ')$')
mtxt = '; '.join(f'{fmt_m(k)}: {v:.2f}' for k, v in sorted(morse.items()))
sky = 'sky position at the host' if st['known_sky'] else 'free sky position'
T = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccc@{}}', r'\toprule',
     r'Parameter & Truth & Median & 90\% interval\\', r'\midrule',
     r'\multicolumn{4}{@{}l}{\emph{GW: joint parameter estimation, ' + sky + r'}}\\']
T += rows(GW)
mtrue = fmt_m(f"({st['true_morse_a'] - 0.25:g},{st['true_morse_b'] - 0.25:g})")
T += [r'Morse indices $(n_A,n_B)$ & ' + mtrue + r' & \multicolumn{2}{c}{' + mtxt + r'}\\',
      r'\midrule', r'\multicolumn{4}{@{}l}{\emph{GW+EM: EPL+shear lens model, $\kappa_{\rm ext}$ and $h$}}\\']
T += rows(LENS)
T += [r'\bottomrule', r'\end{tabular*}']
open(f'{OUT}/table_fullpost{SUF}.tex', 'w').write('\n'.join(T) + '\n')
json.dump(N, open(f'{OUT}/numbers_fullpost{SUF}.json', 'w'), indent=1)
for k, v in N.items():
    print(k, v)
