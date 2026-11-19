"""Supplemental Material: true values and posteriors of every analysed system (60 ET, 59 O5).
GW: joint parameter estimation of both images (ET with the sky position of the host, O5 free sky).
GW+EM: two-stage analysis with the model of the simulated family, source-position prior normalised, h free;
h is given with the selection term (as in Fig. 5). Image labels A, B are those of the GW injection.
Writes persystem.csv (all parameters), table_persystem.tex (longtable), table_coverage.tex, numbers_persystem.json.
usage: table_persystem.py OUT_DIR"""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2lib import S, FAMS, FLAB, SUM, KS, post, quantiles  # noqa: E402

OUT = sys.argv[1]
PE = f'{S}/p2/pe/results/pe'
smp = pd.read_csv(f'{S}/p2/sample.csv')
smp['fam_i'] = smp.family.map(FAMS.index)
smp = smp.sort_values(['scenario', 'fam_i', 'event_id'])


def sky_sep(ra1, dec1, ra2, dec2):
    c = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


rows = []
for _, r in smp.iterrows():
    sc, eid, fam = r.scenario, r.event_id, r.family
    base = f'{PE}/{sc}/{sc}_{eid}' + ('_knownsky' if sc == 'ET' else '')
    p = pd.read_parquet(base + '_posterior.parquet')
    s = json.load(open(base + '_summary.json'))
    d = dict(scenario=sc, family=fam, event_id=eid, z_l=r.z_l, z_s=r.z_s, theta_E=r.theta_E, gamma_true=r.gamma, q_true=r.q,
             n_images=r.n_images, sep_arcsec=r.sep, dt_days=abs(s['true_delta_t']) / 86400, d_L_Mpc=r.d_l,
             m1_src=r.m1, m2_src=r.m2, snr_a=s['snr_a'], snr_b=s['snr_b'],
             morse_a_true=s['true_morse_a'] - 0.25, morse_b_true=s['true_morse_b'] - 0.25)
    # true magnifications in the labels of the injection: D_A = d_L / sqrt(mu_A), ln(mu_B/mu_A)
    d['mu_a_true'] = (r.d_l / s['true_luminosity_distance']) ** 2
    d['mu_b_true'] = d['mu_a_true'] * np.exp(s['true_ln_mu_rel'])
    gw = dict(chirp_mass=(p.chirp_mass, s['true_chirp_mass']), mass_ratio=(p.mass_ratio, s['true_mass_ratio']),
              chi_1=(p.chi_1, s['true_chi_1']), chi_2=(p.chi_2, s['true_chi_2']), theta_jn=(p.theta_jn, s['true_theta_jn']),
              psi=(p.psi, s['true_psi']), phase=(p.get('phase'), s['true_phase']),
              D_A_Mpc=(p.luminosity_distance, s['true_luminosity_distance']), ln_mu_rel=(p.ln_mu_rel, s['true_ln_mu_rel']),
              dt_offset_ms=((p.delta_t - s['true_delta_t']) * 1e3, 0.0),
              tA_offset_ms=((p.geocent_time - s['true_geocent_time']) * 1e3 if 'geocent_time' in p else None, 0.0))
    for k, (x, t) in gw.items():
        if x is None:  # eight early O5 runs stored neither the phase nor the arrival time
            continue
        q05, q50, q95 = np.quantile(np.asarray(x, float), [0.05, 0.5, 0.95])
        d.update({f'{k}_true': t, f'{k}_q05': q05, f'{k}_q50': q50, f'{k}_q95': q95})
    if sc == 'O5':
        sep = sky_sep(p.ra, p.dec, s['true_ra'], s['true_dec'])
        d['sky_offset_deg_q50'], d['sky_offset_deg_q90'] = np.quantile(sep, [0.5, 0.9])
    pm = pd.crosstab(p.morse_n_a, p.morse_n_b, normalize=True)
    d['P_morse_true'] = float(pm.loc[d['morse_a_true'], d['morse_b_true']]) \
        if d['morse_a_true'] in pm.index and d['morse_b_true'] in pm.columns else 0.0
    dn = (p.morse_n_b - p.morse_n_a).to_numpy()
    d['P_dmorse_true'] = float(np.mean(np.isclose(dn, d['morse_b_true'] - d['morse_a_true'])))
    # GW+EM, model of the simulated family
    key = f'lcdm_k0.025_{fam}_h0_sn{KS[sc]}|{sc}_{eid}'
    g = SUM[key]
    if abs(g["true_mu_a"] / d["mu_a_true"] - 1) > 1e-6:
        print("LABEL MISMATCH", key, g["true_mu_a"], d["mu_a_true"])
    d.update(mu_a_gwem_q05=np.exp(g['ln_mu_a_q05']), mu_a_gwem_q50=np.exp(g['ln_mu_a_median']), mu_a_gwem_q95=np.exp(g['ln_mu_a_q95']),
             h_nosel_q05=g['h_rel_q05'], h_nosel_q50=g['h_rel_median'], h_nosel_q95=g['h_rel_q95'])
    d['h_q05'], d['h_q50'], d['h_q95'] = quantiles(post(key))
    if fam == 'EPL_shear':
        d['gamma_gwem_median'], d['gamma_gwem_sd'] = g['gamma_median'], g['gamma_sd']
    rows.append(d)
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/persystem.csv', index=False, float_format='%.6g')

# ---------------------------------------------------------------- coverage of the 90% intervals
N = {}
COV = [('chirp_mass', r'Chirp mass $\mathcal M$'), ('mass_ratio', r'Mass ratio $m_2/m_1$'), ('chi_1', r'Spin $\chi_1$'),
       ('chi_2', r'Spin $\chi_2$'), ('theta_jn', r'Inclination $\theta_{JN}$'), ('D_A_Mpc', r'Apparent distance $D_A$'),
       ('ln_mu_rel', r'$\ln|\mu_B/\mu_A|$'), ('tA_offset_ms', r'Arrival time $t_A$'), ('dt_offset_ms', r'Delay $\Delta t$'),
       ('mu_a_gwem', r'$\mu_A$ (GW+EM)'), ('h', r'$h$ (GW+EM)')]
for sc in ('ET', 'O5'):
    q = df[df.scenario == sc]
    N[sc] = dict(n=len(q))
    for k, _ in COV:
        t = q['mu_a_true'] if k == 'mu_a_gwem' else (1.0 if k == 'h' else q[f'{k}_true'])
        ok = q[f'{k}_q05'].notna()
        N[sc][k] = float(np.mean(((q[f'{k}_q05'] <= t) & (t <= q[f'{k}_q95']))[ok]))
        N[sc][f'{k}_n'] = int(ok.sum())
    N[sc]['P_morse_true_median'] = float(q.P_morse_true.median())
    N[sc]['P_dmorse_true_gt_0.5'] = float(np.mean(q.P_dmorse_true > 0.5))
    rel = lambda k: (q[f'{k}_q95'] - q[f'{k}_q05']) / 2 / q[f'{k}_q50']  # noqa: E731
    N[sc]['median_halfwidth90_rel'] = dict(chirp_mass=float(rel('chirp_mass').median()), D_A=float(rel('D_A_Mpc').median()),
                                          mu_a_gwem=float(rel('mu_a_gwem').median()), h=float(rel('h').median()))
    N[sc]['median_halfwidth90_dt_ms'] = float(((q.dt_offset_ms_q95 - q.dt_offset_ms_q05) / 2).median())
    if sc == 'O5':
        N[sc]['sky_offset_deg_q90_median'] = float(q.sky_offset_deg_q90.median())
    e = q[q.family == 'EPL_shear']
    z = (e.gamma_gwem_median - e.gamma_true) / e.gamma_gwem_sd
    N[sc]['gamma_cov90_gauss'] = float(np.mean(abs(z) <= 1.645))
binom = lambda n: 1.645 * np.sqrt(0.09 / n)  # noqa: E731
T = [r'\begin{tabular}{@{}lcc@{}}', r'\toprule', r'Parameter & ET (60) & O5 (59)\\', r'\midrule']
for k, lab in COV:
    T.append(f"{lab} & {N['ET'][k]:.2f} & {N['O5'][k]:.2f}\\\\")
T += [r"Slope $\gamma'$ (GW+EM, EPL+shear)$^a$ & " + f"{N['ET']['gamma_cov90_gauss']:.2f} & {N['O5']['gamma_cov90_gauss']:.2f}\\\\",
      r'\bottomrule', r'\end{tabular}']
open(f'{OUT}/table_coverage.tex', 'w').write('\n'.join(T) + '\n')
N['binomial_90pc_halfwidth'] = dict(ET=binom(60), O5=binom(59))


# ---------------------------------------------------------------- per-system longtable
def pm(m, lo, hi, f):
    return f'${f.format(m)}^{{+{f.format(hi - m)}}}_{{-{f.format(m - lo)}}}$'


def num(x, f):
    return '$' + f.format(x) + '$'


def ff(x):  # 2-3 significant digits for magnifications
    return '{:.1f}' if x >= 10 else '{:.2f}'


hdr = (r"System & $z_l$ & $z_s$ & $\theta_E$ & $\gamma'$ & $\Delta t$ & $\mu_A$ & $\mu_B$ & $(n_A,n_B)$ & $\rho_A$ & $\rho_B$ & "
       r"$\ln|\mu_B/\mu_A|$ & $D_A^{\rm true}$ & $D_A$ & $P(n)$ & $\gamma'$ & $\mu_A$ & $h$\\")
units = r" & & & ($''$) & & (d) & & & & & & & (Gpc) & (Gpc) & & & & \\"
L = [r'\begin{longtable}{@{}l' + 'c' * 17 + r'@{}}',
     r'\caption{True values and posteriors of every analyzed system: true lens and source parameters, the joint parameter '
     r'estimation of both GW images (posterior median and 90\% interval of the magnification ratio and of the apparent distance '
     r'of image $A$, and the probability of the true Morse indices), and the GW+EM analysis with the model of the simulated '
     r'family (the slope as median $\pm$ standard deviation for EPL+shear lenses, the magnification of image $A$, and $h$ with '
     r'the selection term). $\Delta t$ is in days, $D_A$ in Gpc and $\rho$ is the optimal network SNR.}\label{tab:persystem}\\',
     r'\toprule', r'& \multicolumn{10}{c}{True values} & \multicolumn{4}{c}{GW: joint parameter estimation} & '
     r'\multicolumn{3}{c}{GW+EM}\\', r'\cmidrule(lr){2-11}\cmidrule(lr){12-15}\cmidrule(l){16-18}', hdr, units, r'\midrule',
     r'\endfirsthead', r'\toprule', hdr, units, r'\midrule', r'\endhead', r'\bottomrule', r'\endfoot']
mfmt = {0.0: '0', 0.5: r'\frac12', 1.0: '1'}
for sc in ('ET', 'O5'):
    L.append(r'\multicolumn{18}{@{}l}{\emph{' + {'ET': 'Einstein Telescope', 'O5': 'O5-like network'}[sc] + r'}}\\')
    for _, d in df[df.scenario == sc].iterrows():
        g = '--' if np.isnan(d.get('gamma_gwem_median', np.nan)) else \
            f"${d.gamma_gwem_median:.2f}\\pm{d.gamma_gwem_sd:.2f}$"
        cells = [f"{FLAB[d.family]} {d.event_id.rsplit('-', 1)[1]}", num(d.z_l, '{:.2f}'), num(d.z_s, '{:.2f}'),
                 num(d.theta_E, '{:.2f}'), num(d.gamma_true, '{:.2f}'),
                 num(d.dt_days, '{:.1f}' if d.dt_days >= 1 else ('{:.2f}' if d.dt_days >= 0.01 else '{:.3f}')),
                 num(d.mu_a_true, ff(d.mu_a_true)), num(d.mu_b_true, ff(d.mu_b_true)),
                 f"$({mfmt[d.morse_a_true]},{mfmt[d.morse_b_true]})$", num(d.snr_a, '{:.0f}'), num(d.snr_b, '{:.0f}'),
                 pm(d.ln_mu_rel_q50, d.ln_mu_rel_q05, d.ln_mu_rel_q95, '{:.2f}'), num(d.D_A_Mpc_true / 1e3, '{:.2f}'),
                 pm(d.D_A_Mpc_q50 / 1e3, d.D_A_Mpc_q05 / 1e3, d.D_A_Mpc_q95 / 1e3, '{:.2f}'), num(d.P_morse_true, '{:.2f}'),
                 g, pm(d.mu_a_gwem_q50, d.mu_a_gwem_q05, d.mu_a_gwem_q95, ff(d.mu_a_gwem_q50)),
                 pm(d.h_q50, d.h_q05, d.h_q95, '{:.2f}')]
        L.append(' & '.join(cells) + r'\\')
L.append(r'\end{longtable}')
open(f'{OUT}/table_persystem.tex', 'w').write('\n'.join(L) + '\n')
json.dump(N, open(f'{OUT}/numbers_persystem.json', 'w'), indent=1)
print(json.dumps(N, indent=1))
