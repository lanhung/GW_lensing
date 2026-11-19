"""Figures and failure-case tables for the relative-magnification audit (Section 8 of the request).

Reads the outputs of audit/relmag_errors.py, audit/pe_matched.py, audit/train_oracle_flow.py and the
published NPE predictions, and writes PNG figures plus failure_cases.csv to --out-dir.

    python audit/figures.py --out out
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({'font.size': 8, 'figure.dpi': 150})
COL = {'ET': 'C0', 'O5': 'C3'}
LEVELS = np.linspace(0.05, 0.99, 40)


def cov_from_pit(pit, lev):
    return np.array([np.mean(np.abs(pit - 0.5) <= l / 2) for l in lev])


def cov_from_z(z, lev):
    return np.array([np.mean(np.abs(z) <= stats.norm.ppf((1 + l) / 2)) for l in lev])


def fig_true_vs_est(ev, out):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), sharex=True, sharey=True)
    sel = ev[(ev.known_sky == (ev.scenario == 'ET'))]  # the analyses the paper relies on
    for a, (col, title) in zip(ax, (('b1', 'B1 naive SNR ratio'), ('b2', 'B2 antenna-corrected (benchmark input)'), ('pe_median', 'joint PE (median, 90% interval)'))):
        for scen, g in sel.groupby('scenario'):
            if col == 'pe_median':
                a.errorbar(g.true_ln_R, g.pe_median, yerr=[g.pe_median - g.pe_q05, g.pe_q95 - g.pe_median], fmt='o', ms=2.5, lw=0.6, color=COL[scen], label=scen, alpha=0.8)
            else:
                a.errorbar(g.true_ln_R, g[col], yerr=1.645 * g.sigma_model_cat if col == 'b2' else None, fmt='o', ms=2.5, lw=0.6, color=COL[scen], label=scen, alpha=0.8)
        lim = [-4, 2.5]
        a.plot(lim, lim, color='grey', lw=0.8)
        a.set_xlim(lim)
        a.set_ylim(lim)
        a.set_title(title, fontsize=7)
        a.set_xlabel(r'true $\ln(\mu_B/\mu_A)$')
    ax[0].set_ylabel(r'estimated $\ln(\mu_B/\mu_A)$')
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'fig1_true_vs_estimated_matched.png', bbox_inches='tight')


def fig_error_dist(pe, out):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    bins = np.linspace(-3, 3, 121)
    for a, scen in zip(ax, ('ET', 'O5')):
        g = pe[pe.scenario == scen]
        for col, lab, c in (('d1', 'B1 naive', 'C1'), ('d2', 'B2 corrected', 'C0'), ('d3', 'B3 true response', 'C2')):
            a.hist(np.clip(g[col], -3, 3), bins=bins, histtype='step', color=c, density=True,
                   label=f'{lab}: med|d| {np.median(np.abs(g[col])):.3f}, >ln2: {100 * np.mean(np.abs(g[col]) > np.log(2)):.1f}%')
        a.set_yscale('log')
        a.set_xlabel(r'$\ln\hat R-\ln R$')
        a.set_title(f'{scen}: {len(g)} selected pairs', fontsize=8)
        a.legend(frameon=False, fontsize=6)
    ax[0].set_ylabel('density')
    fig.tight_layout()
    fig.savefig(out / 'fig2_error_distributions_catalogue.png', bbox_inches='tight')


def fig_error_vs_snr(st, ev, out):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for a, var, xl in zip(ax, ('weaker_image_snr', 'joint_snr'), (r'weaker-image SNR $\hat\rho_B$', r'joint SNR $\rho_J$')):
        for scen in ('ET', 'O5'):
            s = st[(st.scenario == scen) & (st.stratum == var)]
            for est, ls in (('B1_naive', ':'), ('B2_corrected', '-'), ('B3_oracle_response', '--')):
                q = s[s.estimator == est]
                x = np.sqrt(q.lo * np.minimum(q.hi, 2 * q.lo.max()))
                a.plot(x, q.median_abs, ls=ls, color=COL[scen], lw=1, label=f'{scen} {est.split("_")[0]}')
            g = ev[(ev.scenario == scen) & (ev.known_sky == (scen == 'ET'))]
            xv = g.obs_rho_b if var == 'weaker_image_snr' else g.rho_j
            a.scatter(xv, np.abs(g.res_pe_median), s=7, color=COL[scen], alpha=0.6, label=f'{scen} joint PE |res|')
        a.set_xscale('log')
        a.set_yscale('log')
        a.set_xlabel(xl)
        a.legend(frameon=False, fontsize=5, ncol=2)
    ax[0].set_ylabel(r'median $|\ln\hat R-\ln R|$ (lines), PE $|$res$|$ (points)')
    fig.tight_layout()
    fig.savefig(out / 'fig3_error_vs_snr.png', bbox_inches='tight')


def fig_naive_bias(pe, out):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for a, scen in zip(ax, ('ET', 'O5')):
        g = pe[pe.scenario == scen]
        if len(g) > 20000:
            g = g.sample(20000, random_state=0)
        a.scatter(2 * g.ln_resp_ratio_true, g.d1, s=2, alpha=0.3, color=COL[scen])
        a.plot([-3, 3], [-3, 3], color='k', lw=0.8)
        a.set_xlim(-3, 3)
        a.set_ylim(-4, 4)
        a.set_xlabel(r'$2\ln(R_B/R_A)$ (true response ratio)')
        a.set_title(f'{scen}: naive-ratio error = response ratio + SNR noise', fontsize=7)
    ax[0].set_ylabel(r'B1 error $\ln\hat R_{\rm naive}-\ln R$')
    fig.tight_layout()
    fig.savefig(out / 'fig4_naive_bias_vs_response_ratio.png', bbox_inches='tight')


def fig_coverage(ev, pe, npe_dir, oracle_dir, out):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for scen in ('ET', 'O5'):
        g = ev[(ev.scenario == scen) & (ev.known_sky == (scen == 'ET'))]
        ax[0].plot(LEVELS, cov_from_pit(g.pe_pit.to_numpy(), LEVELS), color=COL[scen], label=f'{scen} joint PE (n={len(g)})')
        ax[0].plot(LEVELS, cov_from_z((g.res_b2 / g.sigma_model_cat).to_numpy(), LEVELS), color=COL[scen], ls='--', label=f'{scen} B2 with model sigma (matched)')
        c = pe[pe.scenario == scen]
        ax[0].plot(LEVELS, cov_from_z((c.d2 / c.sigma_model).to_numpy(), LEVELS), color=COL[scen], ls=':', label=f'{scen} B2 with analytic sigma (catalogue)')
        for d, name, ls in ((npe_dir, 'Geometry_GWratio_FU', '-'), (oracle_dir, 'oracle_Geometry_GWratio_FU', '--')):
            f = Path(d) / f'npe_predictions_{scen}_{name}.parquet'
            if f.exists():
                pr = pd.read_parquet(f)
                ax[1].plot(LEVELS, cov_from_pit(pr.pit.to_numpy(), LEVELS), color=COL[scen], ls=ls, label=f'{scen} NPE {"oracle ratio" if "oracle" in name else "SNR-model ratio"} (raw)')
    for a in ax:
        a.plot([0, 1], [0, 1], color='grey', lw=0.8)
        a.set_xlabel('nominal central credibility')
        a.legend(frameon=False, fontsize=5.5)
    ax[0].set_ylabel('empirical coverage')
    ax[0].set_title(r'relative magnification $\ln(\mu_B/\mu_A)$', fontsize=8)
    ax[1].set_title(r'absolute magnification $\ln\mu_A$ (NPE test split, uncalibrated)', fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'fig5_coverage_vs_nominal.png', bbox_inches='tight')


def fig_downstream(ds, out):
    d = ds[(ds.family == 'all') & (ds.near_caustic == 'all') & (ds.known_sky == (ds.scenario == 'ET'))]
    order = [c for c in ['D0 geometry only (NPE)', 'D1 true ratio (NPE retrained on truth)', 'D1a true ratio into current NPE',
                         'D2n naive SNR ratio (NPE)', 'D2 SNR-model ratio (NPE, paper)', 'D3 joint-PE median ratio (NPE)',
                         'D4 Bayesian two-stage, simulated family', 'D4e Bayesian two-stage, EPL+shear for all'] if c in set(d.config)]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for scen in ('ET', 'O5'):
        g = d[d.scenario == scen].set_index('config').reindex(order)
        y = np.arange(len(order)) + (0.15 if scen == 'O5' else -0.15)
        ax[0].barh(y, 100 * g.median_rel, height=0.3, color=COL[scen], label=scen)
        ax[0].errorbar(100 * g.p90_rel, y, fmt='|', color=COL[scen], ms=6)
        ax[1].barh(y, g.cov90, height=0.3, color=COL[scen], label=scen)
    ax[0].set_yticks(np.arange(len(order)))
    ax[0].set_yticklabels(order, fontsize=6)
    ax[1].set_yticks(np.arange(len(order)))
    ax[1].set_yticklabels([])
    ax[0].set_xlabel(r'median (bar) and 90th percentile (tick) relative error of $\mu_A$ [%]')
    ax[1].set_xlabel('coverage of 90% intervals')
    ax[1].axvline(0.9, color='grey', lw=0.8)
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'fig6_downstream_ladder.png', bbox_inches='tight')


def failure_cases(ev, out):
    cols = ['scenario', 'known_sky', 'family', 'event_id', 'true_ln_R', 'pe_median', 'pe_q05', 'pe_q95', 'b2', 'sigma_model_cat', 'res_pe_median', 'res_b2',
            'morse_dn_prob_true', 'sky_offset_deg', 'pe_frac_beyond_3rsd', 'pe_width_ratio_90_to_gauss', 'obs_rho_a', 'obs_rho_b', 'n_images', 'true_mu_a']
    worst_pe = ev.reindex(ev.res_pe_median.abs().sort_values(ascending=False).index).head(12)[cols].assign(kind='largest PE residual')
    worst_z = ev.assign(z=ev.res_pe_median / ev.pe_sd).reindex((ev.res_pe_median / ev.pe_sd).abs().sort_values(ascending=False).index).head(12)[cols].assign(kind='largest PE standardised residual')
    worst_b2 = ev.reindex(ev.res_b2.abs().sort_values(ascending=False).index).head(12)[cols].assign(kind='largest SNR-model residual')
    pd.concat([worst_pe, worst_z, worst_b2]).to_csv(out / 'failure_cases.csv', index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('out'))
    p.add_argument('--npe-dir', type=Path, default=Path('published_results/v2_review_npe2'))
    a = p.parse_args()
    out = a.out / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    pe = pd.read_parquet(a.out / 'relmag' / 'per_event.parquet')
    st = pd.read_csv(a.out / 'relmag' / 'strata.csv')
    ev = pd.read_csv(a.out / 'pe_matched' / 'c_per_event.csv')
    ds = pd.read_csv(a.out / 'pe_matched' / 'd_summary.csv')
    fig_true_vs_est(ev, out)
    fig_error_dist(pe, out)
    fig_error_vs_snr(st, ev, out)
    fig_naive_bias(pe, out)
    fig_coverage(ev, pe, a.npe_dir, a.out / 'oracle_flows', out)
    fig_downstream(ds, out)
    failure_cases(ev, out)
    print('figures written to', out)


if __name__ == '__main__':
    main()
