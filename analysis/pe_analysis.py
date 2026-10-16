"""Compare waveform-level joint PE of lensed pairs with the SNR-based measurement model.

Reads the per-event summaries and posteriors written by gwlens.joint_pe and

* checks the calibration of the joint posteriors on ln(mu_B/mu_A) and the apparent distance
  ln d_A (PIT values and 90 per cent coverage over events);
* compares the PE uncertainties with the measurement model of gwlens.catalogue, whose
  relative-magnification scatter is measured on the full catalogue as a function of the joint
  SNR and whose distance error is the GWTC-calibrated sigma_lnD(rho_J);
* reports how often the relative Morse index is identified;
* feeds the PE relative magnification into the saved NPE flows (analysis/npe.py) in place of the
  SNR-model measurement and compares the end-to-end absolute-magnification errors.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.features import build  # noqa: E402

CAL = Path(__file__).resolve().parents[1] / 'realdata' / 'outputs' / 'distance_error_calibration.json'


def robust_sd(x):
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def model_sigma_mu_rel(cat, scen, bins=12):
    """Robust scatter of the SNR-model ln mu_rel measurement versus joint SNR, from the catalogue."""
    s = scen + '__'
    d = cat[cat[s + 'selected'] == True]  # noqa: E712
    rho_j = np.hypot(d[s + 'obs_rho_a'], d[s + 'obs_rho_b']).to_numpy()
    res = (d[s + 'obs_ln_mu_rel'] - np.log(d[s + 'true_mu_b'] / d[s + 'true_mu_a'])).to_numpy()
    edges = np.unique(np.quantile(np.log(rho_j), np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.digitize(np.log(rho_j), edges) - 1, 0, len(edges) - 2)
    centres = np.array([np.median(np.log(rho_j)[idx == i]) for i in range(len(edges) - 1)])
    sig = np.array([robust_sd(res[idx == i]) for i in range(len(edges) - 1)])
    return lambda r: np.interp(np.log(r), centres, sig)


def load_events(pe_dir, data_dir):
    rows, cats = [], {}
    for f in sorted(Path(pe_dir).glob('*/*_summary.json')):
        s = json.loads(f.read_text())
        post = pd.read_parquet(f.with_name(f.name.replace('_summary.json', '_posterior.parquet')))
        fam, scen = s['family'], s['scenario']
        if fam not in cats:
            cats[fam] = pd.read_parquet(Path(data_dir) / f'{fam}.parquet').set_index('event_id', drop=False)
        row = cats[fam].loc[s['event_id']]
        p = scen + '__'
        lnd = np.log(post.luminosity_distance)
        true_lnd = np.log(s['true_luminosity_distance'])
        r = dict(s)
        r.update(rho_j=float(np.hypot(row[p + 'obs_rho_a'], row[p + 'obs_rho_b'])),
                 pit_ln_mu_rel=float(np.mean(post.ln_mu_rel < s['true_ln_mu_rel'])),
                 pit_ln_d=float(np.mean(lnd < true_lnd)), true_ln_d=float(true_lnd),
                 model_ln_mu_rel=float(row[p + 'obs_ln_mu_rel']), model_ln_d=float(row[p + 'obs_ln_dapp_a']),
                 true_mu_a=float(row[p + 'true_mu_a']), n_images=int(row[p + 'geo_n_images']),
                 cos_iota=float(row.true_cos_iota))
        rows.append(r)
    return pd.DataFrame(rows), cats


def end_to_end(ev, cats, flow_dir, device='cpu'):
    """Absolute-magnification NPE with the SNR-model versus the PE relative magnification."""
    import torch
    from analysis.npe import load_flow, posterior_samples, standardise
    out = []
    for scen, g in ev.groupby('scenario'):
        ck = Path(flow_dir) / f'flow_{scen}_Geometry_GWrel_FU.pt'
        if not ck.exists():
            continue
        flow, meta = load_flow(ck, device)
        feats = pd.concat([build(cats[f].loc[cats[f].event_id.isin(g.event_id)], scen) for f in g.family.unique()])
        feats = feats.set_index('event_id').loc[g.event_id].reset_index()
        pe_med = g.set_index('event_id').loc[feats.event_id, 'ln_mu_rel_median'].to_numpy()
        for tag, x_rel in (('SNR model', feats.gw_ln_mu_rel.to_numpy()), ('joint PE', pe_med)):
            f = feats.copy()
            f['gw_ln_mu_rel'] = x_rel
            x, _, _ = standardise(f[meta['cols']].to_numpy(float), meta['mean'], meta['std'])
            torch.manual_seed(0)
            with torch.no_grad():
                smp = posterior_samples(flow, x, device)
            y = np.log(f.mu.to_numpy())
            qv = np.array([meta['qconf'][fam] for fam in f.family])
            lo, hi = np.quantile(smp, 0.05, 1) - qv, np.quantile(smp, 0.95, 1) + qv
            for fam, m in [(fm, f.family.to_numpy() == fm) for fm in f.family.unique()] + [('all', np.ones(len(f), bool))]:
                med = np.exp(np.median(smp[m], 1))
                out.append(dict(scenario=scen, family=fam, gw_input=tag, n=int(m.sum()),
                                median_rel=float(np.median(np.abs(med - f.mu.to_numpy()[m]) / f.mu.to_numpy()[m])),
                                coverage90_conformal=float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))))
    return pd.DataFrame(out)


def summarise(ev, cats, cal):
    rows = []
    for (scen, fam), g in list(ev.groupby(['scenario', 'family'])) + [((s, 'all'), g) for s, g in ev.groupby('scenario')]:
        cat_all = pd.concat(cats.values()) if fam == 'all' else cats[fam]
        sig_model = model_sigma_mu_rel(cat_all, scen)(g.rho_j.to_numpy())
        sig_d_model = np.sqrt(cal['a'] ** 2 + max(cal['b'] ** 2 - 1, 0) / g.rho_j.to_numpy() ** 2)
        res_pe = g.ln_mu_rel_median - g.true_ln_mu_rel
        res_model = g.model_ln_mu_rel - g.true_ln_mu_rel
        rows.append(dict(
            scenario=scen, family=fam, n=len(g), median_rho_j=float(g.rho_j.median()),
            cov90_ln_mu_rel=float(np.mean((g.true_ln_mu_rel >= g.ln_mu_rel_q05) & (g.true_ln_mu_rel <= g.ln_mu_rel_q95))),
            cov90_ln_d=float(np.mean((g.true_ln_d >= g.luminosity_distance_q05) & (g.true_ln_d <= g.luminosity_distance_q95))),
            pit_ks_ln_mu_rel=float(np.max(np.abs(np.sort(g.pit_ln_mu_rel) - (np.arange(len(g)) + 0.5) / len(g)))),
            pe_sd_ln_mu_rel=float(g.ln_mu_rel_sd.median()), model_sd_ln_mu_rel=float(np.median(sig_model)),
            sd_ratio_ln_mu_rel=float(np.median(g.ln_mu_rel_sd / sig_model)),
            pe_absres_ln_mu_rel=float(np.median(np.abs(res_pe))), model_absres_ln_mu_rel=float(np.median(np.abs(res_model))),
            pe_sd_ln_d=float(g.luminosity_distance_sd.median()), model_sd_ln_d=float(np.median(sig_d_model)),
            sd_ratio_ln_d=float(np.median(g.luminosity_distance_sd / sig_d_model)),
            morse_dn_identified=float(np.mean(g.delta_morse_prob_true > 0.5)),
            morse_dn_prob_mean=float(g.delta_morse_prob_true.mean()),
            sky_offset_deg_median=float(g.sky_offset_deg_median.median()),
            cpu_hours_median=float(g.seconds.median() / 3600 * 8)))
    return pd.DataFrame(rows)


def figure(ev, cats, cal, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.4))
    colours = {'ET': 'C0', 'O5': 'C3'}
    rr = np.geomspace(8, 400, 100)
    for scen, g in ev.groupby('scenario'):
        c = colours.get(scen, 'k')
        sig = model_sigma_mu_rel(pd.concat(cats.values()), scen)
        ax[0].scatter(g.rho_j, g.ln_mu_rel_sd, s=8, color=c, label=f'{scen} joint PE')
        rs = rr[(rr >= g.rho_j.min() * 0.8) & (rr <= g.rho_j.max() * 1.2)]
        ax[0].plot(rs, sig(rs), color=c, lw=1, ls='--', label=f'{scen} SNR model')
        ax[1].scatter(g.rho_j, g.luminosity_distance_sd, s=8, color=c)
        x = np.sort(g.pit_ln_mu_rel)
        ax[2].plot(x, (np.arange(len(x)) + 1) / len(x), color=c, drawstyle='steps-post', label=f'{scen} ln $\\mu_{{\\rm rel}}$')
        x = np.sort(g.pit_ln_d)
        ax[2].plot(x, (np.arange(len(x)) + 1) / len(x), color=c, ls=':', drawstyle='steps-post', label=f'{scen} ln $D_A$')
    ax[1].plot(rr, np.sqrt(cal['a'] ** 2 + (cal['b'] ** 2 - 1) / rr ** 2), color='k', lw=1, ls='--', label='GWTC calibration')
    ax[2].plot([0, 1], [0, 1], color='grey', lw=0.8)
    for a in ax[:2]:
        a.set_xscale('log')
        a.set_yscale('log')
        a.set_xlabel(r'joint SNR $\rho_J$')
    ax[0].set_ylabel(r'$\sigma[\ln(\mu_B/\mu_A)]$')
    ax[1].set_ylabel(r'$\sigma[\ln D_A]$')
    ax[2].set_xlabel('PIT')
    ax[2].set_ylabel('CDF')
    ax[0].legend(fontsize=6, frameon=False)
    ax[1].legend(fontsize=6, frameon=False)
    ax[2].legend(fontsize=6, frameon=False, loc='upper left')
    for a, l in zip(ax, 'abc'):
        a.text(-0.2, 1.04, f'({l})', transform=a.transAxes, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pe-dir', type=Path, default=Path('results/pe'))
    p.add_argument('--data-dir', type=Path, default=Path('data/pe'))
    p.add_argument('--flow-dir', type=Path, default=Path('results/v2_npe_ckpt'))
    p.add_argument('--out-dir', type=Path, default=Path('results/pe_analysis'))
    p.add_argument('--figure', type=Path, default=None)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cal = json.loads(CAL.read_text())
    cal = dict(a=cal.get('a', cal.get('sigma_floor')), b=cal.get('b', cal.get('snr_coefficient')))
    ev, cats = load_events(a.pe_dir, a.data_dir)
    print(f'{len(ev)} PE events')
    ev.to_csv(a.out_dir / 'pe_events.csv', index=False)
    summ = summarise(ev, cats, cal)
    summ.to_csv(a.out_dir / 'pe_summary.csv', index=False)
    e2e = end_to_end(ev, cats, a.flow_dir)
    e2e.to_csv(a.out_dir / 'pe_end_to_end.csv', index=False)
    pd.set_option('display.width', 250)
    print(summ.round(3).to_string())
    print(e2e.round(3).to_string())
    if a.figure:
        figure(ev, cats, cal, a.figure)


if __name__ == '__main__':
    main()
