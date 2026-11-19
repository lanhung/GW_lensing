"""Audit tasks C and D: event-matched comparison of the SNR-based measurement with archived joint PE,
and propagation of each relative-magnification input into the absolute-magnification inference.

Inputs
  --pe-dir       archived gwlens.joint_pe outputs (<scen>/<id>_summary.json, _posterior.parquet; ET free sky,
                 ET known sky [*_knownsky_*], O5 free sky)
  --data-dir     the catalogues the PE systems were drawn from (data/pe)
  --train-dir    the benchmark catalogues (data/v2), for the catalogue-wide scatter of the SNR model vs joint SNR
  --flow-dir     published NPE checkpoints (flow_<scen>_Geometry.pt, flow_<scen>_Geometry_GWratio_FU.pt)
  --oracle-dir   checkpoints of audit/train_oracle_flow.py (flow_<scen>_oracle_Geometry_GWratio_FU.pt)
  --posthoc-dir  two-stage Bayesian GW+EM results (gwem_posthoc_sn/<config>/<scen>_<id>_..._summary.json)

Outputs (out_dir)
  c_per_event.csv      per PE run: truth, PE median/interval, B1/B2/B3 estimates, residuals, widths, z-scores
  c_summary.csv        per scenario x sky x family: coverage, PIT, widths, tails, paired PE-vs-model tests
  d_per_event.csv      per system and configuration D0..D4: ln mu_A posterior median, 68/90 per cent intervals
  d_summary.csv        per scenario x sky x family x configuration: median/90th-pct relative error, MAE, coverage
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.features import FAMILIES, build  # noqa: E402
from analysis.npe import load_flow, posterior_samples, standardise  # noqa: E402
from analysis.pe_analysis import model_sigma_mu_rel  # noqa: E402
from audit.relmag_errors import per_event as catalogue_estimates  # noqa: E402


def load_pe(pe_dir):
    rows = []
    for f in sorted(Path(pe_dir).glob('*/*_summary.json')):
        s = json.loads(f.read_text())
        post = pd.read_parquet(f.with_name(f.name.replace('_summary.json', '_posterior.parquet')))
        x = post.ln_mu_rel.to_numpy()
        med, sd = np.median(x), x.std()
        # non-Gaussianity / multimodality diagnostics of the PE posterior of ln(mu_B/mu_A)
        q05, q95, q16, q84 = np.quantile(x, [0.05, 0.95, 0.16, 0.84])
        far = float(np.mean(np.abs(x - med) > 3 * 0.7413 * (q84 - q16)))   # mass beyond 3 robust sd
        r = dict(event_id=s['event_id'], family=s['family'], scenario=s['scenario'],
                 known_sky=bool(s.get('known_sky', False)) or f.name.endswith('_knownsky_summary.json'),
                 true_ln_R=float(s['true_ln_mu_rel']), pe_median=float(med), pe_sd=float(sd), pe_q05=float(q05), pe_q95=float(q95),
                 pe_q16=float(q16), pe_q84=float(q84), pe_width90=float(q95 - q05), pe_frac_beyond_3rsd=far,
                 pe_width_ratio_90_to_gauss=float((q95 - q05) / (2 * 1.645 * sd)) if sd > 0 else np.nan,
                 pe_pit=float(np.mean(x < s['true_ln_mu_rel'])),
                 morse_dn_prob_true=float(s['delta_morse_prob_true']), sky_offset_deg=float(s.get('sky_offset_deg_median', np.nan)),
                 snr_a=float(s['snr_a']), snr_b=float(s['snr_b']), n_samples=int(s['n_samples']), pe_seconds=float(s['seconds']),
                 true_mu_a_pe=float(np.nan), ln_d_pit=float(np.mean(np.log(post.luminosity_distance) < np.log(s['true_luminosity_distance']))))
        rows.append(r)
    return pd.DataFrame(rows)


def c_summary(ev):
    rows = []
    groups = list(ev.groupby(['scenario', 'known_sky', 'family'])) + [((s, k, 'all'), g) for (s, k), g in ev.groupby(['scenario', 'known_sky'])]
    for (scen, ks, fam), g in groups:
        res = {k: (g[k] - g.true_ln_R).to_numpy() for k in ('pe_median', 'b1', 'b2', 'b3')}
        r = dict(scenario=scen, known_sky=ks, family=fam, n=len(g), median_rho_j=float(g.rho_j.median()))
        for k, v in res.items():
            r[f'medabs_{k}'] = float(np.median(np.abs(v)))
            r[f'p90abs_{k}'] = float(np.quantile(np.abs(v), 0.9))
            r[f'rmse_{k}'] = float(np.sqrt(np.mean(v ** 2)))
            r[f'maxabs_{k}'] = float(np.max(np.abs(v)))
            r[f'bias_{k}'] = float(np.mean(v))
        r['cov90_pe'] = float(np.mean((g.true_ln_R >= g.pe_q05) & (g.true_ln_R <= g.pe_q95)))
        r['cov68_pe'] = float(np.mean((g.true_ln_R >= g.pe_q16) & (g.true_ln_R <= g.pe_q84)))
        r['cov90_b2_model_interval'] = float(np.mean(np.abs(res['b2']) <= 1.645 * g.sigma_model_cat))
        r['cov90_b2_analytic_interval'] = float(np.mean(np.abs(res['b2']) <= 1.645 * g.sigma_model_analytic))
        r['pit_ks_pe'] = float(stats.kstest(g.pe_pit, 'uniform').pvalue)
        r['pe_sd_over_model_sd_median'] = float(np.median(g.pe_sd / g.sigma_model_cat))
        r['pe_width90_over_model_width90_median'] = float(np.median(g.pe_width90 / (2 * 1.645 * g.sigma_model_cat)))
        r['frac_pe_abs_z_gt3'] = float(np.mean(np.abs(res['pe_median']) / g.pe_sd > 3))
        r['frac_b2_abs_z_gt3'] = float(np.mean(np.abs(res['b2']) / g.sigma_model_cat > 3))
        r['frac_pe_multimodal_proxy'] = float(np.mean((g.morse_dn_prob_true < 0.9) | (g.pe_frac_beyond_3rsd > 0.01)))
        r['morse_dn_identified'] = float(np.mean(g.morse_dn_prob_true > 0.5))
        d = np.abs(res['pe_median']) - np.abs(res['b2'])
        r['frac_pe_closer_than_b2'] = float(np.mean(d < 0))
        r['wilcoxon_p_pe_vs_b2'] = float(stats.wilcoxon(d).pvalue) if len(d) > 5 and np.any(d != 0) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def npe_eval(flow, meta, feats, device, qconf_by_family=True):
    x, _, _ = standardise(feats[meta['cols']].to_numpy(float), meta['mean'], meta['std'])
    torch.manual_seed(0)
    with torch.no_grad():
        smp = posterior_samples(flow, x, device)
    q = meta['qconf']
    qv = np.array([q[f] for f in feats.family]) if (qconf_by_family and q) else np.zeros(len(feats))
    # global conformal quantile (no family label needed; what a real-lens analysis can use)
    qg = float(meta.get('qgroup', {}).get('none', {}).get('0', np.nan)) if isinstance(meta.get('qgroup'), dict) else np.nan
    lo, hi = np.quantile(smp, 0.05, 1), np.quantile(smp, 0.95, 1)
    return dict(ln_mu_a_median=np.median(smp, 1), q05=lo - qv, q95=hi + qv, q05_glob=lo - qg, q95_glob=hi + qg,
                q16=np.quantile(smp, 0.16, 1), q84=np.quantile(smp, 0.84, 1), conformal_q=qv, conformal_q_global=qg)


def downstream(ev, cats, flow_dir, oracle_dir, posthoc_dir, device='cpu'):
    rows = []
    for (scen, ks), g in ev.groupby(['scenario', 'known_sky']):
        feats = pd.concat([build(cats[f].loc[cats[f].event_id.isin(g.event_id)], scen) for f in g.family.unique()])
        feats = feats.set_index('event_id').loc[g.event_id].reset_index()
        gi = g.set_index('event_id').loc[feats.event_id]
        y = np.log(feats.mu.to_numpy())
        configs = []
        geo = Path(flow_dir) / f'flow_{scen}_Geometry.pt'
        main = Path(flow_dir) / f'flow_{scen}_Geometry_GWratio_FU.pt'
        orc = Path(oracle_dir) / f'flow_{scen}_oracle_Geometry_GWratio_FU.pt' if oracle_dir else None
        if geo.exists():
            configs.append(('D0 geometry only (NPE)', geo, None))
        if main.exists():
            configs += [('D1a true ratio into current NPE', main, np.log(feats.mu_b / feats.mu).to_numpy()),
                        ('D2 SNR-model ratio (NPE, paper)', main, feats.gw_ln_mu_rel.to_numpy()),
                        ('D2n naive SNR ratio (NPE)', main, gi.b1.to_numpy()),
                        ('D3 joint-PE median ratio (NPE)', main, gi.pe_median.to_numpy())]
        if orc is not None and orc.exists():
            configs.append(('D1 true ratio (NPE retrained on truth)', orc, np.log(feats.mu_b / feats.mu).to_numpy()))
        for name, ck, xrel in configs:
            flow, meta = load_flow(ck, device)
            f = feats.copy()
            if xrel is not None:
                f['gw_ln_mu_rel'] = xrel
            r = npe_eval(flow, meta, f, device)
            for i, eid in enumerate(f.event_id):
                rows.append(dict(scenario=scen, known_sky=ks, event_id=eid, family=f.family.iloc[i], config=name, method='NPE',
                                 true_ln_mu_a=y[i], ln_mu_a_median=r['ln_mu_a_median'][i], q05=r['q05'][i], q95=r['q95'][i],
                                 q05_glob=r['q05_glob'][i], q95_glob=r['q95_glob'][i],
                                 q16=r['q16'][i], q84=r['q84'][i], interval='conformal90 (per true family); _glob: global conformal'))
        # D4: two-stage Bayesian reconstruction with the PE posterior as effective likelihood (own family; EPL+shear for all)
        if posthoc_dir and Path(posthoc_dir).exists():
            suffix = '_ks' if ks else ''
            for label, model_of in (('D4 Bayesian two-stage, simulated family', lambda fam: fam),
                                    ('D4e Bayesian two-stage, EPL+shear for all', lambda fam: 'EPL_shear')):
                for i, eid in enumerate(f.event_id):
                    fam = f.family.iloc[i]
                    d = Path(posthoc_dir) / f'free_k0_{model_of(fam)}_sn{suffix}'
                    hits = list(d.glob(f'{scen}_{eid}_*_summary.json'))
                    if not hits:
                        continue
                    s = json.loads(hits[0].read_text())
                    rows.append(dict(scenario=scen, known_sky=ks, event_id=eid, family=fam, config=label, method='Bayesian',
                                     true_ln_mu_a=np.log(s['true_mu_a']), ln_mu_a_median=s['ln_mu_a_median'], q05=s['ln_mu_a_q05'],
                                     q95=s['ln_mu_a_q95'], q05_glob=s['ln_mu_a_q05'], q95_glob=s['ln_mu_a_q95'],
                                     q16=s['ln_mu_a_median'] - s['ln_mu_a_sd'], q84=s['ln_mu_a_median'] + s['ln_mu_a_sd'],
                                     interval='nominal 90 (68: median +- sd)'))
    return pd.DataFrame(rows)


def d_summary(dp):
    rows = []
    dp = dp.copy()
    dp['rel'] = np.abs(np.exp(dp.ln_mu_a_median - dp.true_ln_mu_a) - 1)
    dp['near_caustic'] = np.exp(dp.true_ln_mu_a) >= 5
    keys = [['scenario', 'known_sky', 'config', 'family'], ['scenario', 'known_sky', 'config'], ['scenario', 'known_sky', 'config', 'near_caustic']]
    for k in keys:
        for key, g in dp.groupby(k):
            r = dict(zip(k, key))
            r.setdefault('family', 'all')
            r.setdefault('near_caustic', 'all')
            r.update(n=len(g), method=g.method.iloc[0], median_rel=float(g.rel.median()), p90_rel=float(g.rel.quantile(0.9)),
                     mae_ln=float(np.mean(np.abs(g.ln_mu_a_median - g.true_ln_mu_a))),
                     cov90=float(np.mean((g.true_ln_mu_a >= g.q05) & (g.true_ln_mu_a <= g.q95))),
                     cov68=float(np.mean((g.true_ln_mu_a >= g.q16) & (g.true_ln_mu_a <= g.q84))),
                     cov90_global_conformal=float(np.mean((g.true_ln_mu_a >= g.q05_glob) & (g.true_ln_mu_a <= g.q95_glob))),
                     width90_ln=float(np.mean(g.q95 - g.q05)))
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pe-dir', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=Path('data/pe'))
    p.add_argument('--train-dir', type=Path, default=Path('data/v2'))
    p.add_argument('--flow-dir', type=Path, required=True)
    p.add_argument('--oracle-dir', type=Path, default=None)
    p.add_argument('--posthoc-dir', type=Path, default=None)
    p.add_argument('--out-dir', type=Path, default=Path('out/pe_matched'))
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    ev = load_pe(a.pe_dir)
    print(f'{len(ev)} PE runs:', ev.groupby(['scenario', 'known_sky']).size().to_dict(), flush=True)
    cats = {f: pd.read_parquet(a.data_dir / f'{f}.parquet').set_index('event_id', drop=False) for f in FAMILIES}
    est = pd.concat([catalogue_estimates(cats[f].reset_index(drop=True), s) for f in FAMILIES for s in ('ET', 'O5')], ignore_index=True)
    est = est.set_index(['scenario', 'event_id'])
    ev = ev.join(est[['b1', 'b2', 'b3', 'ln_R_true', 'rho_j', 'obs_rho_a', 'obs_rho_b', 'true_mu_a', 'true_mu_b', 'dt_ab', 'cos_iota',
                      'n_images', 'sigma_model', 'ln_resp_ratio_true']], on=['scenario', 'event_id'])
    assert np.allclose(ev.ln_R_true, ev.true_ln_R, atol=1e-6), 'PE truth and catalogue truth disagree'
    ev = ev.rename(columns={'sigma_model': 'sigma_model_analytic'})
    train = pd.concat([pd.read_parquet(a.train_dir / f'{f}.parquet') for f in FAMILIES], ignore_index=True)
    sig = {s: model_sigma_mu_rel(train, s) for s in ('ET', 'O5')}
    ev['sigma_model_cat'] = [float(sig[s](r)) for s, r in zip(ev.scenario, ev.rho_j)]
    for k in ('pe_median', 'b1', 'b2', 'b3'):
        ev[f'res_{k}'] = ev[k] - ev.true_ln_R
    ev.to_csv(a.out_dir / 'c_per_event.csv', index=False)
    cs = c_summary(ev)
    cs.to_csv(a.out_dir / 'c_summary.csv', index=False)
    pd.set_option('display.width', 250)
    print(cs[['scenario', 'known_sky', 'family', 'n', 'medabs_pe_median', 'medabs_b1', 'medabs_b2', 'medabs_b3', 'p90abs_pe_median', 'p90abs_b2',
              'cov90_pe', 'cov90_b2_model_interval', 'pe_sd_over_model_sd_median', 'frac_pe_closer_than_b2', 'wilcoxon_p_pe_vs_b2']].round(3).to_string(), flush=True)
    dp = downstream(ev, cats, a.flow_dir, a.oracle_dir, a.posthoc_dir)
    dp.to_csv(a.out_dir / 'd_per_event.csv', index=False)
    ds = d_summary(dp)
    ds.to_csv(a.out_dir / 'd_summary.csv', index=False)
    print(ds[(ds.family == 'all') & (ds.near_caustic == 'all')][['scenario', 'known_sky', 'config', 'n', 'median_rel', 'p90_rel', 'cov68', 'cov90', 'cov90_global_conformal', 'width90_ln']].round(3).to_string())


if __name__ == '__main__':
    main()
