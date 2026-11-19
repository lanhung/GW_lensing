"""Audit task B: errors of the SNR-based relative-magnification measurement on the full catalogues.

For every selected pair the catalogue stores the truth and three estimates of ln R, R = |mu_B|/|mu_A|:

  B1 naive      2 ln(rho_hat_B / rho_hat_A)                              (no antenna correction)
  B2 corrected  2 ln(rho_hat_B / rho_hat_A) - 2 ln(R_hat_B / R_hat_A)    (column obs_ln_mu_rel: the benchmark input)
  B3 oracle     2 ln(rho_hat_B / rho_hat_A) - 2 ln(R_B / R_A)            (column obs_ln_mu_rel_knownsky: true response)

with rho_j = sqrt|mu_j| R_j the optimal SNR, rho_hat_j = rho_j + N(0,1) the measured SNR and R_j the
network response. The residuals decompose exactly:
  d3 = B3 - truth                 pure SNR-measurement noise
  d2 - d3                         response-estimation (sky/orientation) error
  d1 - d3 = 2 ln(R_B / R_A)       the bias of the naive ratio = the true response ratio
Outputs (out_dir): per_event.parquet, summary.csv (per scenario x family x estimator) and strata.csv
(per scenario, pooled families, binned by weaker-image SNR, joint SNR, delay, inclination, image
multiplicity and response ratio).

    python audit/relmag_errors.py --data-dir data/v2 --out-dir out/relmag
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwlens.detectors import SCENARIOS  # noqa: E402

FAMILIES = ['SIS', 'SIE', 'SIE_shear', 'EPL_shear']
EST = {'B1_naive': 'b1', 'B2_corrected': 'b2', 'B3_oracle_response': 'b3'}
DAY = 86400.0


def per_event(df, scen):
    s = scen + '__'
    d = df[df[s + 'selected'] == True]  # noqa: E712
    o = pd.DataFrame({'event_id': d.event_id.to_numpy(), 'family': d.family.to_numpy(), 'scenario': scen})
    o['true_mu_a'], o['true_mu_b'] = d[s + 'true_mu_a'].to_numpy(float), d[s + 'true_mu_b'].to_numpy(float)
    o['ln_R_true'] = np.log(o.true_mu_b / o.true_mu_a)
    ra, rb = d[s + 'obs_rho_a'].to_numpy(float), d[s + 'obs_rho_b'].to_numpy(float)
    ta, tb = d[s + 'true_rho_a'].to_numpy(float), d[s + 'true_rho_b'].to_numpy(float)
    o['obs_rho_a'], o['obs_rho_b'], o['true_rho_a'], o['true_rho_b'] = ra, rb, ta, tb
    o['rho_j'] = np.hypot(ra, rb)
    o['b1'] = 2 * np.log(rb / ra)
    o['b2'] = d[s + 'obs_ln_mu_rel'].to_numpy(float)
    o['b3'] = d[s + 'obs_ln_mu_rel_knownsky'].to_numpy(float)
    o['ln_resp_ratio_true'] = 0.5 * (o.b1 - o.b3)        # ln(R_B/R_A)
    o['ln_resp_ratio_hat'] = 0.5 * (o.b1 - o.b2)         # ln(R_hat_B/R_hat_A)
    o['d1'], o['d2'], o['d3'] = o.b1 - o.ln_R_true, o.b2 - o.ln_R_true, o.b3 - o.ln_R_true
    o['d_response'] = o.d2 - o.d3
    o['sigma_model'] = 2 * np.sqrt(1 / ta ** 2 + 1 / tb ** 2)  # Gaussian-noise expectation used by the paper
    o['z3'] = o.d3 / o.sigma_model
    o['dt_ab'] = d[s + 'obs_dt_ab'].to_numpy(float)
    # the 2026-10-16 benchmark catalogues predate the stored inclination; the PE catalogues have it
    o['cos_iota'] = d.true_cos_iota.to_numpy(float) if 'true_cos_iota' in d.columns else np.nan
    o['n_images'] = d[s + 'geo_n_images'].to_numpy(int)
    o['y_true'] = d.true_y.to_numpy(float)
    o['z_s'] = d.true_z_s.to_numpy(float)
    o['sky_sigma_deg'] = SCENARIOS[scen]['sky_sigma_deg_at_rho20'] * 20.0 / o.rho_j
    return o


def metrics(d):
    d = np.asarray(d, float)
    a = np.abs(d)
    return dict(n=len(d), mean=float(d.mean()), median=float(np.median(d)), sd=float(d.std()),
                mad_sd=float(1.4826 * np.median(np.abs(d - np.median(d)))), rmse=float(np.sqrt(np.mean(d ** 2))),
                median_abs=float(np.median(a)), p68_abs=float(np.quantile(a, 0.68)), p90_abs=float(np.quantile(a, 0.90)),
                p95_abs=float(np.quantile(a, 0.95)), p99_abs=float(np.quantile(a, 0.99)), max_abs=float(a.max()),
                median_rel_R=float(np.median(np.abs(np.exp(d) - 1))),
                frac_abs_gt_0p5=float(np.mean(a > 0.5)), frac_abs_gt_ln2=float(np.mean(a > np.log(2))),
                frac_abs_gt_1=float(np.mean(a > 1.0)))


def summarise(pe):
    rows = []
    for scen, g0 in pe.groupby('scenario'):
        for fam, g in list(g0.groupby('family')) + [('all', g0)]:
            for name, col in EST.items():
                rows.append(dict(scenario=scen, family=fam, estimator=name, **metrics(g[col] - g.ln_R_true)))
            rows.append(dict(scenario=scen, family=fam, estimator='response_error_only(d2-d3)', **metrics(g.d_response)))
            rows.append(dict(scenario=scen, family=fam, estimator='true_response_ratio_2ln(RB/RA)', **metrics(2 * g.ln_resp_ratio_true)))
            m = metrics(g.z3)
            rows.append(dict(scenario=scen, family=fam, estimator='z3_standardised_SNR_noise', **m))
    return pd.DataFrame(rows)


def strata(pe):
    bins = {
        'weaker_image_snr': (pe.obs_rho_b, [8, 10, 12, 15, 20, 30, 50, np.inf]),
        'joint_snr': (pe.rho_j, [0, 15, 20, 30, 50, 100, np.inf]),
        'abs_delay_days': (np.abs(pe.dt_ab) / DAY, [0, 1 / 24, 0.25, 1, 10, 30, 100, 366]),
        'abs_cos_iota': (np.abs(pe.cos_iota), [0, 0.3, 0.6, 0.8, 0.95, 1.0001]),
        'n_images': (pe.n_images, [1.5, 2.5, 3.5, 4.5, 10]),
        'abs_ln_response_ratio': (np.abs(pe.ln_resp_ratio_true), [0, 0.1, 0.25, 0.5, 1.0, np.inf]),
        'true_mu_a': (pe.true_mu_a, [0, 2, 3, 5, 10, 20, np.inf]),
    }
    rows = []
    for scen, g0 in pe.groupby('scenario'):
        for var, (x, edges) in bins.items():
            x = x.loc[g0.index]
            idx = np.digitize(x, edges) - 1
            for i in range(len(edges) - 1):
                g = g0[idx == i]
                if len(g) < 30:
                    continue
                for name, col in EST.items():
                    rows.append(dict(scenario=scen, stratum=var, lo=edges[i], hi=edges[i + 1], estimator=name,
                                     **metrics(g[col] - g.ln_R_true)))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, default=Path('data/v2'))
    p.add_argument('--out-dir', type=Path, default=Path('out/relmag'))
    p.add_argument('--scenarios', nargs='+', default=['ET', 'O5'])
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for fam in FAMILIES:
        df = pd.read_parquet(a.data_dir / f'{fam}.parquet')
        for scen in a.scenarios:
            parts.append(per_event(df, scen))
        print(fam, {s: int(df[f'{s}__selected'].sum()) for s in a.scenarios}, flush=True)
    pe = pd.concat(parts, ignore_index=True)
    pe.to_parquet(a.out_dir / 'per_event.parquet', index=False)
    summ = summarise(pe)
    summ.to_csv(a.out_dir / 'summary.csv', index=False)
    st = strata(pe)
    st.to_csv(a.out_dir / 'strata.csv', index=False)
    # Consistency of the stored columns with the documented formulae (exact relations).
    chk = dict(
        b2_minus_b3_equals_response_term=bool(np.allclose(pe.b2 - pe.b3, -2 * (pe.ln_resp_ratio_hat - pe.ln_resp_ratio_true))),
        true_rho_ratio_identity_max_abs=float(np.max(np.abs(2 * np.log(pe.true_rho_b / pe.true_rho_a) - 2 * pe.ln_resp_ratio_true - pe.ln_R_true))),
        n_pairs={s: int((pe.scenario == s).sum()) for s in a.scenarios},
    )
    (a.out_dir / 'checks.json').write_text(json.dumps(chk, indent=1))
    pd.set_option('display.width', 250)
    cols = ['scenario', 'family', 'estimator', 'n', 'median', 'mad_sd', 'median_abs', 'p90_abs', 'p99_abs', 'median_rel_R', 'frac_abs_gt_ln2']
    print(summ[cols].round(3).to_string())
    print(json.dumps(chk, indent=1))


if __name__ == '__main__':
    main()
