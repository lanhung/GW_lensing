"""Audit task E (real-noise part): stratify the published O4a injection results.

Reads realdata/outputs/o4a_injections.csv (exact-template matched-filter recovery of lensed image
pairs injected into public O4a H1/L1 strain and into Gaussian noise with the same PSDs) and reports
the relative-magnification error ln(R_hat/R) by weaker-image SNR, SNR ratio and noise type, with
tails and outlier counts. These injections use the exact template, so they test the noise model
only, not waveform mismatch or free parameter estimation.

    python audit/o4a_strata.py --csv realdata/outputs/o4a_injections.csv --out out/o4a_strata.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def summary(g):
    e, z = g.ln_mu_rel_err.to_numpy(), g.z.to_numpy()
    a = np.abs(e)
    return dict(n=len(g), median_err=float(np.median(e)), median_abs_err=float(np.median(a)), p90_abs=float(np.quantile(a, 0.9)),
                p99_abs=float(np.quantile(a, 0.99)) if len(a) >= 100 else np.nan, max_abs=float(a.max()),
                z_sd_robust=float(1.4826 * np.median(np.abs(z - np.median(z)))), z_sd=float(z.std()),
                frac_abs_z_gt3=float(np.mean(np.abs(z) > 3)), frac_abs_err_gt_ln2=float(np.mean(a > np.log(2))),
                ks_p_vs_normal=float(stats.kstest(z, 'norm').pvalue))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', type=Path, default=Path('realdata/outputs/o4a_injections.csv'))
    p.add_argument('--out', type=Path, default=Path('out/o4a_strata.csv'))
    a = p.parse_args()
    df = pd.read_csv(a.csv)
    df = df[df.rho_b > 8].copy()  # both images above threshold, as in the paper
    df['snr_ratio'] = df.rho_b / df.rho_a
    rows = []
    for noise, g0 in df.groupby('noise'):
        rows.append(dict(noise=noise, stratum='all', lo=np.nan, hi=np.nan, **summary(g0)))
        for var, edges in (('rho_b', [8, 10, 12, 15, 20, 30, 1e9]), ('rho_a', [10, 15, 20, 30, 45, 1e9]),
                           ('snr_ratio', [0, 0.4, 0.6, 0.8, 1.01])):
            idx = np.digitize(g0[var], edges) - 1
            for i in range(len(edges) - 1):
                g = g0[idx == i]
                if len(g) >= 10:
                    rows.append(dict(noise=noise, stratum=var, lo=edges[i], hi=edges[i + 1], **summary(g)))
        # excluding the single glitch outlier documented in the paper
        g = g0[np.abs(g0.z) < 10]
        rows.append(dict(noise=noise, stratum='all, |z|<10', lo=np.nan, hi=np.nan, **summary(g)))
    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    pd.set_option('display.width', 250)
    print(out.round(3).to_string())


if __name__ == '__main__':
    main()
