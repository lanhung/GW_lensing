"""Calibrate luminosity-distance measurement errors on real GWTC events.

Queries the GWOSC event API for all confident BBH candidates in GWTC-1 to GWTC-4.0 and fits
the log-distance uncertainty as a function of network matched-filter SNR,

    sigma_lnD(rho) = sqrt(a^2 + (b / rho)^2),

where sigma_lnD is the half width of the published 90 per cent credible interval of ln d_L
divided by 1.645. The fit is used by the simulation to draw apparent-distance errors for each
image. The constant term absorbs the inclination-distance degeneracy that does not average down
with SNR; the 1/rho term is the pure amplitude-noise contribution.
"""
import argparse, json, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CATALOGS = ['GWTC-1-confident', 'GWTC-2.1-confident', 'GWTC-3-confident', 'GWTC-4.0']


def fetch(catalog):
    url = f'https://gwosc.org/eventapi/json/{catalog}/'
    events = json.load(urllib.request.urlopen(url, timeout=120))['events']
    rows = []
    for name, e in events.items():
        rows.append({'event': e['commonName'], 'catalog': catalog, **{k: e.get(k) for k in [
            'network_matched_filter_snr', 'luminosity_distance', 'luminosity_distance_lower', 'luminosity_distance_upper',
            'mass_1_source', 'mass_2_source', 'redshift', 'chirp_mass_source', 'chi_eff', 'p_astro']}})
    return pd.DataFrame(rows)


def model(rho, a, b):
    return np.sqrt(a ** 2 + (b / rho) ** 2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, default=Path('realdata/outputs'))
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat([fetch(c) for c in CATALOGS], ignore_index=True)
    df = df.dropna(subset=['network_matched_filter_snr', 'luminosity_distance', 'luminosity_distance_lower', 'luminosity_distance_upper'])
    # Keep binary black holes only (both source masses above 3 solar masses).
    df = df[(df.mass_2_source > 3)].drop_duplicates('event', keep='last').copy()
    d = df.luminosity_distance.to_numpy(float)
    lo = d + df.luminosity_distance_lower.to_numpy(float)
    hi = d + df.luminosity_distance_upper.to_numpy(float)
    df['sigma_lnD'] = (np.log(hi) - np.log(np.maximum(lo, 1.0))) / (2 * 1.645)
    rho = df.network_matched_filter_snr.to_numpy(float)
    (ca, cb), cov = curve_fit(model, rho, df.sigma_lnD.to_numpy(), p0=[0.1, 3.0], bounds=([0, 0], [2, 50]))
    resid = df.sigma_lnD - model(rho, ca, cb)
    fit = {'n_events': int(len(df)), 'catalogs': CATALOGS, 'form': 'sigma_lnD = sqrt(a^2 + (b/rho)^2)',
           'a': float(ca), 'b': float(cb), 'a_err': float(np.sqrt(cov[0, 0])), 'b_err': float(np.sqrt(cov[1, 1])),
           'residual_rms': float(np.sqrt(np.mean(resid ** 2))),
           'median_sigma_lnD': float(df.sigma_lnD.median()), 'median_snr': float(np.median(rho))}
    df.to_csv(a.out_dir / 'gwtc_bbh_events.csv', index=False)
    (a.out_dir / 'distance_error_calibration.json').write_text(json.dumps(fit, indent=2))
    print(json.dumps(fit, indent=2))


if __name__ == '__main__':
    main()
