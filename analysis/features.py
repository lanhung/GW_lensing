"""Observable feature sets for the v2 lensed-BBH catalogues.

Every model input is derived from simulated *measurements* (``obs_`` and ``<scenario>__obs_``/
``__geo_`` columns). Geometric inputs are measured relative to the observed lens-light centre and
are invariant under translations and rotations of the image frame, so the arbitrary simulation
coordinate frame cannot be exploited.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from gwlens.population import luminosity_distance

FAMILIES = ['SIS', 'SIE', 'SIE_shear', 'EPL_shear']
GEO = ['geo_r_a', 'geo_r_b', 'geo_sep', 'geo_cos_ab', 'geo_asym', 'geo_theta_proxy', 'geo_n_images', 'geo_r_other1_n',
       'geo_r_other2_n', 'geo_q_light', 'geo_cos2_a', 'geo_sin2_a', 'geo_cos2_b', 'geo_sin2_b']
GW = ['gw_ln_mu_rel', 'gw_ln_abs_dt', 'gw_dt_sign', 'gw_rho_a', 'gw_rho_b', 'gw_ln_dapp_a', 'gw_ln_dapp_b']
FOLLOWUP = ['fu_z_l', 'fu_z_s', 'fu_ln_sigma_v']
COSMO = ['gw_ln_mu_cosmo']  # 2 [ln d_L(z_s,obs) - ln D_app]: requires z_s and an assumed cosmology
INPUT_SETS = {
    'GW': GW,
    'GW+z': GW + FOLLOWUP + COSMO,
    'Geometry': GEO,
    'Geometry+FU': GEO + FOLLOWUP,
    'Geometry+GW(rel)': GEO + ['gw_ln_mu_rel', 'gw_ln_abs_dt', 'gw_dt_sign', 'gw_rho_a', 'gw_rho_b'],
    'Full': GEO + GW + FOLLOWUP + COSMO,
}
# Ablations of the GW 'relative' inputs. |dt| (the clock) scales with the time-delay distance and the
# SNRs with the absolute amplitude, so both carry distance-scale information and change under a
# mass-sheet transformation; ln(mu_B/mu_A) and the arrival order do not ('ratio').
GW_RATIO = ['gw_ln_mu_rel', 'gw_dt_sign']
ABLATION_SETS = {
    'Geometry+GW(rel)+FU': INPUT_SETS['Geometry+GW(rel)'] + FOLLOWUP,
    'Geometry+GW(ratio)': GEO + GW_RATIO,
    'Geometry+GW(ratio)+FU': GEO + GW_RATIO + FOLLOWUP,
    'Geometry+GW(ratio+clock)+FU': GEO + GW_RATIO + ['gw_ln_abs_dt'] + FOLLOWUP,
    'Geometry+GW(ratio+SNR)+FU': GEO + GW_RATIO + ['gw_rho_a', 'gw_rho_b'] + FOLLOWUP,
}
ALL_SETS = {**INPUT_SETS, **ABLATION_SETS}


def mass_sheet_transform(d, lam):
    """Apply an external-convergence mass-sheet transformation (lam = 1 - kappa_ext) to a feature table.

    Image geometry, the relative magnification, the arrival order, redshifts and the lens velocity
    dispersion are unchanged; mu -> mu / lam^2, dt -> lam dt, SNR -> SNR / lam and the apparent
    distances -> lam D_app. The selection of the transformed population is not re-applied.
    """
    t = d.copy()
    t['gw_ln_abs_dt'] = d.gw_ln_abs_dt + np.log(lam)
    t['gw_rho_a'], t['gw_rho_b'] = d.gw_rho_a / lam, d.gw_rho_b / lam
    t['gw_ln_dapp_a'], t['gw_ln_dapp_b'] = d.gw_ln_dapp_a + np.log(lam), d.gw_ln_dapp_b + np.log(lam)
    t['gw_ln_mu_cosmo'] = d.gw_ln_mu_cosmo - 2 * np.log(lam)
    t['mu'], t['mu_b'] = d.mu / lam ** 2, d.mu_b / lam ** 2
    return t


def build(df, scenario, sky='estimated'):
    """Select a scenario's detected, resolved pairs and derive model inputs and targets."""
    s = scenario + '__'
    d = df[df[s + 'selected'] == True].copy()  # noqa: E712
    out = pd.DataFrame({'event_id': d.event_id.to_numpy(), 'family': d.family.to_numpy()})
    xa, ya, xb, yb = (d[s + k].to_numpy(float) for k in ('geo_xa', 'geo_ya', 'geo_xb', 'geo_yb'))
    ra, rb = np.hypot(xa, ya), np.hypot(xb, yb)
    sep = np.hypot(xa - xb, ya - yb)
    out['geo_r_a'], out['geo_r_b'], out['geo_sep'] = ra, rb, sep
    out['geo_cos_ab'] = (xa * xb + ya * yb) / np.maximum(ra * rb, 1e-12)
    out['geo_asym'] = (ra - rb) / np.maximum(ra + rb, 1e-12)
    out['geo_theta_proxy'] = 0.5 * (ra + rb)
    out['geo_n_images'] = d[s + 'geo_n_images'].to_numpy(float)
    out['geo_r_other1_n'] = d[s + 'geo_r_other1'].to_numpy(float) / out.geo_theta_proxy
    out['geo_r_other2_n'] = d[s + 'geo_r_other2'].to_numpy(float) / out.geo_theta_proxy
    out['geo_q_light'] = d.obs_q_light.to_numpy(float)
    pa = d.obs_pa_light.to_numpy(float)
    for tag, x, y in (('a', xa, ya), ('b', xb, yb)):
        ang = np.arctan2(y, x) - pa
        out[f'geo_cos2_{tag}'], out[f'geo_sin2_{tag}'] = np.cos(2 * ang), np.abs(np.sin(2 * ang))
    rel = s + ('obs_ln_mu_rel' if sky == 'estimated' else 'obs_ln_mu_rel_knownsky')
    out['gw_ln_mu_rel'] = d[rel].to_numpy(float)
    dt = d[s + 'obs_dt_ab'].to_numpy(float)
    out['gw_ln_abs_dt'] = np.log(np.maximum(np.abs(dt), 1.0))
    out['gw_dt_sign'] = np.sign(dt)
    out['gw_rho_a'], out['gw_rho_b'] = d[s + 'obs_rho_a'].to_numpy(float), d[s + 'obs_rho_b'].to_numpy(float)
    out['gw_ln_dapp_a'], out['gw_ln_dapp_b'] = d[s + 'obs_ln_dapp_a'].to_numpy(float), d[s + 'obs_ln_dapp_b'].to_numpy(float)
    zs = np.clip(d.obs_z_s.to_numpy(float), 1e-3, None)
    out['fu_z_l'], out['fu_z_s'] = d.obs_z_l.to_numpy(float), zs
    out['fu_ln_sigma_v'] = np.log(np.clip(d.obs_sigma_v.to_numpy(float), 10, None))
    out['gw_ln_mu_cosmo'] = 2 * (np.log(luminosity_distance(zs)) - out.gw_ln_dapp_a)
    # Targets and truth used only for evaluation and diagnostics.
    out['mu'] = d[s + 'true_mu_a'].to_numpy(float)
    out['mu_b'] = d[s + 'true_mu_b'].to_numpy(float)
    out['y_true'] = d.true_y.to_numpy(float)
    out['dl_true'] = d.true_dl.to_numpy(float)
    out['z_s_true'] = d.true_z_s.to_numpy(float)
    out['n_images_true'] = d.true_n_images.to_numpy(int)
    out['morse_a'] = d[s + 'true_morse_a'].to_numpy(float)
    return out.reset_index(drop=True)


def load(data_dir, scenario, families=FAMILIES, max_per_family=None, seed=0, sky='estimated'):
    parts = []
    for fam in families:
        df = pd.read_parquet(Path(data_dir) / f'{fam}.parquet')
        b = build(df, scenario, sky)
        if max_per_family and len(b) > max_per_family:
            b = b.sample(max_per_family, random_state=seed)
        parts.append(b)
    return pd.concat(parts, ignore_index=True)


def check_inputs(cols):
    bad = [c for c in cols if not c.startswith(('geo_', 'gw_', 'fu_'))]
    if bad:
        raise ValueError(f'Forbidden model inputs: {bad}')
    return cols
