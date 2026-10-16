"""Generate catalogues of lensed BBH image pairs with GW and electromagnetic observables.

Every row is an independent lens system (no reused source or lens identities). Each system is
evaluated for every detector scenario; the per-scenario GW measurements and detection flags are
stored in columns prefixed with the scenario name.

Usage:
    python -m gwlens.catalogue --family SIE_shear --n-systems 40000 --seed 1 --out data/v2/SIE_shear.parquet
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import detectors as det
from . import lenses
from . import population as pop

CAL_PATH = Path(__file__).resolve().parents[1] / 'realdata' / 'outputs' / 'distance_error_calibration.json'
YEAR = 365.25 * 86400.0
OBS = dict(
    rho_threshold=8.0,          # network matched-filter SNR threshold per image
    t_obs=YEAR,                 # both images must arrive within one observing year
    min_separation=0.1,         # arcsec, resolvable image pair (HST/JWST/Euclid-like PSF)
    min_em_mu=0.1,              # images fainter than this are not counted as resolved EM images
    sigma_img=0.01,             # arcsec, astrometric error per image
    sigma_center=0.02,          # arcsec, lens light-centre error
    sigma_z=0.001,              # spectroscopic redshift error per (1+z)
    sigma_vdisp=0.10,           # fractional velocity-dispersion error
    sigma_q_light=0.05,         # light axis-ratio error and light-mass mismatch
    sigma_pa_light_deg=10.0,    # light-mass position-angle misalignment
)


def calibration():
    c = json.loads(CAL_PATH.read_text())
    return c['a'], c['b']


def _gw_block(rng, scen, src, img, cal_a, cal_b):
    """Optimal and measured SNRs, detection and GW measurements for one scenario."""
    i0 = src['i0'][scen]
    gps = src['t0'] + img['t']
    rho_unl = np.array([det.network_snr(i0, det.antenna(scen, src['ra'], src['dec'], src['psi'], g), src['cosi']) for g in gps])
    rho = np.sqrt(np.abs(img['mu'])) * rho_unl
    rho_hat = rho + rng.normal(0, 1, len(rho))
    detected = rho_hat >= OBS['rho_threshold']
    out = {f'{scen}__n_detected': int(detected.sum()), f'{scen}__selected': False}
    if detected.sum() < 2:
        return out, None
    # Observing window opens with the first detected image.
    t_first = img['t'][detected].min()
    inwin = detected & (img['t'] - t_first <= OBS['t_obs'])
    if inwin.sum() < 2:
        return out, None
    idx = np.flatnonzero(inwin)
    a_i, b_i = idx[np.argsort(-rho_hat[idx])][:2]
    # Joint parameter estimation shares sky position, polarisation and inclination between images.
    rho_j = float(np.hypot(rho_hat[a_i], rho_hat[b_i]))
    scale = 20.0 / rho_j
    sky = np.deg2rad(det.SCENARIOS[scen]['sky_sigma_deg_at_rho20']) * scale
    v = np.array([np.cos(src['dec']) * np.cos(src['ra']), np.cos(src['dec']) * np.sin(src['ra']), np.sin(src['dec'])])
    v = v + rng.normal(0, sky, 3)
    v /= np.linalg.norm(v)
    ra_h, dec_h = np.arctan2(v[1], v[0]) % (2 * np.pi), np.arcsin(v[2])
    psi_h = src['psi'] + rng.normal(0, min(np.pi / 2, 0.5 * scale))
    cosi_h = float(np.clip(src['cosi'] + rng.normal(0, min(1.0, 0.3 * scale)), -1, 1))
    resp = lambda i, ra, dec, psi, ci: det.network_snr(i0, det.antenna(scen, ra, dec, psi, gps[i]), ci)
    r_true = np.array([resp(i, src['ra'], src['dec'], src['psi'], src['cosi']) for i in (a_i, b_i)])
    r_hat = np.array([resp(i, ra_h, dec_h, psi_h, cosi_h) for i in (a_i, b_i)])
    # Apparent distance d_L/sqrt|mu| from the amplitude: SNR noise + antenna-model error + a
    # shared inclination-distance term calibrated on GWTC-1..4.0 events (realdata/).
    sig_c = np.sqrt(cal_a ** 2 + max(cal_b ** 2 - 1.0, 0.0) / rho_j ** 2)
    eps_c = rng.normal(0, sig_c)
    d_app = src['dl'] / np.sqrt(np.abs(img['mu'][[a_i, b_i]]))
    ln_d_hat = np.log(d_app) + np.log(np.maximum(rho[[a_i, b_i]], 1e-3) / rho_hat[[a_i, b_i]]) + np.log(r_hat / np.maximum(r_true, 1e-12)) + eps_c
    out.update({
        f'{scen}__selected': True,
        f'{scen}__img_a': int(a_i), f'{scen}__img_b': int(b_i),
        f'{scen}__true_mu_a': float(abs(img['mu'][a_i])), f'{scen}__true_mu_b': float(abs(img['mu'][b_i])),
        f'{scen}__true_morse_a': float(img['morse'][a_i]), f'{scen}__true_morse_b': float(img['morse'][b_i]),
        f'{scen}__true_rho_a': float(rho[a_i]), f'{scen}__true_rho_b': float(rho[b_i]),
        f'{scen}__obs_rho_a': float(rho_hat[a_i]), f'{scen}__obs_rho_b': float(rho_hat[b_i]),
        f'{scen}__obs_ln_mu_rel': float(2 * (np.log(rho_hat[b_i] / rho_hat[a_i]) - np.log(r_hat[1] / r_hat[0]))),
        # Optimistic bound: sky position, polarisation and inclination known exactly.
        f'{scen}__obs_ln_mu_rel_knownsky': float(2 * (np.log(rho_hat[b_i] / rho_hat[a_i]) - np.log(r_true[1] / r_true[0]))),
        f'{scen}__obs_ln_dapp_a': float(ln_d_hat[0]), f'{scen}__obs_ln_dapp_b': float(ln_d_hat[1]),
        f'{scen}__obs_dt_ab': float(img['t'][b_i] - img['t'][a_i]),
    })
    return out, (a_i, b_i)


def simulate_system(rng, family, scenarios, cal):
    """One lens system: source, lens, images, GW and EM observables (all scenarios)."""
    while True:
        zs = float(pop.sample_lensed_source_redshift(rng, 1)[0])
        zl = float(pop.sample_lens_redshift(rng, [zs])[0])
        sig = float(pop.sample_velocity_dispersion(rng, 1)[0])
        te = float(pop.einstein_radius_arcsec(sig, zl, zs))
        if te < 0.05:
            continue
        shape = lenses.sample_lens_shape(rng, family)
        sol = lenses.solve(rng, family, te, shape)
        if sol is not None:
            break
    ddt = float(pop.time_delay_distance_seconds(zl, zs))
    t = lenses.time_delays_seconds(sol['fermat'], ddt)
    img = dict(x=sol['x'], y=sol['y'], mu=sol['mu'], morse=sol['morse'], kappa=sol['kappa'], t=t)
    m1, m2 = pop.sample_masses(rng, 1)
    c1, c2 = pop.sample_aligned_spins(rng, 1)
    dl = float(pop.luminosity_distance(zs))
    src = dict(ra=rng.uniform(0, 2 * np.pi), dec=np.arcsin(rng.uniform(-1, 1)), psi=rng.uniform(0, np.pi),
               cosi=rng.uniform(-1, 1), t0=1.9e9 + rng.uniform(0, YEAR), dl=dl)
    src['i0'] = {s: det.face_on_integrals(s, m1[0] * (1 + zs), m2[0] * (1 + zs), c1[0], c2[0], dl) for s in scenarios}

    # Electromagnetic observables. The true lens centre and the frame orientation are random.
    rot = rng.uniform(0, 2 * np.pi)
    cr, sr = np.cos(rot), np.sin(rot)
    cx, cy = rng.normal(0, 5.0, 2)
    xr = cx + cr * img['x'] - sr * img['y']
    yr = cy + sr * img['x'] + cr * img['y']
    xo = xr + rng.normal(0, OBS['sigma_img'], len(xr))
    yo = yr + rng.normal(0, OBS['sigma_img'], len(yr))
    cxo, cyo = cx + rng.normal(0, OBS['sigma_center']), cy + rng.normal(0, OBS['sigma_center'])
    q_light = float(np.clip(shape['q'] + rng.normal(0, OBS['sigma_q_light']), 0.2, 1.0))
    pa_light = (shape['phi_q'] + rot + np.deg2rad(rng.normal(0, OBS['sigma_pa_light_deg']))) % np.pi
    if family == 'SIS':
        pa_light = rng.uniform(0, np.pi)
    row = dict(family=family, true_z_s=zs, true_z_l=zl, true_sigma_v=sig, true_theta_e=te, true_y=sol['y_offset'],
               true_q=shape['q'], true_gamma=shape['gamma'], true_gamma_ext=shape['gamma_ext'], true_dl=dl,
               true_m1=float(m1[0]), true_m2=float(m2[0]), true_n_images=len(img['mu']),
               true_chi1z=float(c1[0]), true_chi2z=float(c2[0]), true_ra=float(src['ra']), true_dec=float(src['dec']),
               true_psi=float(src['psi']), true_cos_iota=float(src['cosi']), true_t0=float(src['t0']),
               obs_z_s=zs + rng.normal(0, OBS['sigma_z'] * (1 + zs)), obs_z_l=zl + rng.normal(0, OBS['sigma_z'] * (1 + zl)),
               obs_sigma_v=sig * (1 + rng.normal(0, OBS['sigma_vdisp'])), obs_q_light=q_light, obs_pa_light=pa_light)
    for s in scenarios:
        gw, pair = _gw_block(rng, s, src, img, *cal)
        row.update(gw)
        if pair is None:
            continue
        a_i, b_i = pair
        sep = float(np.hypot(xo[a_i] - xo[b_i], yo[a_i] - yo[b_i]))
        bright = np.abs(img['mu']) >= OBS['min_em_mu']
        resolved = sep >= OBS['min_separation']
        row[f'{s}__selected'] = bool(resolved)
        others = [k for k in np.flatnonzero(bright) if k not in (a_i, b_i)]
        r_other = sorted(float(np.hypot(xo[k] - cxo, yo[k] - cyo)) for k in others)
        row.update({
            f'{s}__geo_xa': xo[a_i] - cxo, f'{s}__geo_ya': yo[a_i] - cyo,
            f'{s}__geo_xb': xo[b_i] - cxo, f'{s}__geo_yb': yo[b_i] - cyo,
            f'{s}__geo_n_images': int(bright.sum()),
            f'{s}__geo_r_other1': r_other[0] if len(r_other) > 0 else np.nan,
            f'{s}__geo_r_other2': r_other[1] if len(r_other) > 1 else np.nan,
            f'{s}__true_kappa_a': float(img['kappa'][a_i]), f'{s}__true_kappa_b': float(img['kappa'][b_i]),
            f'{s}__true_dt_ab': float(t[b_i] - t[a_i]),
            f'{s}__true_t_a': float(t[a_i]),
            f'{s}__true_mu_signed_a': float(img['mu'][a_i]), f'{s}__true_mu_signed_b': float(img['mu'][b_i]),
        })
    return row


def _worker(args):
    family, scenarios, n, seed = args
    rng = np.random.default_rng(seed)
    cal = calibration()
    return [simulate_system(rng, family, scenarios, cal) for _ in range(n)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--family', choices=lenses.FAMILIES, required=True)
    p.add_argument('--scenarios', nargs='+', default=list(det.SCENARIOS))
    p.add_argument('--n-systems', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--chunk', type=int, default=200)
    p.add_argument('--processes', type=int, default=32)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; choose a new output path')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n_chunks = int(np.ceil(a.n_systems / a.chunk))
    seeds = np.random.SeedSequence([a.seed, lenses.FAMILIES.index(a.family)]).generate_state(n_chunks)
    jobs = [(a.family, tuple(a.scenarios), min(a.chunk, a.n_systems - i * a.chunk), int(s)) for i, s in enumerate(seeds)]
    t0 = time.time()
    with Pool(a.processes) as pool:
        rows = [r for chunk in pool.imap(_worker, jobs) for r in chunk]
    df = pd.DataFrame(rows)
    df.insert(0, 'event_id', [f'{a.family}-{a.seed}-{i:07d}' for i in range(len(df))])
    df.to_parquet(a.out, index=False)
    meta = dict(family=a.family, scenarios=a.scenarios, n_systems=len(df), seed=a.seed, seconds=time.time() - t0,
                observation_model=OBS, distance_error_calibration=json.loads(CAL_PATH.read_text()),
                selected={s: int(df[f'{s}__selected'].sum()) for s in a.scenarios})
    a.out.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta['selected']), f'{meta["seconds"]:.0f}s')


if __name__ == '__main__':
    main()
