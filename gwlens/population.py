"""Source and lens populations for lensed binary-black-hole simulations.

Sources follow the GWTC-4.0 fiducial BBH population (Broken Power Law + 2 Peaks mass model,
aligned spin components, Madau-Dickinson-shaped merger-rate density). Source redshifts are
weighted by the strong-lensing optical depth, and lenses follow the SDSS velocity-dispersion
function weighted by the isothermal cross-section.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy.cosmology import Planck18
from scipy.special import gamma as gamma_fn

C_KM_S = 299792.458
ARCSEC = np.pi / 180 / 3600
CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'gwtc4_bp2p.json'


def load_bp2p(path=CONFIG):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Cosmology tables
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def cosmology_table(zmax=12.0, n=4000):
    z = np.concatenate([[0.0], np.geomspace(1e-4, zmax, n - 1)])
    dc = Planck18.comoving_distance(z).value  # Mpc
    dvdz = Planck18.differential_comoving_volume(z).value * 4 * np.pi  # Mpc^3
    return z, dc, dvdz


def comoving_distance(z):
    zz, dc, _ = cosmology_table()
    return np.interp(z, zz, dc)


def luminosity_distance(z):
    return (1 + np.asarray(z)) * comoving_distance(z)


def angular_diameter_distance(z):
    return comoving_distance(z) / (1 + np.asarray(z))


def angular_diameter_distance_z1z2(z1, z2):
    """Flat universe: D_A(z1,z2) = (D_C(z2) - D_C(z1)) / (1 + z2)."""
    return (comoving_distance(z2) - comoving_distance(z1)) / (1 + np.asarray(z2))


def redshift_from_luminosity_distance(dl):
    zz = np.linspace(0, 12, 6000)
    return np.interp(dl, luminosity_distance(zz), zz)


# ---------------------------------------------------------------------------
# GWTC-4.0 BP2P mass model
# ---------------------------------------------------------------------------
def smoothing(m, mlow, delta):
    """Low-mass tapering used by LVK population models (S = 0 below mlow, 1 above mlow+delta)."""
    m = np.asarray(m, float)
    out = np.zeros_like(m)
    hi = m >= mlow + delta
    out[hi] = 1.0
    mid = (m > mlow) & ~hi
    x = m[mid] - mlow
    with np.errstate(over='ignore'):
        out[mid] = 1.0 / (np.exp(delta / x + delta / (x - delta)) + 1.0)
    return out


def bp2p_primary_pdf(m, h):
    m = np.asarray(m, float)
    b = h['break_mass']
    pl = np.where(m < b, (m / b) ** (-h['alpha_1']), (m / b) ** (-h['alpha_2']))
    pl = np.where((m >= h['mlow_1']) & (m <= h['mhigh']), pl, 0.0)
    grid = np.linspace(h['mlow_1'], h['mhigh'], 20000)
    plg = np.where(grid < b, (grid / b) ** (-h['alpha_1']), (grid / b) ** (-h['alpha_2']))
    pl = pl / np.trapezoid(plg, grid)
    g1 = np.exp(-0.5 * ((m - h['mpp_1']) / h['sigpp_1']) ** 2) / (np.sqrt(2 * np.pi) * h['sigpp_1'])
    g2 = np.exp(-0.5 * ((m - h['mpp_2']) / h['sigpp_2']) ** 2) / (np.sqrt(2 * np.pi) * h['sigpp_2'])
    lam0, lam1 = h['lam_0'], h['lam_1']
    lam2 = max(0.0, 1.0 - lam0 - lam1)
    return (lam0 * pl + lam1 * g1 + lam2 * g2) * smoothing(m, h['mlow_1'], h['delta_m_1'])


@lru_cache(maxsize=None)
def _m1_inverse_cdf():
    h = load_bp2p()
    grid = np.linspace(h['mlow_1'], h['mhigh'], 40000)
    pdf = bp2p_primary_pdf(grid, h)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    return grid, cdf / cdf[-1]


def sample_masses(rng, n):
    h = load_bp2p()
    grid, cdf = _m1_inverse_cdf()
    m1 = np.interp(rng.random(n), cdf, grid)
    m2 = np.empty(n)
    for i, mm in enumerate(m1):
        # Pairing p(q) ~ q^beta with a smoothed lower edge on the secondary mass.
        while True:
            q = rng.random() ** (1.0 / (h['beta'] + 1.0))
            cand = q * mm
            if cand >= h['mlow_2'] and rng.random() < smoothing(np.array([cand]), h['mlow_2'], h['delta_m_2'])[0]:
                m2[i] = cand
                break
    return m1, m2


def sample_aligned_spins(rng, n):
    """Aligned components chi_z = a cos(tilt) under the GWTC-4.0 default spin model."""
    h = load_bp2p()
    out = []
    for _ in range(2):
        a = np.empty(n)
        k = 0
        while k < n:
            d = rng.normal(h['mu_chi'], h['sigma_chi'], 2 * n)
            d = d[(d >= 0) & (d <= 1)]
            take = min(n - k, len(d))
            a[k:k + take] = d[:take]
            k += take
        iso = rng.random(n) > h['xi_spin']
        ct = np.empty(n)
        ct[iso] = rng.uniform(-1, 1, iso.sum())
        m = ~iso
        vals = []
        while len(vals) < m.sum():
            d = rng.normal(h['mu_spin'], h['sigma_spin'], 4 * max(1, m.sum()))
            vals.extend(d[(d >= -1) & (d <= 1)].tolist())
        ct[m] = np.array(vals[:m.sum()])
        out.append(a * ct)
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Redshifts weighted by merger rate and lensing optical depth
# ---------------------------------------------------------------------------
def merger_rate_shape(z, kappa):
    """Madau-Dickinson shape normalised to (1+z)^kappa at low redshift (peak near z~2)."""
    zp = 1.9
    return (1 + z) ** kappa / (1 + ((1 + z) / (1 + zp)) ** (kappa + 2.9))


@lru_cache(maxsize=None)
def _lensed_source_redshift_icdf(zmin=0.1, zmax=10.0):
    h = load_bp2p()
    z = np.linspace(zmin, zmax, 6000)
    zz, _, dvdz_table = cosmology_table()
    dvdz = np.interp(z, zz, dvdz_table)
    # Isothermal optical depth for a non-evolving comoving lens population: tau ~ D_C(z_s)^3.
    tau = comoving_distance(z) ** 3
    pdf = merger_rate_shape(z, h['kappa']) / (1 + z) * dvdz * tau
    cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(z))])
    return z, cdf / cdf[-1]


def sample_lensed_source_redshift(rng, n, zmax=10.0):
    z, cdf = _lensed_source_redshift_icdf(zmax=zmax)
    return np.interp(rng.random(n), cdf, z)


def sample_lens_redshift(rng, zs):
    """p(z_l | z_s) for isothermal lenses: (D_C,l D_C,ls / D_C,s)^2 dD_C,l."""
    out = np.empty(len(zs))
    for i, z in enumerate(zs):
        zl = np.linspace(1e-3, z * 0.999, 400)
        dl = comoving_distance(zl)
        ds = comoving_distance(z)
        w = (dl * (ds - dl) / ds) ** 2 * np.gradient(dl)
        c = np.cumsum(w)
        out[i] = np.interp(rng.random() * c[-1], c, zl)
    return out


# ---------------------------------------------------------------------------
# Lens velocity dispersions
# ---------------------------------------------------------------------------
def sample_velocity_dispersion(rng, n, smin=100.0, smax=400.0):
    """SDSS early-type VDF (Choi, Park & Vogeley 2007) weighted by sigma^4 cross-section."""
    sstar, alpha, beta = 161.0, 2.32, 2.67
    s = np.linspace(smin, smax, 4000)
    phi = (s / sstar) ** alpha * np.exp(-(s / sstar) ** beta) * beta / gamma_fn(alpha / beta) / s
    pdf = phi * s ** 4
    cdf = np.cumsum(pdf)
    return np.interp(rng.random(n) * cdf[-1], cdf, s)


def einstein_radius_arcsec(sigma_v, zl, zs):
    return 4 * np.pi * (sigma_v / C_KM_S) ** 2 * angular_diameter_distance_z1z2(zl, zs) / angular_diameter_distance(zs) / ARCSEC


def time_delay_distance_seconds(zl, zs):
    """(1+z_l) D_l D_s / (c D_ls) in seconds per rad^2."""
    mpc_m = 3.0856775814913673e22
    c = 299792458.0
    return (1 + zl) * angular_diameter_distance(zl) * angular_diameter_distance(zs) / angular_diameter_distance_z1z2(zl, zs) * mpc_m / c
