"""What a known lens predicts for a pair of GW images: delay and apparent distances.

A cataloged galaxy lens with Einstein radius theta_E (measured from its lensed arc) and lens
redshift z_l is modelled as a singular isothermal sphere (SIS). If the GW host is the lensed arc
(source redshift z_s), a source at scaled position y = beta / theta_E < 1 gives two images,

    first  (minimum):  mu_1 = 1 + 1/y
    second (saddle):   mu_2 = 1 - 1/y   (|mu_2| = 1/y - 1)
    delay:             dt   = 2 y theta_E^2 D_dt(z_l, z_s)

so the first image is always the brighter one. With the intrinsic source position uniform in area
(p(y) = 2y on [0, 1]), the observed delay fixes y, and the luminosity distance D_L(z_s) together with
mu_1 and mu_2 predicts the apparent distances D_L / sqrt|mu| that each GW event measures.

pair_likelihood() returns the probability density of the observed delay and the two apparent-distance
likelihoods under "lensed by this lens", marginalised over the source redshift when it is unknown.
Detection selection is not modelled here (stage-1 ranking only).
"""
import numpy as np

from gwlens.population import (ARCSEC, einstein_radius_arcsec, luminosity_distance, time_delay_distance_seconds,
                               _lensed_source_redshift_icdf)

SIGMA_V_MAX = 450.0   # km/s; a galaxy-scale lens cannot be heavier
SIGMA_V_LN_SD = 0.2   # scatter of theta_E about the SIS value for a measured velocity dispersion


def sis_images(y):
    y = np.asarray(y, float)
    return 1 + 1 / y, 1 - 1 / y


def source_redshift_grid(z_l, z_s=None, n=200, zmax=8.0, theta_e_arcsec=None, sigma_v=None):
    """Grid and weights for z_s: a delta at a known z_s, else the lensed-source prior above z_l.

    For unknown z_s the observed theta_E must be produced by a plausible galaxy: with a measured
    velocity dispersion, theta_E(sigma_v, z_l, z_s) must match theta_E within SIGMA_V_LN_SD (log); without
    one, sigma_v <= SIGMA_V_MAX (sources just behind the lens would need an impossibly heavy lens).
    """
    if z_s is not None and np.isfinite(z_s):
        return np.array([float(z_s)]), np.array([1.0])
    z, cdf = _lensed_source_redshift_icdf()
    grid = np.linspace(z_l + 0.02, zmax, n)
    pdf = np.gradient(np.interp(grid, z, cdf), grid)
    w = pdf * np.gradient(grid)
    if theta_e_arcsec is not None:
        if sigma_v is not None and np.isfinite(sigma_v):
            r = np.log(einstein_radius_arcsec(sigma_v, z_l, grid) / theta_e_arcsec)
            w = w * np.exp(-0.5 * (r / SIGMA_V_LN_SD) ** 2)
        else:
            w = w * (einstein_radius_arcsec(SIGMA_V_MAX, z_l, grid) >= theta_e_arcsec)
    if w.sum() <= 0:
        return grid[:0], w[:0]
    return grid, w / w.sum()


def pair_likelihood(dt_obs, theta_e_arcsec, z_l, z_s, loglike_d1, loglike_d2, n_zs=200, sigma_v=None):
    """Density p(dt_obs, d1, d2 | lens) up to the common normalisation of the distance likelihoods.

    dt_obs        delay of the second event after the first [s] (> 0)
    loglike_d1/2  callables: log-likelihood of apparent luminosity distance [Mpc] for event 1 / 2
    Returns (value, info) where info holds the best source redshift and y.
    """
    if dt_obs <= 0 or not np.isfinite(theta_e_arcsec) or theta_e_arcsec <= 0:
        return 0.0, {}
    zs, w = source_redshift_grid(z_l, z_s, n=n_zs, theta_e_arcsec=theta_e_arcsec, sigma_v=sigma_v)
    if len(zs) == 0:
        return 0.0, {}
    th2 = (theta_e_arcsec * ARCSEC) ** 2
    ddt = time_delay_distance_seconds(z_l, zs)
    scale = 2 * th2 * ddt                      # dt = scale * y
    y = dt_obs / scale
    ok = (y > 1e-4) & (y < 1)
    if not ok.any():
        return 0.0, {}
    mu1, mu2 = sis_images(y[ok])
    dl = luminosity_distance(zs[ok])
    ll = loglike_d1(dl / np.sqrt(mu1)) + loglike_d2(dl / np.sqrt(np.abs(mu2)))
    dens = w[ok] * 2 * y[ok] / scale[ok] * np.exp(ll)   # p(y) = 2y, Jacobian 1/scale
    k = int(np.argmax(dens))
    return float(dens.sum()), dict(z_s=float(zs[ok][k]), y=float(y[ok][k]), mu1=float(mu1[k]), mu2=float(mu2[k]),
                                   dt_max_days=float(scale[ok][k] / 86400))


def max_delay_days(theta_e_arcsec, z_l, z_s=None, sigma_v=None):
    """Largest SIS delay (y -> 1) over the allowed source redshifts."""
    zs, w = source_redshift_grid(z_l, z_s, theta_e_arcsec=theta_e_arcsec, sigma_v=sigma_v)
    zs = zs[w > 1e-6 * w.max()] if len(zs) else zs
    if len(zs) == 0:
        return 0.0
    return float(np.max(2 * (theta_e_arcsec * ARCSEC) ** 2 * time_delay_distance_seconds(z_l, zs)) / 86400)


# ---------------------------------------------------------------------------
# Realistic lens geometry: EPL + external shear (library of scale-free solutions, lensearch.inject library)
# ---------------------------------------------------------------------------
class LensGeometry:
    """Smoothed density p(ln dphi, ln|mu_1|, ln|mu_2|) of the two brightest images of EPL+shear lenses with theta_E = 1",
    image 1 arriving first; dphi is their Fermat-potential difference in arcsec^2, so dt = D_dt dphi theta_E^2 ARCSEC^2.
    Source positions are uniform over the multiple-image region (no detection selection)."""

    def __init__(self, library, n_phi=48, n_mu=30, smooth=1.0):
        import pandas as pd
        from scipy.ndimage import gaussian_filter
        lib = pd.read_parquet(library) if not hasattr(library, 'columns') else library
        rows = []
        for mu, fer in zip(lib['mu'], lib['fermat']):
            mu, fer = np.asarray(mu), np.asarray(fer)
            k = np.argsort(-np.abs(mu))[:2]
            k = k[np.argsort(fer[k])]
            dphi = fer[k[1]] - fer[k[0]]
            if dphi > 0:
                rows.append((np.log(dphi), np.log(abs(mu[k[0]])), np.log(abs(mu[k[1]]))))
        x = np.array(rows)
        lo, hi = np.quantile(x[:, 0], [0.0005, 0.9995])
        self.phi_edges = np.linspace(lo - 0.5, hi + 0.5, n_phi + 1)
        self.mu_edges = np.linspace(-1.5, 7.0, n_mu + 1)
        H, _ = np.histogramdd(x, bins=(self.phi_edges, self.mu_edges, self.mu_edges))
        H = gaussian_filter(H, smooth, mode='constant')
        self.dphi_bin = np.diff(self.phi_edges)[0]
        self.dmu_bin = np.diff(self.mu_edges)[0]
        self.density = H / (H.sum() * self.dphi_bin * self.dmu_bin ** 2)
        self.phi_c = 0.5 * (self.phi_edges[1:] + self.phi_edges[:-1])
        self.lnmu = 0.5 * (self.mu_edges[1:] + self.mu_edges[:-1])
        self.n_entries = len(x)

    def slices(self, lnphi):
        """Density in (ln|mu_1|, ln|mu_2|) at each ln dphi (linear interpolation; zero outside the range)."""
        lnphi = np.atleast_1d(lnphi)
        f = (lnphi - self.phi_c[0]) / self.dphi_bin
        i0 = np.clip(np.floor(f).astype(int), 0, len(self.phi_c) - 2)
        t = np.clip(f - i0, 0, 1)[:, None, None]
        out = (1 - t) * self.density[i0] + t * self.density[i0 + 1]
        out[(lnphi < self.phi_edges[0]) | (lnphi > self.phi_edges[-1])] = 0.0
        return out


def pair_likelihood_geom(dt_obs, theta_e_arcsec, z_l, z_s, loglike_d1, loglike_d2, geom, n_zs=40, sigma_v=None):
    """As pair_likelihood, with the EPL+shear geometry density instead of the SIS relations.

    p(dt, D_1, D_2 | lens) = sum_zs w(zs) int p(ln dphi*, ln mu_1, ln mu_2) L_1(D_L/sqrt mu_1) L_2(D_L/sqrt mu_2) / dt,
    with ln dphi* = ln[dt / (D_dt theta_E^2 ARCSEC^2)].
    """
    if dt_obs <= 0 or not np.isfinite(theta_e_arcsec) or theta_e_arcsec <= 0:
        return 0.0, {}
    zs, w = source_redshift_grid(z_l, z_s, n=n_zs, theta_e_arcsec=theta_e_arcsec, sigma_v=sigma_v)
    if len(zs) == 0:
        return 0.0, {}
    ddt = time_delay_distance_seconds(z_l, zs)
    lnphi = np.log(dt_obs) - np.log(ddt * (theta_e_arcsec * ARCSEC) ** 2)
    S = geom.slices(lnphi)                                    # [nzs, n_mu, n_mu]
    if not S.any():
        return 0.0, {}
    dl = luminosity_distance(zs)
    scale = np.exp(-0.5 * geom.lnmu)
    l1 = np.exp(loglike_d1((dl[:, None] * scale[None, :]).ravel()).reshape(len(zs), -1))
    l2 = np.exp(loglike_d2((dl[:, None] * scale[None, :]).ravel()).reshape(len(zs), -1))
    per_zs = np.einsum('kij,ki,kj->k', S, l1, l2) * geom.dmu_bin ** 2
    val = float(np.sum(w * per_zs) / dt_obs)
    k = int(np.argmax(w * per_zs))
    m = S[k] * l1[k][:, None] * l2[k][None, :]
    i, j = np.unravel_index(np.argmax(m), m.shape)
    return val, dict(z_s=float(zs[k]), mu1=float(np.exp(geom.lnmu[i])), mu2=float(np.exp(geom.lnmu[j])),
                     lnphi=float(lnphi[k]))
