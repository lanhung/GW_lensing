"""Stage 1: rank all pairs of GW events for strong lensing, with and without known lenses.

For events a (earlier) and b (later):

  ln B_ms    overlap of detector-frame chirp mass, mass ratio and effective spin,
             B = E_{x ~ p_a}[p_b(x) / pi_ref(x)] (Haris et al. 2018), x = (ln Mc_det, q, chi_eff), pi_ref the
             LVK prior (uniform detector-frame component masses, isotropic spins); exact when both events used
             priors uniform in component masses (any bounds). Set to 0 when an event has no mass posterior.
  ln B_sky   4 pi int p_a p_b dOmega (isotropic sky prior).
  Lambda_c   p(dt, D_a, D_b | lens cell c) / [p_U(dt) E_a E_b] for an SIS lens with the Einstein radius and lens
             redshift of cell c (lensearch.lenspred; source redshift marginalised unless known), where
             E_i = int pi_det(D) l_i(D) dD and p_U(dt) = 2 (T - dt) / T^2.
  ln B_cat   ln B_ms + ln[ sum_L pi_L (4 pi)^2 p_a(Omega_L) p_b(Omega_L) Lambda_{c(L)} ]   (lensed by a cataloged lens)
  ln B_pop   ln B_ms + ln B_sky + ln sum_c w_pop(c) Lambda_c                            (lensed by any galaxy)

pi_L is proportional to theta_E^2 over the catalog, w_pop is the SIS lens population of gwlens.population
(sigma^4-weighted SDSS velocity-dispersion function, lens redshift given the lensed-source redshift).

Background: the lens catalog is rotated in right ascension N times; ln B_cat of every pair is recomputed for each
rotation. The fraction of rotations whose largest ln B_cat exceeds the observed one estimates the probability that
a coincidence as strong arises by chance.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp
from scipy.stats import gaussian_kde

from gwlens.population import cosmology_table
from lensearch.lenspred import LensGeometry, pair_likelihood, pair_likelihood_geom

NSIDE = 128
NPIX = hp.nside2npix(NSIDE)
PIXAREA = hp.nside2pixarea(NSIDE)
NSIDE_LENS = 64                                # lens term: 0.84 deg2 pixels, far finer than any GW sky map
NPIX_LENS = hp.nside2npix(NSIDE_LENS)
SMOOTH_DEG = 1.0
FOURPI = 4 * np.pi
N_KDE = 3000
EVENT_CACHE_VERSION = 2
TEMPLATE_LNMC_SD = 0.15                        # template chirp-mass likelihood width (ln Mc_det)
LND = np.linspace(np.log(5.0), np.log(60000.0), 500)
THETA_GRID = np.geomspace(0.3, 3.0, 13)        # arcsec, cell centres
ZL_GRID = np.linspace(0.05, 1.45, 15)
CHUNK = 4000                                   # pairs per lens-term chunk (memory)
JOINT_MASS = 0.999                             # lens term: pixels holding this fraction of the joint sky density


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------
def reference_prior_kde(n=400000, seed=1):
    rng = np.random.default_rng(seed)
    m1, m2 = rng.uniform(1, 500, (2, n))
    m1, m2 = np.maximum(m1, m2), np.minimum(m1, m2)
    q = m2 / m1
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    a1, a2 = rng.uniform(0, 0.99, (2, n))
    c1, c2 = rng.uniform(-1, 1, (2, n))
    chi = (a1 * c1 + q * a2 * c2) / (1 + q)
    keep = q > 0.05
    return gaussian_kde(np.vstack([np.log(mc[keep]), q[keep], chi[keep]])[:, :80000])


def log_dl_prior(dl):
    """Uniform in comoving volume and source-frame time (GWTC-2.1 cosmo files onward)."""
    zz, dc, dvdz = cosmology_table()
    dlt = (1 + zz) * dc
    z = np.interp(dl, dlt, zz)
    return np.log(np.interp(z, zz, dvdz) / np.interp(z, zz, np.gradient(dlt, zz)) / (1 + z))


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
class Event:
    kind = 'pe'

    def loglike_d(self, d):
        return np.interp(np.log(d), LND, self.lnl_d, left=-np.inf, right=-np.inf)


def _density_map(prob):
    m = np.clip(np.asarray(prob, float), 0, None)
    m = m / m.sum() / PIXAREA
    return m.astype(np.float32)


def _pe_skymap(npz, df):
    """PE sky map at NSIDE in RING order; the stored ordering is identified with the posterior samples."""
    if 'skymap' in npz.files:
        sk = np.asarray(npz['skymap'], float)
        nside = hp.npix2nside(len(sk))
        th, ph = np.pi / 2 - df.dec.values[:2000], df.ra.values[:2000]
        best = None
        for nest in (False, True):
            score = np.mean(sk[hp.ang2pix(nside, th, ph, nest=nest)])
            if best is None or score > best[0]:
                best = (score, nest)
        if best[1]:
            sk = hp.reorder(sk, n2r=True)
        sk = hp.ud_grade(sk, NSIDE, power=-2)
        return _density_map(sk)
    pix = hp.ang2pix(NSIDE, np.pi / 2 - df.dec.values, df.ra.values)
    m = np.bincount(pix, minlength=NPIX).astype(float)
    return _density_map(np.clip(hp.smoothing(m, fwhm=np.radians(SMOOTH_DEG)), 0, None))


def load_pe_event(directory, name):
    e = Event()
    df = pd.read_parquet(directory / f'{name}.parquet')
    npz = np.load(directory / f'{name}.npz')
    meta = json.loads((directory / f'{name}.json').read_text())
    e.name, e.catalog = name, meta['catalog']
    e.t = float(np.median(df.geocent_time))
    sub = df.sample(min(N_KDE, len(df)), random_state=1)
    e.x = np.vstack([np.log(sub.chirp_mass), sub.mass_ratio, sub.chi_eff])
    e.kde_x = gaussian_kde(e.x)
    if meta.get('dl_prior') == 'powerlaw2':           # injections analysed with a D^2 prior
        w = df.luminosity_distance.values ** -2.0
    else:                                              # LVK releases: uniform in comoving volume and source-frame time
        w = np.exp(-log_dl_prior(df.luminosity_distance.values))
    kde = gaussian_kde(np.log(df.luminosity_distance.values), weights=w / w.sum())
    e.lnl_d = np.log(np.maximum(kde(LND), 1e-300))
    e.sky = _pe_skymap(npz, df)
    e.mc_det = float(np.median(df.chirp_mass))
    e.dl = float(np.median(df.luminosity_distance))
    return e


def load_alert(directory, sid, t0):
    """Public alert: sky map and distance ansatz only (no mass posterior)."""
    e = Event()
    e.kind = 'alert'
    z = np.load(directory / f'{sid}.npz')
    e.name, e.catalog, e.t = sid, 'O4c-alert', float(t0)
    e.x = e.kde_x = None
    prob = np.nan_to_num(z['prob'])
    e.sky = _density_map(prob)
    # distance ansatz p(r | pixel) = DISTNORM r^2 N(r; DISTMU, DISTSIGMA) with prior r^2, so the likelihood of r
    # marginalised over the sky is sum_pixels PROB DISTNORM N(r; DISTMU, DISTSIGMA)
    mu, sig, nrm = np.nan_to_num(z['distmu']), np.nan_to_num(z['distsigma']), np.nan_to_num(z['distnorm'])
    ok = (prob > 1e-6 * prob.max()) & (sig > 0) & np.isfinite(mu) & (nrm > 0)
    d = np.exp(LND)
    like = (prob[ok, None] * nrm[ok, None] * np.exp(-0.5 * ((d[None, :] - mu[ok, None]) / sig[ok, None]) ** 2)
            / sig[ok, None]).sum(0)
    e.lnl_d = np.log(np.maximum(like / like.max(), 1e-300))
    e.mc_det, e.dl = np.nan, float(np.exp(LND[np.argmax(e.lnl_d)]))
    return e


def load_search_candidate(path_fits, row, rng):
    """Marginal candidate without PE: search sky map (with distance layers when present) and the template chirp mass
    as a Gaussian likelihood in ln Mc_det (TEMPLATE_LNMC_SD)."""
    from ligo.skymap.bayestar import rasterize
    from ligo.skymap.io import read_sky_map
    e = Event()
    e.kind = 'search'
    e.name, e.catalog, e.t = row.gw_name, row.release, float(row.gps_time)
    try:
        sm = read_sky_map(str(path_fits), moc=True, distances=True)
        r = rasterize(sm, order=int(np.log2(NSIDE)))
        cols = {c.lower(): hp.reorder(np.asarray(r[c], float), n2r=True) for c in r.colnames}
    except Exception:
        prob, _ = read_sky_map(str(path_fits), nest=False)
        cols = {'prob': hp.ud_grade(np.asarray(prob, float), NSIDE, power=-2)}
    prob = np.nan_to_num(cols['prob'])
    e.sky = _density_map(prob)
    d = np.exp(LND)
    if {'distmu', 'distsigma', 'distnorm'} <= set(cols):
        mu, sig, nrm = (np.nan_to_num(cols[k]) for k in ('distmu', 'distsigma', 'distnorm'))
        ok = (prob > 1e-6 * prob.max()) & (sig > 0) & (nrm > 0)
        like = (prob[ok, None] * nrm[ok, None] * np.exp(-0.5 * ((d[None, :] - mu[ok, None]) / sig[ok, None]) ** 2)
                / sig[ok, None]).sum(0)
        e.lnl_d = np.log(np.maximum(like / like.max(), 1e-300))
    else:
        e.lnl_d = np.zeros_like(LND)                       # no distance information (e.g. cWB)
    mc = float(row.template_mchirp) if np.isfinite(row.template_mchirp) else float(row.mchirp)
    e.x = e.kde_x = None
    # template chirp mass as a likelihood in ln Mc_det (width 0.15); mass ratio and spins unconstrained
    e.mc_mu, e.mc_sig = (np.log(mc), TEMPLATE_LNMC_SD) if np.isfinite(mc) and mc > 0 else (None, None)
    e.mc_det, e.dl = mc, float(np.exp(LND[np.argmax(e.lnl_d)]))
    return e


# ---------------------------------------------------------------------------
# Pair ingredients
# ---------------------------------------------------------------------------
N_OVERLAP = 800


def _has_template(e):
    return getattr(e, 'mc_mu', None) is not None


def ln_overlap_ms(a, b, ref=None):
    """ln of the mass-spin coherence of events a and b.

    PE-PE:              E_{x~p_a}[p_b(x) / pi_ref(x)] (symmetrised)
    PE-template:        E_{x~p_a}[L_b(ln Mc)] / Z_b,   Z_b = int L_b pi_ref(ln Mc) d ln Mc
    template-template:  int L_a L_b pi_ref / (Z_a Z_b)
    where L is the template likelihood N(ln Mc; mu, sd) and ref = (grid, density) of ln Mc under the reference prior.
    Returns 0 when an event has no mass information."""
    pa, pb = a.kde_x is not None, b.kde_x is not None
    ta, tb = _has_template(a), _has_template(b)
    if pa and pb:
        va = b.kde_x(a.x[:, :N_OVERLAP]) / a.x_prior
        vb = a.kde_x(b.x[:, :N_OVERLAP]) / b.x_prior
        return float(np.log(max(0.5 * (va.mean() + vb.mean()), 1e-300)))
    g, dens = ref
    dg = g[1] - g[0]
    gauss = lambda x, e: np.exp(-0.5 * ((x - e.mc_mu) / e.mc_sig) ** 2) / (np.sqrt(2 * np.pi) * e.mc_sig)
    if (pa and tb) or (pb and ta):
        pe, tp = (a, b) if pa else (b, a)
        z = np.sum(gauss(g, tp) * dens) * dg
        return float(np.log(max(np.mean(gauss(pe.x[0], tp)) / z, 1e-300)))
    if ta and tb:
        num = np.sum(gauss(g, a) * gauss(g, b) * dens) * dg
        za, zb = np.sum(gauss(g, a) * dens) * dg, np.sum(gauss(g, b) * dens) * dg
        return float(np.log(max(num / (za * zb), 1e-300)))
    return 0.0


def reference_lnmc_density(n=400000, seed=1):
    """Marginal density of ln Mc_det under the reference prior (uniform component masses in [1, 500])."""
    rng = np.random.default_rng(seed)
    m1, m2 = rng.uniform(1, 500, (2, n))
    m1, m2 = np.maximum(m1, m2), np.minimum(m1, m2)
    keep = m2 / m1 > 0.05
    lnmc = np.log((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2)[keep]
    g = np.linspace(np.log(0.5), np.log(450), 2000)
    h, edges = np.histogram(lnmc, bins=np.append(g - (g[1] - g[0]) / 2, g[-1] + (g[1] - g[0]) / 2), density=True)
    from scipy.ndimage import gaussian_filter1d
    return g, np.maximum(gaussian_filter1d(h, 5), 1e-12)


def _prior_at(x, prior_kde):
    return np.maximum(prior_kde(x[:, :N_OVERLAP]), 1e-300)


def ln_evidence_pop(e, ln_pi_det):
    """ln int pi_det(D) l(D) dD on the ln D grid (pi_det as a density in ln D)."""
    return float(logsumexp(e.lnl_d + ln_pi_det) + np.log(LND[1] - LND[0]))


ZS_EDGES = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.5, 6.0])


def cells_from_lenses(lens):
    """Lens cells: (theta_E, z_l) grid with z_s marginalised when z_s is unknown, and a (theta_E, z_l, z_s) grid when the
    source redshift is measured (cell centres; theta_E defaults to 1" when unmeasured)."""
    th = lens.theta_e.fillna(1.0).clip(THETA_GRID[0], THETA_GRID[-1]).values
    zl = lens.z_l.fillna(0.5).clip(ZL_GRID[0], ZL_GRID[-1]).values
    it = np.abs(np.log(th)[:, None] - np.log(THETA_GRID)[None, :]).argmin(1)
    iz = np.abs(zl[:, None] - ZL_GRID[None, :]).argmin(1)
    ng = len(THETA_GRID) * len(ZL_GRID)
    cell = it * len(ZL_GRID) + iz
    grid = [dict(theta_e=t, z_l=z, z_s=np.nan, sigma_v=np.nan) for t in THETA_GRID for z in ZL_GRID]
    zs = lens.z_s.values
    known = np.isfinite(zs)
    ks = np.clip(np.searchsorted(ZS_EDGES, zs[known]) - 1, 0, len(ZS_EDGES) - 2)
    cell[known] = ng + cell[known] * (len(ZS_EDGES) - 1) + ks
    zs_c = 0.5 * (ZS_EDGES[1:] + ZS_EDGES[:-1])
    grid3 = [dict(theta_e=t, z_l=z, z_s=s, sigma_v=np.nan) for t in THETA_GRID for z in ZL_GRID for s in zs_c]
    cells = grid + grid3
    used = np.unique(cell)                                   # keep only occupied cells (and the full 2D grid)
    keep = np.union1d(np.arange(ng), used)
    remap = {c: k for k, c in enumerate(keep)}
    return np.array([remap[c] for c in cell]), [cells[c] for c in keep]


def population_cell_weights(n=200000, seed=3):
    from gwlens.population import (einstein_radius_arcsec, sample_lens_redshift, sample_lensed_source_redshift,
                                   sample_velocity_dispersion)
    rng = np.random.default_rng(seed)
    zs = sample_lensed_source_redshift(rng, n // 20)
    zl = sample_lens_redshift(rng, zs)
    sv = sample_velocity_dispersion(rng, len(zs))
    th = einstein_radius_arcsec(sv, zl, zs)
    ok = (th >= THETA_GRID[0] / 1.1) & (th <= THETA_GRID[-1] * 1.1)
    it = np.abs(np.log(th[ok])[:, None] - np.log(THETA_GRID)[None, :]).argmin(1)
    iz = np.abs(zl[ok][:, None] - ZL_GRID[None, :]).argmin(1)
    w = np.bincount(it * len(ZL_GRID) + iz, minlength=len(THETA_GRID) * len(ZL_GRID)).astype(float)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
_G = {}


def _init_ms(events, ref):
    _G['events'], _G['ref'] = events, ref


def _ms_pair(pair):
    return ln_overlap_ms(_G['events'][pair[0]], _G['events'][pair[1]], _G['ref'])


def _init(events, cells, ln_e, ln_pu_args, geometry=None):
    _G.update(events=events, cells=cells, ln_e=ln_e, ln_pu_args=ln_pu_args,
              geom=LensGeometry(geometry) if geometry is not None else None)


def _lambda_row(pair):
    i, j = pair
    ev, cells = _G['events'], _G['cells']
    a, b = ev[i], ev[j]
    dt = b.t - a.t
    T = _G['ln_pu_args']
    ln_pu = np.log(2 * max(T - dt, 1.0) / T ** 2)
    out = np.full(len(cells), -np.inf)
    for c, cell in enumerate(cells):
        if _G['geom'] is not None:
            lam, _ = pair_likelihood_geom(dt, cell['theta_e'], cell['z_l'], cell['z_s'], a.loglike_d, b.loglike_d,
                                          _G['geom'], n_zs=40, sigma_v=cell['sigma_v'])
        else:
            lam, _ = pair_likelihood(dt, cell['theta_e'], cell['z_l'], cell['z_s'], a.loglike_d, b.loglike_d, n_zs=60,
                                     sigma_v=cell['sigma_v'])
        if lam > 0:
            out[c] = np.log(lam) - ln_pu - _G['ln_e'][i] - _G['ln_e'][j]
    return out


class _Shifted:
    """An event with another arrival time (for time shuffles); everything else is shared."""

    def __init__(self, e, t):
        self.__dict__.update(e.__dict__)
        self.t = t

    def loglike_d(self, d):
        return np.interp(np.log(d), LND, self.lnl_d, left=-np.inf, right=-np.inf)


def _shift(e, t):
    return _Shifted(e, t)


def _lambda_row_ordered(pair):
    """Lambda for a pair whose time order may have changed (shuffles): evaluate with the earlier event first."""
    i, j = pair
    ev = _G['events']
    return _lambda_row((i, j) if ev[i].t <= ev[j].t else (j, i))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pe-dir', type=Path, required=True)
    p.add_argument('--alerts-dir', type=Path)
    p.add_argument('--cand-dirs', type=Path, nargs='*', help='lensearch.fetch_candidates outputs (one per release)')
    p.add_argument('--lenses', type=Path, required=True, help='parquet/csv with ra, dec, theta_e, z_l, z_s, sigma_v')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n-rot', type=int, default=200)
    p.add_argument('--ms-min', type=float, default=-2.0, help='lens terms only for pairs with ln B_ms above this')
    p.add_argument('--sky-min', type=float, default=-3.0, help='... and ln B_sky above this')
    p.add_argument('--sky-min-nomass', type=float, default=0.5, help='ln B_sky cut for pairs without mass information')
    p.add_argument('--n-shuffle', type=int, default=0, help='time shuffles for the background of ln B_pop')
    p.add_argument('--shuffle-top', type=int, default=5000, help='shuffles use the pairs with the largest ln B_ms + ln B_sky')
    p.add_argument('--npool', type=int, default=48)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--high-grade-only', action='store_true')
    p.add_argument('--geometry', type=Path, help='EPL+shear lens library (lensearch.inject library); default: SIS relations')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    # events (cached: loading ~850 events with sky maps takes ~30 min)
    import pickle
    names = sorted(f.stem for f in a.pe_dir.glob('GW*.json'))
    ev_cache = a.out / 'events_cache.pkl'
    key = (names, [str(d) for d in (a.cand_dirs or [])], str(a.alerts_dir), EVENT_CACHE_VERSION)
    if ev_cache.exists():
        k0, cached = pickle.loads(ev_cache.read_bytes())
        events = cached if k0 == key else None
    else:
        events = None
    if events is not None:
        print('events read from cache', flush=True)
    else:
        events = [load_pe_event(a.pe_dir, n) for n in names]
        rng = np.random.default_rng(a.seed + 1)
        for cdir in a.cand_dirs or []:
            cand_tab = pd.read_csv(cdir / 'candidates.csv')
            cand_tab = cand_tab[cand_tab.skymap_found & (cand_tab.p_BBH > 0.5 * cand_tab.p_astro)]
            cand_tab['release'] = cdir.name
            for r in cand_tab.itertuples():
                try:
                    events.append(load_search_candidate(cdir / 'skymaps' / r.skymap_local, r, rng))
                except Exception as ex:
                    print(json.dumps(dict(candidate=r.gw_name, error=repr(ex))), flush=True)
        if a.alerts_dir:
            al = pd.read_csv(a.alerts_dir / 'superevents.csv')
            al = al[~al.retracted & al.skymap.notna()]
            events += [load_alert(a.alerts_dir, r.superevent, r.t_0) for r in al.itertuples()]
        ev_cache.write_bytes(pickle.dumps((key, events)))
    pe_times = np.array([e.t for e in events if e.kind == 'pe'])
    dup = [e.name for e in events if e.kind != 'pe' and np.min(np.abs(pe_times - e.t)) < 1.0]
    if dup:
        print(f'dropping {len(dup)} alerts/candidates that duplicate events with PE: {dup}', flush=True)
    events = [e for e in events if e.name not in set(dup)]
    events.sort(key=lambda e: e.t)
    n = len(events)
    T = events[-1].t - events[0].t
    kinds = pd.Series([e.kind for e in events]).value_counts().to_dict()
    print(f'{n} events {kinds}, span {T / 86400 / 365.25:.1f} yr', flush=True)

    # detected apparent-distance distribution: mean of the PE distance posteriors (density in ln D)
    post = np.mean([np.exp(e.lnl_d + log_dl_prior(np.exp(LND))) / np.sum(np.exp(e.lnl_d + log_dl_prior(np.exp(LND))))
                    for e in events if e.kind == 'pe'], axis=0)
    ln_pi_det = np.log(np.maximum(post / (post.sum() * (LND[1] - LND[0])), 1e-300))
    ln_e = np.array([ln_evidence_pop(e, ln_pi_det) for e in events])

    # sky and mass overlaps for all pairs
    S = np.vstack([e.sky for e in events]).astype(np.float64)
    ln_bsky = np.log(np.maximum(FOURPI * (S * PIXAREA) @ S.T, 1e-300))
    prior_kde = reference_prior_kde()
    with Pool(a.npool) as pool:
        xp = pool.starmap(_prior_at, [(e.x, prior_kde) for e in events if e.kde_x is not None])
    it = iter(xp)
    for e in events:
        e.x_prior = next(it) if e.kde_x is not None else None
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    cache = a.out / 'ln_bms_cache.npz'
    names_now = np.array([e.name for e in events])
    if cache.exists() and np.array_equal(np.load(cache)['names'], names_now):
        ln_bms = list(np.load(cache)['ln_bms'])
        print('mass-spin overlaps read from cache', flush=True)
    else:
        ref = reference_lnmc_density()
        with Pool(a.npool, initializer=_init_ms, initargs=(events, ref)) as pool:
            ln_bms = pool.map(_ms_pair, pairs, chunksize=256)
        np.savez(cache, names=names_now, ln_bms=np.array(ln_bms))
    tab = pd.DataFrame(dict(a=[events[i].name for i, _ in pairs], b=[events[j].name for _, j in pairs],
                            i=[i for i, _ in pairs], j=[j for _, j in pairs],
                            dt_days=[(events[j].t - events[i].t) / 86400 for i, j in pairs],
                            ln_bms=ln_bms, ln_bsky=[ln_bsky[i, j] for i, j in pairs],
                            kind=[events[i].kind + '-' + events[j].kind for i, j in pairs]))
    has_mass = np.array([e.kde_x is not None or _has_template(e) for e in events])
    no_mass = np.array([not (has_mass[i] and has_mass[j]) for i, j in pairs])
    keep = (tab.ln_bms > a.ms_min) & (tab.ln_bsky > a.sky_min) & (~no_mass | (tab.ln_bsky > a.sky_min_nomass))
    cand = tab[keep].reset_index(drop=True)
    print(f'{len(tab)} pairs; {len(cand)} pass ln B_ms > {a.ms_min} and ln B_sky > {a.sky_min}', flush=True)

    # lens catalog and cells
    lens = pd.read_parquet(a.lenses) if str(a.lenses).endswith('.parquet') else pd.read_csv(a.lenses)
    for c in ('theta_e', 'z_l', 'z_s', 'sigma_v'):
        if c not in lens:
            lens[c] = np.nan
    lens = lens[np.isfinite(lens.ra) & np.isfinite(lens.dec)]
    if 'type' in lens:                                         # galaxy-scale SIS model; clusters need their own maps
        lens = lens[lens.type != 'cluster']
    if a.high_grade_only:
        lens = lens[lens.high_grade]
    lens = lens.reset_index(drop=True)
    cell_of_lens, cells = cells_from_lenses(lens)
    th_use = lens.theta_e.fillna(1.0).values
    ln_pi_l = 2 * np.log(th_use) - np.log(np.sum(th_use ** 2))
    w_pop = population_cell_weights()
    print(f'{len(lens)} lenses, {len(cells)} cells', flush=True)

    # Lambda for every candidate pair and cell
    lam_cache = a.out / 'lambda_cells.npy'
    if lam_cache.exists() and np.load(lam_cache, mmap_mode='r').shape == (len(cand), len(cells)):
        lam = np.load(lam_cache)
        print('lens terms read from cache', flush=True)
    else:
        with Pool(a.npool, initializer=_init, initargs=(events, cells, ln_e, T, a.geometry)) as pool:
            lam = np.array(pool.map(_lambda_row, list(zip(cand.i, cand.j)), chunksize=4))
        np.save(lam_cache, lam)
    ng = len(THETA_GRID) * len(ZL_GRID)
    cand['ln_lambda_pop'] = logsumexp(lam[:, :ng] + np.log(np.maximum(w_pop, 1e-300))[None, :], axis=1)
    cand['ln_bpop'] = cand.ln_bms + cand.ln_bsky + cand.ln_lambda_pop
    cand['ln_bms_sky'] = cand.ln_bms + cand.ln_bsky

    # lens term with the catalogue at its true position (rotation 0) and rotated in right ascension (background),
    # computed in chunks of pairs: J (pairs x pixels, joint sky density) @ N (pixels x cells, catalogue weights)
    S_l = np.vstack([hp.ud_grade(e.sky.astype(np.float64), NSIDE_LENS) for e in events]).astype(np.float32)
    lam_max = lam.max(axis=1)
    lam_max = np.where(np.isfinite(lam_max), lam_max, 0.0)
    lam_lin = np.exp(lam - lam_max[:, None]).astype(np.float32)   # per-pair scaling for numerical range (-inf -> 0)
    rng = np.random.default_rng(a.seed)
    shifts = np.concatenate([[0.0], rng.uniform(10, 350, a.n_rot)])
    theta_l = np.radians(90 - lens.dec.values)
    w_l = np.exp(ln_pi_l).astype(np.float32)
    N_rot = [sparse.csr_matrix((w_l, (hp.ang2pix(NSIDE_LENS, theta_l, np.radians((lens.ra.values + sh) % 360)), cell_of_lens)),
                               shape=(NPIX_LENS, len(cells)), dtype=np.float32) for sh in shifts]
    out_rot = np.full((len(shifts), len(cand)), -np.inf, dtype=np.float32)
    ci, cj = cand.i.values, cand.j.values
    for c0 in range(0, len(cand), CHUNK):
        rows, cols, vals = [], [], []
        for k in range(c0, min(c0 + CHUNK, len(cand))):
            jm = S_l[ci[k]] * S_l[cj[k]]
            order = np.argsort(jm)[::-1]
            csum = np.cumsum(jm[order])
            nz = order[:np.searchsorted(csum, JOINT_MASS * csum[-1]) + 1]
            rows.append(np.full(len(nz), k - c0)), cols.append(nz), vals.append(jm[nz] * FOURPI ** 2)
        J = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(len(rows), NPIX_LENS), dtype=np.float32)
        sl = slice(c0, c0 + len(rows))
        for r, N in enumerate(N_rot):
            s = np.asarray((J @ N).multiply(lam_lin[sl]).sum(axis=1)).ravel()
            with np.errstate(divide='ignore'):
                out_rot[r, sl] = np.log(s) + lam_max[sl] + cand.ln_bms.values[sl]
        print(f'lens term: pairs {c0}-{c0 + len(rows)} of {len(cand)} ({J.nnz} sky entries)', flush=True)
    cand['ln_bcat'] = out_rot[0]
    bg = out_rot[1:]
    np.save(a.out / 'background_ln_bcat.npy', bg)
    bg_max = np.nanmax(np.where(np.isfinite(bg), bg, -np.inf), axis=1)
    cand['p_chance'] = [(bg_max >= x).mean() if np.isfinite(x) else 1.0 for x in cand.ln_bcat]

    if a.n_shuffle:
        ng = len(THETA_GRID) * len(ZL_GRID)
        grid_cells = cells[:ng]
        times0 = np.array([e.t for e in events])
        top = cand.nlargest(min(a.shuffle_top, len(cand)), columns='ln_bms_sky')
        rng_s = np.random.default_rng(a.seed + 5)
        best_bg = []
        for s in range(a.n_shuffle):
            perm = rng_s.permutation(n)
            shuffled = [_shift(e, times0[perm[k]]) for k, e in enumerate(events)]
            with Pool(a.npool, initializer=_init, initargs=(shuffled, grid_cells, ln_e, T, a.geometry)) as pool:
                lam_s = np.array(pool.map(_lambda_row_ordered, list(zip(top.i, top.j)), chunksize=16))
            lp = logsumexp(lam_s + np.log(np.maximum(w_pop, 1e-300))[None, :], axis=1)
            best_bg.append(float(np.nanmax(top.ln_bms.values + top.ln_bsky.values + lp)))
            print(f'shuffle {s}: max ln B_pop {best_bg[-1]:.2f}', flush=True)
        np.save(a.out / 'background_ln_bpop_shuffle.npy', np.array(best_bg))
        cand['p_chance_pop'] = [(np.array(best_bg) >= x).mean() for x in cand.ln_bpop]
    best = cand.ln_bcat.idxmax() if np.isfinite(cand.ln_bcat).any() else None
    tab.to_parquet(a.out / 'pairs_all.parquet', index=False)
    cand.sort_values('ln_bcat', ascending=False).to_csv(a.out / 'pairs_ranked.csv', index=False)
    lens.assign(cell=cell_of_lens).to_parquet(a.out / 'lenses_used.parquet', index=False)
    json.dump(dict(n_events=n, n_pairs=len(tab), n_candidates=len(cand), n_lenses=len(lens), n_rot=a.n_rot,
                   top=cand.loc[best].to_dict() if best is not None else None), open(a.out / 'summary.json', 'w'), default=str,
              indent=1)
    print(cand.sort_values('ln_bcat', ascending=False).head(15).to_string(), flush=True)


if __name__ == '__main__':
    main()
