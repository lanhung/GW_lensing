"""Joint GW + EM Bayesian inference of the absolute magnification of a lensed GW image pair.

The electromagnetic data are the observed positions of all resolved images relative to the
lens-light centre, the light axis ratio and position angle, the stellar velocity dispersion and
the lens and source redshifts (the observation model of gwlens.catalogue). The lens model is an
elliptical power law (or one of its special cases) plus external shear, an external convergence
kappa_ext (mass-sheet degeneracy) and the source position, which is marginalised analytically
in the linearised image-plane likelihood

    -2 ln L_pos = sum_i |A_i^-1 (beta_i - beta)|^2 / sigma_img^2 + const,

where beta_i is image i traced back to the source plane and A_i the lensing Jacobian. Predicted
image positions are refined by one Newton step before the magnifications, Morse indices and
Fermat potentials are evaluated. The GW data enter through

    ln(mu_B / mu_A)          (amplitude ratio),
    Morse indices n_A, n_B   (waveform phase; identified with higher modes),
    D_A = d_L / sqrt|mu_A|   (apparent distance; only with a cosmology),
    Delta t = (1 - kappa_ext) D_dt Delta phi   (only its sign without a cosmology).

Cosmology options
    free  no cosmology: the GW amplitude constrains only mu_B/mu_A and the arrival order is
          used; kappa_ext ~ N(0, sigma_kext) sets the mass-sheet uncertainty. This is the
          setting of the paper's main benchmark.
    lcdm  the simulation cosmology with the observed redshifts: d_L(z_s) makes the GW apparent
          distance a direct magnification measurement, and the millisecond-precise delay fixes
          (1 - kappa_ext) Delta phi; kappa_ext is then derived from Delta t (requires
          sigma_kext > 0). With --h0-free, h_rel = H0 / H0_sim is sampled (flat in [0.5, 1.5]) and
          all distances scale as 1/h_rel at fixed redshift. Both the delay and the apparent distance
          then depend on (1 - kappa_ext) / h_rel, so the kappa_ext prior sets the per-event floor.
          The EPL slope is degenerate with h_rel; --gamma-prior MEAN SD replaces the intrinsic
          population prior N(2.08, 0.15) by, e.g., the slope distribution of detected systems.

Sub-commands
    prepare  write self-contained per-event JSON inputs from gwlens.catalogue catalogues (the
             catalogue chunk is replayed to recover every image and the true lens parameters).
    posthoc  EM lens model x GW likelihood summarised from a gwlens.joint_pe posterior
             (seconds to minutes per event). Exact up to the Gaussian/discrete summary of the
             GW likelihood, because in geometric optics the GW likelihood depends on the lens
             only through (mu_A, mu_B, Delta t, n_A, n_B).
    full     simultaneous sampling of the strain likelihood of both images (IMRPhenomXHM,
             relative binning) and the EM lens model. Expensive; intended for a cluster
             (one event per node, see scripts/slurm_gwem.sh).

Example
    python -m gwlens.joint_gwem prepare --catalogue data/pe/EPL_shear.parquet --scenario ET \\
        --event-ids EPL_shear-7777-0005239 --out-dir inputs/gwem
    python -m gwlens.joint_gwem posthoc --event inputs/gwem/ET_EPL_shear-7777-0005239.json \\
        --gw-posterior results/pe/ET/ET_EPL_shear-7777-0005239_posterior.parquet --out-dir results/gwem
    python -m gwlens.joint_gwem full --event inputs/gwem/ET_EPL_shear-7777-0005239.json \\
        --out-dir results/gwem_full --npool 32
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import population as pop

LENS_MODELS = ['SIS', 'SIE', 'SIE_shear', 'EPL_shear']
TWO_PI = 2 * np.pi


# ---------------------------------------------------------------------------
# Event inputs
# ---------------------------------------------------------------------------
def _jsonable(v):
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def prepare_event(catalogue, event_id, scenario):
    """Self-contained event description: catalogue row, EM data and truths."""
    from .catalogue import replay
    row = replay(catalogue, event_id)
    if not row.get(f'{scenario}__selected'):
        raise ValueError(f'{event_id} is not a selected {scenario} pair')
    em = row.pop('em')
    a_i, b_i = em['pairs'][scenario]
    keep = [k for k in range(len(em['x'])) if em['bright'][k] or k in (a_i, b_i)]
    idx = {k: n for n, k in enumerate(keep)}
    data = dict(x=[em['x'][k] for k in keep], y=[em['y'][k] for k in keep], gw_a=idx[a_i], gw_b=idx[b_i],
                sigma_img=em['obs']['sigma_img'], sigma_center=em['obs']['sigma_center'],
                q_light=row['obs_q_light'], sigma_q_light=em['obs']['sigma_q_light'],
                pa_light=row['obs_pa_light'], sigma_pa_light=np.deg2rad(em['obs']['sigma_pa_light_deg']),
                sigma_v=row['obs_sigma_v'], sigma_v_frac=em['obs']['sigma_vdisp'],
                z_l=row['obs_z_l'], z_s=row['obs_z_s'], light_pa_informative=row['family'] != 'SIS')
    truth = dict(em['lens'], mu=[em['mu'][k] for k in keep], morse=[em['morse'][k] for k in keep],
                 t=[em['t'][k] for k in keep], mu_a=abs(em['mu'][a_i]), mu_b=abs(em['mu'][b_i]), d_l=row['true_dl'],
                 kappa_ext=0.0)
    return _jsonable(dict(event_id=event_id, family=row['family'], scenario=scenario, em=data, truth=truth, row=row))


def degrade_astrometry(event, sigma_img_gw, seed):
    """Astrometry sensitivity test: give the GW-source images a positional error sigma_img_gw (arcsec).

    The image positions of the BBH need not coincide with directly observed host features; they are
    reconstructed with an uncertainty that may exceed the 0.01-arcsec default. The observed GW-image
    positions receive the extra scatter sqrt(sigma_img_gw^2 - sigma_img^2), so that data and
    likelihood stay consistent; other host images keep sigma_img.
    """
    em = dict(event['em'])
    extra = np.sqrt(max(sigma_img_gw ** 2 - em['sigma_img'] ** 2, 0.0))
    rng = np.random.default_rng(seed)
    x, y = list(em['x']), list(em['y'])
    for k in (em['gw_a'], em['gw_b']):
        x[k] += float(rng.normal(0, extra))
        y[k] += float(rng.normal(0, extra))
    em.update(x=x, y=y, sigma_img_gw=float(sigma_img_gw))
    return dict(event, em=em)


# ---------------------------------------------------------------------------
# Lens model
# ---------------------------------------------------------------------------
class LensForward:
    """Lens-model likelihood of the image positions and derived image properties."""

    def __init__(self, model, em, exact_positions=False):
        from lenstronomy.LensModel.lens_model import LensModel
        self.model, self.exact_positions = model, exact_positions
        names = ['SIS'] if model == 'SIS' else ['EPL_NUMBA'] + ([] if model == 'SIE' else ['SHEAR'])
        self.lm = LensModel(names)
        self.x, self.y = np.asarray(em['x'], float), np.asarray(em['y'], float)
        sig = np.full(len(self.x), float(em['sigma_img']))
        if em.get('sigma_img_gw'):
            sig[[em['gw_a'], em['gw_b']]] = em['sigma_img_gw']
        self.s2 = sig ** 2
        self.em = em
        self.theta_e_per_sigma2 = float(pop.einstein_radius_arcsec(1.0, em['z_l'], em['z_s']))

    def kwargs(self, p):
        te = self.theta_e_per_sigma2 * p['sigma_v'] ** 2
        cx, cy = p['center_x'], p['center_y']
        if self.model == 'SIS':
            return [dict(theta_E=te, center_x=cx, center_y=cy)]
        e = (1 - p['q']) / (1 + p['q'])
        kw = [dict(theta_E=te, gamma=p.get('gamma', 2.0), e1=e * np.cos(2 * p['phi_q']), e2=e * np.sin(2 * p['phi_q']),
                   center_x=cx, center_y=cy)]
        if self.model != 'SIE':
            g = np.exp(p['ln_gamma_ext'])
            kw.append(dict(gamma1=g * np.cos(2 * p['phi_ext']), gamma2=g * np.sin(2 * p['phi_ext']), ra_0=cx, dec_0=cy))
        return kw

    def _jacobian(self, x, y, kw):
        fxx, fxy, fyx, fyy = self.lm.hessian(x, y, kw)
        a11, a12, a21, a22 = 1 - fxx, -fxy, -fyx, 1 - fyy
        return a11, a12, a21, a22, a11 * a22 - a12 * a21

    def _exact_images(self, p, kw, sx, sy, W11, W12, W22, detW):
        """Image-plane likelihood without linearisation, for an explicitly sampled source position.

        The source position is beta = beta_lin + L u, where beta_lin and the Cholesky factor L of its
        covariance come from the linearised likelihood and u = (u_x, u_y) is sampled uniformly; log|det L|
        makes the implied prior flat in beta. Each model image is found by Newton iteration of the lens
        equation from the observed position, and the likelihood compares model and observed positions.
        """
        c11, c12, c22 = W22 / detW, -W12 / detW, W11 / detW
        l11 = np.sqrt(c11)
        l21 = c12 / l11
        l22 = np.sqrt(max(c22 - l21 ** 2, 1e-300))
        bx, by = sx + l11 * p['u_x'], sy + l21 * p['u_x'] + l22 * p['u_y']
        xs, ys = self.x.copy(), self.y.copy()
        for _ in range(40):
            ax, ay = self.lm.alpha(xs, ys, kw)
            rx, ry = xs - ax - bx, ys - ay - by
            if np.max(np.abs(rx)) < 1e-10 and np.max(np.abs(ry)) < 1e-10:
                break
            a11, a12, a21, a22, det = self._jacobian(xs, ys, kw)
            if np.any(np.abs(det) < 1e-12):
                return None
            xs, ys = xs - (a22 * rx - a12 * ry) / det, ys - (-a21 * rx + a11 * ry) / det
        else:
            return None
        if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
            return None
        sep = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :]) + np.eye(len(xs))
        if np.min(sep) < 1e-4:
            return None  # two observed images mapped onto the same model image
        chi2 = float(np.sum(((xs - self.x) ** 2 + (ys - self.y) ** 2) / self.s2))
        log_l = -0.5 * chi2 - np.sum(np.log(TWO_PI * self.s2)) + np.log(l11 * l22)
        return log_l, xs, ys, bx, by

    def evaluate(self, p):
        """Log-likelihood of the EM data and the model image properties (None if invalid)."""
        kw = self.kwargs(p)
        x, y = self.x, self.y
        ax, ay = self.lm.alpha(x, y, kw)
        bx, by = x - ax, y - ay
        a11, a12, a21, a22, det = self._jacobian(x, y, kw)
        if np.any(np.abs(det) < 1e-8) or not np.all(np.isfinite(det)):
            return None
        m11, m12, m21, m22 = a22 / det, -a12 / det, -a21 / det, a11 / det
        w11, w12, w22 = (m11 ** 2 + m21 ** 2) / self.s2, (m11 * m12 + m21 * m22) / self.s2, (m12 ** 2 + m22 ** 2) / self.s2
        W11, W12, W22 = w11.sum(), w12.sum(), w22.sum()
        detW = W11 * W22 - W12 ** 2
        if not detW > 0:
            return None
        c1, c2 = (w11 * bx + w12 * by).sum(), (w12 * bx + w22 * by).sum()
        sx, sy = (W22 * c1 - W12 * c2) / detW, (W11 * c2 - W12 * c1) / detW
        dx, dy = bx - sx, by - sy
        chi2 = float(np.sum(w11 * dx ** 2 + 2 * w12 * dx * dy + w22 * dy ** 2))
        log_l = -0.5 * chi2 - 0.5 * np.log(detW) + np.log(TWO_PI) - np.sum(np.log(TWO_PI * self.s2))
        if self.exact_positions:
            sol = self._exact_images(p, kw, sx, sy, W11, W12, W22, detW)
            if sol is None:
                return None
            log_l, xs, ys, sx, sy = sol
        else:
            # One Newton step to the model image positions, then image properties there.
            xs, ys = x - (m11 * dx + m12 * dy), y - (m21 * dx + m22 * dy)
        a11, a12, a21, a22, det = self._jacobian(xs, ys, kw)
        kappa = 1 - 0.5 * (a11 + a22)
        morse = np.where(det < 0, 0.5, np.where(kappa > 1, 1.0, 0.0))
        fermat = self.lm.fermat_potential(xs, ys, kw, sx, sy)
        # Light-profile and kinematic constraints.
        em = self.em
        if self.model != 'SIS':
            log_l += -0.5 * ((em['q_light'] - p['q']) / em['sigma_q_light']) ** 2
            if em.get('light_pa_informative', True):
                dpa = (em['pa_light'] - p['phi_q'] + np.pi / 2) % np.pi - np.pi / 2
                log_l += -0.5 * (dpa / em['sigma_pa_light']) ** 2
        sv = p['sigma_v']
        log_l += -0.5 * ((em['sigma_v'] - sv) / (em['sigma_v_frac'] * sv)) ** 2 - np.log(sv)
        return dict(log_l=float(log_l), mu=1 / det, morse=morse, fermat=np.asarray(fermat), beta=(sx, sy),
                    theta_e=kw[0]['theta_E'])


def lens_priors(model, em, exact_positions=False, gamma_prior=None):
    import bilby
    from bilby.core.prior import Gaussian, Interped, TruncatedGaussian, Uniform
    pr = bilby.core.prior.PriorDict()
    if exact_positions:  # source position relative to its linearised estimate, in units of its uncertainty
        pr['u_x'] = Uniform(-8.0, 8.0, name='u_x')
        pr['u_y'] = Uniform(-8.0, 8.0, name='u_y')
    s = np.linspace(100.0, 400.0, 2000)
    sstar, alpha, beta = 161.0, 2.32, 2.67
    pdf = (s / sstar) ** alpha * np.exp(-(s / sstar) ** beta) / s * s ** 4
    pr['sigma_v'] = Interped(s, pdf, minimum=100.0, maximum=400.0, name='sigma_v', latex_label=r'$\sigma_v$')
    pr['center_x'] = Gaussian(0.0, em['sigma_center'], name='center_x')
    pr['center_y'] = Gaussian(0.0, em['sigma_center'], name='center_y')
    if model != 'SIS':
        pr['q'] = TruncatedGaussian(0.75, 0.15, 0.35, 0.99, name='q')
        pr['phi_q'] = Uniform(0, np.pi, boundary='periodic', name='phi_q')
    if model in ('SIE_shear', 'EPL_shear'):
        pr['ln_gamma_ext'] = TruncatedGaussian(np.log(0.05), 0.6, np.log(1e-3), np.log(0.25), name='ln_gamma_ext')
        pr['phi_ext'] = Uniform(0, np.pi, boundary='periodic', name='phi_ext')
    if model == 'EPL_shear':
        g_mu, g_sd = gamma_prior or (2.08, 0.15)
        pr['gamma'] = TruncatedGaussian(g_mu, g_sd, 1.6, 2.6, name='gamma')
    return pr


# ---------------------------------------------------------------------------
# Combination with the GW data
# ---------------------------------------------------------------------------
class Combiner:
    """Maps a lens evaluation and an image assignment to GW-side quantities and prior weights."""

    def __init__(self, em, cosmology, sigma_kext, assignment):
        self.em, self.cosmology, self.sigma_kext = em, cosmology, sigma_kext
        n = len(em['x'])
        self.pairs = [(em['gw_a'], em['gw_b'])] if assignment == 'known' else \
            [(i, j) for i in range(n) for j in range(n) if i != j]
        self.d_l = float(pop.luminosity_distance(em['z_s']))
        self.ddt = float(pop.time_delay_distance_seconds(em['z_l'], em['z_s'])) * pop.ARCSEC ** 2
        if cosmology == 'lcdm' and sigma_kext <= 0:
            raise ValueError('cosmology=lcdm needs sigma_kext > 0 (kappa_ext is derived from the delay)')

    def image_pair(self, ev, pair, delta_t, kappa_ext=0.0, h_rel=1.0):
        """GW-side quantities of an assignment, plus the log prior weight of derived kappa_ext."""
        i, j = self.pairs[pair]
        dphi = ev['fermat'][j] - ev['fermat'][i]
        log_w = 0.0
        ddt = self.ddt / h_rel
        if self.cosmology == 'lcdm':
            one_minus_k = delta_t / (ddt * dphi)
            if not one_minus_k > 0:
                return None
            kappa_ext = 1 - one_minus_k
            log_w = -0.5 * (kappa_ext / self.sigma_kext) ** 2 - np.log(ddt * abs(dphi))
        elif np.sign(dphi) != np.sign(delta_t):
            return None  # arrival order must agree with the Fermat potential
        mu_a = abs(ev['mu'][i]) / (1 - kappa_ext) ** 2
        mu_b = abs(ev['mu'][j]) / (1 - kappa_ext) ** 2
        out = dict(i=i, j=j, mu_a=mu_a, mu_b=mu_b, ln_mu_rel=float(np.log(mu_b / mu_a)), n_a=float(ev['morse'][i]),
                   n_b=float(ev['morse'][j]), kappa_ext=float(kappa_ext), log_w=float(log_w), dphi=float(dphi))
        if self.cosmology == 'lcdm':
            out['d_a'] = self.d_l / h_rel / np.sqrt(mu_a)
        return out


class GWSummary:
    """GW likelihood in (ln mu_rel, ln D_A, n_A, n_B) summarised from a joint_pe posterior.

    The posterior is divided by the joint_pe priors (uniform in ln mu_rel, p(D_A) ~ D_A^2,
    uniform Morse indices): a Gaussian in (ln mu_rel, ln D_A) fitted with importance weights
    D_A^-3 times the posterior probabilities of the nine Morse combinations.
    """

    def __init__(self, post):
        x = np.column_stack([post.ln_mu_rel, np.log(post.luminosity_distance)])
        w = np.exp(-3 * (x[:, 1] - x[:, 1].mean()))
        w /= w.sum()
        self.mean = w @ x
        d = x - self.mean
        self.cov = (w[:, None] * d).T @ d / (1 - np.sum(w ** 2))
        self.icov = np.linalg.inv(self.cov)
        self.sd_rel = np.sqrt(self.cov[0, 0])
        self.ln_frac = {}
        n = len(post)
        for na in (0.0, 0.5, 1.0):
            for nb in (0.0, 0.5, 1.0):
                k = np.sum(w[(post.morse_n_a.to_numpy() == na) & (post.morse_n_b.to_numpy() == nb)])
                self.ln_frac[(na, nb)] = float(np.log((k * n + 0.5) / (n + 4.5)))
        self.delta_t = float(np.median(post.delta_t))

    def log_l(self, g, use_distance):
        ll = self.ln_frac[(g['n_a'], g['n_b'])]
        if use_distance:
            d = np.array([g['ln_mu_rel'], np.log(g['d_a'])]) - self.mean
            return ll - 0.5 * d @ self.icov @ d - 0.5 * np.log(np.linalg.det(TWO_PI * self.cov))
        return ll - 0.5 * ((g['ln_mu_rel'] - self.mean[0]) / self.sd_rel) ** 2 - np.log(np.sqrt(TWO_PI) * self.sd_rel)


def _logsumexp(v):
    v = np.asarray(v)
    m = v.max()
    return m + np.log(np.sum(np.exp(v - m)))


def make_posthoc_likelihood(lens, comb, gw, sample_kext):
    import bilby

    class PosthocLikelihood(bilby.Likelihood):
        def __init__(self):
            super().__init__()

        def terms(self, p):
            ev = lens.evaluate(p)
            if ev is None:
                return None, []
            k = p.get('kappa_ext', 0.0) if sample_kext else 0.0
            out = []
            for pair in range(len(comb.pairs)):
                g = comb.image_pair(ev, pair, gw.delta_t, k, p.get('h_rel', 1.0))
                if g is not None:
                    out.append((g, ev['log_l'] + g['log_w'] + gw.log_l(g, comb.cosmology == 'lcdm')))
            return ev, out

        def log_likelihood(self, parameters=None):
            _, out = self.terms(parameters if parameters is not None else self.parameters)
            if not out:
                return -np.inf
            return float(_logsumexp([v for _, v in out]) - np.log(len(comb.pairs)))

    return PosthocLikelihood()


def make_full_likelihood(lens, comb, gw_like, sample_kext):
    import bilby

    class FullLikelihood(bilby.Likelihood):
        """Strain likelihood of both images times the EM lens-model likelihood."""

        def __init__(self):
            super().__init__()

        def joint_terms(self, p):
            ev = lens.evaluate(p)
            if ev is None:
                return None, None
            pair = int(np.clip(np.floor(p.get('pair_index', 0.0)), 0, len(comb.pairs) - 1))
            g = comb.image_pair(ev, pair, p['delta_t'], p.get('kappa_ext', 0.0) if sample_kext else 0.0,
                                p.get('h_rel', 1.0))
            return ev, g

        def log_likelihood_ratio(self, parameters=None):
            p = parameters if parameters is not None else self.parameters
            ev, g = self.joint_terms(p)
            if g is None:
                return -np.inf
            q = dict(p, ln_mu_rel=g['ln_mu_rel'], morse_a=g['n_a'] + 0.25, morse_b=g['n_b'] + 0.25)
            q['luminosity_distance'] = g['d_a'] if comb.cosmology == 'lcdm' else p['luminosity_distance']
            return gw_like.log_likelihood_ratio(parameters=q) + ev['log_l'] + g['log_w']

        def noise_log_likelihood(self):
            return gw_like.noise_log_likelihood()

        def log_likelihood(self, parameters=None):
            return self.log_likelihood_ratio(parameters) + self.noise_log_likelihood()

    return FullLikelihood()


# ---------------------------------------------------------------------------
# Derived quantities and summaries
# ---------------------------------------------------------------------------
def derived(post, lens, comb, delta_t_col, gw, sample_kext, rng):
    """Re-evaluate the lens model on posterior samples; draw the image assignment if marginalised."""
    rows = []
    for _, s in post.iterrows():
        p = s.to_dict()
        dt = p[delta_t_col] if delta_t_col else gw.delta_t
        k = p.get('kappa_ext', 0.0) if sample_kext else 0.0
        ev = lens.evaluate(p)
        cands = []
        for pair in range(len(comb.pairs)):
            if 'pair_index' in p and pair != int(np.clip(np.floor(p['pair_index']), 0, len(comb.pairs) - 1)):
                continue
            g = comb.image_pair(ev, pair, dt, k, p.get('h_rel', 1.0)) if ev is not None else None
            if g is not None:
                ll = ev['log_l'] + g['log_w'] + (gw.log_l(g, comb.cosmology == 'lcdm') if gw is not None else 0.0)
                cands.append((g, ll))
        if not cands:
            rows.append({})
            continue
        lls = np.array([c[1] for c in cands])
        pr = np.exp(lls - lls.max())
        g = cands[rng.choice(len(cands), p=pr / pr.sum())][0]
        rows.append(dict(mu_a=g['mu_a'], mu_b=g['mu_b'], ln_mu_a=np.log(g['mu_a']), ln_mu_rel_model=g['ln_mu_rel'],
                         morse_n_a=g['n_a'], morse_n_b=g['n_b'], kappa_ext_derived=g['kappa_ext'], gw_image_a=g['i'],
                         gw_image_b=g['j'], theta_e=ev['theta_e']))
    return pd.concat([post.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def summarise(post, event, extra):
    t = event['truth']
    y = np.log(t['mu_a'])
    x = post.ln_mu_a.dropna().to_numpy()
    s = dict(event_id=event['event_id'], family=event['family'], scenario=event['scenario'], n_samples=int(len(x)),
             true_mu_a=t['mu_a'], true_mu_b=t['mu_b'], ln_mu_a_median=float(np.median(x)), ln_mu_a_sd=float(np.std(x)),
             ln_mu_a_q05=float(np.quantile(x, 0.05)), ln_mu_a_q95=float(np.quantile(x, 0.95)),
             pit=float(np.mean(x < y)), rel_err=float(abs(np.exp(np.median(x)) - t['mu_a']) / t['mu_a']),
             assignment_correct=float(np.mean((post.gw_image_a == event['em']['gw_a']) & (post.gw_image_b == event['em']['gw_b']))),
             **extra)
    s['covered90'] = bool(s['ln_mu_a_q05'] <= y <= s['ln_mu_a_q95'])
    if 'gamma' in post:
        s.update(true_gamma=t['gamma'], gamma_median=float(post.gamma.median()), gamma_sd=float(post.gamma.std()))
    if 'h_rel' in post:
        h = post.h_rel.to_numpy()
        s.update(h_rel_median=float(np.median(h)), h_rel_sd=float(np.std(h)), h_rel_q05=float(np.quantile(h, 0.05)),
                 h_rel_q95=float(np.quantile(h, 0.95)), h_rel_pit=float(np.mean(h < 1.0)))
    return s


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
def _setup(event, a):
    em = event['em']
    model = a.lens_model or event['family']
    lens = LensForward(model, em, a.exact_positions)
    comb = Combiner(em, a.cosmology, a.sigma_kext, a.assignment)
    priors = lens_priors(model, em, a.exact_positions, a.gamma_prior)
    sample_kext = a.cosmology == 'free' and a.sigma_kext > 0
    if sample_kext:
        from bilby.core.prior import TruncatedGaussian
        priors['kappa_ext'] = TruncatedGaussian(0.0, a.sigma_kext, -0.5, 0.5, name='kappa_ext')
    if a.h0_free:
        if a.cosmology != 'lcdm':
            raise ValueError('--h0-free needs --cosmology lcdm')
        from bilby.core.prior import Uniform
        priors['h_rel'] = Uniform(0.5, 1.5, name='h_rel', latex_label=r'$H_0/H_0^{\rm sim}$')
    return model, lens, comb, priors, sample_kext


def _label(event, a, mode):
    tag = (f"_ast{a.sigma_img_gw:g}" if a.sigma_img_gw else '') + ('_h0' if a.h0_free else '') + \
        ('_xpos' if a.exact_positions else '') + (f"_g{a.gamma_prior[0]:g}" if a.gamma_prior else '')
    return f"{event['scenario']}_{event['event_id']}_{mode}_{a.cosmology}_{a.lens_model or event['family']}{tag}"


def _sampler_kwargs(a):
    kw = dict(sampler=a.sampler, nlive=a.nlive, npool=a.npool, resume=True, check_point_delta_t=600, plot=False,
              save='hdf5', conversion_function=None, seed=a.seed)
    if a.sampler == 'dynesty':
        kw.update(sample='acceptance-walk', naccept=a.naccept)
    return kw


def run_posthoc(event, gw_post, a):
    import bilby
    t0 = time.time()
    model, lens, comb, priors, sample_kext = _setup(event, a)
    gw = GWSummary(gw_post)
    like = make_posthoc_likelihood(lens, comb, gw, sample_kext)
    label = _label(event, a, 'posthoc')
    res = bilby.run_sampler(likelihood=like, priors=priors, outdir=str(a.out_dir), label=label, **_sampler_kwargs(a))
    post = derived(res.posterior, lens, comb, None, gw, sample_kext, np.random.default_rng(a.seed))
    post.to_parquet(a.out_dir / f'{label}_posterior.parquet', index=False)
    s = summarise(post, event, dict(mode='posthoc', cosmology=a.cosmology, lens_model=model, sigma_kext=a.sigma_kext,
                                    sigma_img_gw=a.sigma_img_gw, assignment=a.assignment, seconds=time.time() - t0,
                                    log_evidence=float(res.log_evidence)))
    (a.out_dir / f'{label}_summary.json').write_text(json.dumps(s, indent=1))
    return s


def run_full(event, a):
    import bilby
    from bilby.core.prior import LogUniform, Uniform
    from .joint_pe import event_seed, gw_likelihood, priors_for
    t0 = time.time()
    a.seed = event_seed(a.seed, event['event_id'])  # same noise as gwlens.joint_pe for this event
    model, lens, comb, priors, sample_kext = _setup(event, a)
    row = pd.Series(event['row'])
    gw_like, base, truth, duration, snrs = gw_likelihood(row, event['scenario'], a.seed)
    gpr = priors_for(truth)
    for k in ('ln_mu_rel', 'morse_a', 'morse_b'):
        gpr.pop(k)
    if a.cosmology == 'lcdm':
        gpr.pop('luminosity_distance')
    else:
        d = truth['luminosity_distance']
        gpr['luminosity_distance'] = LogUniform(0.1 * d, 10 * d, name='luminosity_distance')
    for k, v in gpr.items():
        priors[k] = v
    if len(comb.pairs) > 1:
        priors['pair_index'] = Uniform(0, len(comb.pairs), name='pair_index')
    like = make_full_likelihood(lens, comb, gw_like, sample_kext)
    label = _label(event, a, 'full')
    res = bilby.run_sampler(likelihood=like, priors=priors, outdir=str(a.out_dir), label=label, **_sampler_kwargs(a))
    post = derived(res.posterior, lens, comb, 'delta_t', None, sample_kext, np.random.default_rng(a.seed))
    if a.cosmology == 'free':
        post['luminosity_distance_source'] = post.luminosity_distance * np.sqrt(post.mu_a)
    post.to_parquet(a.out_dir / f'{label}_posterior.parquet', index=False)
    s = summarise(post, event, dict(mode='full', cosmology=a.cosmology, lens_model=model, sigma_kext=a.sigma_kext,
                                    sigma_img_gw=a.sigma_img_gw, assignment=a.assignment, seconds=time.time() - t0, snr_a=snrs[0], snr_b=snrs[1],
                                    duration=duration, log_bayes_factor=float(res.log_bayes_factor)))
    (a.out_dir / f'{label}_summary.json').write_text(json.dumps(s, indent=1))
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    pp = sub.add_parser('prepare', help='write per-event JSON inputs from a catalogue')
    pp.add_argument('--catalogue', type=Path, required=True)
    pp.add_argument('--scenario', required=True)
    pp.add_argument('--event-ids', nargs='+', required=True)
    pp.add_argument('--out-dir', type=Path, required=True)
    for name in ('posthoc', 'full'):
        q = sub.add_parser(name)
        q.add_argument('--event', type=Path, required=True, help='JSON from the prepare step')
        q.add_argument('--out-dir', type=Path, required=True)
        q.add_argument('--cosmology', choices=['free', 'lcdm'], default='free')
        q.add_argument('--sigma-kext', type=float, default=0.0,
                       help='prior width of the external convergence (0: no mass sheet, as in the simulations)')
        q.add_argument('--lens-model', choices=LENS_MODELS, default=None, help='default: the simulated family')
        q.add_argument('--assignment', choices=['known', 'marginalise'], default='known',
                       help='which EM images host the GW images: known, or summed over all ordered pairs')
        q.add_argument('--sigma-img-gw', type=float, default=0.0,
                       help='astrometry test: positional error (arcsec) of the GW-source images, with matching extra '
                            'scatter added to their observed positions (0: the simulated 0.01 arcsec)')
        q.add_argument('--exact-positions', action='store_true',
                       help='image-position likelihood without linearisation (source position sampled)')
        q.add_argument('--gamma-prior', type=float, nargs=2, metavar=('MEAN', 'SD'), default=None,
                       help='Gaussian prior on the EPL slope, truncated to [1.6, 2.6] (default: 2.08 0.15)')
        q.add_argument('--h0-free', action='store_true',
                       help='with --cosmology lcdm: sample h_rel = H0 / H0_sim, flat in [0.5, 1.5]')
        q.add_argument('--sampler', default='dynesty')
        q.add_argument('--nlive', type=int, default=500 if name == 'posthoc' else 300)
        q.add_argument('--naccept', type=int, default=30)
        q.add_argument('--npool', type=int, default=1 if name == 'posthoc' else 16)
        q.add_argument('--seed', type=int, default=1234)
        if name == 'posthoc':
            q.add_argument('--gw-posterior', type=Path, required=True, help='posterior parquet from gwlens.joint_pe')
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.cmd == 'prepare':
        cat = Path(a.catalogue)
        for eid in a.event_ids:
            ev = prepare_event(cat, eid, a.scenario)
            (a.out_dir / f'{a.scenario}_{eid}.json').write_text(json.dumps(ev))
            print(a.out_dir / f'{a.scenario}_{eid}.json', len(ev['em']['x']), 'images')
        return
    event = json.loads(a.event.read_text())
    if a.sigma_img_gw:
        from .joint_pe import event_seed
        event = degrade_astrometry(event, a.sigma_img_gw, event_seed(a.seed + 1, event['event_id']))
    if a.cmd == 'posthoc':
        s = run_posthoc(event, pd.read_parquet(a.gw_posterior), a)
    else:
        s = run_full(event, a)
    print(json.dumps({k: s[k] for k in ('event_id', 'true_mu_a', 'ln_mu_a_median', 'ln_mu_a_sd', 'rel_err', 'covered90', 'seconds')}))


if __name__ == '__main__':
    main()
