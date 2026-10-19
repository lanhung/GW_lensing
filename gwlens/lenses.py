"""Strong-lens image solutions for four galaxy-lens families.

Families
--------
SIS        singular isothermal sphere (analytic images)
SIE        singular isothermal ellipsoid, SLACS-like axis ratios
SIE_shear  SIE plus external shear
EPL_shear  elliptical power law (free radial slope) plus external shear

Sources are drawn uniformly in area inside a disk of radius ``r_max * theta_E`` around the lens
centre and redrawn (up to ``max_tries`` times) until the configuration is multiply imaged, so the
source is uniform over the multiple-image region of each lens and practically every lens is kept:
the population is weighted by lens size through the sigma_v^4 factor of the velocity-dispersion
distribution, but not by the dependence of the cross-section on ellipticity, shear or slope. Each
image carries its signed magnification, convergence, Morse index and arrival-time delay.
"""
import numpy as np

from .population import ARCSEC

FAMILIES = ['SIS', 'SIE', 'SIE_shear', 'EPL_shear']
R_MAX = {'SIS': 1.0, 'SIE': 1.6, 'SIE_shear': 1.8, 'EPL_shear': 1.8}
MAX_TRIES = 60

_SOLVERS = {}


def _solver(family):
    if family not in _SOLVERS:
        from lenstronomy.LensModel.lens_model import LensModel
        from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
        models = ['EPL_NUMBA'] if family == 'SIE' else ['EPL_NUMBA', 'SHEAR']
        lm = LensModel(models)
        _SOLVERS[family] = (lm, LensEquationSolver(lm))
    return _SOLVERS[family]


def sample_lens_shape(rng, family):
    """Mass-profile shape parameters (dimensionless)."""
    q, phi_q, gamma, g_ext, phi_ext = 1.0, 0.0, 2.0, 0.0, 0.0
    if family != 'SIS':
        # SLACS-like mass axis ratios (Bolton et al. 2008; Sonnenfeld et al. 2013).
        while True:
            q = rng.normal(0.75, 0.15)
            if 0.35 <= q <= 0.99:
                break
        phi_q = rng.uniform(0, np.pi)
    if family in ('SIE_shear', 'EPL_shear'):
        g_ext = min(0.25, float(np.exp(rng.normal(np.log(0.05), 0.6))))
        phi_ext = rng.uniform(0, np.pi)
    if family == 'EPL_shear':
        # Total-density slopes of massive early-type lenses (Auger et al. 2010).
        gamma = float(np.clip(rng.normal(2.08, 0.15), 1.6, 2.6))
    return dict(q=q, phi_q=phi_q, gamma=gamma, gamma_ext=g_ext, phi_ext=phi_ext)


def _kwargs(theta_e, shape, family):
    e = (1 - shape['q']) / (1 + shape['q'])
    kw = [dict(theta_E=theta_e, gamma=shape['gamma'], e1=e * np.cos(2 * shape['phi_q']),
               e2=e * np.sin(2 * shape['phi_q']), center_x=0.0, center_y=0.0)]
    if family != 'SIE':
        kw.append(dict(gamma1=shape['gamma_ext'] * np.cos(2 * shape['phi_ext']),
                       gamma2=shape['gamma_ext'] * np.sin(2 * shape['phi_ext']), ra_0=0, dec_0=0))
    return kw


def _sis_images(theta_e, bx, by):
    b = np.hypot(bx, by)
    y = b / theta_e
    if y >= 1 or y <= 0:
        return None
    ux, uy = bx / b, by / b
    r = np.array([theta_e * (1 + y), -theta_e * (1 - y)])
    x, yy = r * ux, r * uy
    mu = np.array([1 + 1 / y, 1 - 1 / y])
    kappa = theta_e / (2 * np.abs(r))
    fermat = 0.5 * ((x - bx) ** 2 + (yy - by) ** 2) - theta_e * np.abs(r)
    return x, yy, mu, kappa, fermat


def solve(rng, family, theta_e, shape, max_tries=MAX_TRIES):
    """Draw a source position and return the multiply imaged solution (lens centred at 0)."""
    rmax = R_MAX[family] * theta_e
    for _ in range(max_tries):
        rad = rmax * np.sqrt(rng.random())
        ang = rng.uniform(0, 2 * np.pi)
        bx, by = rad * np.cos(ang), rad * np.sin(ang)
        if family == 'SIS':
            sol = _sis_images(theta_e, bx, by)
            if sol is None:
                continue
            x, y, mu, kappa, fermat = sol
        else:
            lm, solver = _solver(family)
            kw = _kwargs(theta_e, shape, family)
            x, y = solver.image_position_from_source(bx, by, kw, solver='lenstronomy', search_window=4 * theta_e * R_MAX[family],
                                                     min_distance=theta_e / 40, precision_limit=1e-10, num_iter_max=200)
            if len(x) < 2:
                continue
            fxx, fxy, fyx, fyy = lm.hessian(x, y, kw)
            det = (1 - fxx) * (1 - fyy) - fxy * fyx
            mu = 1 / det
            kappa = 0.5 * (fxx + fyy)
            fermat = lm.fermat_potential(x, y, kw, bx, by)
            # Discard spurious roots: the lens equation residual must be negligible.
            ax, ay = lm.alpha(x, y, kw)
            ok = np.hypot(x - ax - bx, y - ay - by) < 1e-6 * theta_e
            x, y, mu, kappa, fermat = x[ok], y[ok], mu[ok], kappa[ok], fermat[ok]
            if len(x) < 2:
                continue
        det = 1 / mu
        morse = np.where(det < 0, 0.5, np.where(kappa > 1, 1.0, 0.0))
        return dict(beta_x=bx, beta_y=by, y_offset=rad / theta_e, x=np.asarray(x), y=np.asarray(y), mu=np.asarray(mu),
                    kappa=np.asarray(kappa), morse=morse, fermat=np.asarray(fermat))
    return None


def time_delays_seconds(fermat_arcsec2, ddt_seconds):
    """Arrival time relative to the first image (seconds)."""
    t = ddt_seconds * np.asarray(fermat_arcsec2) * ARCSEC ** 2
    return t - t.min()
