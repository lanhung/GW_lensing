"""Normalisation of the source-position prior of the simulated lens population.

gwlens.lenses.solve draws the source uniformly in a disk of radius R_MAX * theta_E and redraws it (up to
MAX_TRIES times, after which the whole lens is redrawn by gwlens.catalogue) until the lens equation has at
least two solutions. For a given lens the source density is therefore 1 / A_mult over its multiple-image
region, with A_mult = a * pi * (R_MAX * theta_E)^2 and a the accepted fraction of the disk, and the lens
is kept with probability 1 - (1 - a)^MAX_TRIES. The fraction a rises steeply with the radial slope (for
EPL+shear with q = 0.75 about 0.09 at gamma = 1.8, 0.30 at 2.0 and 0.84 at 2.2), because steeper profiles produce a
second, faint image near the centre for sources far outside the Einstein radius. An inference with a flat
source-position prior that is not normalised per lens therefore weights each lens model by A_mult and
favours steep slopes. log_weight returns ln[(1 - (1 - a)^MAX_TRIES) / A_mult], the term that makes the
lens-model posterior consistent with the simulated population.

The table of a (source_area.npz next to this file) is computed by scripts/source_area_table.py with the
solver call and acceptance test of lenses.solve, on a fixed lattice of source positions.
"""
from pathlib import Path

import numpy as np

from .lenses import MAX_TRIES, R_MAX

_TABLE = {}


def _table():
    if not _TABLE:
        from scipy.interpolate import RegularGridInterpolator
        t = np.load(Path(__file__).with_name('source_area.npz'))
        axes = (t['gamma'], t['q'], np.log(t['gamma_ext']), t['dphi'])
        _TABLE['epl'] = RegularGridInterpolator(axes, np.log(t['frac_epl_shear']))
        _TABLE['sie'] = (t['q'], np.log(t['frac_sie']))
        _TABLE['bounds'] = [(ax[0], ax[-1]) for ax in axes]
    return _TABLE


def accepted_fraction(model, q=0.75, gamma_ext=0.0, dphi=0.0, gamma=2.0):
    """Fraction a of the source disk in which the simulation accepts a source (theta_E-independent)."""
    if model == 'SIS':
        return np.ones_like(np.asarray(q, float))
    tab = _table()
    if model == 'SIE':
        return np.exp(np.interp(q, *tab['sie']))
    if model == 'SIE_shear':
        gamma = 2.0
    d = np.abs(np.asarray(dphi, float)) % np.pi
    d = np.minimum(d, np.pi - d)  # the area is symmetric under reflection of the shear angle
    pts = np.broadcast_arrays(gamma, q, np.log(np.maximum(gamma_ext, 1e-12)), d)
    pts = np.stack([np.clip(np.asarray(v, float), lo, hi) for v, (lo, hi) in zip(pts, tab['bounds'])], axis=-1)
    return np.exp(tab['epl'](pts.reshape(-1, 4)).reshape(pts.shape[:-1]))


def log_weight(model, theta_e, q=0.75, phi_q=0.0, gamma_ext=0.0, phi_ext=0.0, gamma=2.0):
    """ln[(1 - (1 - a)^MAX_TRIES) / (a pi (R_MAX theta_E)^2)] for the simulated population of `model`."""
    a = accepted_fraction(model, q, gamma_ext, np.asarray(phi_ext) - np.asarray(phi_q), gamma)
    keep = -np.expm1(MAX_TRIES * np.log1p(-np.minimum(a, 1 - 1e-12)))
    return np.log(keep) - np.log(a * np.pi * (R_MAX[model] * np.asarray(theta_e, float)) ** 2)
