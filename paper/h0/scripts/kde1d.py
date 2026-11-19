"""Per-event log posterior of h on a grid: weighted Gaussian KDE with Silverman's bandwidth of the samples
themselves, reflected at the prior bounds 0.5 and 1.5 (the reflection must not enter the bandwidth)."""
import numpy as np
from scipy.stats import gaussian_kde

GRID = np.linspace(0.5, 1.5, 2001)


def logpost(x, w=None, grid=GRID, floor=1e-12):
    x = np.asarray(x, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    w = w / w.sum()
    n_eff = 1 / np.sum(w ** 2)
    m = w @ x; sd = np.sqrt(w @ (x - m) ** 2)
    o = np.argsort(x); c = np.cumsum(w[o]); iqr = np.interp(0.75, c, x[o]) - np.interp(0.25, c, x[o])
    bw = 0.9 * min(sd, iqr / 1.34) * n_eff ** -0.2
    xx = np.concatenate([x, 1.0 - x, 3.0 - x]); ww = np.tile(w, 3) / 3
    mm = ww @ xx; sdd = np.sqrt(ww @ (xx - mm) ** 2)
    p = gaussian_kde(xx, bw_method=bw / sdd, weights=ww)(grid)
    return np.log(np.maximum(p, floor * p.max()))


def stats(L, grid=GRID):
    p = np.exp(L - L.max()); c = np.cumsum(p); c /= c[-1]
    q = lambda a: grid[np.searchsorted(c, a)]
    return q(0.5), q(0.16), q(0.84)
