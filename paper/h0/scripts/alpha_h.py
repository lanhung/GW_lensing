"""Selection fraction alpha(h) of the simulated lensed population versus the Hubble-constant scale h = H0/H0_true.

At fixed redshifts the optimal SNR of every image scales as h (rho ~ 1/d_L) and the delays as 1/h; the lens and
source populations in redshift do not depend on h. For each simulated system we record the optimal SNRs, the
selection noise draw of gwlens.catalogue._gw_block (same random state), the delays and the observed image positions,
and re-apply the selection (rho_hat >= 8 for >= 2 images within one year of the first, brightest pair resolved).
usage: alpha_h.py FAMILY N SEED NPROC OUT.npz
"""
import sys
from multiprocessing import Pool
import numpy as np
from gwlens import catalogue as cat, detectors as det, lenses

H = np.round(np.linspace(0.5, 1.5, 101), 4)
OBS = cat.OBS
_rec = []
_orig = cat._gw_block


def _patched(rng, scen, src, img, cal_a, cal_b):
    g = np.random.Generator(type(rng.bit_generator)())
    g.bit_generator.state = rng.bit_generator.state
    i0 = src['i0'][scen]
    gps = src['t0'] + img['t']
    rho_unl = np.array([det.network_snr(i0, det.antenna(scen, src['ra'], src['dec'], src['psi'], t), src['cosi']) for t in gps])
    rho = np.sqrt(np.abs(img['mu'])) * rho_unl
    _rec.append((scen, rho, g.normal(0, 1, len(rho)), np.asarray(img['t'], float)))
    return _orig(rng, scen, src, img, cal_a, cal_b)


cat._gw_block = _patched


def selected(rho, n, t, x, y, h):
    rh = rho * h + n
    d = rh >= OBS['rho_threshold']
    if d.sum() < 2:
        return False
    th = t / h
    inwin = d & (th - th[d].min() <= OBS['t_obs'])
    if inwin.sum() < 2:
        return False
    idx = np.flatnonzero(inwin)
    a, b = idx[np.argsort(-rh[idx])][:2]
    return np.hypot(x[a] - x[b], y[a] - y[b]) >= OBS['min_separation']


def work(args):
    fam, n, seed = args
    rng = np.random.default_rng(seed)
    np.random.seed(seed % 2 ** 32)
    cal = cat.calibration()
    scen = tuple(det.SCENARIOS)
    out = {s: np.zeros((n, len(H)), bool) for s in scen}
    check = 0
    for k in range(n):
        _rec.clear()
        row = cat.simulate_system(rng, fam, scen, cal, full=True)
        x, y = np.array(row['em']['x']), np.array(row['em']['y'])
        for s, rho, nz, t in _rec:
            out[s][k] = [selected(rho, nz, t, x, y, h) for h in H]
            check += out[s][k][50] != row[f'{s}__selected']
    return out, check


if __name__ == '__main__':
    fam, N, seed, npr, fout = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    chunk = 100
    seeds = np.random.SeedSequence([seed, lenses.FAMILIES.index(fam)]).generate_state(N // chunk)
    with Pool(npr) as p:
        res = p.map(work, [(fam, chunk, int(s)) for s in seeds])
    sel = {s: np.concatenate([r[0][s] for r in res]) for s in det.SCENARIOS}
    mism = sum(r[1] for r in res)
    np.savez(fout, h=H, **{f'sel_{s}': v for s, v in sel.items()})
    print(fam, 'systems', N, 'mismatches with catalogue selection at h=1:', mism,
          ' '.join(f'{s}: alpha(1)={v[:, 50].mean():.4f} dln_alpha/dln_h={np.polyfit(np.log(H[40:61]), np.log(v[:, 40:61].mean(0)), 1)[0]:.2f}' for s, v in sel.items()))
