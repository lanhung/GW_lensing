"""Table of the accepted source fraction a used by gwlens.source_area.

For each lens shape (theta_E = 1, mass axis along x, shear angle dphi relative to it) the source is
placed on a fixed lattice of N positions, uniform in area within the disk of radius R_MAX theta_E, and
a is the fraction for which the solver call and acceptance test of gwlens.lenses.solve find at least
two images. The shared lattice keeps the table smooth in the lens parameters.

    python scripts/source_area_table.py run part0.pkl --slice 0/5 --par 30   # one slice per host
    python scripts/source_area_table.py merge gwlens/source_area.npz part*.pkl
"""
import argparse
import itertools
import pickle
import time

import numpy as np

GAMMA = np.round(np.arange(1.60, 2.601, 0.05), 3)
Q = np.array([0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99])
GAMMA_EXT = np.array([0.001, 0.005, 0.02, 0.05, 0.1, 0.17, 0.25])
DPHI = np.array([0.0, np.pi / 4, np.pi / 2])
N_SRC = 1200


def tasks():
    t = [('EPL_shear', g, q, ge, dp) for g, q, ge, dp in itertools.product(GAMMA, Q, GAMMA_EXT, DPHI)]
    return t + [('SIE', 2.0, q, 0.0, 0.0) for q in Q]


def frac(task):
    from gwlens import lenses
    fam, gam, q, gext, dphi = task
    np.random.seed(12345)  # lenstronomy's solver draws from the global generator
    k = np.arange(N_SRC)
    r, t = lenses.R_MAX[fam] * np.sqrt((k + 0.5) / N_SRC), k * np.pi * (3 - np.sqrt(5))
    lm, solver = lenses._solver(fam)
    kw = lenses._kwargs(1.0, dict(q=q, phi_q=0.0, gamma=gam, gamma_ext=gext, phi_ext=dphi), fam)
    ok = 0
    for bx, by in zip(r * np.cos(t), r * np.sin(t)):
        x, y = solver.image_position_from_source(bx, by, kw, solver='lenstronomy', search_window=4 * lenses.R_MAX[fam],
                                                 min_distance=1 / 40, precision_limit=1e-10, num_iter_max=200)
        if len(x) < 2:
            continue
        ax, ay = lm.alpha(x, y, kw)
        ok += np.sum(np.hypot(x - ax - bx, y - ay - by) < 1e-6) >= 2
    return task, ok / N_SRC


def key(task):
    return (task[0],) + tuple(round(float(v), 6) for v in task[1:])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('out')
    r.add_argument('--slice', default='0/1')
    r.add_argument('--par', type=int, default=8)
    m = sub.add_parser('merge')
    m.add_argument('out')
    m.add_argument('parts', nargs='+')
    a = p.parse_args()
    if a.cmd == 'run':
        from multiprocessing import Pool
        i, n = map(int, a.slice.split('/'))
        todo, res, t0 = tasks()[i::n], {}, time.time()
        with Pool(a.par) as pool:
            for j, (task, f) in enumerate(pool.imap_unordered(frac, todo)):
                res[key(task)] = f
                if j % 50 == 0:
                    print(j, len(todo), round(time.time() - t0), flush=True)
        pickle.dump(res, open(a.out, 'wb'))
        return
    res = {}
    for f in a.parts:
        res.update({key(k): v for k, v in pickle.load(open(f, 'rb')).items()})
    epl = np.array([res[key(('EPL_shear', g, q, ge, dp))] for g, q, ge, dp in itertools.product(GAMMA, Q, GAMMA_EXT, DPHI)])
    sie = np.array([res[key(('SIE', 2.0, q, 0.0, 0.0))] for q in Q])
    np.savez(a.out, gamma=GAMMA, q=Q, gamma_ext=GAMMA_EXT, dphi=DPHI, n_src=N_SRC,
             frac_epl_shear=epl.reshape(len(GAMMA), len(Q), len(GAMMA_EXT), len(DPHI)), frac_sie=sie)
    print('saved', a.out, 'min a =', epl.min(), sie.min())


if __name__ == '__main__':
    main()
