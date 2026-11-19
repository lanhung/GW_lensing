"""Stage-1 statistics of the injected lensed pairs, with the code and settings of the real ranking (lensearch.rank).

Each injected image was analysed alone (lensearch.realpe single) and converted to the event format by
lensearch.inj_events. For every injected pair with both images available this script evaluates

  ln B_ms, ln B_sky, ln Lambda_pop, ln B_pop      exactly as rank.py does for real pairs (same reference prior,
                                                   same lens-population cells, same EPL+shear geometry library,
                                                   same detected-distance distribution of the real events)
  ln B_cat(true lens)                              ln B_cat the pair would have if its lens were in the catalogue:
                                                   the lens is added to the real catalogue with its Einstein radius
                                                   and lens redshift (source redshift unknown, as for a lens found in
                                                   imaging), pi_L = theta_E^2 / sum theta_E^2 over the enlarged
                                                   catalogue (ln B_cat(true)); ln B_cat(chance) is the value from the
                                                   real catalogue alone and ln B_cat(full) the sum of both, i.e. the
                                                   statistic the search would have computed.

and compares them with the real search: the stage-1 cuts (ln B_ms > 0, ln B_sky > -2), the thresholds that selected
the stage-2 pairs (the n-th largest value among real PE-PE pairs) and the largest real values.

    python -m lensearch.inj_rank --rank results/rank --inj-events data/inj_events --placed results/inject/placed_s1.csv \
        --geometry results/inject/library_EPL_shear.parquet --out results/analysis/injections_stage1.csv
"""
import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from lensearch import rank as R

MS_MIN, SKY_MIN = 0.0, -2.0                      # cuts of the real run (logs/rank.log)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rank', type=Path, required=True, help='stage-1 output of the real search')
    p.add_argument('--inj-events', type=Path, required=True, help='lensearch.inj_events output')
    p.add_argument('--placed', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n-sel', type=int, default=25, help='stage-2 took the n largest PE-PE pairs of each statistic')
    p.add_argument('--npool', type=int, default=8)
    a = p.parse_args()

    # the real events: detected-distance distribution and time span, as in rank.main
    sys.modules['__main__'].Event = R.Event           # the cache was written by `python -m lensearch.rank`
    _, events = pickle.loads((a.rank / 'events_cache.pkl').read_bytes())
    events.sort(key=lambda e: e.t)
    T = events[-1].t - events[0].t
    post = np.mean([np.exp(e.lnl_d + R.log_dl_prior(np.exp(R.LND))) / np.sum(np.exp(e.lnl_d + R.log_dl_prior(np.exp(R.LND))))
                    for e in events if e.kind == 'pe'], axis=0)
    ln_pi_det = np.log(np.maximum(post / (post.sum() * (R.LND[1] - R.LND[0])), 1e-300))
    ranked = pd.read_csv(a.rank / 'pairs_ranked.csv')
    pepe = ranked[ranked.kind == 'pe-pe']
    thr = dict(max_cat=float(ranked.ln_bcat.max()), max_pop=float(ranked.ln_bpop.max()),
               sel_cat=float(pepe.ln_bcat.nlargest(a.n_sel).min()), sel_pop=float(pepe.ln_bpop.nlargest(a.n_sel).min()))
    lenses = pd.read_parquet(a.rank / 'lenses_used.parquet')
    sum_th2 = float(np.sum(lenses.theta_e.fillna(1.0).values ** 2))

    # injected images with a single-image analysis
    placed = pd.read_csv(a.placed)
    have = {f.stem for f in a.inj_events.glob('*.json')}
    placed = placed[[f'{s}A' in have and f'{s}B' in have for s in placed.system]].reset_index(drop=True)
    print(f'{len(placed)} injected pairs with both single-image analyses', flush=True)
    if len(placed) == 0:
        return
    names = [f'{s}{im}' for s in placed.system for im in 'AB']
    inj = [R.load_pe_event(a.inj_events, n) for n in names]
    prior_kde = R.reference_prior_kde()
    ref = R.reference_lnmc_density()
    for e in inj:
        e.x_prior = R._prior_at(e.x, prior_kde)
    # the images sit in noise slots at arbitrary times (sky position preserved by whole sidereal-day shifts); the lens
    # term needs the true arrival times of the lensed images
    for k, r in enumerate(placed.itertuples()):
        inj[2 * k].t, inj[2 * k + 1].t = float(r.t_a), float(r.t_b)
    ln_e = np.array([R.ln_evidence_pop(e, ln_pi_det) for e in inj])
    pairs = [(2 * k, 2 * k + 1) for k in range(len(placed))]
    ln_bms = np.array([R.ln_overlap_ms(inj[i], inj[j], ref) for i, j in pairs])
    ln_bsky = np.array([np.log(max(R.FOURPI * np.sum(inj[i].sky.astype(float) * inj[j].sky.astype(float)) * R.PIXAREA, 1e-300))
                        for i, j in pairs])

    # lens term for every cell of the real catalogue (the first ng cells are the population grid, as in rank.py)
    ng = len(R.THETA_GRID) * len(R.ZL_GRID)
    cell_of_lens, cells = R.cells_from_lenses(lenses)
    with Pool(a.npool, initializer=R._init, initargs=(inj, cells, ln_e, T, a.geometry)) as pool:
        lam_all = np.array(pool.map(R._lambda_row, pairs, chunksize=1))
    lam = lam_all[:, :ng]
    w_pop = R.population_cell_weights()
    ln_lambda_pop = logsumexp(lam + np.log(np.maximum(w_pop, 1e-300))[None, :], axis=1)
    ln_bpop = ln_bms + ln_bsky + ln_lambda_pop

    # ln B_cat with the true lens in the catalogue
    th = np.clip(placed.theta_e.values, R.THETA_GRID[0], R.THETA_GRID[-1])
    zl = np.clip(placed.z_l.values, R.ZL_GRID[0], R.ZL_GRID[-1])
    it = np.abs(np.log(th)[:, None] - np.log(R.THETA_GRID)[None, :]).argmin(1)
    iz = np.abs(zl[:, None] - R.ZL_GRID[None, :]).argmin(1)
    cell = it * len(R.ZL_GRID) + iz
    lam_true = lam[np.arange(len(placed)), cell]
    ln_pi_l = 2 * np.log(th) - np.log(sum_th2 + th ** 2)
    pix = hp.ang2pix(R.NSIDE_LENS, np.pi / 2 - placed.dec.values, placed.ra.values)      # placed ra, dec in radians
    sky_l = np.array([hp.ud_grade(inj[i].sky.astype(np.float64), R.NSIDE_LENS)[px] * hp.ud_grade(inj[j].sky.astype(np.float64), R.NSIDE_LENS)[px]
                      for (i, j), px in zip(pairs, pix)])
    with np.errstate(divide='ignore'):
        ln_bcat_true = ln_bms + ln_pi_l + 2 * np.log(R.FOURPI) + np.log(sky_l) + lam_true
    # the same with the lens at its exact Einstein radius and redshift instead of the grid-cell centre (shows how much
    # of the cell result is discretisation)
    from lensearch.lenspred import LensGeometry, pair_likelihood_geom
    geom = LensGeometry(a.geometry)
    lam_exact = np.full(len(placed), -np.inf)
    for k, ((i, j), r) in enumerate(zip(pairs, placed.itertuples())):
        dt = inj[j].t - inj[i].t
        ln_pu = np.log(2 * max(T - dt, 1.0) / T ** 2)
        v, _ = pair_likelihood_geom(dt, float(r.theta_e), float(r.z_l), np.nan, inj[i].loglike_d, inj[j].loglike_d, geom, n_zs=40)
        if v > 0:
            lam_exact[k] = np.log(v) - ln_pu - ln_e[i] - ln_e[j]

    # ln B_cat from the real catalogue alone (chance level of the injected pair) and with the true lens added
    from scipy import sparse
    S_l = np.vstack([hp.ud_grade(e.sky.astype(np.float64), R.NSIDE_LENS) for e in inj]).astype(np.float32)
    lam_max = lam_all.max(axis=1)
    lam_max = np.where(np.isfinite(lam_max), lam_max, 0.0)
    lam_lin = np.exp(lam_all - lam_max[:, None]).astype(np.float32)
    w_l = np.exp(2 * np.log(lenses.theta_e.fillna(1.0).values) - np.log(sum_th2)).astype(np.float32)
    N = sparse.csr_matrix((w_l, (hp.ang2pix(R.NSIDE_LENS, np.radians(90 - lenses.dec.values), np.radians(lenses.ra.values)),
                                 cell_of_lens)), shape=(R.NPIX_LENS, len(cells)), dtype=np.float32)
    ln_bcat_chance = np.full(len(placed), -np.inf)
    for k, (i, j) in enumerate(pairs):
        jm = S_l[i] * S_l[j]
        order = np.argsort(jm)[::-1]
        csum = np.cumsum(jm[order])
        nz = order[:np.searchsorted(csum, R.JOINT_MASS * csum[-1]) + 1]
        J = sparse.csr_matrix((jm[nz] * R.FOURPI ** 2, (np.zeros(len(nz), int), nz)), shape=(1, R.NPIX_LENS), dtype=np.float32)
        val = float(np.asarray((J @ N).multiply(lam_lin[k:k + 1]).sum(axis=1)).ravel()[0])
        if val > 0:
            ln_bcat_chance[k] = np.log(val) + lam_max[k] + ln_bms[k]
    ln_bcat_full = np.logaddexp(ln_bcat_chance, ln_bcat_true)

    out = placed[['system', 'run', 'z_s', 'z_l', 'theta_e', 'delta_t', 'mu_a', 'mu_b', 'snr_a', 'snr_b', 'dl']].copy()
    out['ln_bms'], out['ln_bsky'], out['ln_lambda_pop'], out['ln_bpop'] = ln_bms, ln_bsky, ln_lambda_pop, ln_bpop
    out['ln_lambda_true'], out['ln_bcat_true'] = lam_true, ln_bcat_true
    out['ln_lambda_true_exact'] = lam_exact
    out['ln_bcat_chance'], out['ln_bcat_full'] = ln_bcat_chance, ln_bcat_full
    out['ln_sky_lens'], out['ln_pi_lens'] = np.log(sky_l), ln_pi_l
    out['pass_s1'] = (out.ln_bms > MS_MIN) & (out.ln_bsky > SKY_MIN)
    for k, v in thr.items():
        col = 'ln_bcat_full' if k.endswith('cat') else 'ln_bpop'
        out[f'above_{k}'] = out.pass_s1 & (out[col] > v)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    summ = dict(n_pairs=len(out), thresholds=thr, pass_s1=float(out.pass_s1.mean()),
                **{f'frac_{k}': float(out[f'above_{k}'].mean()) for k in thr},
                ln_bpop_median=float(np.median(out.ln_bpop)), ln_bcat_true_median=float(np.nanmedian(out.ln_bcat_true)),
                ln_bcat_full_median=float(np.nanmedian(out.ln_bcat_full)), ln_bcat_chance_median=float(np.nanmedian(out.ln_bcat_chance)),
                n_true_lens_zero=int(np.sum(~np.isfinite(out.ln_lambda_true))))
    (a.out.with_suffix('.json')).write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    print(out.sort_values('ln_bpop', ascending=False).to_string(), flush=True)


if __name__ == '__main__':
    main()
