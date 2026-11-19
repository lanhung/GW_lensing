"""Selection-matched unlensed background for stage 2: unrelated pairs of twin images that pass the real stage-1 cuts.

The pair-matched twins of lensearch.twins are not a valid null on their own. A twin image is injected at a posterior
sample of its real event and measured again in new noise, so it scatters about two noise widths from the real event,
and two twin images of a pair are less consistent than the real pair. More importantly the real pairs reached stage 2
*because* their posteriors agree (stage-1 selection), and the twins do not inherit that selection. Measured on the
first 34 pairs: the standardised A-B chirp-mass separation is 1.3 for real pairs and 2.8 for twins, and real ln C
exceed their own twins by a median 4.7.

The correct null is the stage-2 statistic of unrelated pairs that pass the same stage-1 selection. Every twin image is
an unrelated, realistic unlensed event in real noise with a finished single-image analysis. This script

  1. forms all pairs of twin images from different real source events (images of the same real event are excluded,
     they would mimic lensing), A = the image whose source event is earlier, time delay = that of the source events;
  2. evaluates ln B_ms, ln B_sky, ln B_pop and ln B_cat (real catalogue) with the code and settings of lensearch.rank,
     as lensearch.inj_rank does;
  3. selects the pairs that pass the cuts and exceed the stage-2 selection thresholds of the real search
     (n-th largest real PE-PE value of ln B_cat or ln B_pop);
  4. for each selected pair, rescales the two single-image evidences to the prior of the new pairing (the priors differ
     only in the chirp-mass and distance ranges, set from the posteriors of the two real source events; the rescaling
     is the ratio of prior normalisations, exact when the posterior lies inside both ranges, which is checked) and
     writes the joint task lines (lensearch.realpe joint, images ordered by data time).

    python -m lensearch.null_pairs --rank results/rank --events data/bg_events --twins results/bg/twins_s1.csv \
        --pe-dir data/pe --collected results/collected/bg_pe --out results/bg/null_pairs.csv \
        --tasks results/bg/tasks_null.txt --injections results/bg/twins_s1.csv
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
from scipy import sparse
from scipy.special import logsumexp

from lensearch import rank as R

MS_MIN, SKY_MIN = 0.0, -2.0


def prior_norms(posts):
    """ln normalisations of the chirp-mass (UniformInComponentsChirpMass, density ~ Mc) and distance (PowerLaw 2)
    priors of lensearch.realpe.priors for the given real-event posteriors, and their ranges."""
    mc = np.concatenate([p.chirp_mass.values for p in posts])
    lo, hi = 0.8 * float(np.quantile(mc, 0.0005)), 1.2 * float(np.quantile(mc, 0.9995))
    d = np.concatenate([p.luminosity_distance.values for p in posts])
    dlo, dhi = max(10.0, 0.2 * float(d.min())), 3.0 * float(d.max())
    return np.log(hi ** 2 - lo ** 2) + np.log(dhi ** 3 - dlo ** 3), (lo, hi, dlo, dhi)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rank', type=Path, required=True)
    p.add_argument('--events', type=Path, required=True, help='lensearch.inj_events output for the twin singles')
    p.add_argument('--twins', type=Path, required=True)
    p.add_argument('--pe-dir', type=Path, required=True)
    p.add_argument('--collected', type=Path, required=True, help='twin single-image results (summaries, posteriors)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--injections', default='results/bg/twins_s1.csv')
    p.add_argument('--noise-dir', default='data/inject_noise_s1')
    p.add_argument('--outdir', default='results/bg_null')
    p.add_argument('--nlive', type=int, default=500)
    p.add_argument('--naccept', type=int, default=40)
    p.add_argument('--n-sel', type=int, default=25)
    p.add_argument('--min-maxl', type=float, default=28.0,
                   help='detectability: an image enters only if its maximum log-likelihood ratio reaches this (all real '
                        'stage-2 events but one, GW200308_173609 at SNR 4.7, have >= 28.5)')
    p.add_argument('--npool', type=int, default=8)
    p.add_argument('--geometry', type=Path, default=Path('results/inject/library_EPL_shear.parquet'))
    a = p.parse_args()

    sys.modules['__main__'].Event = R.Event
    _, events = pickle.loads((a.rank / 'events_cache.pkl').read_bytes())
    events.sort(key=lambda e: e.t)
    T = events[-1].t - events[0].t
    post = np.mean([np.exp(e.lnl_d + R.log_dl_prior(np.exp(R.LND))) / np.sum(np.exp(e.lnl_d + R.log_dl_prior(np.exp(R.LND))))
                    for e in events if e.kind == 'pe'], axis=0)
    ln_pi_det = np.log(np.maximum(post / (post.sum() * (R.LND[1] - R.LND[0])), 1e-300))
    ranked = pd.read_csv(a.rank / 'pairs_ranked.csv')
    pepe = ranked[ranked.kind == 'pe-pe']
    thr = dict(sel_cat=float(pepe.ln_bcat.nlargest(a.n_sel).min()), sel_pop=float(pepe.ln_bpop.nlargest(a.n_sel).min()),
               max_cat=float(ranked.ln_bcat.max()), max_pop=float(ranked.ln_bpop.max()))
    lenses = pd.read_parquet(a.rank / 'lenses_used.parquet')
    sum_th2 = float(np.sum(lenses.theta_e.fillna(1.0).values ** 2))

    # twin images with a single-image analysis: image -> (source event, source time, data time, summary)
    tw = pd.read_csv(a.twins)
    have = {f.stem for f in a.events.glob('*.json')}
    img = []
    for r in tw.itertuples():
        for im in 'ab':
            name = f'{r.system}{im.upper()}'
            if name not in have:
                continue
            s = list(a.collected.glob(f'single_{name}_for_*_summary.json'))
            if not s:
                continue
            maxl = float(pd.read_parquet(next(a.collected.glob(f'single_{name}_for_*_posterior.parquet')), columns=['log_likelihood']).log_likelihood.max())
            if maxl < a.min_maxl:      # not a detectable event: a random posterior sample can have a low optimal SNR
                continue
            img.append(dict(image=name, system=r.system, src=getattr(r, f'src_{im}'), t_src=float(getattr(r, f't_real_{im}')),
                            t_data=float(getattr(r, f't_{im}')), partner_a=r.src_a, partner_b=r.src_b,
                            ln_z=float(json.loads(s[0].read_text())['log_evidence'])))
    img = pd.DataFrame(img).sort_values('t_src').reset_index(drop=True)
    print(f'{len(img)} detectable twin images with single-image analyses', flush=True)
    ev = [R.load_pe_event(a.events, n) for n in img.image]
    prior_kde, ref = R.reference_prior_kde(), R.reference_lnmc_density()
    for e, t in zip(ev, img.t_src):
        e.x_prior = R._prior_at(e.x, prior_kde)
        e.t = t                                    # the lens term needs the time delay of the source events
    ln_e = np.array([R.ln_evidence_pop(e, ln_pi_det) for e in ev])
    pairs = [(i, j) for i in range(len(img)) for j in range(i + 1, len(img))
             if img.src[i] != img.src[j] and img.t_src[j] > img.t_src[i]]
    print(f'{len(pairs)} unrelated pairs', flush=True)

    ln_bms = np.array([R.ln_overlap_ms(ev[i], ev[j], ref) for i, j in pairs])
    ln_bsky = np.array([np.log(max(R.FOURPI * np.sum(ev[i].sky.astype(float) * ev[j].sky.astype(float)) * R.PIXAREA, 1e-300))
                        for i, j in pairs])
    ok = (ln_bms > MS_MIN) & (ln_bsky > SKY_MIN)
    print(f'{ok.sum()} pass the stage-1 cuts', flush=True)
    idx = np.flatnonzero(ok)
    sel_pairs = [pairs[k] for k in idx]
    ng = len(R.THETA_GRID) * len(R.ZL_GRID)
    cell_of_lens, cells = R.cells_from_lenses(lenses)
    with Pool(a.npool, initializer=R._init, initargs=(ev, cells, ln_e, T, a.geometry)) as pool:
        lam_all = np.array(pool.map(R._lambda_row, sel_pairs, chunksize=4))
    w_pop = R.population_cell_weights()
    ln_bpop = ln_bms[idx] + ln_bsky[idx] + logsumexp(lam_all[:, :ng] + np.log(np.maximum(w_pop, 1e-300))[None, :], axis=1)
    S_l = np.vstack([hp.ud_grade(e.sky.astype(np.float64), R.NSIDE_LENS) for e in ev]).astype(np.float32)
    lam_max = np.max(lam_all, axis=1)
    lam_max = np.where(np.isfinite(lam_max), lam_max, 0.0)
    lam_lin = np.exp(lam_all - lam_max[:, None]).astype(np.float32)
    w_l = np.exp(2 * np.log(lenses.theta_e.fillna(1.0).values) - np.log(sum_th2)).astype(np.float32)
    N = sparse.csr_matrix((w_l, (hp.ang2pix(R.NSIDE_LENS, np.radians(90 - lenses.dec.values), np.radians(lenses.ra.values)),
                                 cell_of_lens)), shape=(R.NPIX_LENS, len(cells)), dtype=np.float32)
    ln_bcat = np.full(len(idx), -np.inf)
    for k, (i, j) in enumerate(sel_pairs):
        jm = S_l[i] * S_l[j]
        order = np.argsort(jm)[::-1]
        csum = np.cumsum(jm[order])
        nz = order[:np.searchsorted(csum, R.JOINT_MASS * csum[-1]) + 1]
        J = sparse.csr_matrix((jm[nz] * R.FOURPI ** 2, (np.zeros(len(nz), int), nz)), shape=(1, R.NPIX_LENS), dtype=np.float32)
        v = float(np.asarray((J @ N).multiply(lam_lin[k:k + 1]).sum(axis=1)).ravel()[0])
        if v > 0:
            ln_bcat[k] = np.log(v) + lam_max[k] + ln_bms[idx[k]]

    out = pd.DataFrame(dict(a=img.image.values[[i for i, _ in sel_pairs]], b=img.image.values[[j for _, j in sel_pairs]],
                            src_a=img.src.values[[i for i, _ in sel_pairs]], src_b=img.src.values[[j for _, j in sel_pairs]],
                            dt_days=(img.t_src.values[[j for _, j in sel_pairs]] - img.t_src.values[[i for i, _ in sel_pairs]]) / 86400,
                            ln_bms=ln_bms[idx], ln_bsky=ln_bsky[idx], ln_bpop=ln_bpop, ln_bcat=ln_bcat))
    out['original_twin'] = [img.system[i] == img.system[j] for i, j in sel_pairs]
    out['selected'] = (out.ln_bcat > thr['sel_cat']) | (out.ln_bpop > thr['sel_pop'])

    # single-image evidences rescaled to the prior of the new pairing
    posts = {}
    def real_post(e):
        if e not in posts:
            posts[e] = pd.read_parquet(a.pe_dir / f'{e}.parquet', columns=['chirp_mass', 'luminosity_distance'])
        return posts[e]
    rows = img.set_index('image')
    lnz_a, lnz_b, inside = [], [], []
    for r in out.itertuples():
        new_n, (lo, hi, dlo, dhi) = prior_norms([real_post(r.src_a), real_post(r.src_b)])
        vals, ins = [], True
        for im in (r.a, r.b):
            q = rows.loc[im]
            old_n, _ = prior_norms([real_post(q.partner_a), real_post(q.partner_b)])
            s = pd.read_parquet(next(a.collected.glob(f'single_{im}_for_*_posterior.parquet')), columns=['chirp_mass', 'luminosity_distance'])
            ins &= bool(s.chirp_mass.between(lo, hi).all() and s.luminosity_distance.between(dlo, dhi).all())
            vals.append(q.ln_z + old_n - new_n)
        lnz_a.append(vals[0]); lnz_b.append(vals[1]); inside.append(ins)
    out['ln_z_a'], out['ln_z_b'], out['posterior_inside_prior'] = lnz_a, lnz_b, inside
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    # joint tasks for the selected pairs that are not already analysed (the original twin pairs are)
    data_t = rows.t_data.to_dict()
    opts = f'--injections {a.injections} --noise-dir {a.noise_dir} --nlive {a.nlive} --naccept {a.naccept} --outdir {a.outdir}'
    todo = out[out.selected & ~out.original_twin & out.posterior_inside_prior]
    lines = [f'joint {x} {y} {opts}' for x, y in (sorted((r.a, r.b), key=data_t.get) for r in todo.itertuples())]
    # a selected pair whose single-image posterior is not inside the new prior: run its singles with that prior
    redo = out[out.selected & ~out.original_twin & ~out.posterior_inside_prior]
    for r in redo.itertuples():
        x, y = sorted((r.a, r.b), key=data_t.get)
        lines += [f'single {x} --prior-events {x} {y} {opts}', f'single {y} --prior-events {x} {y} {opts}', f'joint {x} {y} {opts}']
    a.tasks.write_text('\n'.join(lines) + ('\n' if lines else ''))
    summ = dict(images=len(img), unrelated_pairs=len(pairs), pass_cuts=int(ok.sum()), thresholds=thr,
                selected=int(out.selected.sum()), selected_original_twins=int((out.selected & out.original_twin).sum()),
                selected_outside_prior=int((out.selected & ~out.posterior_inside_prior).sum()), new_tasks=len(lines),
                images_below_min_maxl=int(2 * len(tw) - len(img)),
                real_selection_fraction=float(len(pd.unique(pd.concat([pepe.nlargest(a.n_sel, 'ln_bcat').index.to_series(),
                                                                        pepe.nlargest(a.n_sel, 'ln_bpop').index.to_series()]))) / len(pepe)),
                null_selection_fraction=float(out.selected.sum() / len(pairs)))
    a.out.with_suffix('.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
