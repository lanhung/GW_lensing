"""Accuracy of the relative-binning likelihood on joint-PE posterior samples.

For each event, re-create the injected data (same seed as gwlens.joint_pe), evaluate the
relative-binning likelihood (with its exact fallback) and the exact likelihood on random posterior
samples, and importance-reweight the samples by exp(lnL_exact - lnL_rb). Reports the likelihood
differences, the effective sample fraction and the shift of the ln(mu_B/mu_A) and ln D_A posteriors.
Usage: python analysis/relbin_check.py --pe-dir results/pe --events ET:SIS:SIS-7777-0000507:1 ... --out out.csv
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gwlens.joint_pe import event_seed, gw_likelihood, image_parameters

SAMPLED = ['chirp_mass', 'mass_ratio', 'chi_1', 'chi_2', 'ra', 'dec', 'psi', 'theta_jn', 'luminosity_distance',
           'ln_mu_rel', 'delta_t', 'phase', 'geocent_time']


def wq(x, w, q):
    i = np.argsort(x)
    c = np.cumsum(w[i]) / w.sum()
    return float(np.interp(q, c, x[i]))


def check(scen, fam, eid, ks, pe_dir, n, seed):
    row = pd.read_parquet(f'data/pe/{fam}.parquet').set_index('event_id', drop=False).loc[eid]
    like, base, truth, duration, snrs = gw_likelihood(row, scen, event_seed(1234, eid))
    suf = '_knownsky' if ks else ''
    post = pd.read_parquet(Path(pe_dir) / scen / f'{scen}_{eid}{suf}_posterior.parquet')
    s = post.sample(min(n, len(post)), random_state=seed)
    rb, ex, t0 = [], [], time.time()
    for _, r in s.iterrows():
        p = {k: float(r[k]) for k in SAMPLED}
        p.update(morse_a=float(r.morse_n_a) + 0.25, morse_b=float(r.morse_n_b) + 0.25)
        rb.append(like.log_likelihood_ratio(p))
        ex.append(sum(like.exact[k].log_likelihood_ratio(parameters=image_parameters(p, 'ab'[k])) for k in (0, 1)))
    rb, ex = np.array(rb), np.array(ex)
    d = ex - rb
    w = np.exp(d - d.max())
    ess = w.sum() ** 2 / np.sum(w ** 2) / len(w)
    out = dict(scenario=scen, family=fam, event_id=eid, known_sky=ks, n=len(s), seconds=time.time() - t0,
               dlogl_mean=float(d.mean()), dlogl_sd=float(d.std()), dlogl_absmax=float(np.abs(d).max()), ess_frac=float(ess))
    lnd = np.log(s.luminosity_distance.to_numpy())
    for name, x, tr in (('ln_mu_rel', s.ln_mu_rel.to_numpy(), truth['ln_mu_rel']),
                        ('ln_d', lnd, np.log(truth['luminosity_distance']))):
        one = np.ones_like(x)
        sd = np.std(x)
        for tag, ww in (('rb', one), ('exact', w)):
            q05, q50, q95 = (wq(x, ww, q) for q in (0.05, 0.5, 0.95))
            out.update({f'{name}_{tag}_q05': q05, f'{name}_{tag}_q50': q50, f'{name}_{tag}_q95': q95,
                        f'{name}_{tag}_covered': bool(q05 <= tr <= q95)})
        out[f'{name}_median_shift_sigma'] = (out[f'{name}_exact_q50'] - out[f'{name}_rb_q50']) / sd
        out[f'{name}_width_ratio'] = (out[f'{name}_exact_q95'] - out[f'{name}_exact_q05']) / (out[f'{name}_rb_q95'] - out[f'{name}_rb_q05'])
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--pe-dir', default='results/pe')
    a.add_argument('--events', nargs='+', required=True, help='scenario:family:event_id:known_sky')
    a.add_argument('--n', type=int, default=200)
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--out', type=Path, required=True)
    args = a.parse_args()
    rows = []
    for e in args.events:
        scen, fam, eid, ks = e.split(':')
        rows.append(check(scen, fam, eid, ks == '1', args.pe_dir, args.n, args.seed))
        print(json.dumps({k: rows[-1][k] for k in ('event_id', 'dlogl_mean', 'dlogl_sd', 'ess_frac', 'ln_mu_rel_median_shift_sigma', 'seconds')}), flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
