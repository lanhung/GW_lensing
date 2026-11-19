"""Split the joint log-likelihood of a stage-2 pair into its image-A and image-B parts (exact likelihoods), and compare
with the single-image fits: shows which data set the joint fit gains on.
    python lensearch/ops/split_joint_logl.py RESULTS_DIR A B [--candidates CSV --cand-noise-dir DIR] [--n 150]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lensearch import realpe as R
from gwlens.joint_pe import image_parameters

p = argparse.ArgumentParser()
p.add_argument('results', type=Path); p.add_argument('a'); p.add_argument('b')
p.add_argument('--pe-dir', type=Path, default=Path('data/pe')); p.add_argument('--cache', type=Path, default=Path('data/strain'))
p.add_argument('--candidates', type=Path); p.add_argument('--cand-noise-dir', type=Path)
p.add_argument('--n', type=int, default=150); p.add_argument('--approximant', default='IMRPhenomXPHM')
a = p.parse_args()
if a.candidates is not None:
    for r in pd.read_csv(a.candidates).to_dict('records'):
        R.CANDIDATES[r['name']] = dict(r, noise_dir=str(a.cand_noise_dir))
    R.EXTRA_CACHES.append(a.cand_noise_dir)
likes = {}
for ev in (a.a, a.b):
    s, ps, _, best = R.release_settings(a.pe_dir, ev)
    rb, exact, ceil, _ = R.single_likelihood(s, ps, best, a.cache, a.approximant)
    likes[ev] = exact if exact is not None else rb
tag = f'{a.a}__{a.b}_{a.approximant}'
J = pd.read_parquet(a.results / f'joint_{tag}_posterior.parquet').sort_values('log_likelihood', ascending=False)
def ll(ev, prm):
    return likes[ev].log_likelihood_ratio(parameters=prm)
rows = []
for _, r in J.head(a.n).iterrows():
    prm = r.to_dict()
    rows.append(dict(joint=r.log_likelihood, la=ll(a.a, image_parameters(prm, 'a')), lb=ll(a.b, image_parameters(prm, 'b')),
                     mc=r.chirp_mass, dl=r.luminosity_distance, ln_mu=r.get('ln_mu_rel', np.nan)))
d = pd.DataFrame(rows)
print('joint top samples: logL_joint, logL_A, logL_B quantiles (10,50,90, max)')
for c in ('joint', 'la', 'lb'):
    print(f'  {c:6s}', np.round(np.quantile(d[c], [0.1, 0.5, 0.9]), 2), round(d[c].max(), 2))
print('  joint - (A+B) median', round(float(np.median(d.joint - d.la - d.lb)), 3))
for ev in (a.a, a.b):
    S = pd.read_parquet(a.results / f'single_{ev}_for_{tag}_posterior.parquet')
    print(f'single {ev}: max logL {S.log_likelihood.max():.2f}, median {S.log_likelihood.median():.2f}; '
          f'Mc {np.quantile(S.chirp_mass,[0.05,0.5,0.95]).round(3)}, dL {np.quantile(S.luminosity_distance,[0.05,0.5,0.95]).round(0)}')
    top = S.sort_values('log_likelihood', ascending=False).head(5)
    print('   recomputed logL of its 5 best samples', [round(ll(ev, image_parameters(x.to_dict(), 'a')), 2) for _, x in top.iterrows()],
          'stored', top.log_likelihood.round(2).tolist())
