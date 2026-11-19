"""Convert single-image PE results of injections (lensearch.realpe) into the standardised event format of
lensearch.fetch_pe, so that injected pairs pass through exactly the same stage-1 ranking as the real events.

Writes <out>/<image>.parquet (samples incl. chi_eff), <out>/<image>.json (catalog 'injection', analysis, the distance
prior used: PowerLaw(alpha=2)) and an empty <out>/<image>.npz (the sky map is built from the samples).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', type=Path, required=True, help='realpe output directory with single_* posteriors')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(a.results.glob('single_*_posterior.parquet')):
        image = f.name[len('single_'):].split('_for_')[0]
        df = pd.read_parquet(f)
        if 'chi_eff' not in df:
            q = df.mass_ratio
            c1 = df.a_1 * np.cos(df.tilt_1) if 'tilt_1' in df else df.get('chi_1', 0.0)
            c2 = df.a_2 * np.cos(df.tilt_2) if 'tilt_2' in df else df.get('chi_2', 0.0)
            df['chi_eff'] = (c1 + q * c2) / (1 + q)
        keep = [c for c in ('chirp_mass', 'mass_ratio', 'chi_eff', 'ra', 'dec', 'luminosity_distance', 'theta_jn', 'psi',
                            'geocent_time', 'log_likelihood') if c in df]
        df[keep].to_parquet(a.out / f'{image}.parquet', index=False)
        meta = dict(event=image, catalog='injection', analysis='realpe single', priors=dict(
            luminosity_distance="PowerLaw(alpha=2, ...)"), dl_prior='powerlaw2', source=f.name)
        (a.out / f'{image}.json').write_text(json.dumps(meta))
        np.savez(a.out / f'{image}.npz')
        n += 1
    print(f'{n} injected images converted')


if __name__ == '__main__':
    main()
