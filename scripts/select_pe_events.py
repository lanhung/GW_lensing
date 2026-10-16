"""Draw a random subset of selected lensed pairs for joint parameter estimation.

Writes one line per event, "<scenario> <family> <event_id>", for scripts/run_pe_batch.sh.
Events whose analysis segment would exceed --max-duration seconds are skipped to bound the cost
(this removes only the lightest detector-frame binaries; the fraction removed is printed).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gwlens.joint_pe import SCENARIO_FMIN, segment_duration

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-dir', type=Path, default=Path('data/pe'))
p.add_argument('--families', nargs='+', default=['SIS', 'SIE', 'SIE_shear', 'EPL_shear'])
p.add_argument('--scenario', required=True)
p.add_argument('--n-per-family', type=int, required=True)
p.add_argument('--max-duration', type=float, default=256.0)
p.add_argument('--seed', type=int, default=2026)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()

rng = np.random.default_rng(a.seed)
lines = []
for fam in a.families:
    d = pd.read_parquet(a.data_dir / f'{fam}.parquet')
    d = d[d[f'{a.scenario}__selected']]
    z = d.true_z_s
    dur = np.array([segment_duration(m1, m2, SCENARIO_FMIN[a.scenario])
                    for m1, m2 in zip(d.true_m1 * (1 + z), d.true_m2 * (1 + z))])
    ok = d[dur <= a.max_duration]
    pick = ok.iloc[rng.permutation(len(ok))[:a.n_per_family]]
    print(f'{fam}: {len(d)} selected, {1 - len(ok) / len(d):.3f} over {a.max_duration:.0f} s, picked {len(pick)}')
    lines += [f'{a.scenario} {fam} {e}' for e in pick.event_id]
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text('\n'.join(lines) + '\n')
