"""Stage-2 task list from the stage-1 ranking.

Takes the top --n-cat pairs by ln B_cat (lensed by a cataloged lens) and the top --n-pop pairs by ln B_pop (lensed by
any galaxy), both events with LVK PE, and writes for each pair three lensearch.realpe tasks that share the pair's
prior ranges: single A, single B and joint A+B. Also writes the events whose strain must be cut
(lensearch.fetch_strain).
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ranked', type=Path, help='stage-1 pairs_ranked.csv (real events)')
    p.add_argument('--placed', type=Path, help='placed injections (lensearch.inject place) instead of real pairs')
    p.add_argument('--joint-fraction', type=float, default=1.0, help='injections: fraction of pairs that also get a joint run')
    p.add_argument('--n-cat', type=int, default=25)
    p.add_argument('--n-pop', type=int, default=25)
    p.add_argument('--out', type=Path, required=True, help='task file')
    p.add_argument('--events-out', type=Path)
    a = p.parse_args()
    if a.placed is not None:
        pl = pd.read_csv(a.placed)
        step = max(1, int(round(1 / a.joint_fraction))) if a.joint_fraction > 0 else 0
        lines, n_joint = [], 0
        for k, r in enumerate(pl.itertuples()):      # joint runs spread evenly over the list (both runs)
            A, B = r.system + 'A', r.system + 'B'
            lines += [f'single {A} --prior-events {A} {B}', f'single {B} --prior-events {A} {B}']
            if step and k % step == 0:
                lines.append(f'joint {A} {B}')
                n_joint += 1
        a.out.write_text('\n'.join(lines) + '\n')
        print(f'{len(pl)} injected pairs, {len(lines)} tasks ({n_joint} joint)')
        return
    c = pd.read_csv(a.ranked)
    c = c[c.kind == 'pe-pe']
    top = pd.concat([c.nlargest(a.n_cat, 'ln_bcat').assign(why='cat'), c.nlargest(a.n_pop, 'ln_bpop').assign(why='pop')])
    top = top.groupby(['a', 'b'], as_index=False).agg(why=('why', lambda w: '+'.join(sorted(set(w)))),
                                                       ln_bcat=('ln_bcat', 'first'), ln_bpop=('ln_bpop', 'first'))
    lines = []
    for r in top.itertuples():
        lines += [f'single {r.a} --prior-events {r.a} {r.b}', f'single {r.b} --prior-events {r.a} {r.b}', f'joint {r.a} {r.b}']
    a.out.write_text('\n'.join(lines) + '\n')
    a.events_out.write_text('\n'.join(sorted(set(top.a) | set(top.b))) + '\n')
    top.to_csv(a.out.with_suffix('.pairs.csv'), index=False)
    print(f'{len(top)} pairs, {len(lines)} tasks, {len(set(top.a) | set(top.b))} events')


if __name__ == '__main__':
    main()
