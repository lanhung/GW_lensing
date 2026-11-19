"""Unlensed-twin background for stage 2 (lensearch.realpe).

For a real pair (A, B) the null hypothesis is that A and B are unrelated binaries whose data happen to look as
they do. Its pair-matched realisation is an *unlensed twin*: one posterior sample of A and one, independent,
posterior sample of B (LVK release posteriors, all 15 parameters), injected as two unrelated signals into real
O4 noise and analysed with exactly the pipeline, settings and priors of the real pair (the prior ranges come
from the real posteriors).

Caveat (measured 2026-11-12): the twins are NOT a valid null for the real pairs. A twin image sits at a posterior
sample and is measured again in new noise, so twin images scatter about twice as far apart as the real ones (A-B
standardised chirp-mass separation 2.8 vs 1.3), and the twins do not inherit the stage-1 selection that picked the
real pairs for agreeing posteriors; their ln C is ~4.7 below the real pairs'. Some twin images are also undetectable
(a posterior sample can have a low optimal SNR). The twin single-image analyses are the raw material of the
selection-matched background of lensearch.null_pairs; the pair-matched twins are kept only as a diagnostic.

Noise: the twin images reuse the real-noise slots cut for the lensed injection set (lensearch.inject cut), one
slot per image, no slot used twice. Each image keeps its detector-frame sky position: the right ascension is
rotated by the Earth's rotation between the real and the slot time, so the antenna response is unchanged.
Detectors are those of the real analysis; the segment is 4, 8 or 16 s from 20 Hz (the LVK runs of the lightest
events used 32 s from ~13 Hz).

    python -m lensearch.twins --pe-dir data/pe --pairs results/stage2_tasks.txt --placed results/inject/placed_s1.csv \
        --out results/bg/twins_s1.csv --tasks results/bg/tasks_bg_s1.txt --extra GW240601_061200+GW240612_081540:5 ...
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lensearch.inject import PAD, RUNS, SIDEREAL_DAY

F_MIN = 20.0
PAR = ['mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'theta_jn', 'phase', 'psi', 'ra', 'dec',
       'luminosity_distance', 'geocent_time']


def ra_shift(t_from, t_to):
    """Change of right ascension that keeps the detector-frame sky position when the signal moves from t_from to t_to."""
    return 2 * np.pi * np.mod(t_to - t_from, SIDEREAL_DAY) / SIDEREAL_DAY


def event_detectors(pe_dir, ev):
    npz = np.load(Path(pe_dir) / f'{ev}.npz')
    return sorted(k[4:] for k in npz.files if k.startswith('psd_'))


def slot_table(placed):
    """One row per noise slot of the lensed injection set: image name, run, slot time, segment duration, file span."""
    rows = []
    for r in placed.itertuples():
        for im in 'ab':
            rows.append(dict(noise=f'{r.system}{im.upper()}', run=r.run, t_noise=float(getattr(r, f't_noise_{im}')),
                             slot_duration=float(r.duration), span=float(r.duration) + 2 * PAD))
    return pd.DataFrame(rows)


def twin_duration(m1, m2):
    from gwlens.joint_pe import segment_duration
    return min(max(segment_duration(m1, m2, F_MIN), 4.0), 16.0)


def pick_slot(slots, used, ifos, duration, rng):
    """A free slot whose run has all required detectors and whose noise file holds `duration` s ending 2 s after
    the trigger. An 8-s twin fits a 4-s slot (file span 12 s) and a 16-s twin fits an 8-s slot (span 16 s) when the
    trigger is moved to the end of the file; otherwise the trigger stays at the slot time."""
    ok = slots[~slots.noise.isin(used) & slots.run.map(lambda r: set(ifos) <= set(RUNS[r]['ifos']))]
    ok = ok[ok.span >= duration - 1e-6]
    if len(ok) == 0:
        raise RuntimeError(f'no free slot for {ifos} and {duration} s')
    ok = ok[ok.span == ok.span.min()]          # the shortest file that fits, so long files stay for long twins
    s = ok.iloc[int(rng.integers(len(ok)))]
    if s.slot_duration >= duration:
        t = s.t_noise
    else:   # move the trigger so that the analysis window [t + 2 - duration, t + 2] ends with the file
        t = s.t_noise + PAD
        file_start = s.t_noise + 2.0 - s.slot_duration - PAD
        assert t + 2.0 - duration >= file_start - 1e-6
    return s.noise, s.run, float(t)


def build(pairs, pe_dir, placed, rng, n_extra=None, used=None, prefix='BG'):
    n_extra = n_extra or {}
    slots = slot_table(placed)
    used, rows, posts, dets = list(used or []), [], {}, {}
    for i, (a, b) in enumerate(pairs):
        for ev in (a, b):
            if ev not in posts:
                posts[ev] = pd.read_parquet(Path(pe_dir) / f'{ev}.parquet')
                dets[ev] = event_detectors(pe_dir, ev)
        n = 1 + n_extra.get(f'{a}+{b}', 0)
        for k in range(n):
            row = dict(system=f'{prefix}{i:02d}-{k}', pair=f'{a}+{b}', src_a=a, src_b=b, run='')
            for im, ev in (('a', a), ('b', b)):
                p = posts[ev].iloc[int(rng.integers(len(posts[ev])))]
                m1, m2 = float(p.mass_1), float(p.mass_2)
                dur = twin_duration(m1, m2)
                ifos = dets[ev]
                noise, run, t = pick_slot(slots, used, ifos, dur, rng)
                used.append(noise)
                row.update({f'noise_{im}': noise, f'run_{im}': run, f't_{im}': t, f'duration_{im}': dur,
                            f'fs_{im}': 4096.0 if m1 + m2 < 60 else 2048.0, f'ifos_{im}': '+'.join(ifos),
                            f'm1_det_{im}': m1, f'm2_det_{im}': m2, f'dl_{im}': float(p.luminosity_distance),
                            f'ra_{im}': float(np.mod(p.ra + ra_shift(float(p.geocent_time), t), 2 * np.pi)),
                            f'dec_{im}': float(p.dec), f'psi_{im}': float(p.psi), f'theta_jn_{im}': float(p.theta_jn),
                            f'phase_{im}': float(p.phase), f'a1_{im}': float(p.a_1), f'a2_{im}': float(p.a_2),
                            f'tilt1_{im}': float(p.tilt_1), f'tilt2_{im}': float(p.tilt_2), f'phi12_{im}': float(p.phi_12),
                            f'phijl_{im}': float(p.phi_jl), f't_real_{im}': float(p.geocent_time),
                            f'snr_sample_{im}': float(p.get('network_optimal_snr', np.nan)),
                            f'logl_sample_{im}': float(p.get('log_likelihood', np.nan))})
            row['run'] = row['run_a'] if row['run_a'] == row['run_b'] else f"{row['run_a']}/{row['run_b']}"
            rows.append(row)
    return pd.DataFrame(rows)


def tasks(df, csv, noise_dir, outdir, nlive=500, naccept=40):
    opts = f'--injections {csv} --noise-dir {noise_dir} --nlive {nlive} --naccept {naccept} --outdir {outdir}'
    lines = []
    for s in df.system:
        lines.append(f'single {s}A --prior-events {s}A {s}B {opts}')
        lines.append(f'single {s}B --prior-events {s}A {s}B {opts}')
    for s in df.system:
        lines.append(f'joint {s}A {s}B {opts}')
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pe-dir', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True, help='stage-2 task list (joint lines give the pairs)')
    p.add_argument('--placed', type=Path, required=True, help='placed lensed injections whose noise slots are reused')
    p.add_argument('--noise-dir', default='data/inject_noise_s1')
    p.add_argument('--out', type=Path, required=True, help='twin table (realpe --injections)')
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--outdir', default='results/bg_pe')
    p.add_argument('--extra', nargs='*', default=[], help='A+B:n  n additional twins of that pair')
    p.add_argument('--seed', type=int, default=31)
    p.add_argument('--used', type=Path, nargs='*', default=[], help='earlier twin tables whose noise slots must not be reused')
    p.add_argument('--prefix', default='BG', help='system-name prefix (distinct per table)')
    p.add_argument('--only-pairs', nargs='*', help='A+B ... restrict to these pairs of the task list')
    p.add_argument('--nlive', type=int, default=500)
    p.add_argument('--naccept', type=int, default=40)
    a = p.parse_args()
    pairs = [l.split()[1:3] for l in a.pairs.read_text().splitlines() if l.startswith('joint ')]
    if a.only_pairs:
        pairs = [ab for ab in pairs if f'{ab[0]}+{ab[1]}' in set(a.only_pairs)]
    extra = {kv.split(':')[0]: int(kv.split(':')[1]) for kv in a.extra}
    used = [n for u in a.used for t in [pd.read_csv(u)] for n in list(t.noise_a) + list(t.noise_b)]
    df = build(pairs, a.pe_dir, pd.read_csv(a.placed), np.random.default_rng(a.seed), extra, used, a.prefix)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    a.tasks.write_text('\n'.join(tasks(df, a.out, a.noise_dir, a.outdir, a.nlive, a.naccept)) + '\n')
    print(json.dumps(dict(pairs=len(pairs), twins=len(df), tasks=3 * len(df),
                          durations=df.duration_a.tolist() + df.duration_b.tolist(),
                          runs=df.run.value_counts().to_dict())))


if __name__ == '__main__':
    main()
