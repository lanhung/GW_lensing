"""Lensed GW pairs for the search's efficiency and background studies, injected into real O4 noise.

draw      Lensed systems from the populations of gwlens (GWTC-4 BBH population, SIS/SIE/EPL+shear lenses with the
          sigma^4-weighted velocity-dispersion function, lensed-source redshifts). For each system the two images
          with the largest |mu| form the pair; both must reach optimal network SNR >= --snr-min with the median
          O4a or O4b PSDs (lensearch data/psd) at their arrival times. Arrival time of image A is uniform in the run.
place     Each image is moved by a whole number of sidereal days into one of the real-noise blocks whose sidereal
          phase covers it, so its detector response is unchanged. Output: per image the GPS time in real data,
          plus the true delay. Run 'blocks' locally first (lensearch.inject blocks) to pick and cut noise blocks.
blocks    (local, GWOSC) choose N 4096-s stretches with all detectors observing and no catalogued event, and write
          the analysis segments around requested GPS times with a Welch PSD from the surrounding data.

The PE of an injection uses lensearch.realpe with --injection (strain = real noise + signal).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SIDEREAL_DAY = 86164.0905
RUNS = {  # GPS spans and detectors with public strain
    'O4a': dict(span=(1368975618, 1389456018), ifos=['H1', 'L1']),
    'O4b': dict(span=(1396396818, 1422118818), ifos=['H1', 'L1', 'V1']),
}
F_MIN = 20.0
PAD = 4.0        # s of extra noise on each side of an analysis segment


def register_scenario(run, psd_dir):
    from gwlens import detectors as det
    ifos = RUNS[run]['ifos']
    det.SCENARIOS[run] = dict(detectors=ifos, psd={i: str(Path(psd_dir).resolve() / f'{run}_{i}_median_psd.txt') for i in ifos},
                              f_min=F_MIN, sky_sigma_deg_at_rho20=10.0)
    return det


def draw(n_target, run, psd_dir, rng, family='EPL_shear', snr_min=10.0, max_draws=10 ** 7, library=None):
    from gwlens import lenses, population as pop
    det = register_scenario(run, psd_dir)
    t0, t1 = RUNS[run]['span']
    rows, n, n_fast = [], 0, 0
    while len(rows) < n_target and n < max_draws:
        n += 1
        zs = float(pop.sample_lensed_source_redshift(rng, 1)[0])
        zl = float(pop.sample_lens_redshift(rng, np.array([zs]))[0])
        sv = float(pop.sample_velocity_dispersion(rng, 1)[0])
        th = float(pop.einstein_radius_arcsec(sv, zl, zs))
        if library is not None:
            e = library.iloc[int(rng.integers(len(library)))]
            shape = {k[6:]: e[k] for k in library.columns if k.startswith('shape_')}
            sol = dict(mu=np.asarray(e['mu']), fermat=np.asarray(e['fermat']) * th ** 2, morse=np.asarray(e['morse']))
        else:
            shape = lenses.sample_lens_shape(rng, family)
            sol = lenses.solve(rng, family, th, shape)
            if sol is None:
                continue
        mu_faint = np.sort(np.abs(sol['mu']))[-2]            # fainter image of the pair
        m1, m2 = (float(v[0]) for v in pop.sample_masses(rng, 1))
        c1, c2 = (float(v[0]) for v in pop.sample_aligned_spins(rng, 1))
        dl = float(pop.luminosity_distance(zs))
        ra, dec = rng.uniform(0, 2 * np.pi), np.arcsin(rng.uniform(-1, 1))
        psi, cosi = rng.uniform(0, np.pi), rng.uniform(-1, 1)
        ta = rng.uniform(t0, t1)
        i0 = det.face_on_integrals(run, m1 * (1 + zs), m2 * (1 + zs), c1, c2, dl)
        # exact bound for the fainter image: its own |mu| and the best detector response over a sidereal day
        rho0_max = max(det.network_snr(i0, det.antenna(run, ra, dec, psi, ta + h * 3600.0), cosi) for h in range(24))
        if rho0_max * np.sqrt(mu_faint) < snr_min:
            n_fast += 1
            continue
        order = np.argsort(-np.abs(sol['mu']))[:2]
        tdel = lenses.time_delays_seconds(sol['fermat'], pop.time_delay_distance_seconds(zl, zs))
        order = order[np.argsort(tdel[order])]               # A = earlier of the two brightest images
        imgs = []
        for k in order:
            t = ta + tdel[k] - tdel[order[0]]
            rho = det.network_snr(i0, det.antenna(run, ra, dec, psi, t), cosi) * np.sqrt(abs(sol['mu'][k]))
            imgs.append((t, rho, sol['mu'][k], sol['morse'][k]))
        if min(r for _, r, _, _ in imgs) < snr_min:
            continue
        (ta_, ra_snr, mua, na), (tb_, rb_snr, mub, nb) = imgs
        rows.append(dict(system=f'{run}-{family}-{len(rows):05d}', run=run, family=family, z_s=zs, z_l=zl, sigma_v=sv,
                         theta_e=th, n_images=len(sol['mu']), m1_src=m1, m2_src=m2, chi1z=c1, chi2z=c2, dl=dl, ra=ra, dec=dec,
                         psi=psi, theta_jn=float(np.arccos(cosi)), phase=float(rng.uniform(0, 2 * np.pi)),
                         t_a=ta_, t_b=tb_, delta_t=tb_ - ta_, mu_a=float(mua), mu_b=float(mub), morse_a=float(na),
                         morse_b=float(nb), snr_a=float(ra_snr), snr_b=float(rb_snr), **{f'shape_{k}': v for k, v in shape.items()}))
    return pd.DataFrame(rows), n


def _library_chunk(args):
    family, n, seed = args
    from gwlens import lenses
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        shape = lenses.sample_lens_shape(rng, family)
        sol = lenses.solve(rng, family, 1.0, shape)
        if sol is None:
            continue
        out.append(dict(mu=sol['mu'].tolist(), fermat=sol['fermat'].tolist(), morse=sol['morse'].tolist(),
                        n_images=len(sol['mu']), **{f'shape_{k}': v for k, v in shape.items()}))
    return out


def build_library(family, n, npool, seed):
    """Scale-free lens solutions (theta_E = 1): magnifications, Fermat potentials (units theta_E^2) and Morse indices.
    Shape and source position are drawn independently of the source and lens redshifts, so an entry combined with any
    theta_E reproduces lenses.solve exactly (delays scale as theta_E^2)."""
    from multiprocessing import Pool
    chunks = [(family, n // npool + 1, seed + k) for k in range(npool)]
    with Pool(npool) as pool:
        rows = [r for part in pool.map(_library_chunk, chunks) for r in part]
    return pd.DataFrame(rows[:n])


def sidereal_phase(t):
    return np.mod(t, SIDEREAL_DAY)


def place(systems, blocks, rng, margin=200.0, spacing=150.0):
    """Assign each image to a noise block with the same sidereal phase (shift by whole sidereal days)."""
    used = {b: [] for b in blocks.block}
    out = []
    for r in systems.itertuples():
        rec = r._asdict()
        ok = True
        for im in ('a', 'b'):
            t = getattr(r, f't_{im}')
            cands = []
            for b in blocks.itertuples():
                if b.run != r.run:
                    continue
                k = np.round((b.start + b.duration / 2 - t) / SIDEREAL_DAY)
                tn = t + k * SIDEREAL_DAY
                if b.start + margin < tn < b.start + b.duration - margin and all(abs(tn - u) > spacing for u in used[b.block]):
                    cands.append((b.block, tn))
            if not cands:
                ok = False
                break
            blk, tn = cands[rng.integers(len(cands))]
            used[blk].append(tn)
            rec[f'block_{im}'], rec[f't_noise_{im}'] = blk, tn
        if ok:
            from gwlens.joint_pe import segment_duration
            zs = r.z_s
            rec['duration'] = segment_duration(r.m1_src * (1 + zs), r.m2_src * (1 + zs), F_MIN)
            out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Local part: noise blocks from GWOSC
# ---------------------------------------------------------------------------
def choose_blocks(run, n, rng, event_times):
    """4096-s GWOSC file starts with all detectors in DATA and CBC_CAT1 for the whole file and no catalogued event."""
    from gwosc.timeline import get_segments
    ifos = RUNS[run]['ifos']
    t0, t1 = RUNS[run]['span']
    segs = {i: get_segments(f'{i}_CBC_CAT1', t0, t1) for i in ifos}
    starts = np.arange((t0 // 4096 + 1) * 4096, t1 - 4096, 4096)
    good = []
    for s in starts:
        if any(abs(e - (s + 2048)) < 2048 + 64 for e in event_times):
            continue
        if all(any(a <= s and s + 4096 <= b for a, b in segs[i]) for i in ifos):
            good.append(int(s))
    sel = rng.choice(good, size=min(n, len(good)), replace=False)
    return pd.DataFrame(dict(block=[f'{run}-{s}' for s in sel], run=run, start=sel.astype(float), duration=4096.0))


def cut_noise(requests, out, tmp, psd_span=512.0, fs=4096):
    """For each (block, ifo, t, duration) cut the analysis segment and a Welch PSD from psd_span s around it."""
    import urllib.request
    from gwosc.locate import get_urls
    from gwpy.timeseries import TimeSeries
    from scipy import signal
    out.mkdir(parents=True, exist_ok=True)
    for (run, start), grp in requests.groupby(['run', 'start']):
        for ifo in RUNS[run]['ifos']:
            url = get_urls(ifo, int(start) + 10, int(start) + 11, sample_rate=4096)[0]
            path = tmp / url.split('/')[-1]
            if not path.exists():
                urllib.request.urlretrieve(url, path)
            ts = TimeSeries.read(str(path), format='hdf5.gwosc')
            for r in grp.itertuples():
                seg_start = r.t_noise + 2.0 - r.duration - PAD
                span = r.duration + 2 * PAD
                name = out / f'{ifo}-{r.image}-{seg_start:.3f}-{span:.0f}.hdf5'
                if name.exists():
                    continue
                seg = ts.crop(seg_start, seg_start + span)
                lo = max(float(ts.t0.value), r.t_noise - psd_span / 2)
                ref = ts.crop(lo, min(lo + psd_span, float(ts.t0.value) + 4096)).value
                f, p = signal.welch(ref, fs=fs, nperseg=int(4 * fs), noverlap=int(2 * fs), average='median')
                seg.write(str(name), format='hdf5', overwrite=True)
                np.savetxt(out / f'{ifo}-{r.image}-psd.txt', np.column_stack([f, p]))
            path.unlink()
            print(json.dumps(dict(run=run, block=int(start), ifo=ifo, images=len(grp))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('draw')
    d.add_argument('--run', choices=list(RUNS), required=True)
    d.add_argument('--n', type=int, default=100)
    d.add_argument('--family', default='EPL_shear')
    d.add_argument('--snr-min', type=float, default=10.0)
    d.add_argument('--psd-dir', type=Path, required=True)
    d.add_argument('--out', type=Path, required=True)
    d.add_argument('--seed', type=int, default=11)
    d.add_argument('--library', type=Path, help='lens-solution library (lensearch.inject library)')
    lb = sub.add_parser('library')
    lb.add_argument('--family', default='EPL_shear')
    lb.add_argument('--n', type=int, default=30000)
    lb.add_argument('--npool', type=int, default=16)
    lb.add_argument('--out', type=Path, required=True)
    lb.add_argument('--seed', type=int, default=21)
    b = sub.add_parser('blocks')
    b.add_argument('--run', choices=list(RUNS), required=True)
    b.add_argument('--n', type=int, default=60)
    b.add_argument('--events', type=Path, required=True, help='GWOSC allevents.json (to avoid catalogued events)')
    b.add_argument('--out', type=Path, required=True)
    b.add_argument('--seed', type=int, default=12)
    pl = sub.add_parser('place')
    pl.add_argument('--systems', type=Path, required=True)
    pl.add_argument('--blocks', type=Path, required=True)
    pl.add_argument('--out', type=Path, required=True)
    pl.add_argument('--seed', type=int, default=13)
    c = sub.add_parser('cut')
    c.add_argument('--placed', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--tmp', type=Path, required=True)
    a = p.parse_args()
    if a.cmd == 'draw':
        lib = pd.read_parquet(a.library) if a.library else None
        df, n = draw(a.n, a.run, a.psd_dir, np.random.default_rng(a.seed), a.family, a.snr_min, library=lib)
        df.to_csv(a.out, index=False)
        print(f'{len(df)} detected lensed pairs from {n} lensed systems ({len(df) / max(n, 1):.4f})')
    elif a.cmd == 'library':
        lib = build_library(a.family, a.n, a.npool, a.seed)
        lib.to_parquet(a.out, index=False)
        print(f'{len(lib)} {a.family} solutions; image multiplicity {lib.n_images.value_counts().to_dict()}')
    elif a.cmd == 'blocks':
        ev = [v['GPS'] for v in json.load(open(a.events))['events'].values()]
        choose_blocks(a.run, a.n, np.random.default_rng(a.seed), ev).to_csv(a.out, index=False)
    elif a.cmd == 'place':
        df = place(pd.read_csv(a.systems), pd.read_csv(a.blocks), np.random.default_rng(a.seed))
        df.to_csv(a.out, index=False)
        print(f'{len(df)} pairs placed')
    elif a.cmd == 'cut':
        pl = pd.read_csv(a.placed)
        req = []
        for r in pl.itertuples():
            for im in ('a', 'b'):
                run, start = getattr(r, f'block_{im}').split('-')
                req.append(dict(run=run, start=float(start), image=f'{r.system}{im.upper()}', t_noise=getattr(r, f't_noise_{im}'),
                                duration=float(r.duration)))
        a.tmp.mkdir(parents=True, exist_ok=True)
        cut_noise(pd.DataFrame(req), a.out, a.tmp)


if __name__ == '__main__':
    main()
