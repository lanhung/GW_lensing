"""Marginal search candidates (no LVK PE) for stage-2 parameter estimation with our own settings.

prepare  (GPU side) records for the candidates of selected pairs: GPS time, detectors, template masses and aligned spins,
         and the sky-map maximum and distance estimate (search sky map).
cut      (where GWOSC is fast) analysis segment and a Welch PSD (512 s around the candidate) per detector, written in the
         same layout as injected noise (<ifo>-<name>-<start>-<span>.hdf5, <ifo>-<name>-psd.txt); a detector without
         public data at that time is skipped.
lensearch.realpe --candidates records.csv --noise-dir <cut output> analyses them like any other event.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAD = 4.0
PSD_SPAN = 512.0


def prepare(pairs, cand_dirs, out):
    from ligo.skymap.io import read_sky_map
    import healpy as hp
    names = set(pairs.a) | set(pairs.b)
    rows = []
    for d in cand_dirs:
        c = pd.read_csv(d / 'candidates.csv')
        for r in c[c.gw_name.isin(names)].itertuples():
            prob, meta = read_sky_map(str(d / 'skymaps' / r.skymap_local), nest=False)
            ipix = int(np.argmax(prob))
            th, ph = hp.pix2ang(hp.npix2nside(len(prob)), ipix)
            mc = r.template_mchirp if np.isfinite(r.template_mchirp) else r.mchirp
            m1, m2 = r.mass1, r.mass2
            if not (np.isfinite(m1) and np.isfinite(m2)):         # cWB: equal masses with the estimated chirp mass
                m1 = m2 = mc * 2 ** 0.2
            rows.append(dict(name=r.gw_name, release=d.name, gps_time=r.gps_time, instruments=r.instruments,
                             mass1=m1, mass2=m2, spin1z=r.spin1z if np.isfinite(r.spin1z) else 0.0,
                             spin2z=r.spin2z if np.isfinite(r.spin2z) else 0.0, mchirp=mc, ra=float(ph),
                             dec=float(np.pi / 2 - th), distance=float(meta.get('distmean', np.nan)), snr=r.snr,
                             p_astro=r.p_astro, far=r.far))
    df = pd.DataFrame(rows)
    from gwlens.joint_pe import segment_duration
    df['duration'] = [segment_duration(r.mass1, r.mass2, 20.0) for r in df.itertuples()]
    df['sampling_frequency'] = np.where(df.mass1 + df.mass2 < 15, 4096.0, 2048.0)   # BBH: 2048 Hz keeps exact likelihoods
    df.to_csv(out, index=False)
    print(f'{len(df)} candidate records written to {out}')


def cut(records, out, tmp):
    import urllib.request
    from gwosc.locate import get_urls
    from gwpy.timeseries import TimeSeries
    from scipy import signal
    out.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    for r in records.itertuples():
        t = float(r.gps_time)
        seg_start = t + 2.0 - r.duration - PAD
        span = r.duration + 2 * PAD
        done = []
        for ifo in [s.strip() for s in str(r.instruments).split(',')]:
            target = out / f'{ifo}-{r.name}-{seg_start:.3f}-{span:.0f}.hdf5'
            if target.exists():
                done.append(ifo)
                continue
            lo, hi = t - PSD_SPAN / 2, t + PSD_SPAN / 2
            try:
                urls = sorted({u for u in get_urls(ifo, int(lo), int(hi), sample_rate=4096) if '-4096.' in u})
            except Exception:
                urls = []
            if not urls:
                continue
            parts = []
            for url in urls:
                path = tmp / url.split('/')[-1]
                if not path.exists():
                    urllib.request.urlretrieve(url, path)
                parts.append(TimeSeries.read(str(path), format='hdf5.gwosc').crop(max(lo, seg_start) - 1, hi + 1))
                path.unlink()
            ts = parts[0]
            for p in parts[1:]:
                ts = ts.append(p, gap='pad', pad=np.nan, inplace=False)
            seg = ts.crop(seg_start, seg_start + span)
            if not np.isfinite(seg.value).all():
                print(json.dumps(dict(candidate=r.name, ifo=ifo, skipped='non-finite strain in the analysis segment')), flush=True)
                continue
            ref = ts.crop(lo, hi).value
            ref = ref[np.isfinite(ref)]
            f, p = signal.welch(ref, fs=4096, nperseg=4 * 4096, noverlap=2 * 4096, average='median')
            seg.write(str(target), format='hdf5', overwrite=True)
            np.savetxt(out / f'{ifo}-{r.name}-psd.txt', np.column_stack([f, p]))
            done.append(ifo)
        print(json.dumps(dict(candidate=r.name, detectors=done)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    pr = sub.add_parser('prepare')
    pr.add_argument('--pairs', type=Path, required=True)
    pr.add_argument('--cand-dirs', type=Path, nargs='+', required=True)
    pr.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('cut')
    c.add_argument('--records', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--tmp', type=Path, required=True)
    a = p.parse_args()
    if a.cmd == 'prepare':
        prepare(pd.read_csv(a.pairs), a.cand_dirs, a.out)
    else:
        cut(pd.read_csv(a.records), a.out, a.tmp)


if __name__ == '__main__':
    main()
