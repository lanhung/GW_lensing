"""Cut the analysis segments of real events out of GWOSC strain files.

Run where GWOSC is fast. For each event the 4096-s GWOSC files (4 kHz, or 16 kHz when the LVK analysis used a
higher sampling rate) covering [trigger + post_trigger - duration - PAD, trigger + post_trigger + PAD] are
downloaded, cropped to that interval and written as <out>/<ifo>-<event>-<start>-<duration>.hdf5 (gwpy, float64);
the large files are deleted. Analysis settings come from the event's lensearch.fetch_pe JSON.
"""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

PAD = 8.0


def segment_for(meta):
    cfg = meta['config'].get('config', meta['config'])
    t = float(cfg.get('trigger-time', cfg.get('trigger_time')))
    dur = float(cfg.get('duration'))
    post = float(cfg.get('post-trigger-duration', cfg.get('post_trigger_duration', 2.0)))
    fs = float(cfg.get('sampling-frequency', cfg.get('sampling_frequency', 4096)))
    dets = cfg.get('detectors')
    if isinstance(dets, str):
        dets = [d.strip(" '[]\"") for d in dets.replace(',', ' ').split() if d.strip(" '[]\"")]
    return t + post - dur - PAD, dur + 2 * PAD, fs, dets


def cut(event, meta, out, tmp):
    from gwosc.locate import get_urls
    from gwpy.timeseries import TimeSeries
    start, span, fs, dets = segment_for(meta)
    rate = 4096                      # lensearch.realpe analyses at most at 4096 Hz
    written = []
    for ifo in dets:
        target = out / f'{ifo}-{event}-{start:.3f}-{span:.0f}.hdf5'
        if target.exists():
            written.append(target.name)
            continue
        urls = get_urls(ifo, int(start) - 1, int(start + span) + 1, sample_rate=rate)
        # keep the 4096-s bulk files only (O1-O3 also list event-centred files covering the same times)
        bulk = sorted({u for u in urls if '-4096.' in u}, key=lambda u: int(u.split('-')[-2]))
        urls = bulk or urls[:1]
        parts = []
        for url in urls:
            path = tmp / url.split('/')[-1]
            if not path.exists():
                urllib.request.urlretrieve(url, path)
            parts.append(TimeSeries.read(str(path), format='hdf5.gwosc').crop(start - 1, start + span + 1))
            path.unlink()
        ts = parts[0]
        for p in parts[1:]:
            ts = ts.append(p, gap='raise', inplace=False)
        ts = ts.crop(start, start + span)
        ts.write(str(target), format='hdf5', overwrite=True)
        written.append(target.name)
        if not np.isfinite(ts.value).all():
            print(json.dumps(dict(event=event, ifo=ifo, warning='non-finite samples in segment')), flush=True)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('events', nargs='+')
    p.add_argument('--pe-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tmp', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.tmp.mkdir(parents=True, exist_ok=True)
    for ev in a.events:
        meta = json.loads((a.pe_dir / f'{ev}.json').read_text())
        try:
            print(json.dumps(dict(event=ev, files=cut(ev, meta, a.out, a.tmp))), flush=True)
        except Exception as e:
            print(json.dumps(dict(event=ev, error=repr(e))), flush=True)


if __name__ == '__main__':
    main()
