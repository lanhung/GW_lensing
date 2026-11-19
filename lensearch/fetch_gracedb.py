"""Public O4c (and later) significant alerts from GraceDB: sky maps with distance, and source classification.

For every significant, non-retracted superevent in a GPS range we keep the latest Bilby multi-order sky map
(else Bayestar), rasterised to HEALPix nside NSIDE (ring ordering): probability per pixel and the
distance-ansatz parameters (DISTMU, DISTSIGMA, DISTNORM), plus the preferred pipeline's p_astro.

Output: <out>/<superevent>.npz, <out>/<superevent>.json, <out>/superevents.csv
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = 'https://gracedb.ligo.org/api/superevents/'
NSIDE = 128
O4C = (1422118818, 1447520418)


def get(url, **kw):
    for i in range(5):
        try:
            r = requests.get(url, timeout=120, **kw)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except Exception:
            time.sleep(5 * (i + 1))
    raise RuntimeError(url)


def list_superevents(t0, t1):
    q = f'gpstime: {t0} .. {t1} SIGNIF_LOCKED'
    url, out = API, []
    params = dict(query=q, format='json', count=100)
    while url:
        r = get(url, params=params).json()
        out += r['superevents']
        url, params = r['links'].get('next'), None
    return out


def rasterise(fits_bytes):
    from astropy.table import Table
    from ligo.skymap.bayestar import rasterize
    from ligo.skymap.io import read_sky_map
    import healpy as hp
    sm = read_sky_map(io.BytesIO(fits_bytes), moc=True, distances=True)
    r = rasterize(sm, order=int(np.log2(NSIDE)))
    out = {}
    for col in ('PROB', 'DISTMU', 'DISTSIGMA', 'DISTNORM'):
        if col in r.colnames:
            x = np.asarray(r[col], float)
            out[col.lower()] = hp.reorder(x, n2r=True)  # rasterize returns NESTED order
    return out, dict(sm.meta)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gps', type=float, nargs=2, default=O4C)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in list_superevents(*a.gps):
        sid = s['superevent_id']
        retracted = 'ADVNO' in s['labels']
        row = dict(superevent=sid, t_0=s['t_0'], far=s['far'], retracted=retracted, pipeline=s['preferred_event_data']['pipeline'],
                   instruments=s['preferred_event_data']['instruments'], labels=';'.join(s['labels']))
        if retracted:
            rows.append(row)
            continue
        files = get(f'{API}{sid}/files/').json()
        names = list(files)
        choice = next((n for n in ('Bilby.multiorder.fits', 'Bilby.offline1.multiorder.fits', 'Bilby.offline0.multiorder.fits',
                                   'bayestar.multiorder.fits') if n in names), None)
        row['skymap'] = choice
        if choice and not (a.out / f'{sid}.npz').exists():
            maps, meta = rasterise(get(f'{API}{sid}/files/{choice}').content)
            np.savez_compressed(a.out / f'{sid}.npz', **maps)
            (a.out / f'{sid}.json').write_text(json.dumps(dict(row, skymap_meta=meta), default=str))
        pa = next((n for n in names if n.endswith('.p_astro.json') and ',' not in n and n.lower().startswith(row['pipeline'].lower())),
                  next((n for n in names if n.endswith('.p_astro.json') and ',' not in n), None))
        if pa:
            row.update({f'p_{k}': v for k, v in get(f'{API}{sid}/files/{pa}').json().items()})
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ('superevent', 'skymap', 'p_BBH', 'p_Terrestrial')}), flush=True)
    pd.DataFrame(rows).to_csv(a.out / 'superevents.csv', index=False)
    print(len(rows), 'superevents;', sum(not r['retracted'] for r in rows), 'not retracted')


if __name__ == '__main__':
    main()
