"""Extract what the lensing search needs from LVK PE releases on Zenodo, without downloading the files.

The release files (100 MB to 6 GB each) are read remotely with HTTP byte-range requests. For each event
and one analysis (IMRPhenomXPHM preferred, the same waveform family for every catalog) we keep:

  <out>/<event>.parquet      posterior samples (selected columns, at most N_MAX rows)
  <out>/<event>.npz          PSDs (one array per detector), sky map (HEALPix), calibration envelopes
  <out>/<event>.json         catalog, file, analysis, analysis settings (config file), prior strings,
                             maximum-likelihood sample (all columns)

Records are resolved to their latest Zenodo version. Existing outputs are skipped.
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

RECORDS = {  # catalog: (Zenodo record, file-name filter)
    'GWTC-2.1': (6513631, '_mixed_cosmo.h5'),
    'GWTC-3': (22685054, '_mixed_cosmo.h5'),
    'GWTC-4.1': (20275769, '-combined_PEDataRelease.hdf5'),
    'GWTC-5.0a': (20348005, '-combined_PEDataRelease.hdf5'),
    'GWTC-5.0b': (20348006, '-combined_PEDataRelease.hdf5'),
    'O4c-GW250207': (18600070, 'GW250207_combinedPHM_cal_metafile.hdf5'),
}
PREFERRED = ['C00:IMRPhenomXPHM-SpinTaylor', 'C01:IMRPhenomXPHM', 'C00:IMRPhenomXPHM', 'C01:IMRPhenomXPHM-SpinTaylor',
             'C00:IMRPhenomXPNR', 'C00:SEOBNRv5PHM', 'C01:SEOBNRv4PHM', 'C00:NRSur7dq4', 'C01:Mixed', 'C00:Mixed']
COLUMNS = ['chirp_mass', 'mass_ratio', 'mass_1', 'mass_2', 'chi_eff', 'chi_p', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12',
           'phi_jl', 'theta_jn', 'psi', 'phase', 'ra', 'dec', 'luminosity_distance', 'geocent_time', 'redshift',
           'network_matched_filter_snr', 'network_optimal_snr', 'log_likelihood', 'log_prior']
N_MAX = 10000


def zenodo_json(url):
    import requests
    for i in range(5):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(5 * (i + 1))
    raise RuntimeError(url)


def file_list():
    rows = []
    for cat, (rec, filt) in RECORDS.items():
        d = zenodo_json(f'https://zenodo.org/api/records/{rec}/versions/latest')
        for f in d['files']:
            if f['key'].endswith(filt):
                name = f['key']
                m = re.search(r'GW\d{6}(_\d{6})?', name)
                event = m.group(0) if m else name
                event = {'GW250207': 'GW250207_115645'}.get(event, event)
                rows.append(dict(catalog=cat.rstrip('ab'), record=d['id'], file=name, size=f['size'], event=event,
                                 url=f"https://zenodo.org/api/records/{d['id']}/files/{name}/content"))
    return pd.DataFrame(rows)


def _decode(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        v = v.tolist()
        if isinstance(v, list) and len(v) == 1:
            v = v[0]
        return _decode(v) if isinstance(v, bytes) else v
    return v.item() if hasattr(v, 'item') else v


def _group_to_dict(g, depth=0):
    out = {}
    for k, v in g.items():
        if isinstance(v, h5py.Group):
            if depth < 2:
                out[k] = _group_to_dict(v, depth + 1)
        else:
            try:
                out[k] = _decode(v[()])
            except Exception:
                pass
    return out


def choose(f):
    labels = [k for k in f.keys() if isinstance(f[k], h5py.Group) and 'posterior_samples' in f[k]]
    for p in PREFERRED:
        if p in labels:
            return p, labels
    return sorted(labels)[0], labels


def extract(row, out):
    import fsspec
    ev = row['event']
    if (out / f'{ev}.json').exists():
        return dict(event=ev, status='exists')
    t0 = time.time()
    fs = fsspec.filesystem('https', block_size=2 ** 22)
    with fs.open(row['url'], 'rb', cache_type='blockcache') as fo, h5py.File(fo, 'r') as f:
        label, labels = choose(f)
        g = f[label]
        ds = g['posterior_samples']
        n_total = len(ds)
        # bilby posterior samples are independent draws in random order: the first N_MAX rows are a fair subset and
        # reading a row slice keeps memory and transfer small for the largest releases
        arr = ds[:N_MAX]
        names = arr.dtype.names
        df = pd.DataFrame({c: arr[c] for c in COLUMNS if c in names})
        imax = int(np.argmax(arr['log_likelihood'])) if 'log_likelihood' in names else None
        maxl = {n: _decode(arr[n][imax]) for n in names} if imax is not None else {}
        npz = {}
        if 'psds' in g:
            for ifo in g['psds']:
                npz[f'psd_{ifo}'] = np.asarray(g['psds'][ifo][()], float)
        if 'calibration_envelope' in g:
            for ifo in g['calibration_envelope']:
                try:
                    npz[f'cal_{ifo}'] = np.asarray(g['calibration_envelope'][ifo][()], float)
                except Exception:
                    pass
        sky_meta = {}
        if 'skymap' in g and 'data' in g['skymap']:
            npz['skymap'] = np.asarray(g['skymap']['data'][()], float)
            if 'meta_data' in g['skymap']:
                sky_meta = _group_to_dict(g['skymap']['meta_data'])
        cfg = _group_to_dict(g['config_file']) if 'config_file' in g else {}
        pri = _group_to_dict(g['priors']['analytic']) if 'priors' in g and 'analytic' in g['priors'] else {}
        prior_df = None
        if 'priors' in g and 'samples' in g['priors']:
            ps = g['priors']['samples']
            cols = [c for c in ('chirp_mass', 'mass_ratio', 'chi_eff', 'luminosity_distance') if c in ps]
            if cols:
                prior_df = pd.DataFrame({c: np.asarray(ps[c][:20000]) for c in cols})
        meta = _group_to_dict(g['meta_data']['meta_data']) if 'meta_data' in g and 'meta_data' in g['meta_data'] else {}
    if len(df) > N_MAX:
        df = df.sample(N_MAX, random_state=0).reset_index(drop=True)
    df.to_parquet(out / f'{ev}.parquet', index=False)
    if prior_df is not None:
        prior_df.to_parquet(out / f'{ev}_prior.parquet', index=False)
    np.savez_compressed(out / f'{ev}.npz', **npz)
    info = dict(event=ev, catalog=row['catalog'], record=int(row['record']), file=row['file'], analysis=label,
                analyses=labels, n_samples_total=int(n_total), columns_available=list(names), config=cfg, priors=pri,
                meta_data=meta, skymap_meta=sky_meta, maxl=maxl, seconds=time.time() - t0)
    (out / f'{ev}.json').write_text(json.dumps(info, default=str))
    return dict(event=ev, status='ok', analysis=label, n=len(df), sky='skymap' in npz, psds=[k for k in npz if k.startswith('psd')],
                seconds=round(time.time() - t0, 1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--only', nargs='*', help='event names to extract (default: all)')
    p.add_argument('--list-only', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    files = a.out / 'files.csv'
    df = pd.read_csv(files) if files.exists() else file_list()
    df.to_csv(files, index=False)
    if a.only:
        df = df[df.event.isin(a.only)]
    print(f'{len(df)} files, {df["size"].sum() / 1e9:.1f} GB in the releases', flush=True)
    if a.list_only:
        return

    rows = [dict(r) for _, r in df.iterrows()]
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(_job, rows, [a.out] * len(rows)):
            print(json.dumps(res), flush=True)


def _job(r, out):
    err = None
    for attempt in range(6):
        try:
            return extract(r, out)
        except Exception as e:  # mostly Zenodo rate limiting (133 requests per minute)
            err = repr(e)
            time.sleep(75)
    return dict(event=r['event'], status='failed', error=err)


if __name__ == '__main__':
    sys.exit(main())
