"""Standardised per-event posterior samples from LVK PE data releases (PESummary HDF5 files).

For every event we keep, from the preferred analysis of the release:
  chirp_mass (detector frame), mass_ratio, chi_eff, ra, dec, luminosity_distance, theta_jn, psi,
  geocent_time, and the log prior weight of the luminosity distance (to turn the posterior into a
  likelihood), plus the sky map (HEALPix posterior density) when the release provides one.

Output: <out>/<event>.parquet (samples) and <out>/<event>.json (metadata, chosen analysis, PSD/calibration
availability). Files are written once; reruns skip existing outputs.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Preference order for the analysis label inside a release file (combined samples first).
PREFERRED = ['C00:Mixed', 'C01:Mixed', 'C00:Mixed+XO4a', 'C00:IMRPhenomXPHM-SpinTaylor', 'C01:IMRPhenomXPHM',
             'C00:IMRPhenomXPHM', 'C01:IMRPhenomPv2', 'C00:IMRPhenomPv2', 'C01:SEOBNRv4PHM', 'C00:SEOBNRv5PHM',
             'C00:NRSur7dq4', 'C01:NRSur7dq4']
KEEP = ['chirp_mass', 'mass_ratio', 'chi_eff', 'ra', 'dec', 'luminosity_distance', 'theta_jn', 'psi', 'geocent_time',
        'mass_1', 'mass_2', 'log_likelihood']
N_MAX = 20000


def choose_analysis(f):
    labels = [k for k in f.keys() if isinstance(f[k], h5py.Group) and 'posterior_samples' in f[k]]
    for p in PREFERRED:
        if p in labels:
            return p, labels
    mixed = [l for l in labels if 'Mixed' in l]
    return (mixed or sorted(labels))[0], labels


def read_samples(grp):
    ps = grp['posterior_samples']
    if isinstance(ps, h5py.Dataset):
        arr = ps[()]
        df = pd.DataFrame({n: arr[n] for n in arr.dtype.names if n in KEEP})
    else:  # older layout: group with 'parameter_names' and 'samples'
        names = [n.decode() if isinstance(n, bytes) else n for n in ps['parameter_names'][()]]
        s = ps['samples'][()]
        df = pd.DataFrame({n: s[:, i] for i, n in enumerate(names) if n in KEEP})
    if 'chirp_mass' not in df and {'mass_1', 'mass_2'} <= set(df):
        m1, m2 = df.mass_1, df.mass_2
        df['chirp_mass'] = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        df['mass_ratio'] = m2 / m1
    return df


def read_skymap(grp):
    if 'skymap' not in grp:
        return None
    sk = grp['skymap']
    data = sk['data'][()] if 'data' in sk else None
    return None if data is None else np.asarray(data, float)


def standardise(path, out):
    name = Path(path).name
    with h5py.File(path, 'r') as f:
        label, labels = choose_analysis(f)
        g = f[label]
        df = read_samples(g)
        sky = read_skymap(g)
        meta = dict(file=name, analysis=label, analyses=labels, n_samples=int(len(df)),
                    has_psds='psds' in g, has_calibration='calibration_envelope' in g or 'priors' in g and 'calibration' in g['priors'],
                    has_skymap=sky is not None)
        if 'config_file' in g:
            cfg = g['config_file']
            for sec in cfg:
                for key in ('trigger-time', 'trigger_time', 'detectors', 'channel-dict', 'channel_dict', 'duration',
                            'sampling-frequency', 'sampling_frequency', 'minimum-frequency', 'minimum_frequency',
                            'reference-frequency', 'reference_frequency', 'waveform-approximant', 'waveform_approximant'):
                    if key in cfg[sec]:
                        v = cfg[sec][key][()]
                        v = v[0] if hasattr(v, '__len__') and not isinstance(v, (bytes, str)) and len(v) == 1 else v
                        meta[key.replace('-', '_')] = v.decode() if isinstance(v, bytes) else (v.tolist() if hasattr(v, 'tolist') else v)
    if len(df) > N_MAX:
        df = df.sample(N_MAX, random_state=0).reset_index(drop=True)
    return df, sky, meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('files', nargs='+', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--event-from', default='GW', help='event name = substring of the file name starting here')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for path in a.files:
        stem = path.name
        i = stem.find(a.event_from)
        ev = stem[i:].split('_PEDataRelease')[0].split('-')[0] if i >= 0 else path.stem
        ev = '_'.join(ev.split('_')[:2])
        if (a.out / f'{ev}.parquet').exists():
            continue
        try:
            df, sky, meta = standardise(path, a.out)
        except Exception as e:  # report and continue with the next file
            print(json.dumps(dict(event=ev, file=path.name, error=repr(e))), flush=True)
            continue
        df.to_parquet(a.out / f'{ev}.parquet', index=False)
        if sky is not None:
            np.save(a.out / f'{ev}_skymap.npy', sky.astype(np.float32))
        meta['event'] = ev
        (a.out / f'{ev}.json').write_text(json.dumps(meta, indent=1, default=str))
        print(json.dumps(dict(event=ev, analysis=meta['analysis'], n=meta['n_samples'], sky=meta['has_skymap'])), flush=True)


if __name__ == '__main__':
    main()
