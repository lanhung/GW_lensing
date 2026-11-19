"""Marginal candidates without PE: search sky maps and template parameters from the LVK candidate releases.

The archive (1-1.5 GB tar.gz on Zenodo) is streamed and only the sky maps of the selected candidates are kept, so
nothing large is stored. Selection: p_astro = 1 - p_Terrestrial above --p-astro and no PE release for the event.
For each candidate the preferred search's sky map (the summary table's skymap_file) is written to <out>/skymaps/,
and <out>/candidates.csv holds GPS time, FAR, SNR, p_astro, source probabilities and the template parameters
(detector-frame chirp mass, mass1, mass2, spin1z, spin2z) of the pipeline that produced the sky map.
"""
import argparse
import tarfile
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

RELEASES = {
    'GWTC-5.0': ('https://zenodo.org/api/records/20348004/files/IGWN-GWTC5p0-29ebe06b7_25-SearchSummaryTable.hdf5/content',
                 'https://zenodo.org/api/records/20348004/files/IGWN-GWTC5p0-29ebe06b7_25-Archived_SearchResults.tar.gz/content'),
    'GWTC-4.1': ('https://zenodo.org/api/records/20276095/files/IGWN-GWTC4p1-18965dda8_5-SearchSummaryTable.hdf5/content',
                 'https://zenodo.org/api/records/20276095/files/IGWN-GWTC4p1-18965dda8_5-Archived_SearchResults.tar.gz/content'),
}


def table(f, name):
    df = pd.DataFrame(f[name][()])
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.decode('utf-8')
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('release', choices=list(RELEASES))
    p.add_argument('--pe-files', type=Path, required=True, help='files.csv of lensearch.fetch_pe (events with PE)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--p-astro', type=float, default=0.1)
    a = p.parse_args()
    out = a.out / a.release
    (out / 'skymaps').mkdir(parents=True, exist_ok=True)
    summ_url, tar_url = RELEASES[a.release]
    summ = out / 'search_summary.hdf5'
    if not summ.exists():
        urllib.request.urlretrieve(summ_url, summ)
    with h5py.File(summ, 'r') as f:
        s = table(f, 'search_summary')
        pipes = {k: table(f, k) for k in f.keys() if k != 'search_summary'}
    pe = set(pd.read_csv(a.pe_files).event)
    s['p_astro'] = 1 - s.p_Terrestrial
    sel = s[(s.p_astro > a.p_astro) & ~s.gw_name.isin(pe)].copy()
    for col in ('mass1', 'mass2', 'spin1z', 'spin2z', 'template_mchirp'):
        sel[col] = np.nan
    for i, r in sel.iterrows():
        t = pipes.get(r.pipeline)
        if t is None or 'mass1' not in t:
            continue
        m = t[t.gw_name == r.gw_name]
        if len(m):
            for col in ('mass1', 'mass2', 'spin1z', 'spin2z'):
                sel.at[i, col] = float(m[col].iloc[0])
            sel.at[i, 'template_mchirp'] = float(m['mchirp'].iloc[0])
    wanted = set(sel.skymap_file)
    print(f'{a.release}: {len(s)} candidates, {len(sel)} selected; streaming the archive for {len(wanted)} sky maps', flush=True)
    got = 0
    with urllib.request.urlopen(tar_url, timeout=600) as resp, tarfile.open(fileobj=resp, mode='r|gz') as tf:
        for member in tf:
            name = member.name.lstrip('./')
            key = next((w for w in wanted if name.endswith(w)), None)
            if key is None or not member.isfile():
                continue
            data = tf.extractfile(member).read()
            (out / 'skymaps' / Path(key).name).write_bytes(data)
            got += 1
    sel['skymap_local'] = [Path(k).name for k in sel.skymap_file]
    sel['skymap_found'] = [(out / 'skymaps' / Path(k).name).exists() for k in sel.skymap_file]
    sel.to_csv(out / 'candidates.csv', index=False)
    print(f'{got} sky maps written; {int(sel.skymap_found.sum())} of {len(sel)} candidates have one', flush=True)


if __name__ == '__main__':
    main()
