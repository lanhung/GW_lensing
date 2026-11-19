"""Merge public strong-lens catalogues into one table of unique systems with physical parameters.

Inputs: the directory of downloaded catalogues (see the lensing-search notes). Every entry gives a position, a
grade flag (high = grade A/B or 'confident'), a type (galaxy or cluster) and, where the catalogue has them, the lens
redshift z_l, source redshift z_s, Einstein radius theta_e [arcsec] and velocity dispersion sigma_v [km/s].
Entries within 5 arcsec are one system (friends of friends). Each parameter of a system is taken from the most
precise catalogue that has it (PRIORITY: spectroscopic lens samples, DESI spectroscopic searches, Euclid and HSC
models, compilations); a photometric redshift is used only when no spectroscopic one exists.

Output: parquet with ra, dec, n_entries, catalogues, high_grade, type, z_l, z_l_spec, z_s, z_s_spec, theta_e, sigma_v.
"""
import argparse
import csv
import json
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
PRIORITY = ['spec_precise', 'desi_pair', 'desi_sfl', 'euclid', 'sugohi', 'slcomp', 'lenscat', 'cluster_desi', 'imaging']


def fnum(x):
    try:
        v = float(str(x).strip().split()[0])
    except (ValueError, IndexError):
        return np.nan
    return v if math.isfinite(v) and v > -90 else np.nan


class Collector:
    def __init__(self):
        self.rows = []

    def add(self, cat, tier, ra, dec, high=False, kind='galaxy', z_l=np.nan, z_s=np.nan, theta_e=np.nan, sigma_v=np.nan,
            zl_spec=True, zs_spec=True):
        ra, dec = fnum(ra), fnum(dec)
        if not (np.isfinite(ra) and np.isfinite(dec)):
            return
        z_l, z_s, theta_e, sigma_v = fnum(z_l), fnum(z_s), fnum(theta_e), fnum(sigma_v)
        if np.isfinite(theta_e) and not 0.05 < theta_e < 60:
            theta_e = np.nan
        if np.isfinite(sigma_v) and not 50 < sigma_v < 2000:
            sigma_v = np.nan
        if np.isfinite(z_l) and np.isfinite(z_s) and z_s <= z_l:
            z_s = np.nan
        self.rows.append(dict(cat=cat, tier=tier, ra=ra % 360, dec=dec, high=bool(high), kind=kind, z_l=z_l, z_s=z_s,
                              theta_e=theta_e, sigma_v=sigma_v, zl_spec=bool(zl_spec), zs_spec=bool(zs_spec)))


def rows(fn, delim=','):
    return list(csv.DictReader(open(fn, errors='ignore'), delimiter=delim))


def viz(fn, table=None):
    out, hdr, cur, skip = [], None, None, 0
    for line in open(fn, errors='ignore'):
        line = line.rstrip('\n')
        if line.startswith('#Table'):
            cur = line.split('\t')[1].rstrip(':') if '\t' in line else line
            hdr = None
            continue
        if line.startswith('#') or not line.strip():
            continue
        if hdr is None:
            hdr, skip = line.split('\t'), 2
            continue
        if skip:
            skip -= 1
            continue
        if table and table not in (cur or ''):
            continue
        out.append(dict(zip(hdr, [x.strip() for x in line.split('\t')])))
    return out


def pick(r, *names):
    for n in names:
        for k in r:
            if k.lower() == n.lower():
                return r[k]
    return None


def read_fits_table(fn):
    from astropy.table import Table
    return Table.read(fn, hdu=1)


def collect(d):
    C = Collector()
    # --- precise spectroscopic samples (VizieR tables with z_l, z_s, theta_E, sigma) ---
    skip = {'J_AJ_143_119', 'J_A_A_618_A56'}          # SQLS parent quasar sample; Gaia GraL candidates
    for fn in sorted((d / 'spec' / 'viz').glob('*.tsv')):
        if fn.stem in skip:
            continue
        for r in viz(fn):
            ra = pick(r, '_RA', 'RAJ2000', 'RAdeg', 'RA')
            de = pick(r, '_DE', 'DEJ2000', 'DEdeg', 'DE', 'Dec')
            zl = pick(r, 'zl', 'z_l', 'zL', 'zlens', 'zd', 'z(d)', 'zFG', 'z_FG', 'zL1', 'zdef')
            zs = pick(r, 'zs', 'z_s', 'zS', 'zsrc', 'zsource', 'z(s)', 'zBG', 'z_BG')
            te = pick(r, 'thetaE', 'bSIE', 'REin', 'RE', 'b', 'theta_E', 'ThetaE', 'Rein', 'bSIS')
            sv = pick(r, 'sigap', 'sigma', 'sig', 'sigSDSS', 'sigma*', 'vdisp', 'Vd', 'sigma0', 'sigSIE')
            if ra is None or de is None:
                continue
            grade = (pick(r, 'Grade', 'Class', 'Gr', 'Type') or '').strip()
            if fn.stem == 'J_MNRAS_520_3305' and not ('lens' in grade.lower() or 'quad' in grade.lower()):
                continue
            if fn.stem == 'J_ApJS_269_61' and grade not in ('A', 'B', 'C'):
                continue
            precise = zl is not None and zs is not None and fnum(zl) == fnum(zl) and fnum(zs) == fnum(zs)
            high = precise or (fn.stem == 'J_ApJS_269_61' and grade in ('A', 'B')) or fn.stem == 'J_MNRAS_520_3305'
            C.add(fn.stem, 'spec_precise' if precise else 'imaging', ra, de, high=high, z_l=zl, z_s=zs, theta_e=te, sigma_v=sv)
    # --- DESI DR1 pair-wise spectroscopic search (Hsu+25) ---
    for line in open(d / 'desi' / 'hsu25_fullcat.mrt', errors='ignore'):
        if line.startswith('DESI-'):
            g, dim = line[42:43].strip(), line[60:61] == '1'
            if g in ('A', 'B', 'C') or dim:
                C.add('desi_hsu25', 'desi_pair', line[22:31], line[32:41], high=g in ('A', 'B'), z_l=line[44:50],
                      z_s=line[51:57], theta_e=line[71:77], sigma_v=line[93:100])
    # --- DESI single-fibre lens VAC (Karp+): lens z, background [O II] z, SIS theta_E, sigma ---
    t = read_fits_table(d / 'spec' / 'desi_sfl_vac_unique_v2.fits')
    for r in t:
        C.add('desi_sfl', 'desi_sfl', r['TARGET_RA'], r['TARGET_DEC'], high=float(r['LENS_PROBABILITY']) > 0.5
              if 'LENS_PROBABILITY' in t.colnames else False, z_l=r['Z'], z_s=r['BGRND_Z'] if 'BGRND_Z' in t.colnames else np.nan,
              theta_e=r['EINSTEIN_RAD'], sigma_v=r['SIGMA'])
    # --- Euclid Q1 (grades) with lens-model Einstein radii ---
    mass = {r['id_str']: r['einstein_radius_effective_median_pdf'] for r in rows(d / 'q1_mass.csv')}
    for r in rows(d / 'q1_cat.csv'):
        if r['grade'] and r['grade'] in 'ABC':
            C.add('euclid_q1', 'euclid', r['right_ascension'], r['declination'], high=r['grade'] in 'AB',
                  theta_e=mass.get(r['id_str'], np.nan))
    for fn, cat in [('ml/euclid_q1_sldeF_lens_table.csv', 'euclid_F')]:
        for r in rows(d / fn):
            C.add(cat, 'euclid', pick(r, 'RA_deg', 'ra'), pick(r, 'Dec_deg', 'dec'), high=True,
                  z_l=pick(r, 'z_lens', 'zl', 'z_l', 'zlens'), z_s=pick(r, 'z_source', 'zs', 'z_s', 'zsource'),
                  theta_e=pick(r, 'theta_E', 'thetaE', 'einstein_radius', 'theta_E_arcsec'))
    for r in rows(d / 'ml/euclid_q1_astrovink_missed.csv'):
        C.add('euclid_vink', 'imaging', r['ra'], r['dec'], high=r['grade'] in 'AB')
    # --- SuGOHI (HSC): cols 3/4 spec z_l/z_s, 5/6 photo z_l/z_s, 7 theta_E, 12 grade ---
    for line in open(d / 'ground' / 'sugohi_public.csv', errors='ignore'):
        f = line.rstrip('\n').split(',')
        if len(f) > 12:
            zl, zls = (f[3], True) if fnum(f[3]) == fnum(f[3]) else (f[5], False)
            zs, zss = (f[4], True) if fnum(f[4]) == fnum(f[4]) else (f[6], False)
            C.add('sugohi', 'sugohi', f[1], f[2], high=f[12] in ('A', 'B'), z_l=zl, z_s=zs, theta_e=f[7],
                  zl_spec=zls, zs_spec=zss)
    # --- LaStBeRu/slcomp public dashboard (z_L, z_S) ---
    for o in json.load(open(d / 'spec' / 'slcomp_dashboard_catalog.json'))['objects']:
        C.add('slcomp', 'slcomp', o['RA'], o['DEC'], high=False, z_l=o.get('z_L'), z_s=o.get('z_S'))
    # --- lenscat (z_l; type galaxy/cluster/group) ---
    for r in rows(d / 'lenscat_catalog.csv'):
        C.add('lenscat', 'lenscat', r['RA [deg]'], r['DEC [deg]'], high=r['grading'] == 'confident',
              kind='cluster' if r['type'] in ('cluster', 'group') else 'galaxy', z_l=r['zlens'])
    # --- imaging-only candidate lists (positions and grades) ---
    for fn, cat, rk, dk, gk, hg in [('ml/desi_dr9_storfer.csv', 'storfer', 'ra', 'dec', 'grade', 'AB'),
                                    ('ml/desi_dr10_inchausti2025_811.csv', 'inchausti', 'ra', 'dec', 'grade', 'AB'),
                                    ('ml/unions_glue1_storfer2025.csv', 'glue', 'ra', 'dec', 'grade', 'AB'),
                                    ('ml/kids_dr4_unet_resnet_2026_result.csv', 'kids_liu26', 'RAJ2000', 'DECJ2000', 'CLASS', 'AB'),
                                    ('ml/kids_astronomaly_grespan2026.csv', 'kids_astronomaly', 'RA', 'DEC', 'Grade', 'AB')]:
        for r in rows(d / fn):
            g = (r.get(gk) or '').strip()
            if g in ('A', 'B', 'C'):
                C.add(cat, 'imaging', r[rk], r[dk], high=g in hg)
    for r in rows(d / 'ml/desi_ls_stein2022_new_lenses.tsv', '\t'):
        C.add('stein', 'imaging', r['ra'], r['dec'], high=r['grade'] in ('A', 'B'))
    for r in viz(d / 'ground/viz_J_A+A_699_A350.tsv'):
        if fnum(r.get('G')) >= 1.5:
            C.add('holismokes16', 'imaging', r['RAJ2000'], r['DEJ2000'], high=True)
    for tab in ('tablea1', 'tablec1'):
        for r in viz(d / 'ground/viz_J_A+A_688_A34.tsv', tab):
            C.add('teglie', 'imaging', r.get('RAJ2000'), r.get('DEJ2000'), high=True)
    for r in viz(d / 'z23.tsv'):
        if r.get('Type', '').strip() == 'lens' and r.get('Grade') in ('A', 'B', 'C'):
            C.add('delve', 'imaging', r['RAJ2000'], r['DEJ2000'], high=r['Grade'] in 'AB',
                  theta_e=pick(r, 'thetaE', 'RE', 'Rein'))
    for line in open(d / 'ground/gonzalez2025_t4_mrt.txt', errors='ignore'):
        if line.startswith('DESJ'):
            s = fnum(line[51:55])
            if s >= 1:
                C.add('gonzalez', 'imaging', line[29:39], line[40:50], high=s >= 2)
    # --- cluster-scale lenses (DESI Legacy Surveys) ---
    for r in rows(d / 'ml/desi_ls_clusterlens_2026.csv'):
        C.add('desi_cluster', 'cluster_desi', r['RA_BCG'], r['DEC_BCG'], high=r['GRADE'] in ('A', 'B'), kind='cluster',
              z_l=pick(r, 'Z_BCG', 'z_bcg', 'zl', 'Z'), theta_e=pick(r, 'THETA_E', 'theta_E', 'RE', 'Einstein_radius'))
    return pd.DataFrame(C.rows)


def merge(df, radius_arcsec=5.0):
    """Friends-of-friends within radius_arcsec (k-d tree on unit vectors), then per-system parameters by priority."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    ra, dec = np.radians(df.ra.values), np.radians(df.dec.values)
    xyz = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    pairs = cKDTree(xyz).query_pairs(2 * np.sin(np.radians(radius_arcsec / 3600) / 2), output_type='ndarray')
    n = len(df)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, group = connected_components(graph, directed=False)
    df = df.assign(group=group, prio=df.tier.map({t: k for k, t in enumerate(PRIORITY)})).sort_values(['group', 'prio'])
    g = df.groupby('group', sort=False)
    out = g.agg(ra=('ra', 'first'), dec=('dec', 'first'), n_entries=('cat', 'size'), high_grade=('high', 'any'),
                is_cluster=('kind', lambda k: (k == 'cluster').any()), catalogues=('cat', lambda c: ';'.join(sorted(set(c)))))
    for col, flag in (('z_l', 'zl_spec'), ('z_s', 'zs_spec')):
        spec = df[col].where(df[flag]).groupby(df.group, sort=False).first()
        anyv = g[col].first()
        out[col] = spec.fillna(anyv)
        out[col + '_spec'] = spec.notna()
    for col in ('theta_e', 'sigma_v'):
        out[col] = g[col].first()
    out['type'] = np.where(out.pop('is_cluster'), 'cluster', 'galaxy')
    return out.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--catalog-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    raw = collect(a.catalog_dir)
    print('entries', len(raw), raw.groupby('cat').size().sort_values(ascending=False).head(40).to_dict())
    m = merge(raw)
    m.to_parquet(a.out, index=False)
    print('systems', len(m), 'high grade', int(m.high_grade.sum()), 'clusters', int((m.type == 'cluster').sum()))
    for c in ('z_l', 'z_s', 'theta_e', 'sigma_v'):
        print(c, int(np.isfinite(m[c]).sum()))
    print('z_l and z_s and theta_e', int((np.isfinite(m.z_l) & np.isfinite(m.z_s) & np.isfinite(m.theta_e)).sum()))


if __name__ == '__main__':
    main()
