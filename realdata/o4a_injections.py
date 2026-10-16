"""Validate the relative-amplitude error model with injections into real O4a detector noise.

Lensed image pairs (same source, different magnifications, different arrival times) are injected
into public LIGO O4a strain from GWOSC (H1 and L1, 4 kHz). Each image is recovered with a
matched filter using the exact template, maximised over coalescence phase and a +-5 ms time
window, with the PSD estimated by Welch averaging of neighbouring data. The recovered relative
magnification ln(mu_B/mu_A) = 2 ln(rho_B/rho_A) is compared with the Gaussian-noise expectation
used by the simulation (sd = 2 sqrt(1/rho_A^2 + 1/rho_B^2)). The same injections are repeated in
coloured Gaussian noise generated from the same PSDs as a control.
"""
import argparse
import json
import urllib.request
from pathlib import Path

import h5py
import numpy as np
from scipy import signal, stats

FS = 4096
SEG = 64            # analysis segment (s)
PSD_SPAN = 512      # data used for the PSD estimate (s)
F_LOW = 20.0


def find_blocks(n_blocks, rng, start=1368195220, stop=1389456018):
    from gwosc.locate import get_urls
    blocks = []
    tried = 0
    while len(blocks) < n_blocks and tried < 400:
        tried += 1
        t = int(rng.uniform(start, stop - 8192))
        try:
            uh = get_urls('H1', t, t + 1, sample_rate=FS)
            ul = get_urls('L1', t, t + 1, sample_rate=FS)
        except Exception:
            continue
        if uh and ul and uh[0].split('-')[-2] == ul[0].split('-')[-2]:
            if uh[0] not in [b[0] for b in blocks]:
                blocks.append((uh[0], ul[0]))
    return blocks


def download(url, cache):
    path = cache / url.split('/')[-1]
    if not path.exists():
        tmp = path.with_suffix('.part')
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def load(path):
    with h5py.File(path, 'r') as f:
        x = f['strain/Strain'][:]
        gps0 = f['strain/Strain'].attrs['Xstart']
        dq = f['quality/simple/DQmask'][:]
    good = (dq & 0b11) == 0b11  # bits 0 and 1: DATA and CBC_CAT1 (1 Hz quality flags)
    return x, float(gps0), good


def clean_windows(x, good, need):
    """Start indices (s) of windows with finite data and good quality for `need` seconds."""
    ok = good.copy()
    ok &= np.array([np.isfinite(x[i * FS:(i + 1) * FS]).all() for i in range(len(good))])
    starts = []
    run = 0
    for i, v in enumerate(ok):
        run = run + 1 if v else 0
        if run >= need:
            starts.append(i - need + 1)
    return starts


def psd_welch(x):
    f, p = signal.welch(x, fs=FS, nperseg=4 * FS, noverlap=2 * FS, average='median')
    return f, p


def fd_waveform(m1, m2, chi1, chi2, dl, iota, df, fmax):
    import lal
    import lalsimulation as ls
    hp, hc = ls.SimInspiralChooseFDWaveform(m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0, chi1, 0, 0, chi2, dl * 1e6 * lal.PC_SI,
                                            iota, 0.0, 0, 0, 0, df, F_LOW, fmax, F_LOW, None, ls.IMRPhenomXAS)
    return hp.data.data, hc.data.data


_IFOS = {}


def project(ifo_name, hp, hc, f, ra, dec, psi, gps, t_in_seg):
    import bilby
    if ifo_name not in _IFOS:
        _IFOS[ifo_name] = bilby.gw.detector.get_empty_interferometer(ifo_name)
    ifo = _IFOS[ifo_name]
    fp = ifo.antenna_response(ra, dec, gps, psi, 'plus')
    fc = ifo.antenna_response(ra, dec, gps, psi, 'cross')
    dt = ifo.time_delay_from_geocenter(ra, dec, gps)
    return (fp * hp + fc * hc) * np.exp(-2j * np.pi * f * (t_in_seg + dt))


def matched_filter(dseg_f, h_f, psd, f, t_true):
    """|z| maximised over phase and +-5 ms around the injected time (single detector)."""
    band = (f >= F_LOW) & (f <= 1000)
    w = np.zeros_like(psd)
    w[band] = 1 / psd[band]
    norm = np.sqrt(4 * np.sum(np.abs(h_f) ** 2 * w) / SEG)
    zc = np.fft.ifft(np.concatenate([np.conj(h_f) * dseg_f * w, np.zeros(SEG * FS - len(f))])) * SEG * FS * 4 / SEG
    # Template already carries the arrival time; the peak is at lag 0 (wrap-around indices).
    lags = np.r_[np.arange(0, int(0.005 * FS) + 1), np.arange(SEG * FS - int(0.005 * FS), SEG * FS)]
    return float(np.max(np.abs(zc[lags])) / norm), float(norm)


def run_pair(rng, segs, psds, src, mus, gaussian):
    """Inject two images (magnifications mus) in two independent segments; return recovered SNRs."""
    f = np.fft.rfftfreq(SEG * FS, 1 / FS)
    df = 1 / SEG
    hp, hc = fd_waveform(src['m1'], src['m2'], src['c1'], src['c2'], src['dl'], src['iota'], df, FS / 2)
    n = min(len(hp), len(f))
    hp_full = np.zeros(len(f), complex)
    hc_full = np.zeros(len(f), complex)
    hp_full[:n], hc_full[:n] = hp[:n], hc[:n]
    out = []
    win = signal.windows.tukey(SEG * FS, 0.1)
    for k, mu in enumerate(mus):
        rho2_opt, rho2_hat = 0.0, 0.0
        for det in ('H1', 'L1'):
            x, gps_start = segs[k][det]
            psd_f, psd = psds[k][det]
            psd_i = np.interp(f, psd_f, psd)
            t_in = SEG / 2
            h = np.sqrt(mu) * project(det, hp_full, hc_full, f, src['ra'], src['dec'], src['psi'], gps_start + t_in, t_in)
            if gaussian:
                # Continuous-FT Gaussian noise: <|n(f)|^2> = T S(f) / 2.
                d_f_scaled = (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))) * np.sqrt(psd_i * SEG / 4) + h
            else:
                d_f_scaled = np.fft.rfft(x * win) / FS + h
            band = (f >= F_LOW) & (f <= 1000)
            rho_opt = np.sqrt(4 * np.sum(np.abs(h[band]) ** 2 / psd_i[band]) / SEG)
            rho_hat, _ = matched_filter(d_f_scaled, h / max(np.sqrt(mu), 1e-9), psd_i, f, t_in)
            rho2_opt += rho_opt ** 2
            rho2_hat += rho_hat ** 2
        out.append((np.sqrt(rho2_opt), np.sqrt(rho2_hat)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=Path('realdata/o4a_cache'))
    p.add_argument('--out-dir', type=Path, default=Path('realdata/outputs'))
    p.add_argument('--blocks', type=int, default=4)
    p.add_argument('--pairs', type=int, default=400)
    p.add_argument('--seed', type=int, default=4096)
    p.add_argument('--summarize', action='store_true', help='Only combine existing o4a_injections_seed*.csv files')
    a = p.parse_args()
    if a.summarize:
        summarize(a.out_dir)
        return
    rng = np.random.default_rng(a.seed)
    a.cache.mkdir(parents=True, exist_ok=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cached = sorted(a.cache.glob('H-H1_GWOSC_O4a_4KHZ_R1-*-4096.hdf5'))
    if cached:
        # Use pre-downloaded files (same 4096-s block for H1 and L1).
        blocks = [(str(h), str(h).replace('H-H1_', 'L-L1_')) for h in cached if Path(str(h).replace('H-H1_', 'L-L1_')).exists()]
    else:
        blocks = find_blocks(a.blocks, rng)
    data = []
    for uh, ul in blocks:
        xs = {d: load(Path(u) if Path(u).exists() else download(u, a.cache)) for d, u in (('H1', uh), ('L1', ul))}
        good = xs['H1'][2] & xs['L1'][2]
        starts = clean_windows(np.minimum(xs['H1'][0], xs['L1'][0]), good, PSD_SPAN + SEG)
        data.append((xs, starts, [uh, ul]))
    usable = [(xs, st, u) for xs, st, u in data if st]
    if not usable:
        raise SystemExit('No clean O4a windows found')

    def draw_segment():
        xs, st, _ = usable[rng.integers(len(usable))]
        s0 = st[rng.integers(len(st))]
        seg, psd = {}, {}
        for det in ('H1', 'L1'):
            x, gps0, _ = xs[det]
            psd_part = x[s0 * FS:(s0 + PSD_SPAN) * FS]
            seg_part = x[(s0 + PSD_SPAN) * FS:(s0 + PSD_SPAN + SEG) * FS]
            seg[det] = (seg_part, gps0 + s0 + PSD_SPAN)
            psd[det] = psd_welch(psd_part)
        return seg, psd

    rows = []
    for i in range(a.pairs):
        src = dict(m1=rng.uniform(20, 60), m2=0, c1=rng.uniform(-0.3, 0.3), c2=rng.uniform(-0.3, 0.3),
                   iota=np.arccos(rng.uniform(-1, 1)), ra=rng.uniform(0, 2 * np.pi), dec=np.arcsin(rng.uniform(-1, 1)),
                   psi=rng.uniform(0, np.pi), dl=1000.0)
        src['m2'] = src['m1'] * rng.uniform(0.4, 1.0)
        mu_a = 10 ** rng.uniform(0, 1.3)
        mu_b = mu_a * 10 ** rng.uniform(-1, 0)
        s1, p1 = draw_segment()
        s2, p2 = draw_segment()
        # Scale the distance so the brighter image has network SNR in [10, 60].
        probe = run_pair(rng, [s1, s2], [p1, p2], src, [mu_a, mu_b], gaussian=True)
        target = rng.uniform(10, 60)
        src['dl'] *= probe[0][0] / target
        for mode in ('o4a', 'gaussian'):
            res = run_pair(rng, [s1, s2], [p1, p2], src, [mu_a, mu_b], gaussian=(mode == 'gaussian'))
            (ra_opt, ra_hat), (rb_opt, rb_hat) = res
            rows.append(dict(pair=i, noise=mode, mu_a=mu_a, mu_b=mu_b, rho_a=ra_opt, rho_b=rb_opt, rho_a_hat=ra_hat, rho_b_hat=rb_hat))
    import pandas as pd
    pd.DataFrame(rows).assign(seed=a.seed).to_csv(a.out_dir / f'o4a_injections_seed{a.seed}.csv', index=False)
    (a.out_dir / f'o4a_blocks_seed{a.seed}.json').write_text(json.dumps([[Path(x).name for x in u] for _, _, u in usable]))
    summarize(a.out_dir)


def summarize(out_dir):
    import pandas as pd
    df = pd.concat([pd.read_csv(f) for f in sorted(out_dir.glob('o4a_injections_seed*.csv'))], ignore_index=True)
    blocks = sorted({tuple(b) for f in out_dir.glob('o4a_blocks_seed*.json') for b in json.loads(f.read_text())})
    df = df[(df.rho_b > 4)]
    df['ln_mu_rel_err'] = 2 * (np.log(df.rho_b_hat / df.rho_a_hat) - np.log(df.rho_b / df.rho_a))
    df['model_sd'] = 2 * np.sqrt(1 / df.rho_a ** 2 + 1 / df.rho_b ** 2)
    df['z'] = df.ln_mu_rel_err / df.model_sd
    df.to_csv(out_dir / 'o4a_injections.csv', index=False)
    summary = {'blocks': [list(b) for b in blocks], 'n_pairs': int(df.groupby(['seed', 'pair']).ngroups)}
    for mode, g in df.groupby('noise'):
        g = g[g.rho_b > 8]
        summary[mode] = dict(n=int(len(g)), z_mean=float(g.z.mean()), z_sd=float(g.z.std()),
                             z_sd_robust=float(1.4826 * np.median(np.abs(g.z - g.z.median()))),
                             frac_abs_z_gt3=float((g.z.abs() > 3).mean()), ks_pvalue_vs_normal=float(stats.kstest(g.z, 'norm').pvalue),
                             median_abs_err=float(g.ln_mu_rel_err.abs().median()))
    (out_dir / 'o4a_injection_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
