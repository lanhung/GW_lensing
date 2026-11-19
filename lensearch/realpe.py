"""Stage 2: joint lensed and single-image parameter estimation of real GW event pairs.

For a pair of events (A earlier, B later) the strain around each event is read from GWOSC, and the
PSDs, detectors, segment duration, sampling rate and minimum frequency are taken from the LVK PE release
of that event, so that each single-image likelihood matches the published analysis up to calibration
marginalisation (not included here). The joint model is that of gwlens.joint_pe: shared source parameters,
image B with luminosity distance d_A / sqrt(mu_rel), arrival time t_A + delta_t and its own Morse index.

    ln C = ln Z_joint - ln Z_A - ln Z_B        (coherence ratio, lensed versus two unrelated events)

The single-image runs use the same waveform, priors and data, so the ratio is internally consistent.
Relative binning uses the maximum-likelihood sample of the LVK posterior as fiducial point.
"""
import argparse
import json
import time
from pathlib import Path

import bilby
import numpy as np
import pandas as pd

from gwlens.joint_pe import (RB_MARGIN, JointLensedLikelihood, discrete_morse, image_parameters, lensed_bbh,
                             lensed_bbh_relative_binning)

F_REF = 20.0
EXACT_MAX_SAMPLES = 32768   # duration x sampling rate up to which the exact likelihood is used (e.g. 8 s at 4096 Hz)


# ---------------------------------------------------------------------------
# LVK release: analysis settings, PSDs and fiducial point
# ---------------------------------------------------------------------------
def waveform_flags(cfg):
    """LALSimulation flags of the LVK run (e.g. PhenomXPrecVersion 320 = SpinTaylor precession), passed to bilby."""
    import ast
    v = cfg.get('waveform-arguments-dict', cfg.get('waveform_arguments_dict'))
    if not v or str(v).strip() in ('None', '{}'):
        return {}
    try:
        d = ast.literal_eval(str(v))
    except (ValueError, SyntaxError):
        return {}
    return {k: x for k, x in d.items() if k not in ('minimum_frequency', 'reference_frequency', 'waveform_approximant')}


INJECTIONS = {}   # image name -> injection record (set by --injections)


def injection_settings(image, pe_dir=None):
    """Settings, PSDs, a stand-in 'posterior' (for prior ranges) and the injection parameters of an injected image:
    a lensed image of lensearch.inject (aligned spins, apparent distance d_L/sqrt|mu|, Morse index of the image) or an
    unlensed twin of lensearch.twins (a posterior sample of a real event, whose posterior then sets the prior ranges)."""
    from gwlens.joint_pe import segment_duration
    from lensearch.inject import RUNS
    r = INJECTIONS[image]
    im = image[-1].lower()
    noise_dir = Path(r['noise_dir'])
    if f'src_{im}' in r:   # unlensed twin
        m1, m2 = float(r[f'm1_det_{im}']), float(r[f'm2_det_{im}'])
        d_app, t, morse = float(r[f'dl_{im}']), float(r[f't_{im}']), 0.0
        duration, fs = float(r[f'duration_{im}']), float(r[f'fs_{im}'])
        ifos, psd_name = str(r[f'ifos_{im}']).split('+'), r[f'noise_{im}']
        spins = dict(a_1=float(r[f'a1_{im}']), a_2=float(r[f'a2_{im}']), tilt_1=float(r[f'tilt1_{im}']),
                     tilt_2=float(r[f'tilt2_{im}']), phi_12=float(r[f'phi12_{im}']), phi_jl=float(r[f'phijl_{im}']))
        stand_in = pd.read_parquet(Path(pe_dir) / f"{r[f'src_{im}']}.parquet")
    else:                  # lensed image
        zs = r['z_s']
        m1, m2 = r['m1_src'] * (1 + zs), r['m2_src'] * (1 + zs)
        mu = abs(r[f'mu_{im}'])
        d_app, t, morse = r['dl'] / np.sqrt(mu), float(r[f't_noise_{im}']), float(r[f'morse_{im}'])
        duration = float(r.get('duration') or segment_duration(m1, m2, 20.0))
        fs = 4096.0 if m1 + m2 < 60 else 2048.0
        ifos, psd_name = RUNS[r['run']]['ifos'], image
        spins = dict(a_1=abs(r['chi1z']), a_2=abs(r['chi2z']), tilt_1=0.0 if r['chi1z'] >= 0 else np.pi,
                     tilt_2=0.0 if r['chi2z'] >= 0 else np.pi, phi_12=0.0, phi_jl=0.0)
        stand_in = None
    psds = {i: np.loadtxt(noise_dir / f'{i}-{psd_name}-psd.txt') for i in ifos}
    settings = dict(event=image, analysis='injection', detectors=sorted(psds), trigger_time=t,
                    duration=duration, sampling_frequency=fs, minimum_frequency=20.0, maximum_frequency=None,
                    post_trigger_duration=2.0, reference_frequency=20.0, waveform_flags={}, injection=True)
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    inj = dict(mass_1=m1, mass_2=m2, chirp_mass=mc, mass_ratio=m2 / m1, **spins,
               theta_jn=float(r[f'theta_jn_{im}'] if f'src_{im}' in r else r['theta_jn']),
               phase=float(r[f'phase_{im}'] if f'src_{im}' in r else r['phase']),
               psi=float(r[f'psi_{im}'] if f'src_{im}' in r else r['psi']),
               ra=float(r[f'ra_{im}'] if f'src_{im}' in r else r['ra']),
               dec=float(r[f'dec_{im}'] if f'src_{im}' in r else r['dec']),
               luminosity_distance=d_app, geocent_time=t, morse_n=morse)
    if stand_in is None:
        stand_in = pd.DataFrame(dict(chirp_mass=[0.7 * mc, 1.3 * mc], luminosity_distance=[0.3 * d_app, 2.0 * d_app]))
    return settings, psds, stand_in, dict(inj, log_likelihood=np.nan)


CANDIDATES = {}   # marginal search candidate -> record (set by --candidates)


def candidate_settings(name):
    """Settings, Welch PSDs, a stand-in 'posterior' (prior ranges) and a template-based fiducial point for a marginal
    candidate without LVK PE (lensearch.candidates)."""
    r = CANDIDATES[name]
    noise_dir = Path(r['noise_dir'])
    ifos = [i.strip() for i in str(r['instruments']).split(',') if (noise_dir / f'{i.strip()}-{name}-psd.txt').exists()]
    psds = {i: np.loadtxt(noise_dir / f'{i}-{name}-psd.txt') for i in ifos}
    settings = dict(event=name, analysis='candidate', detectors=sorted(psds), trigger_time=float(r['gps_time']),
                    duration=float(r['duration']), sampling_frequency=float(r['sampling_frequency']), minimum_frequency=20.0,
                    maximum_frequency=None, post_trigger_duration=2.0, reference_frequency=20.0, waveform_flags={})
    m1, m2 = float(r['mass1']), float(r['mass2'])
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    d = float(r['distance']) if np.isfinite(r['distance']) and r['distance'] > 0 else 2000.0
    best = dict(mass_1=m1, mass_2=m2, chirp_mass=mc, mass_ratio=min(m1, m2) / max(m1, m2), a_1=abs(r['spin1z']),
                a_2=abs(r['spin2z']), tilt_1=0.0 if r['spin1z'] >= 0 else np.pi, tilt_2=0.0 if r['spin2z'] >= 0 else np.pi,
                phi_12=0.0, phi_jl=0.0, theta_jn=0.5, phase=0.0, psi=0.0, ra=float(r['ra']), dec=float(r['dec']),
                luminosity_distance=d, geocent_time=float(r['gps_time']), log_likelihood=np.nan)
    stand_in = pd.DataFrame(dict(chirp_mass=[0.6 * mc, 1.5 * mc], luminosity_distance=[0.2 * d, 3.0 * d]))
    return settings, psds, stand_in, best


def release_settings(pe_dir, event):
    if event in INJECTIONS:
        return injection_settings(event, pe_dir)
    if event in CANDIDATES:
        return candidate_settings(event)
    """Analysis settings, PSDs, posterior samples and maximum-likelihood sample from lensearch.fetch_pe outputs."""
    meta = json.loads((pe_dir / f'{event}.json').read_text())
    cfg = meta['config'].get('config', meta['config'])
    npz = np.load(pe_dir / f'{event}.npz')
    psds = {k[4:]: np.asarray(npz[k]) for k in npz.files if k.startswith('psd_')}
    settings = dict(event=event, analysis=meta['analysis'], detectors=sorted(psds),
                    trigger_time=float(cfg.get('trigger-time', cfg.get('trigger_time'))),
                    duration=float(cfg['duration']),
                    sampling_frequency=float(cfg.get('sampling-frequency', cfg.get('sampling_frequency', 4096))),
                    minimum_frequency=cfg.get('minimum-frequency', cfg.get('minimum_frequency')),
                    maximum_frequency=cfg.get('maximum-frequency', cfg.get('maximum_frequency')),
                    post_trigger_duration=float(cfg.get('post-trigger-duration', cfg.get('post_trigger_duration', 2.0))),
                    reference_frequency=float(cfg.get('reference-frequency', cfg.get('reference_frequency', F_REF))),
                    waveform_flags=waveform_flags(cfg))
    post = pd.read_parquet(pe_dir / f'{event}.parquet')
    best = {k: float(v) for k, v in meta['maxl'].items() if isinstance(v, (int, float))}
    return settings, psds, post, best


def parse_fmin(v, key, default=20.0):
    """minimum-frequency is a number or a loose dict string such as '{ H1:20,L1:20, waveform: 13.33 }'."""
    import re
    if v is None:
        return default
    if isinstance(v, (int, float, np.floating)):
        return float(v)
    d = dict((k, float(x)) for k, x in re.findall(r"['\"]?(\w+)['\"]?\s*:\s*([0-9.eE+-]+)", str(v)))
    if d:
        return d.get(key, min(x for k, x in d.items() if k != 'waveform'))
    return float(v)


EXTRA_CACHES = []   # further directories with cut strain (candidates)


def strain_segment(ifo, start, duration, fs, cache):
    """Strain of one detector for [start, start + duration) from the cropped segments of lensearch.fetch_strain."""
    from gwpy.timeseries import TimeSeries
    paths = [p for c in [cache] + EXTRA_CACHES for p in sorted(Path(c).glob(f'{ifo}-*.hdf5'))]
    for path in paths:
        t0, span = path.stem.rsplit('-', 2)[1:]
        if float(t0) <= start + 1e-3 and start + duration <= float(t0) + float(span) + 1e-3:
            ts = TimeSeries.read(str(path), format='hdf5').crop(start, start + duration)
            if ts.sample_rate.value != fs:
                ts = ts.resample(fs)
            if not np.isfinite(ts.value).all():
                raise ValueError(f'{ifo}: non-finite strain in the analysis segment')
            return ts
    raise FileNotFoundError(f'no cropped {ifo} strain covering {start}..{start + duration} in {cache}')


def interferometers(settings, psds, cache):
    duration, fs = settings['duration'], min(settings['sampling_frequency'], 4096.0)
    start = settings['trigger_time'] + settings['post_trigger_duration'] - duration
    ifos = bilby.gw.detector.InterferometerList(settings['detectors'])
    for ifo in ifos:
        ifo.set_strain_data_from_gwpy_timeseries(strain_segment(ifo.name, start, duration, fs, cache))
        f, p = psds[ifo.name][:, 0], psds[ifo.name][:, 1]
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array=f, psd_array=p)
        ifo.minimum_frequency = parse_fmin(settings['minimum_frequency'], ifo.name)
        ifo.maximum_frequency = min(float(f.max()), 0.875 * fs / 2)
    if settings.get('injection'):
        from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
        gen = bilby.gw.WaveformGenerator(duration=duration, sampling_frequency=fs, start_time=start,
                                         frequency_domain_source_model=lensed_bbh,
                                         parameter_conversion=convert_to_lal_binary_black_hole_parameters,
                                         waveform_arguments=dict(waveform_approximant='IMRPhenomXPHM', reference_frequency=20.0,
                                                                 minimum_frequency=20.0))
        p = {k: v for k, v in settings['injection_parameters'].items() if k not in ('chirp_mass', 'mass_ratio', 'log_likelihood')}
        ifos.inject_signal(waveform_generator=gen, parameters=p, raise_error=False)
    return ifos, start


# ---------------------------------------------------------------------------
# Likelihoods and priors
# ---------------------------------------------------------------------------
PRECESSING = ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl']


def fiducial(best):
    from bilby.gw.conversion import generate_all_bbh_parameters
    keys = ['chirp_mass', 'mass_ratio', 'mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
            'theta_jn', 'phase', 'psi', 'ra', 'dec', 'luminosity_distance', 'geocent_time']
    p = {k: best[k] for k in keys if k in best}
    p.setdefault('phase', 0.0)
    for k in PRECESSING:
        p.setdefault(k, 0.0)
    p['morse_n'] = 0.0
    return p


def single_likelihood(settings, psds, best, cache, approximant, epsilon=0.025):
    from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
    ifos, start = interferometers(settings, psds, cache)
    fmin = min(ifo.minimum_frequency for ifo in ifos)
    fmin_wf = min(fmin, parse_fmin(settings['minimum_frequency'], 'waveform', fmin))
    wf_args = dict(waveform_approximant=approximant, reference_frequency=settings.get('reference_frequency', F_REF),
                   minimum_frequency=fmin_wf, **settings.get('waveform_flags', {}))
    kw = dict(duration=settings['duration'], sampling_frequency=min(settings['sampling_frequency'], 4096.0), start_time=start,
              parameter_conversion=convert_to_lal_binary_black_hole_parameters)
    exact_gen = bilby.gw.WaveformGenerator(frequency_domain_source_model=lensed_bbh, waveform_arguments=dict(wf_args), **kw)
    rb_gen = bilby.gw.WaveformGenerator(frequency_domain_source_model=lensed_bbh_relative_binning,
                                        waveform_arguments=dict(wf_args), **kw)
    fid = fiducial(best)
    low_fref = wf_args['reference_frequency'] < min(ifo.minimum_frequency for ifo in ifos)   # binning unreliable then
    if settings['duration'] * kw['sampling_frequency'] <= EXACT_MAX_SAMPLES or low_fref:
        exact = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos, waveform_generator=exact_gen)
        return exact, None, np.inf, ifos      # short segment: the exact likelihood is cheap and always right
    # Relative binning needs one frequency band for all detectors, and SpinTaylor precession needs it to start at or
    # below f_ref. Each detector keeps its own LVK band: its PSD is made effectively infinite outside it.
    f_lo = min([ifo.minimum_frequency for ifo in ifos] + [wf_args['reference_frequency']])
    f_hi = max(ifo.maximum_frequency for ifo in ifos)
    for ifo in ifos:
        lo, hi = ifo.minimum_frequency, ifo.maximum_frequency
        f = np.asarray(ifo.power_spectral_density.frequency_array, float)
        psd = np.asarray(ifo.power_spectral_density.psd_array, float).copy()
        psd[(f < lo) | (f > hi)] = 1e20 * np.nanmax(psd)
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array=f, psd_array=psd)
        ifo.minimum_frequency, ifo.maximum_frequency = f_lo, f_hi
    wf_args['minimum_frequency'] = min(wf_args['minimum_frequency'], f_lo)
    exact_gen = bilby.gw.WaveformGenerator(frequency_domain_source_model=lensed_bbh, waveform_arguments=dict(wf_args), **kw)
    rb_gen = bilby.gw.WaveformGenerator(frequency_domain_source_model=lensed_bbh_relative_binning,
                                        waveform_arguments=dict(wf_args), **kw)
    exact = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos, waveform_generator=exact_gen)
    rb = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(interferometers=ifos, waveform_generator=rb_gen,
                                                                     fiducial_parameters=fid, epsilon=epsilon)
    ceiling = exact.log_likelihood_ratio(dict(fid)) + RB_MARGIN
    return rb, exact, ceiling, ifos


MC_RANGE = None   # (lo, hi): diagnostic restriction of the chirp-mass prior (--mc-range), else None
DL_MIN = None     # lower edge of the distance prior (--dl-min), else 0.2 x the smallest posterior distance


def priors(posts, t_ref, dt_ab=None, precessing=True):
    """Priors shared by the single-image and joint runs of a pair: chirp-mass and distance ranges cover the
    posteriors of all events in `posts`; times are +-0.1 s around t_ref (and t_ref + dt_ab for image B).
    With MC_RANGE set, the chirp-mass prior is cut to that window (same density, ∝ Mc): the evidence of a single
    mode, to be weighted by the window's prior mass (hi^2-lo^2 ratio) when added to the full-prior evidence."""
    from bilby.core.prior import Cosine, PowerLaw, Sine, Uniform
    pr = bilby.gw.prior.BBHPriorDict(aligned_spin=not precessing)
    for k in list(pr):
        pr.pop(k)
    mc = np.concatenate([p.chirp_mass.values for p in posts])
    lo, hi = float(np.quantile(mc, 0.0005)), float(np.quantile(mc, 0.9995))
    lo, hi = 0.8 * lo, 1.2 * hi
    if MC_RANGE is not None:
        print(f'chirp-mass prior {lo:.4f}-{hi:.4f} cut to {MC_RANGE[0]}-{MC_RANGE[1]}', flush=True)
        lo, hi = max(lo, MC_RANGE[0]), min(hi, MC_RANGE[1])
    pr['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(lo, hi, name='chirp_mass')
    pr['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(0.05, 1.0, name='mass_ratio')
    if precessing:
        pr['a_1'], pr['a_2'] = Uniform(0, 0.99, name='a_1'), Uniform(0, 0.99, name='a_2')
        pr['tilt_1'], pr['tilt_2'] = Sine(name='tilt_1'), Sine(name='tilt_2')
        pr['phi_12'] = Uniform(0, 2 * np.pi, boundary='periodic', name='phi_12')
        pr['phi_jl'] = Uniform(0, 2 * np.pi, boundary='periodic', name='phi_jl')
    else:
        pr['chi_1'] = bilby.gw.prior.AlignedSpin(a_prior=Uniform(0, 0.99), name='chi_1')
        pr['chi_2'] = bilby.gw.prior.AlignedSpin(a_prior=Uniform(0, 0.99), name='chi_2')
    pr['ra'] = Uniform(0, 2 * np.pi, boundary='periodic', name='ra')
    pr['dec'] = Cosine(name='dec')
    pr['psi'] = Uniform(0, np.pi, boundary='periodic', name='psi')
    pr['theta_jn'] = Sine(name='theta_jn')
    pr['phase'] = Uniform(0, 2 * np.pi, boundary='periodic', name='phase')
    d = np.concatenate([p.luminosity_distance.values for p in posts])
    d_lo = max(10.0, 0.2 * float(d.min())) if DL_MIN is None else DL_MIN
    pr['luminosity_distance'] = PowerLaw(2.0, d_lo, 3.0 * float(d.max()), name='luminosity_distance')
    pr['geocent_time'] = Uniform(t_ref - 0.1, t_ref + 0.1, name='geocent_time')
    if dt_ab is not None:
        pr['delta_t'] = Uniform(dt_ab - 0.1, dt_ab + 0.1, name='delta_t')
        pr['ln_mu_rel'] = Uniform(-5, 5, name='ln_mu_rel')
        pr['morse_a'] = Uniform(0, 1.5, name='morse_a')
        pr['morse_b'] = Uniform(0, 1.5, name='morse_b')
    else:
        pr['morse_a'] = Uniform(0, 1.5, name='morse_a')
    return pr


class SingleImageLikelihood(bilby.Likelihood):
    """One image with a free Morse index (same parameterisation as image A of the joint model)."""

    def __init__(self, rb, exact, ceiling):
        super().__init__()
        self.rb, self.exact, self.ceiling = rb, exact, ceiling

    def log_likelihood_ratio(self, parameters=None):
        q = image_parameters(parameters, 'a')
        logl = self.rb.log_likelihood_ratio(parameters=q)
        if self.exact is not None and not logl <= self.ceiling:
            logl = self.exact.log_likelihood_ratio(parameters=q)
        return logl

    def noise_log_likelihood(self):
        return self.rb.noise_log_likelihood()

    def log_likelihood(self, parameters=None):
        return self.log_likelihood_ratio(parameters) + self.noise_log_likelihood()


def run(kind, events, pe_dir, outdir, cache, nlive, naccept, npool, approximant, seed, prior_events=None):
    """kind = 'single' (one event) or 'joint' (two events, A earlier). prior_events: the events whose posteriors set
    the shared prior ranges (for a single run that is compared with a joint run: both events of the pair)."""
    t0 = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    rel = [release_settings(pe_dir, ev) for ev in events]
    for s, _, _, b in rel:
        if s.get('injection'):
            s['injection_parameters'] = b
    prior_events = list(prior_events or events)
    posts = [release_settings(pe_dir, ev)[2] for ev in prior_events]
    precessing = approximant.endswith(('PHM', 'Pv2', 'Pv3'))
    likes = [single_likelihood(s, p, b, cache, approximant) for s, p, _, b in rel]
    names = list(events)
    t_a = rel[0][3].get('geocent_time', rel[0][0]['trigger_time'])
    tag = '' if prior_events == list(events) else '_for_' + '__'.join(prior_events)
    if kind == 'single':
        like = SingleImageLikelihood(*likes[0][:3])
        pr = priors(posts, t_a, precessing=precessing)
        label = f'single_{names[0]}{tag}_{approximant}'
    else:
        t_b = rel[1][3].get('geocent_time', rel[1][0]['trigger_time'])
        like = JointLensedLikelihood(likes[0][0], likes[1][0], exact=(likes[0][1], likes[1][1]),
                                     ceilings=(likes[0][2], likes[1][2]))
        pr = priors(posts, t_a, dt_ab=t_b - t_a, precessing=precessing)
        label = f'joint_{names[0]}__{names[1]}_{approximant}'
    res = bilby.run_sampler(likelihood=like, priors=pr, sampler='dynesty', nlive=nlive, sample='acceptance-walk',
                            naccept=naccept, npool=npool, outdir=str(outdir), label=label, save='hdf5', plot=False,
                            resume=True, check_point_delta_t=600, conversion_function=None, seed=seed)
    post = res.posterior
    for k in ('morse_a', 'morse_b'):
        if k in post:
            post[f'{k}_n'] = discrete_morse(post[k])
    post.drop(columns=[c for c in post.columns if c.startswith('recalib')], errors='ignore').to_parquet(
        outdir / f'{label}_posterior.parquet', index=False)
    summary = dict(kind=kind, events=list(events), approximant=approximant, nlive=nlive, naccept=naccept,
                   log_evidence=float(res.log_evidence), log_evidence_err=float(res.log_evidence_err),
                   log_bayes_factor=float(res.log_bayes_factor), log_noise_evidence=float(res.log_noise_evidence),
                   n_samples=len(post), seconds=time.time() - t0, settings=[r[0] for r in rel])
    (outdir / f'{label}_summary.json').write_text(json.dumps(summary, indent=1, default=float))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('kind', choices=['single', 'joint'])
    p.add_argument('events', nargs='+', help='event name(s); for joint: earlier event first')
    p.add_argument('--pe-dir', type=Path, required=True, help='lensearch.fetch_pe outputs')
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True, help='cropped strain from lensearch.fetch_strain')
    p.add_argument('--approximant', default='IMRPhenomXPHM')
    p.add_argument('--nlive', type=int, default=1000)
    p.add_argument('--naccept', type=int, default=60)
    p.add_argument('--npool', type=int, default=16)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--prior-events', nargs='*', help='events whose posteriors set the prior ranges (default: the analysed ones)')
    p.add_argument('--injections', type=Path, help='placed injections (lensearch.inject place); images are <system>A/B')
    p.add_argument('--noise-dir', type=Path, help='cut noise segments and PSDs (lensearch.inject cut)')
    p.add_argument('--candidates', type=Path, help='marginal candidate records (lensearch.candidates prepare)')
    p.add_argument('--cand-noise-dir', type=Path, help='cut strain and PSDs of the candidates (lensearch.candidates cut)')
    p.add_argument('--mc-range', type=float, nargs=2, help='diagnostic: cut the chirp-mass prior to [lo, hi]')
    p.add_argument('--dl-min', type=float, help='lower edge of the distance prior [Mpc] (e.g. a single-image run whose '
                   'posterior piles at the default edge, which the joint model reaches through mu_rel)')
    a = p.parse_args()
    global MC_RANGE, DL_MIN
    if a.mc_range:
        MC_RANGE = tuple(a.mc_range)
    DL_MIN = a.dl_min
    if a.candidates is not None:
        for r in pd.read_csv(a.candidates).to_dict('records'):
            CANDIDATES[r['name']] = dict(r, noise_dir=str(a.cand_noise_dir))
        EXTRA_CACHES.append(a.cand_noise_dir)
    if a.injections is not None:
        for r in pd.read_csv(a.injections).to_dict('records'):
            for im in 'AB':
                INJECTIONS[r['system'] + im] = dict(r, noise_dir=str(a.noise_dir))
        a.cache = a.noise_dir
    s = run(a.kind, a.events, a.pe_dir, a.outdir, a.cache, a.nlive, a.naccept, a.npool, a.approximant, a.seed, a.prior_events)
    print(json.dumps({k: s[k] for k in ('kind', 'events', 'log_evidence', 'log_bayes_factor', 'seconds')}))


if __name__ == '__main__':
    main()
