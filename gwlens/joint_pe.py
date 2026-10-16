"""Waveform-level simulation and joint Bayesian parameter estimation of lensed image pairs.

For a selected lens system, both images are injected into simulated detector data (coloured
Gaussian noise from the scenario PSDs) with IMRPhenomXHM (aligned spins, higher modes), each
with its own magnification, arrival time (hence antenna pattern) and Morse phase. The pair is
then analysed with a joint likelihood in which the images share all intrinsic and extrinsic
source parameters and differ only by

    luminosity distance   d_B = d_A / sqrt(mu_rel),   mu_rel = |mu_B| / |mu_A|,
    arrival time          t_B = t_A + delta_t,
    Morse index           n_A, n_B in {0, 1/2, 1},

where d_A = d_L / sqrt|mu_A| is the apparent distance of image A. Each image uses a relative-
binning likelihood (Zackay, Dai & Venumadhav 2018), and the joint posterior is sampled with
dynesty. The posterior on (ln mu_rel, ln d_A, sky, orientation, Morse indices) replaces the
SNR-based measurement model of gwlens.catalogue for this subset.
"""
import argparse
import json
import time
from pathlib import Path

import bilby
import numpy as np
import pandas as pd

APPROXIMANT = 'IMRPhenomXHM'
F_REF = 20.0
SAMPLING_FREQUENCY = 2048.0
SCENARIO_IFOS = {'ET': ['ET'], 'O5': ['H1', 'L1', 'V1']}
SCENARIO_PSD = {'ET': {'ET': 'ET_D_psd.txt'}, 'O5': {'H1': 'Aplus_asd.txt', 'L1': 'Aplus_asd.txt', 'V1': 'AdV_asd.txt'}}
SCENARIO_FMIN = {'ET': 5.0, 'O5': 10.0}


def _morse(wf, morse_n):
    if wf is None:
        return None
    fac = np.exp(-1j * np.pi * morse_n)
    return {k: v * fac for k, v in wf.items()}


def lensed_bbh(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
               theta_jn, phase, morse_n=0.0, **kwargs):
    """Full-grid lensed image: the unlensed waveform times exp(-i pi n) for positive frequencies."""
    from bilby.gw.source import lal_binary_black_hole
    wf = lal_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2,
                               phi_jl, theta_jn, phase, **kwargs)
    return _morse(wf, morse_n)


def lensed_bbh_relative_binning(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2,
                                phi_jl, theta_jn, phase, morse_n=0.0, **kwargs):
    """Relative-binning version of lensed_bbh (see bilby.gw.source.lal_binary_black_hole_relative_binning)."""
    from bilby.gw.source import lal_binary_black_hole_relative_binning
    wf = lal_binary_black_hole_relative_binning(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12,
                                                a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)
    return _morse(wf, morse_n)


def discrete_morse(x):
    """Map a continuous variable on [0, 1.5) to a Morse index in {0, 1/2, 1}."""
    return 0.5 * np.floor(2.0 * np.clip(x, 0.0, 1.4999))


def image_parameters(p, image):
    """Map joint parameters to the parameters of image 'a' or 'b'."""
    q = dict(p)
    if image == 'a':
        q['morse_n'] = discrete_morse(p['morse_a'])
    else:
        q['luminosity_distance'] = p['luminosity_distance'] * np.exp(-0.5 * p['ln_mu_rel'])
        q['geocent_time'] = p['geocent_time'] + p['delta_t']
        q['morse_n'] = discrete_morse(p['morse_b'])
    for k in ('ln_mu_rel', 'delta_t', 'morse_a', 'morse_b'):
        q.pop(k, None)
    return q


class JointLensedLikelihood(bilby.Likelihood):
    """Product of two single-image likelihoods with shared source parameters."""

    def __init__(self, like_a, like_b):
        super().__init__()
        self.like_a, self.like_b = like_a, like_b

    def log_likelihood_ratio(self, parameters=None):
        return (self.like_a.log_likelihood_ratio(parameters=image_parameters(parameters, 'a'))
                + self.like_b.log_likelihood_ratio(parameters=image_parameters(parameters, 'b')))

    def noise_log_likelihood(self):
        return self.like_a.noise_log_likelihood() + self.like_b.noise_log_likelihood()

    def log_likelihood(self, parameters=None):
        return self.log_likelihood_ratio(parameters) + self.noise_log_likelihood()


def interferometers(scenario, start, duration, seed):
    import os
    import bilby
    np.random.seed(seed)
    ifos = bilby.gw.detector.InterferometerList(SCENARIO_IFOS[scenario])
    base = os.path.join(os.path.dirname(bilby.__file__), 'gw', 'detector', 'noise_curves')
    for ifo in ifos:
        key = 'ET' if ifo.name.startswith('ET') else ifo.name
        fname = os.path.join(base, SCENARIO_PSD[scenario][key])
        if 'asd' in fname:
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=fname)
        else:
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file=fname)
        ifo.minimum_frequency = SCENARIO_FMIN[scenario]
        ifo.maximum_frequency = SAMPLING_FREQUENCY / 2 * 0.9
    ifos.set_strain_data_from_power_spectral_densities(sampling_frequency=SAMPLING_FREQUENCY, duration=duration,
                                                     start_time=start)
    return ifos


def segment_duration(m1_det, m2_det, f_min):
    from bilby.gw.utils import calculate_time_to_merger
    t = calculate_time_to_merger(f_min, m1_det, m2_det, safety=1.2)
    return float(2 ** int(np.ceil(np.log2(t + 4.0))))


def event_parameters(row, scenario):
    """True injection parameters (image A) and the true joint-parameter vector."""
    z = row.true_z_s
    m1, m2 = row.true_m1 * (1 + z), row.true_m2 * (1 + z)
    mu_a, mu_b = abs(row[f'{scenario}__true_mu_signed_a']), abs(row[f'{scenario}__true_mu_signed_b'])
    t_a = row.true_t0 + row[f'{scenario}__true_t_a']
    base = dict(mass_1=m1, mass_2=m2, chi_1=row.true_chi1z, chi_2=row.true_chi2z, ra=row.true_ra, dec=row.true_dec,
                psi=row.true_psi, theta_jn=float(np.arccos(row.true_cos_iota)), phase=1.3,
                luminosity_distance=row.true_dl / np.sqrt(mu_a), geocent_time=t_a)
    truth = dict(base, chirp_mass=(m1 * m2) ** 0.6 / (m1 + m2) ** 0.2, mass_ratio=m2 / m1,
                 ln_mu_rel=float(np.log(mu_b / mu_a)), delta_t=float(row[f'{scenario}__true_dt_ab']),
                 morse_a=float(row[f'{scenario}__true_morse_a']) + 0.25, morse_b=float(row[f'{scenario}__true_morse_b']) + 0.25)
    return base, truth


def priors_for(truth):
    import bilby
    from bilby.core.prior import Uniform, PowerLaw, Sine, Cosine
    pr = bilby.gw.prior.BBHPriorDict(aligned_spin=True)
    for k in ('mass_1', 'mass_2', 'luminosity_distance', 'chirp_mass', 'mass_ratio', 'chi_1', 'chi_2',
              'ra', 'dec', 'psi', 'theta_jn', 'phase'):
        pr.pop(k, None)
    mc = truth['chirp_mass']
    pr['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(0.85 * mc, 1.15 * mc, name='chirp_mass')
    pr['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(0.1, 1.0, name='mass_ratio')
    pr['chi_1'] = bilby.gw.prior.AlignedSpin(a_prior=Uniform(0, 0.99), name='chi_1')
    pr['chi_2'] = bilby.gw.prior.AlignedSpin(a_prior=Uniform(0, 0.99), name='chi_2')
    pr['ra'] = Uniform(0, 2 * np.pi, boundary='periodic', name='ra')
    pr['dec'] = Cosine(name='dec')
    pr['psi'] = Uniform(0, np.pi, boundary='periodic', name='psi')
    pr['theta_jn'] = Sine(name='theta_jn')
    pr['phase'] = Uniform(0, 2 * np.pi, boundary='periodic', name='phase')
    d = truth['luminosity_distance']
    pr['luminosity_distance'] = PowerLaw(2.0, 0.1 * d, 5.0 * d, name='luminosity_distance')
    pr['geocent_time'] = Uniform(truth['geocent_time'] - 0.05, truth['geocent_time'] + 0.05, name='geocent_time')
    pr['delta_t'] = Uniform(truth['delta_t'] - 0.05, truth['delta_t'] + 0.05, name='delta_t')
    pr['ln_mu_rel'] = Uniform(-6, 6, name='ln_mu_rel')
    pr['morse_a'] = Uniform(0, 1.5, name='morse_a')
    pr['morse_b'] = Uniform(0, 1.5, name='morse_b')
    return pr


def gw_likelihood(row, scenario, seed):
    """Inject both images of a catalogue row into simulated noise; return the joint likelihood.

    Returns (likelihood, base, truth, duration, snrs), where base holds the injection parameters
    of image A and truth the true joint parameters (see event_parameters).
    """
    import bilby
    from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
    base, truth = event_parameters(row, scenario)
    duration = segment_duration(base['mass_1'], base['mass_2'], SCENARIO_FMIN[scenario])
    wf_args = dict(waveform_approximant=APPROXIMANT, reference_frequency=F_REF, minimum_frequency=SCENARIO_FMIN[scenario])
    likes, snrs = [], []
    for k, image in enumerate(('a', 'b')):
        p = image_parameters(dict(truth), image)
        start = p['geocent_time'] + 2.0 - duration
        ifos = interferometers(scenario, start, duration, seed + k)
        inj_gen = bilby.gw.WaveformGenerator(duration=duration, sampling_frequency=SAMPLING_FREQUENCY, start_time=start,
                                             frequency_domain_source_model=lensed_bbh,
                                             parameter_conversion=convert_to_lal_binary_black_hole_parameters,
                                             waveform_arguments=wf_args)
        inj = {kk: p[kk] for kk in ('mass_1', 'mass_2', 'chi_1', 'chi_2', 'ra', 'dec', 'psi', 'theta_jn', 'phase',
                                     'luminosity_distance', 'geocent_time', 'morse_n')}
        ifos.inject_signal(waveform_generator=inj_gen, parameters=inj, raise_error=False)
        snrs.append(float(np.sqrt(sum(abs(ifo.meta_data['optimal_SNR']) ** 2 for ifo in ifos))))
        gen = bilby.gw.WaveformGenerator(duration=duration, sampling_frequency=SAMPLING_FREQUENCY, start_time=start,
                                         frequency_domain_source_model=lensed_bbh_relative_binning,
                                         parameter_conversion=convert_to_lal_binary_black_hole_parameters,
                                         waveform_arguments=wf_args)
        fid = dict(inj, chirp_mass=truth['chirp_mass'], mass_ratio=truth['mass_ratio'])
        likes.append(bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
            interferometers=ifos, waveform_generator=gen, fiducial_parameters=fid, epsilon=0.025))
    return JointLensedLikelihood(*likes), base, truth, duration, snrs


def run_event(row, scenario, outdir, seed, nlive, npool, naccept):
    import bilby
    t0 = time.time()
    like, base, truth, duration, snrs = gw_likelihood(row, scenario, seed)
    priors = priors_for(truth)
    label = f'{scenario}_{row.event_id}'
    res = bilby.run_sampler(likelihood=like, priors=priors, sampler='dynesty', nlive=nlive, sample='acceptance-walk',
                            naccept=naccept, npool=npool, outdir=str(outdir), label=label, injection_parameters=None,
                            save='hdf5', plot=False, resume=True, check_point_delta_t=600,
                            conversion_function=None, seed=seed)
    post = res.posterior
    post['morse_n_a'] = discrete_morse(post.morse_a)
    post['morse_n_b'] = discrete_morse(post.morse_b)
    keep = ['chirp_mass', 'mass_ratio', 'chi_1', 'chi_2', 'ra', 'dec', 'psi', 'theta_jn', 'luminosity_distance',
            'ln_mu_rel', 'delta_t', 'morse_n_a', 'morse_n_b']
    post[keep].to_parquet(Path(outdir) / f'{label}_posterior.parquet', index=False)
    summary = dict(event_id=row.event_id, family=row.family, scenario=scenario, duration=duration, snr_a=snrs[0],
                   snr_b=snrs[1], seconds=time.time() - t0, log_bayes_factor=float(res.log_bayes_factor),
                   n_samples=len(post), **{f'true_{k}': float(v) for k, v in truth.items() if np.isscalar(v)})
    for k in ('ln_mu_rel', 'luminosity_distance', 'theta_jn', 'chirp_mass'):
        x = np.log(post[k]) if k == 'luminosity_distance' else post[k]
        summary[f'{k}_median'] = float(np.median(x))
        summary[f'{k}_sd'] = float(np.std(x))
        summary[f'{k}_q05'], summary[f'{k}_q95'] = (float(v) for v in np.quantile(x, [0.05, 0.95]))
    sky = np.column_stack([np.cos(post.dec) * np.cos(post.ra), np.cos(post.dec) * np.sin(post.ra), np.sin(post.dec)])
    true_v = np.array([np.cos(base['dec']) * np.cos(base['ra']), np.cos(base['dec']) * np.sin(base['ra']), np.sin(base['dec'])])
    summary['sky_offset_deg_median'] = float(np.degrees(np.median(np.arccos(np.clip(sky @ true_v, -1, 1)))))
    for im in ('a', 'b'):
        vals, cnt = np.unique(post[f'morse_n_{im}'], return_counts=True)
        summary[f'morse_{im}_map'] = float(vals[np.argmax(cnt)])
        summary[f'morse_{im}_map_prob'] = float(cnt.max() / cnt.sum())
    summary['delta_morse_true'] = truth['morse_b'] - truth['morse_a']
    dn = post.morse_n_b - post.morse_n_a
    summary['delta_morse_prob_true'] = float(np.mean(np.isclose(dn, summary['delta_morse_true'])))
    (Path(outdir) / f'{label}_summary.json').write_text(json.dumps(summary, indent=1))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--catalogue', type=Path, required=True, help='Parquet file from gwlens.catalogue')
    p.add_argument('--scenario', choices=list(SCENARIO_IFOS), required=True)
    p.add_argument('--event-ids', nargs='+', required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--nlive', type=int, default=500)
    p.add_argument('--naccept', type=int, default=60)
    p.add_argument('--npool', type=int, default=4)
    p.add_argument('--seed', type=int, default=1234)
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    cat = pd.read_parquet(a.catalogue).set_index('event_id', drop=False)
    for i, eid in enumerate(a.event_ids):
        if (a.outdir / f'{a.scenario}_{eid}_summary.json').exists():
            continue
        s = run_event(cat.loc[eid], a.scenario, a.outdir, a.seed + 17 * i, a.nlive, a.npool, a.naccept)
        print(json.dumps({k: s[k] for k in ('event_id', 'snr_a', 'snr_b', 'seconds', 'true_ln_mu_rel', 'ln_mu_rel_median', 'ln_mu_rel_sd')}), flush=True)


if __name__ == '__main__':
    main()
