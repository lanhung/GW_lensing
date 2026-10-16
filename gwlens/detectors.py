"""Detector networks and optimal SNRs of lensed images.

For each source a face-on (2,2)-mode waveform is evaluated once on a logarithmic frequency grid and
the noise-weighted inner product I0 = 4 int |h0|^2/S df is stored per detector. The optimal SNR of an
image with magnification mu arriving at time t is then
    rho^2 = |mu| * sum_det I0 [F+^2 ((1+cos^2 i)/2)^2 + Fx^2 cos^2 i](t).
A Morse phase multiplies both polarisations by the same factor and leaves rho unchanged.
"""
import os
from functools import lru_cache

import numpy as np

SCENARIOS = {
    # Advanced LIGO A+ (H1, L1) and Advanced Virgo design sensitivity: an O5-like network.
    'O5': dict(detectors=['H1', 'L1', 'V1'], psd={'H1': 'Aplus_asd.txt', 'L1': 'Aplus_asd.txt', 'V1': 'AdV_asd.txt'},
               f_min=10.0, sky_sigma_deg_at_rho20=2.0),
    # Einstein Telescope (triangle, ET-D) operating alone.
    'ET': dict(detectors=['ET1', 'ET2', 'ET3'], psd={'ET1': 'ET_D_psd.txt', 'ET2': 'ET_D_psd.txt', 'ET3': 'ET_D_psd.txt'},
               f_min=5.0, sky_sigma_deg_at_rho20=15.0),
}
F_MAX = 2048.0


@lru_cache(maxsize=None)
def interferometers(scenario):
    import bilby
    names = SCENARIOS[scenario]['detectors']
    if names[0].startswith('ET'):
        ifos = list(bilby.gw.detector.InterferometerList(['ET']))
    else:
        ifos = list(bilby.gw.detector.InterferometerList(names))
    return ifos


@lru_cache(maxsize=None)
def psd_on_grid(scenario, n=1500):
    import bilby
    cfg = SCENARIOS[scenario]
    f = np.geomspace(cfg['f_min'], F_MAX, n)
    base = os.path.join(os.path.dirname(bilby.__file__), 'gw', 'detector', 'noise_curves')
    out = {}
    for det, fname in cfg['psd'].items():
        data = np.loadtxt(os.path.join(base, fname))
        vals = data[:, 1] ** 2 if 'asd' in fname else data[:, 1]
        out[det] = np.exp(np.interp(np.log(f), np.log(data[:, 0]), np.log(vals), right=np.inf))
    return f, out


def face_on_integrals(scenario, m1_det, m2_det, chi1z, chi2z, dl_mpc):
    """I0 = 4 int |h+|^2/S df per detector for a face-on source (IMRPhenomXAS, (2,2) mode).

    For a (2,2)-mode waveform h+ = h0 (1+cos^2 i)/2 and hx = -i h0 cos i, so the inclination
    dependence of the SNR is analytic and the waveform is generated once per source.
    """
    import lal
    import lalsimulation as ls
    f, psd = psd_on_grid(scenario)
    fv = lal.CreateREAL8Vector(len(f))
    fv.data = f
    hp, _ = ls.SimInspiralChooseFDWaveformSequence(0.0, m1_det * lal.MSUN_SI, m2_det * lal.MSUN_SI, 0, 0, chi1z, 0, 0, chi2z,
                                                   20.0, dl_mpc * 1e6 * lal.PC_SI, 0.0, None, ls.IMRPhenomXAS, fv)
    h2 = np.abs(hp.data.data) ** 2
    df = np.gradient(f)
    return {det: float(np.sum(4 * h2 * df / s)) for det, s in psd.items()}


def antenna(scenario, ra, dec, psi, gps):
    """Antenna patterns (F+, Fx) per detector at a geocentric time."""
    return {ifo.name: (ifo.antenna_response(ra, dec, gps, psi, 'plus'), ifo.antenna_response(ra, dec, gps, psi, 'cross'))
            for ifo in interferometers(scenario)}


def network_snr(i0, patterns, cos_iota):
    """Optimal network SNR of an unlensed source with face-on integrals i0."""
    ap, ac = (1 + cos_iota ** 2) / 2, cos_iota
    tot = sum(i0[d] * (fp * fp * ap * ap + fc * fc * ac * ac) for d, (fp, fc) in patterns.items())
    return float(np.sqrt(max(tot, 0.0)))
