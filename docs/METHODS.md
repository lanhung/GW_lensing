# v2 methods

This document summarises the simulation and inference choices of the v2 pipeline. The
corresponding code paths are given in brackets.

## Sources (`gwlens/population.py`)

* **Masses.** GWTC-4.0 fiducial BBH model "Broken Power Law + 2 Peaks" (LVK 2025,
  arXiv:2508.18083): posterior medians alpha_1 = 1.7, alpha_2 = 4.5, break mass 36 Msun,
  Gaussian peaks at 9.8 (sigma 0.65) and 33 Msun (sigma 3.9), mixing fractions 0.36 (power law) and
  0.59 (low-mass peak), low-mass tapers, pairing q^1.2. Values are parsed from the paper source by
  `realdata/fetch_gwtc4_population.py` and stored in `configs/gwtc4_bp2p.json`.
* **Spins.** Aligned components chi_z = a cos(tilt) with the GWTC-4.0 default magnitude and tilt
  model (only the aligned part enters the IMRPhenomXAS SNR).
* **Redshift.** p(z_s) ∝ psi(z) (1+z)^-1 dV_c/dz × tau(z_s), with psi a Madau-Dickinson-shaped
  rate normalised to (1+z)^3.2 at low z (GWTC-4.0 lambda = 3.2) and tau ∝ D_C(z_s)^3, the optical
  depth of a non-evolving isothermal lens population. 0.1 < z_s < 10.
* **Extrinsic.** Isotropic sky position and orientation, uniform polarisation; geocentric merger
  time uniform over one sidereal year so antenna patterns vary between images.

## Lenses (`gwlens/population.py`, `gwlens/lenses.py`)

* p(z_l | z_s) ∝ (D_C,l D_C,ls / D_C,s)^2 dD_C,l.
* Velocity dispersion from the SDSS early-type velocity-dispersion function (Choi et al. 2007)
  weighted by sigma^4; Einstein radius from the SIS relation (Planck18 cosmology).
* Families: SIS; SIE (axis ratio q ~ N(0.75, 0.15) truncated to [0.35, 0.99], SLACS-like);
  SIE + external shear (|gamma| log-normal, median 0.05, capped at 0.25); EPL + shear (slope
  gamma' ~ N(2.08, 0.15), Auger et al. 2010).
* Source positions uniform in area inside 1.0–1.8 theta_E, redrawn (up to 60 times, then the lens
  is redrawn) until the configuration is multiply imaged. The source is therefore uniform over the
  multiple-image region of each lens and quads appear at their natural rate for that lens, but the
  lenses are not weighted by how the area of that region depends on their shape and slope (only by
  sigma^4). The Bayesian lens analyses must normalise the source prior accordingly
  (`gwlens/source_area.py`, `--source-norm`).
* Images are solved with lenstronomy (EPL_NUMBA); spurious roots are removed with a lens-equation
  residual test. Each image stores signed magnification, convergence, Morse index and Fermat delay.

## Detectors and GW measurements (`gwlens/detectors.py`, `gwlens/catalogue.py`)

* Networks: **O5-like** (H1, L1 at A+ design, V1 at AdV design; f_min = 10 Hz) and **ET-D**
  triangle (f_min = 5 Hz).
* Optimal SNR per image from IMRPhenomXAS, rho_i^2 = |mu_i| Σ_det I0_det [F+^2 ((1+cos^2 i)/2)^2 +
  Fx^2 cos^2 i], with antenna patterns evaluated at the image's arrival time.
* Matched-filter SNR: rho_hat = rho + N(0, 1). Detection: rho_hat ≥ 8 for at least two images
  arriving within one year of the first detected image. The two highest-SNR images form the pair
  (A = brighter in measured SNR).
* **Relative magnification.** ln(mu_B/mu_A)_hat = 2[ln(rho_hat_B/rho_hat_A) − ln(R_hat_B/R_hat_A)],
  where R_hat is the network response at an *estimated* sky position, polarisation and
  inclination. Their errors scale as 20/rho_joint times 2 deg (O5) or 15 deg (ET) on the sky,
  0.5 rad in polarisation and 0.3 in cos i. A known-sky variant gives the optimistic bound.
* **Apparent distance.** ln D_app,i_hat = ln(d_L/sqrt|mu_i|) + ln(rho_i/rho_hat_i) +
  ln(R_hat_i/R_i) + eps, with eps a shared (both images) inclination–distance term,
  N(0, sqrt(a^2 + (b^2 − 1)/rho_joint^2)). a = 0.223, b = 2.74 are fitted to 167 GWTC-1–4.0 BBHs
  (`realdata/calibrate_distance_errors.py`), so a single image reproduces the observed catalogue
  distance uncertainties.
* The relative-amplitude noise model is validated by injections into public O4a H1/L1 strain
  (`realdata/o4a_injections.py`).

## Electromagnetic observations (`gwlens/catalogue.py`)

* The true lens centre is drawn from N(0, 5 arcsec) per axis and the frame is rotated by a random
  angle, so absolute coordinates carry no information.
* Image positions: 0.01 arcsec Gaussian errors; lens-light centre: 0.02 arcsec.
* Lens light axis ratio = mass axis ratio + N(0, 0.05); light position angle = mass angle +
  N(0, 10 deg) (random for SIS).
* Spectroscopic redshifts (sigma_z = 0.001(1+z)) and velocity dispersion (10 per cent).
* Resolvability: the GW image pair must be separated by ≥ 0.1 arcsec. Images with |mu| < 0.1 are
  not counted as resolved EM images.

## Observable features (`analysis/features.py`)

All inputs are translation and rotation invariant: radii of A and B from the observed centre,
their separation, cos of the opening angle, asymmetry, Einstein-radius proxy, number of resolved
images and radii of additional images, light axis ratio and image angles relative to the light
position angle; GW: ln(mu_B/mu_A)_hat, ln|Delta t|, sign of Delta t, SNRs and apparent distances;
follow-up: z_l, z_s, ln sigma_v; and the cosmology estimator ln mu_cosmo = 2[ln d_L(z_s) − ln D_app,A].

| Input set | Contents | Needs an assumed cosmology |
| --- | --- | --- |
| GW | GW features only | no |
| GW+z | GW + follow-up + ln mu_cosmo | **yes** |
| Geometry | image geometry + light profile | no |
| Geometry+FU | geometry + redshifts + sigma_v | no |
| Geometry+GW(rel) | geometry + relative magnification, delay and SNRs (no absolute distance) | no |
| Full | everything | yes |

## Inference (`analysis/benchmark.py`, `analysis/npe.py`)

* Target: ln mu_A (absolute magnification of the brighter GW image).
* HistGB quantile regression (5, 50, 95 per cent; 500 iterations, early stopping on an internal
  10 per cent split) with conformalized quantile regression (Romano et al. 2019) on a separate
  calibration set; global and per-family (Mondrian) calibration.
* Independent events; random 60/20/20 train/calibration/test splits stratified by family; five
  split seeds for the in-family comparison and three for mixed-family runs; bootstrap intervals
  over test events.
* Analytic baselines: SIS asymmetry inversion, SIS relative-magnification inversion and the
  known-redshift cosmology estimator.
* NPE: conditional neural spline flow (zuko; 4 transforms, 3×256 hidden units, 12 bins) on
  ln mu_A; evaluated with central-interval coverage at 50/68/90/95 per cent and PIT histograms.
