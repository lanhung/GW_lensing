# Change log

## 2026-10-16: pre-submission audit and v2 analysis

### Fixes to the v1 benchmarks

1. **Fixed-centre leak in the geometry inputs (SIS and four-family).** Both generators place the
   true lens centre at the coordinate origin, and the "compact geometry" input set contained the
   raw absolute image and lens-centre coordinates. A regressor could therefore measure image radii
   from the exact centre and bypass the simulated 0.02 arcsec centre error.
   `core_experiments/common.py` now builds translation-invariant features (image radii from the
   *observed* centre, image-pair differences, mid-point offset and their angle), and
   `experiments/extended_experiments/evaluate_universal_identifiability.py` uses only radii,
   separation and asymmetry. The old behaviour is available with `--leaky-geometry` for
   reproduction only.
   Effect on the selected-SIS controlled comparison (ten identity splits, same HistGB settings):

   | Input set | MAE, leaky (old) | MAE, leak-free |
   | --- | --- | --- |
   | Image geometry | 0.193 ± 0.038 | 0.270 ± 0.040 |
   | Geometry + follow-up | 0.201 ± 0.038 | 0.272 ± 0.042 |
   | Full observed | 0.207 ± 0.041 | 0.239 ± 0.036 |
   | GW proxies | 0.885 ± 0.076 | 0.885 ± 0.076 |

   The coordinate-likelihood SIS Bayesian baseline (MAE 0.243), which never used the absolute
   frame, now outperforms geometry HistGB, as expected for a correctly specified posterior.

2. **Truth-based image ordering in the four-family generator.** Images were selected and ordered
   by the true absolute magnification, leaking the label for multi-image lenses.
   `generate_universal_latent_benchmark.py` now ranks images by a noisy observed brightness.

3. **Conformal quantile level.** `conformal_phrm.py` (not part of this repository; fixed in the
   server working copies) used the (n+1)·alpha instead of the (n+1)·(1−alpha) order statistic.

### v2 analysis (new)

The v1 GW baseline (waveform-window amplitude ratios in one ET arm, no antenna-pattern
correction), the non-adaptive constant-width intervals, the ad hoc source population and the
inconsistent four-family selection motivated a new pipeline (`gwlens/`, `analysis/`,
`realdata/`):

* GWTC-4.0 BP2P source population, Madau-Dickinson-shaped rate and optical-depth weighting to z=10;
* SIS, SIE, SIE+shear and EPL+shear lenses with SLACS-like shapes and cross-section weighting;
* one consistent selection for all families: both images above network SNR 8 within one observing
  year and a resolvable pair (separation ≥ 0.1 arcsec);
* O5-like and ET-D networks with antenna patterns at each image's arrival time;
* GW relative magnification and apparent distance from SNRs and an antenna-pattern model with
  sky/orientation errors, plus an inclination term calibrated on 167 GWTC BBHs;
* injection validation in public O4a H1/L1 strain;
* random true lens centre and frame rotation; only translation/rotation-invariant features;
* conformalized quantile regression with per-family (Mondrian) calibration and neural posterior
  estimation;
* independent events throughout (no reused source or lens identities).
