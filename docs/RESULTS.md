# v2 results summary

Run on 2026-10-16 with the code at this commit (`scripts/generate_v2.sh`,
`scripts/run_analysis_v2.sh`). Numbers are median relative errors |mu_hat − mu|/mu of the brighter
GW image's absolute magnification, unless stated otherwise.

## Selected samples

| Family | Systems | ET pairs | O5 pairs | ET median mu_A | O5 median mu_A | O5 quad fraction |
| --- | --- | --- | --- | --- | --- | --- |
| SIS | 250 000 | 128 133 | 3031 | 2.68 | 5.21 | 0 |
| SIE | 250 000 | 125 147 | 2457 | 2.48 | 4.36 | 0.33 |
| SIE+shear | 150 000 | 75 377 | 1492 | 2.43 | 4.32 | 0.38 |
| EPL+shear | 150 000 | 63 595 | 1572 | 2.47 | 4.85 | 0.39 |

## Information content (ET, in-family HistGB-CQR; coverage of 90% intervals in brackets)

| Input set | SIS | SIE | SIE+shear | EPL+shear |
| --- | --- | --- | --- | --- |
| SIS analytic (GW relative) | 8.7 | 33.0 | 36.5 | 42.5 |
| SIS analytic (geometry) | 3.3 | 21.0 | 23.8 | 32.9 |
| Known z_s + cosmology | 40.8 | 41.6 | 41.4 | 41.3 |
| GW | 7.7 (0.90) | 21.4 (0.90) | 21.9 (0.90) | 25.8 (0.90) |
| Geometry | 3.2 (0.90) | 7.3 (0.90) | 11.6 (0.90) | 19.9 (0.90) |
| Geometry+GW(rel) | 2.9 (0.90) | 6.9 (0.90) | 9.9 (0.90) | 18.1 (0.90) |
| Full | 2.7 (0.90) | 6.9 (0.90) | 10.0 (0.90) | 15.6 (0.90) |
| **NPE: Geometry+GW(rel)+FU** | **1.7 (0.89)** | **5.9 (0.90)** | **9.7 (0.90)** | **13.6 (0.90)** |

O5-like network: Geometry 10.7 / 16.2 / 21.9 / 27.9; NPE Geometry+GW(rel)+FU 7.2 / 10.7 / 15.0 / 21.3.

## Calibration

* Mixed-family training, Geometry+GW(rel), ET: global conformal coverage 0.95 / 0.97 / 0.91 /
  0.76 (SIS / SIE / SIE+shear / EPL+shear); per-family (Mondrian) 0.90 / 0.90 / 0.90 / 0.89.
* NPE raw coverage 0.93 / 0.93 / 0.84 / 0.76; after per-family conformal recalibration 0.89–0.90.
* Coverage in bins of true mu (ET, in-family): 0.86 (mu<2), 0.94, 0.92, 0.84, 0.74, 0.58 (mu>20);
  calibrating in bins of predicted mu does not change this materially (regression to the mean).

## Lens-model transfer (ET, Geometry; error [coverage])

* SIS-trained models on elliptical lenses: 20.4–30.6 per cent [0.21–0.29].
* EPL+shear-trained models on the other families: 9.4–13.2 per cent [0.96–0.98].
* Leaving EPL+shear out of mixed training: 21.2 per cent [0.55].

## Distance

ET, Geometry+GW(rel): median |delta ln d_L| from the magnification alone 1.4 / 3.5 / 4.9 / 9.1
per cent; total 22–24 per cent, versus 22 per cent with the true magnification (GW apparent-distance
error dominates).

## Real data

* GWTC-1–4.0, 167 BBHs: sigma_lnD = sqrt(0.223^2 + (2.74/rho)^2), rms residual 0.057.
* O4a injections, 321 pairs: standardised relative-magnification error, robust sd 1.02 (Gaussian
  control 0.94); one glitch outlier (0.3 per cent).

## v1 benchmark after removing the fixed-centre leak

Geometry MAE 0.193 → 0.270; full 0.207 → 0.239; SIS Bayesian baseline 0.243 (see `CHANGES.md`).
