# Audit of the relative-magnification measurement in the magnification benchmark

Response to the "Technical Audit and Experimental Revision Request" of 2026-11-02.
Baseline: repository revision 1a1c80c (the audited modules `gwlens/catalogue.py`, `analysis/features.py`,
`analysis/benchmark.py`, `analysis/npe.py`, `analysis/pe_analysis.py`, `realdata/o4a_injections.py` are
byte-identical between 1a1c80c and the current head b2a2716; the later commits touch only the GW+EM
Bayesian code, the source-area normalisation and documentation).

All new scripts are in `audit/`; all new outputs are in a separate workspace (GPU server
`/root/autodl-tmp/audit_relmag/out/`, copies of the small tables and figures alongside this report).
No published output file was modified. No code outside `audit/` was changed.

## 0. Summary and decision

1. **No leakage of simulation truth into the model inputs was found.** Every input is derived from a
   simulated measurement with a declared error model; the input gate rejects every truth column; events
   are independent, identifiers unique, feature rows unique and the splits disjoint. The single-feature
   probes give no input whose error is anywhere near zero; the forbidden truth columns, fed past the gate
   on purpose, give 0.1–0.2% and are what a leak would look like.
2. **The alleged "error ≈ 0.0" could not be traced to any repository output.** The only legitimate
   configurations whose metric prints as 0.0 at one decimal are the SIS rows: their median relative error
   is 0.025–0.035 (2.5–3.5%), which rounds to 0.0. For an SIS lens the magnification is an exact function
   of the image asymmetry, so a 3% error is the expected limit from the 0.01-arcsec astrometry (the closed-form
   SIS inversion gives 3.3%). A rounded SIS number is the most plausible origin of the statement; it is
   not evidence of leakage.
3. **The SNR-based relative magnification used by the benchmark is not an overly clean quantity.** On the
   full catalogues its error has a median of 0.20 in ln(μ_B/μ_A) (a 20% error in the ratio), a 90th
   percentile of 0.6–0.8, a 99th percentile of 1.8, and 8–12% of pairs are off by more than a factor of 2.
   For the ET scenario two thirds of this error comes from the simulated sky/orientation (antenna-response)
   uncertainty, not from SNR noise. The naive SNR ratio without antenna correction is unusable (median error
   0.6; 44% of ET pairs off by more than a factor of 2).
4. **Event by event, joint PE is better than the SNR model, and the model is conservative in width but
   not Gaussian-calibrated in its tails.** With the host sky position, the PE median error is 0.08 (ET)
   and 0.15 (O5) against 0.28 and 0.19 for the SNR model on the same 60 systems; PE is closer to the truth
   in 75% (ET) and 60% (O5) of systems. The SNR model's own 90% error bars contain the truth for 80% (ET) and
   77% (O5) of these systems, and the pure-SNR analytic error bar used in the O4a validation contains it for
   only 43% (ET), because that validation covers only the SNR-noise part of the error budget.
5. **Downstream, the measurement model matters less than the audit assumed, except for SIS lenses.**
   With the exact relative magnification (oracle, NPE retrained on it) the population-level median error of
   μ_A falls from 8.7% to 5.2% (ET) and from 15.1% to 11.4% (O5); the SNR-model ratio recovers only
   0.5–1 point of this over geometry alone, and feeding the PE median into the NPE trained on the SNR
   model recovers hardly more (ET 10.5% vs 10.8%, O5 13.8% vs 17.8% on the matched systems), because that
   NPE discounts the ratio according to the training noise. The Bayesian two-stage analysis, which uses the
   PE posterior as an effective likelihood, does exploit the PE precision (7.9% ET, 13.7% O5). For SIS the
   oracle ratio is decisive (3.4% → 0.2%).
6. **Decision (Section 12 of the request):** retain the calibrated SNR benchmark, with its scope stated
   explicitly, and add the measurement-to-inference ladder (D0–D4) to the paper as one compact table;
   the posterior-aware route (D4) is the method of choice for real events. The measurement model does not
   need to be replaced; two claims need rewording (Section 9 below).

## 1. Reproducibility

| Item | Value |
| --- | --- |
| Code | github.com/lanhung/GW_lensing, head b2a2716; audited modules identical to 1a1c80c |
| Catalogues | `data/v2/{SIS,SIE,SIE_shear,EPL_shear}.parquet`, seed 20261016 (DATA.md); PE systems `data/pe/*.parquet`, seed 7777 |
| Published model outputs | `results/v2b` (HistGB), `results/v2_review_npe2` (NPE checkpoints and predictions), Zenodo 10.5281/zenodo.23473170 |
| Archived PE | `gwlens_pe_gwem_2026-10-18.tgz`: 60 ET free-sky, 60 ET known-sky, 60 O5 joint-PE posteriors; two-stage GW+EM with source-normalised prior from the Zenodo package (`results/gwem_posthoc_sn`) |
| Environment | Python 3.10, bilby 2.8.2, lenstronomy 1.14.2, scikit-learn 1.7.2, torch 2.11, zuko 1.6.0 |
| Commands | `audit/relmag_errors.py`, `audit/leak_probe.py`, `audit/train_oracle_flow.py --scenario ET --epochs 40` and `--scenario O5 --epochs 150`, `audit/pe_matched.py`, `audit/o4a_strata.py`, `audit/figures.py` (arguments in each docstring; seeds 0/1 as in `analysis/`) |

## 2. Task A: feature provenance and leakage audit

### 2.1 Provenance of every model input

Line numbers refer to the current files (identical to 1a1c80c).

| Input | Built in | From | Truth entering it | Class |
| --- | --- | --- | --- | --- |
| `gw_ln_mu_rel` | features.py:79–80 ← catalogue.py:88 | 2 ln(ρ̂_B/ρ̂_A) − 2 ln(R̂_B/R̂_A); ρ̂ = ρ + N(0,1) (l. 50), ρ = √|μ| R ρ_unl (l. 48–49); R̂ from sky/ψ/cos ι perturbed by 20/ρ_J × (2° or 15°, 0.5 rad, 0.3) (l. 63–74) | μ, sky, ψ, ι only through the declared forward model | inferred observable (synthetic measurement) |
| `gw_rho_a`, `gw_rho_b` | features.py:84 ← catalogue.py:87 | measured SNRs ρ̂ | as above | direct observable |
| `gw_ln_dapp_a/b` | features.py:85 ← catalogue.py:77–80 | ln(d_L/√|μ|) + SNR noise + response error + GWTC-calibrated shared term | as above | inferred observable |
| `gw_ln_abs_dt`, `gw_dt_sign` | features.py:81–83 ← catalogue.py:92 | `obs_dt_ab` = t_B − t_A **without noise** | the exact delay | truth used as observable; justified by ms timing but should be declared (see 8.3) |
| `geo_r_a`, `geo_r_b`, `geo_sep`, `geo_cos_ab`, `geo_asym`, `geo_theta_proxy`, `geo_cos2_*`, `geo_sin2_*` | features.py:64–78 ← catalogue.py:127–136, 172–173 | image positions with 0.01″ noise relative to the light centre with 0.02″ noise; light PA with 10° noise; random frame translation and rotation | none beyond the declared noise | direct observables (frame-invariant) |
| `geo_n_images`, `geo_r_other1_n`, `geo_r_other2_n` | features.py:71–73 ← catalogue.py:166, 170, 174–176 | count and observed radii of images with true |μ| ≥ 0.1 | the EM-detectability cut uses the true |μ| of the *other* images (a threshold, not a value) | valid simulation design choice (declared threshold) |
| `geo_q_light` | features.py:74 ← catalogue.py:135 | true q + N(0, 0.05) | declared noise | direct observable |
| `fu_z_l`, `fu_z_s` | features.py:86–87 ← catalogue.py:144 | z + N(0, 10⁻³(1+z)) | declared noise | direct observable |
| `fu_ln_sigma_v` | features.py:88 ← catalogue.py:145 | the SIS-equivalent σ_v that sets θ_E, × (1 + N(0, 0.1)) | σ_v is the generator's lens-strength parameter with 10% noise | valid but optimistic design choice (the paper states it) |
| `gw_ln_mu_cosmo` | features.py:89 | 2[ln d_L(ẑ_s; Planck18) − ln D̂_app,A] | assumed cosmology | assumed prior; declared "not scale-free" |
| image labels A/B | catalogue.py:61 | larger **measured** SNR ρ̂ | — | observable |
| selection | catalogue.py:51–57, 165–168 | ρ̂ ≥ 8 both, one-year window on true times, observed separation ≥ 0.1″ | window uses true arrival times (ms-exact) | design choice |

Targets and evaluation-only truths (`mu`, `mu_b`, `y_true`, `dl_true`, `z_s_true`, `n_images_true`, `morse_a`)
are created in features.py:91–97 and rejected by `check_inputs` (features.py:112–116). Every fit passes
through the gate: `CQR.__init__` (benchmark.py:54; also used by `conditional_calibration.py:46` and
`review_tests.py:73,96`), `npe.py:92`; `pe_analysis.py:91` and `review_distance_scale.py:38` use the column
list stored in the checkpoint at training time. No ungated `fit` call exists in `analysis/`.

### 2.2 Integrity of the data and splits (ET, 60 000 systems per family as in the paper)

| Check | Result |
| --- | --- |
| events / unique identifiers | 240 000 / 240 000 |
| duplicate feature rows (all 25 candidate inputs, 12 decimals) | 0 |
| split (benchmark.py:40–47, seed 0) | 144 000 / 48 000 / 48 000 train / calibration / test, disjoint |
| shared latent variables between events | none: every row is an independent source and lens (catalogue.py:97–124) |
| input gate on declared sets | all 11 sets accepted |
| input gate on truth columns | `mu`, `mu_b`, `y_true`, `dl_true`, `z_s_true`, `n_images_true`, `morse_a`, `event_id`, `family` all rejected |

### 2.3 Rank correlations with the target

The largest |Spearman| of any input with ln μ_A is 0.60 (`gw_ln_mu_cosmo`, EPL+shear) and the largest with
ln μ_B is 0.96 (`geo_asym`, SIS), which is the analytic SIS relation μ = 1 ± 1/y, not a leak
(`out/leak/correlations.csv`).

### 2.4 Single-input and leak-demonstration probes (ET, HistGB median regressor, 20 000 training systems, test split)

Median relative error of μ_A:

| Input | SIS | SIE | SIE+shear | EPL+shear |
| --- | --- | --- | --- | --- |
| prior only (training median) | 0.186 | 0.245 | 0.245 | 0.317 |
| best single observable | 0.035 (`geo_asym`) | 0.174 (`geo_asym`) | 0.187 (`geo_asym`) | 0.256 (`geo_asym`) |
| `gw_ln_mu_rel` alone | 0.087 | 0.242 | 0.250 | 0.304 |
| Geometry (declared) | 0.033 | 0.072 | 0.118 | 0.203 |
| Geometry + μ ratio (declared, paper's main set) | 0.031 | 0.069 | 0.103 | 0.191 |
| All observables (declared) | 0.028 | 0.070 | 0.099 | 0.161 |
| **Geometry + true μ_B (forbidden)** | 0.001 | 0.067 | 0.101 | 0.163 |
| **Geometry + true y (forbidden)** | 0.001 | 0.056 | 0.109 | 0.122 |
| **Geometry + true μ_A (forbidden)** | 0.001 | 0.002 | 0.002 | 0.002 |

The declared sets reproduce the published Table (results/v2b: Geometry 0.032 / 0.073 / 0.116 / 0.199; All
0.027 / 0.069 / 0.100 / 0.156) to within 0.5 points with a third of the training data. No observable or declared
combination approaches zero. The forbidden columns show what leakage looks like: 0.1–0.2% for μ_A itself, and
0.1% for SIS from μ_B or y, because for an SIS |μ_A| = |μ_B| + 2 and μ_A = 1 + 1/y exactly.

### 2.5 The "error ≈ 0.0" statement

Not reproducible from any file in the repository or the published outputs. The published HistGB metrics
(`results/v2b/metrics.csv`, 1 848 rows) contain 63 rows whose median relative error rounds to 0.0 at one
decimal: 62 of them are SIS rows with errors 0.025–0.035, one is an EPL+shear row of the learning-curve
experiment (`out/leak/published_rounds_to_zero.csv`). Seven rows have a bias that rounds to 0.0. The
"removal of direct true-magnification inputs" in the statement matches the v1 fix of 2026-10-16 (docs/CHANGES.md),
after which the v1 SIS geometry error *rose* from 0.19 to 0.27 in ln μ. Unless the meeting record names a
metric, revision, population and split, the statement should be treated as a rounded SIS relative error.

## 3. Task B: relative-magnification errors on the full catalogues

Every selected pair of the four benchmark catalogues (392 252 ET pairs, 8 552 O5 pairs). Estimators of
ln R, R = |μ_B|/|μ_A|: B1 naive 2 ln(ρ̂_B/ρ̂_A); B2 antenna-corrected (the benchmark input); B3 noisy SNR with
the true response (oracle diagnostic). The residuals decompose exactly: B3 − truth is pure SNR noise,
B2 − B3 is the response-estimation error, B1 − B3 = 2 ln(R_B/R_A) is the true response ratio
(`out/relmag/checks.json` verifies the stored columns against these identities).

| Scenario | Estimator | median | robust sd | median |d| | p68 |d| | p90 |d| | p95 |d| | p99 |d| | median rel. error of R | fraction |d| > ln 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ET (392 252) | B1 naive | −0.14 | 0.87 | 0.588 | 0.94 | 1.58 | 1.80 | 2.16 | 53% | 44.1% |
| | B2 corrected | −0.00 | 0.30 | 0.201 | 0.36 | 0.77 | 1.09 | 1.81 | 20% | 12.1% |
| | B3 true response | +0.00 | 0.13 | 0.090 | 0.15 | 0.28 | 0.35 | 0.54 | 9% | 0.2% |
| | response error alone (B2−B3) | −0.00 | 0.24 | 0.159 | 0.30 | 0.72 | 1.05 | 1.75 | — | 10.8% |
| O5 (8 552) | B1 naive | −0.20 | 0.56 | 0.399 | 0.61 | 1.10 | 1.36 | 1.77 | 37% | 27.9% |
| | B2 corrected | −0.01 | 0.29 | 0.193 | 0.30 | 0.62 | 0.90 | 1.78 | 19% | 8.1% |
| | B3 true response | −0.01 | 0.22 | 0.150 | 0.23 | 0.41 | 0.50 | 0.73 | 15% | 1.2% |
| | response error alone (B2−B3) | −0.00 | 0.10 | 0.069 | 0.14 | 0.46 | 0.72 | 1.69 | — | 6.0% |

(p68 values from `out/relmag/summary.csv`; per-family rows differ from the pooled rows by at most 0.01 in
the medians.) The standardised SNR-noise residual (B3 − truth)/[2(ρ_A⁻² + ρ_B⁻²)^½] has robust sd 1.00 (ET) and
0.97 (O5), so the analytic error of the paper describes the SNR-noise part exactly, as it must by construction.

Findings:

* The naive ratio is biased by the full response ratio: its error equals 2 ln(R_B/R_A) plus noise (Fig. 4), and
  R_B/R_A exceeds a factor √2 (|2 ln| > 0.69) for 44% of ET and 27% of O5 pairs because the Earth rotates
  between images.
* The corrected ratio has heavy tails. 12% (ET) and 8% (O5) of pairs are wrong by more than a factor of two;
  the 99th percentile of 1.8 is a factor of 6 in R. For ET the tails come almost entirely from the response
  term, whose scatter is set by the ad hoc sky prescription (15° × 20/ρ_J for a single ET site; the sky error
  is isotropic Gaussian in the unit vector and is identical for both images). This prescription is the
  least validated ingredient of the model.
* Stratification (`out/relmag/strata.csv`, Fig. 3): the ET B2 error falls from 0.34 (ρ̂_B 8–10) to 0.05
  (ρ̂_B > 50) and from 0.60 (ρ_J < 15) to 0.05 (ρ_J > 100); at ρ_J < 15 44% of ET pairs are off by more
  than a factor 2. It is 0.03 for delays under an hour (same antenna response) and 0.22 beyond 100 days;
  0.21 for doubles and 0.12 for quads; 0.25 at μ_A < 2 and 0.07 at μ_A > 20. O5 is dominated by SNR noise
  at all SNRs (B3 ≈ B2 except at ρ_J < 15, where the response error adds a tail). The inclination stratum
  is empty for the benchmark catalogues because they predate the stored `true_cos_iota` column.
* Machine-readable per-pair output: `out/relmag/per_event.parquet` (truth, B1, B2, B3, SNRs, delay,
  multiplicity, response ratios, analytic σ).

## 4. Task C: event-matched comparison with archived joint PE

120 systems (15 per family and scenario), each with a free-sky PE; ET also with the host sky fixed. No
sampler was rerun. The SNR-model quantities for the same systems come from the catalogue rows they were drawn
from; "model σ" is the catalogue-wide robust scatter of B2 at the system's joint SNR, as used in the paper's
Fig. 4; "analytic σ" is the pure-SNR expression 2(ρ_A⁻² + ρ_B⁻²)^½.

| | ET free sky | ET host sky | O5 free sky |
| --- | --- | --- | --- |
| n | 60 | 60 | 60 |
| median |res| PE / B1 / B2 / B3 | 0.285 / 0.48 / 0.277 / 0.093 | **0.082** / 0.48 / 0.277 / 0.093 | **0.154** / 0.41 / 0.192 / 0.169 |
| 90th pct |res| PE / B2 | 1.47 / 0.86 | 0.40 / 0.86 | 0.40 / 0.91 |
| max |res| PE / B2 | 4.2 / 1.9 | 0.69 / 1.9 | 0.74 / 2.0 |
| bias PE / B2 | +0.18 / −0.04 | +0.02 / −0.04 | −0.01 / −0.07 |
| coverage of PE 90% / 68% intervals | 0.67 / 0.48 | 0.87 / 0.68 | 0.92 / 0.72 |
| coverage of B2 ± 1.645 model σ | 0.80 | 0.80 | 0.77 |
| coverage of B2 ± 1.645 analytic σ | 0.43 | 0.43 | 0.68 |
| PIT of PE, KS p-value | 0.003 | 0.92 | 0.60 |
| PE sd / model σ (median) | 0.70 | 0.41 | 0.88 |
| PE 90% width / model 90% width (median) | 0.70 | 0.41 | 0.87 |
| PE residual beyond 3 PE sd | 17% | 0% | 0% |
| B2 residual beyond 3 model σ | 5% | 5% | 8% |
| multimodal PE proxy (Δn prob < 0.9 or mass beyond 3 robust sd) | 20% | 0% | 12% |
| Morse-index difference identified | 0.82 | 1.00 | 0.98 |
| PE closer to truth than B2; Wilcoxon p | 45%; 0.20 | 75%; < 10⁻³ | 60%; 0.07 |

Findings:

* With the host sky, PE beats the SNR model clearly for ET (factor 3.4 in median error, factor 2.4 in the
  90th percentile, no outliers) and modestly for O5 (0.15 vs 0.19; the O5 error is SNR-noise dominated and
  PE recovers it with higher harmonics and shared source parameters).
* Free-sky ET PE is worse than the SNR model in its tails: 17% of systems have the truth more than 3 posterior
  sd away, the PIT is non-uniform, and the worst cases (`out/figures/failure_cases.csv`) are wrong sky modes
  50–110° from the truth with the wrong Morse-index difference (residuals of 1.5–4.2). This reproduces the paper's
  statement and shows why the benchmark's host-sky assumption is load-bearing.
* The SNR model is conservative in width (PE/model 0.41 for ET, 0.88 for O5) but its tails are heavier than
  Gaussian: its own 90% error bars cover 77–80% of the matched truths, and 5–8% of residuals exceed 3σ.
  The analytic SNR-only σ, which is what the O4a injections validate, covers 43% (ET) and 68% (O5), because it
  omits the response term.
* By family (`out/pe_matched/c_summary.csv`): the largest single B2 residuals are at ρ̂_B ≈ 8–9 with
  ρ_J ≈ 15–18 (e.g. ET SIE-7777-0009346: B2 −3.69 vs truth −1.74; host-sky PE −1.41 ± 0.25), i.e. the sky term at
  low joint SNR, and O5 SIS systems with ρ̂_B ≈ 8 (B2 off by 1.8–2.0 where PE is within 0.3).

## 5. Task D: propagation into the absolute-magnification inference

Same 60 systems per scenario (ET with host sky, O5 free sky; the free-sky ET rows are in `out/pe_matched/d_summary.csv`).
NPE = the published scale-free flow (Geometry + μ ratio + z, σ_v), intervals conformally calibrated per true
family as in the paper (global calibration in the last column); D1 uses a flow retrained with the exact ratio,
with the same data, split, architecture and schedule; D4 = the two-stage Bayesian reconstruction with the PE
posterior as effective likelihood (source-normalised prior, the paper's v5 numbers).

| Configuration | ET median err | ET p90 err | ET cov 68 / 90 | ET cov 90 global | O5 median err | O5 p90 err | O5 cov 68 / 90 | O5 cov 90 global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0 geometry only (NPE) | 11.2% | 32% | 0.60 / 0.92 | 0.90 | 15.8% | 48% | 0.65 / 0.90 | 0.90 |
| D1 true ratio (NPE retrained on truth) | **5.7%** | 33% | 0.65 / 0.88 | 0.88 | 15.5% | 36% | 0.60 / 0.87 | 0.90 |
| D1a true ratio into the current NPE | 8.4% | 31% | 0.62 / 0.87 | 0.88 | 15.5% | 47% | 0.62 / 0.87 | 0.88 |
| D2n naive SNR ratio into the current NPE | 11.8% | 39% | 0.60 / 0.83 | 0.85 | 17.3% | 62% | 0.60 / 0.83 | 0.87 |
| D2 SNR-model ratio (NPE, paper) | 10.8% | 41% | 0.57 / 0.85 | 0.88 | 17.8% | 47% | 0.63 / 0.87 | 0.90 |
| D3 joint-PE median ratio (NPE, paper) | 10.5% | 31% | 0.63 / 0.85 | 0.88 | 13.8% | 48% | 0.63 / 0.87 | 0.88 |
| D4 Bayesian two-stage, simulated family (paper) | 7.9% | 30% | 0.58 / 0.82 | — | 13.7% | 35% | 0.61 / 0.92 | — |
| D4e Bayesian two-stage, EPL+shear for all (paper) | 12.4% | 41% | 0.68 / 0.95 | — | 18.6% | 40% | 0.73 / 0.88 | — |

Population level (full test split, 48 000 ET and 1 700 O5 systems; `out/oracle_flows/*.csv` and the published
`npe_*_Geometry_GWratio_FU.csv`, `npe_*_Geometry.csv`):

| | SIS | SIE | SIE+shear | EPL+shear | all |
| --- | --- | --- | --- | --- | --- |
| ET geometry only | 3.4% | 7.3% | 11.9% | 20.9% | 9.2% |
| ET SNR-model ratio (paper) | 3.4% | 7.2% | 10.6% | 19.8% | 8.7% |
| ET true ratio (oracle) | 0.2% | 5.2% | 7.4% | 17.6% | 5.2% |
| O5 geometry only | 11.3% | 14.3% | 22.6% | 24.8% | 16.0% |
| O5 SNR-model ratio (paper) | 10.6% | 13.1% | 17.3% | 27.6% | 15.1% |
| O5 true ratio (oracle) | 7.7% | 10.6% | 13.4% | 21.3% | 11.4% |

Conformal 90% coverage is 0.89–0.90 for every row; the raw (uncalibrated) 68% coverage is 0.42–0.48 for
EPL+shear in every configuration, so the under-dispersion of the raw NPE for that family is independent of the
ratio input.

Findings:

* The published numbers are reproduced exactly (D2 10.8 / 17.8%, D3 10.5 / 13.8%, D4 7.9 / 13.7%, D4e 12.4 / 18.6%).
* The information in the GW ratio is bounded: even an exact ratio improves the population error by 3.5–4.6 points
  (ET 9.2 → 5.2%, O5 16.0 → 11.4%). The SNR-model measurement delivers 0.5–1 point of this; a measurement
  three times more precise (host-sky ET PE) delivers no more *through the NPE trained on the SNR model* (D3 ≈ D2),
  because the flow has learnt to discount the ratio by the training noise. The only paths that convert PE precision
  into magnification precision are retraining the NPE on PE-like noise (D1 shows the ceiling) or the Bayesian
  two-stage analysis (D4), whose 7.9% (ET) sits between the SNR-model and oracle values.
* The dependence on the ratio is family-specific: decisive for SIS (3.4 → 0.2%), moderate for SIE+shear
  (10.6 → 7.4%), small for EPL+shear (19.8 → 17.6%) where the lens model, not the GW data, limits the inference.
* Near-caustic systems (|μ_A| ≥ 5; 6 ET and 36 O5 of the matched systems): errors of 21–35% for every
  configuration except D1 (ET 4%, n = 6); coverage drops to 0.33–0.67 for the ET configurations and is 0.81–0.89
  for O5. The paper's statement that near-caustic systems are under-covered holds for every input.
* A wrong ratio degrades gracefully: the naive ratio fed into the current NPE costs 0.6 (ET) to 1 (O5) point
  in the median and 2–4 points of coverage; the free-sky ET PE ratio gives 12.4% against 11.2% for geometry alone.
* Prior removal: D4 removes the PE distance prior by importance weights before forming the effective likelihood
  (`gwlens/joint_gwem.py`); the NPE configurations D2/D3 use point summaries only, which is one reason D3 cannot
  exploit the PE width.

## 6. Task E: stress tests

Completed with existing data:

* **Real O4a noise, exact-template recovery, stratified** (`out/o4a_strata.csv`; 321 pairs with ρ_B > 8).
  Median |Δ ln R| 0.074 (O4a) vs 0.067 (Gaussian), 90th percentile 0.22 vs 0.20, 99th 0.46 vs 0.37; robust
  standardised scatter 1.02 vs 0.85; one glitch outlier (|Δ ln R| = 4.2, 0.3%). Stratified by the weaker image's
  SNR the median *error* is biased positive at low SNR in both noise types: +0.07 to +0.09 (Gaussian) and
  +0.10 (O4a) at ρ_B = 8–12, +0.10 (O4a, KS p = 0.005) at 12–15, and consistent with zero above 20. This is the
  maximisation bias of a matched-filter SNR (maximised over phase and a ±5 ms window), which the catalogue model
  ρ̂ = ρ + N(0,1) does not contain; it is a 5% bias of the SNR ratio, small against the 20% B2 error but
  systematic. Fainter-image SNR ratio < 0.4: O4a median |error| 0.125 vs 0.085 (Gaussian).
* **Free-sky ET degeneracies** (Section 4): sky–polarisation–Morse-phase degeneracy produces wrong modes in
  20% of systems; identified-host sky removes them.
* **Inclination–distance** (archived PE summaries, not re-analysed here): the paper's Fig. 4(b).
* **Image association**: D4 with marginalised assignment exists in the Paper II archive (`gwem_posthoc_asg_sn`)
  and was not re-run here.

Not done (would need new simulations; proposed): waveform mismatch between injection and template (e.g. XPHM
injections recovered with XAS/XHM, or non-aligned spins), glitch injections with and without vetoes, PE in real
O4a noise, and unequal-SNR pairs with ρ_B < 8 (sub-threshold second image).

## 7. Figures (`out/figures/`)

1. `fig1_true_vs_estimated_matched.png`: true vs estimated ln(μ_B/μ_A) for B1, B2 (with model 90% bars) and
   joint PE (median, 90% interval) on the 120 matched systems.
2. `fig2_error_distributions_catalogue.png`: B1/B2/B3 error distributions on the full catalogues (log scale),
   with median |d| and the fraction beyond a factor 2 in the legend.
3. `fig3_error_vs_snr.png`: median |error| of B1/B2/B3 vs weaker-image SNR and joint SNR (catalogue), with the
   matched PE residuals.
4. `fig4_naive_bias_vs_response_ratio.png`: naive-ratio error against 2 ln(R_B/R_A).
5. `fig5_coverage_vs_nominal.png`: empirical vs nominal coverage of PE, of B2 with the model σ, and of B2 with
   the analytic σ; right panel: raw NPE coverage with the SNR-model and the oracle ratio.
6. `fig6_downstream_ladder.png`: D0–D4 median and 90th-percentile error of μ_A and 90% coverage.
7. `failure_cases.csv`: the twelve largest PE residuals, PE standardised residuals and SNR-model residuals with
   their Morse probabilities, sky offsets, SNRs and multiplicities (reproducible from the archived posteriors).

## 8. Classification of findings

### 8.1 Confirmed bugs
None in the audited code paths.

### 8.2 Suspected leaks
None. The two documented v1 leaks (fixed lens centre, truth-ordered images) are absent from v2: the frame is
random and the inputs are translation- and rotation-invariant (features.py:64–78), images are ordered by
measured SNR (catalogue.py:61), and the one place where a truth label is used at test time is the per-family
conformal calibration, which the paper declares as a diagnostic (and the global calibration gives 0.88–0.90 on
the matched systems, Table in Section 5).

### 8.3 Valid simulation design choices that should be stated more explicitly
* The arrival-time difference enters without noise (`obs_dt_ab`); only its sign is used in the scale-free sets.
* The EM image count uses a threshold on the true |μ| of the other images (0.1).
* The velocity dispersion is the generator's lens-strength parameter with 10% noise (already stated).
* The response error of the GW ratio is a prescription (isotropic sky error 15° or 2° × 20/ρ_J, 0.5 rad in ψ,
  0.3 in cos ι, identical for both images), not a calibrated distribution; it dominates the ET error budget.
* The detection statistic ρ̂ = ρ + N(0,1) has no maximisation bias; real matched filtering adds +0.05 to +0.10
  to ln R at ρ_B < 15.

### 8.4 Claims that need qualification
* "The SNR-based GW measurement model is conservative" (discussion.tex, item 2): true for the width (PE/model
  0.41 ET, 0.88 O5) but its 90% error bars cover 77–80% of matched truths and its tails are heavier than Gaussian.
* "The Gaussian amplitude-error model therefore describes real O4a noise well" (intro_methods.tex:103): holds
  for the SNR-noise part only; the O4a test does not probe the response term, which is two thirds of the ET error.
* "The NPE trained with the SNR-based model therefore works with relative magnifications measured by full PE"
  (pe.tex:13): it accepts them without harm but does not benefit from their precision (D3 ≈ D2 vs oracle 5.7%).
* The median absolute error 0.074 of the O4a injections is a property of exact-template recovery at SNR 10–60; it
  is not the catalogue error (0.20) and should not be quoted as such.

## 9. Section-by-section recommendations for the manuscript

Already supported, keep: the leak-free construction and frame invariance (Sec. II D, III A); the information-content
table and the scale-free/not-scale-free distinction (Sec. III, Table II); the real-data distance calibration (Sec. II E);
the joint-PE validation numbers and the free-sky ET discussion (Sec. IV A); the Bayesian-vs-NPE table (Sec. IV B);
the mass-sheet and transfer results.

Changes that need no new runs (numbers in this report):

1. **Sec. II C (GW measurements)**, after Eq. (2): add the measured error budget of the catalogue ratio: median
   |Δ ln(μ_B/μ_A)| 0.20 (both scenarios), 90th percentile 0.77 (ET) / 0.62 (O5), 12% / 8% of pairs beyond a factor 2;
   for ET two thirds from the sky/orientation term; naive ratio 0.59 / 0.40. One sentence that the response
   error is a prescription, not calibrated on data.
2. **Sec. II E (O4a)**: state that the injections validate the SNR-noise term only (analytic σ covers 43% of the
   full ET error), add the low-SNR maximisation bias (+0.1 at ρ_B < 15), and do not describe 0.074 as the model error.
3. **Sec. IV A**: replace "conservative" by "wider than PE by factors 2.4 (ET) and 1.1 (O5) but with heavier tails:
   its own 90% intervals cover 80% (ET) and 77% (O5) of the 60 systems". Add the paired statistic (PE closer in 75% / 60%).
4. **Sec. IV A, end-to-end paragraph**: add the ladder as one table (D0, D1, D2, D3, D4 for ET host-sky and O5;
   the rows exist in Section 5) and the sentence that the NPE trained on the SNR model does not exploit the PE
   precision, while the Bayesian two-stage analysis does; the oracle ceiling is 5.7% (ET) and 15.5% (O5) on these
   systems and 5.2% / 11.4% on the population. State the family dependence (SIS decisive, EPL+shear small).
5. **Sec. V (discussion), item 2**: reword as in 8.4; add that geometry alone gives 9.2% / 16.0% on the population and
   that the GW ratio is bounded at 3.5–4.6 points even if exact.
6. **Sec. II D / Table I**: state that the delay is noise-free and only its sign enters the scale-free sets; state the
   |μ| ≥ 0.1 EM-image threshold.
7. **Appendix / data availability**: reference the audit outputs (per-pair B1/B2/B3 table) in the Zenodo record when it
   is next versioned.

Claims that would require new results before being made: any statement about PE in real noise, waveform mismatch
or glitch-contaminated PE (Section 6, not done).

## 10. Unresolved issues

* The origin of the "≈ 0.0" statement (no metric, revision or population given).
* The fidelity of the sky/orientation error prescription, which sets the ET tails of the benchmark ratio: the matched
  systems (60 per scenario) only show that it is wider than PE on average and heavier-tailed than Gaussian; a
  calibration of its *distribution* on PE would need a few hundred host-sky PE runs or an analytic Fisher-matrix
  model of the single-site sky error.
* Raw NPE under-dispersion for EPL+shear (68% coverage 0.42–0.48 before conformal correction) in every configuration,
  already known from the paper's Table III; conformal calibration fixes the 90% level only.
* Near-caustic coverage (0.33–0.67 for ET, n = 6) cannot be assessed from the matched sample.

## 11. Code changes

None to published code. New, in `audit/`: `relmag_errors.py` (B), `leak_probe.py` (A), `train_oracle_flow.py`
(D1), `pe_matched.py` (C, D), `o4a_strata.py` (E), `figures.py`. Outputs: `out/relmag/{per_event.parquet, summary.csv,
strata.csv, checks.json}`, `out/leak/{integrity.json, correlations.csv, single_feature.csv, published.csv,
published_rounds_to_zero.csv, published_recomputed_from_predictions.csv}`, `out/oracle_flows/` (two checkpoints,
metrics, predictions), `out/pe_matched/{c_per_event.csv, c_summary.csv, d_per_event.csv, d_summary.csv}`,
`out/o4a_strata.csv`, `out/figures/`.
