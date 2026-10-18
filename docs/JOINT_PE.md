# Waveform-level and joint GW+EM parameter estimation

Two modules go beyond the SNR-based measurement model of `gwlens/catalogue.py`.

| Module | What it samples | Cost per event |
| --- | --- | --- |
| `gwlens/joint_pe.py` | Strain data of both GW images (IMRPhenomXHM, aligned spins, Morse phases, per-image antenna patterns) with shared source parameters and free `ln(mu_B/mu_A)`, `Delta t`, `n_A`, `n_B` | ET: ~3.5 h on 8 cores (nlive 150); O5: less |
| `gwlens/joint_gwem.py posthoc` | EM lens model (image positions, light shape, velocity dispersion, redshifts) times the GW likelihood summarised from a `joint_pe` posterior | minutes on 1 core |
| `gwlens/joint_gwem.py full` | Strain data of both images and the EM lens model simultaneously | ~25 ms per likelihood call; 1–3 days on 32 cores (cluster) |

## Physics

In geometric optics the GW likelihood depends on the lens only through the image magnifications,
arrival times and Morse indices. The joint model therefore has the lens parameters
`(sigma_v -> theta_E, q, phi_q, gamma', gamma_ext, phi_ext, centre)`, an external convergence
`kappa_ext` and the source position. The source position is marginalised analytically in the
linearised image-plane likelihood `sum_i |A_i^-1 (beta_i - beta)|^2 / sigma_img^2`. The model
images are refined by one Newton step, and the image properties are evaluated there.

* `--cosmology free` (default; the paper's benchmark setting): the GW amplitude constrains only
  `mu_B/mu_A` (the GW apparent distance is uninformative under a log-uniform `d_L` prior). The
  arrival order must agree with the Fermat potential. `kappa_ext ~ N(0, sigma_kext)` encodes the
  mass-sheet degeneracy.
* `--cosmology lcdm`: the simulation cosmology plus the observed redshifts gives `d_L(z_s)`, so
  the GW apparent distance measures `|mu_A|` directly. The millisecond-precise delay fixes
  `(1 - kappa_ext) * D_dt * Delta phi`, and `kappa_ext` is derived from it (needs
  `sigma_kext > 0`, e.g. 0.025).
* `--assignment marginalise` sums over which EM images host the two GW images (`posthoc`) or
  samples the assignment as a discrete parameter (`full`).
* Priors match the simulated lens populations: SLACS-like `q`, log-normal shear, `gamma'` ~
  N(2.08, 0.15), the sigma^4-weighted SDSS velocity-dispersion function, and a flat source
  position. Detection selection effects are *not* included in the Bayesian prior; they are
  included implicitly in the NPE training set.

`posthoc` is exact up to the summary of the GW likelihood: a Gaussian in `(ln mu_rel, ln D_A)`
fitted to the `joint_pe` posterior with importance weights that remove its priors, times the
posterior probabilities of the nine Morse combinations. `full` makes no such approximation and
is the reference for checking it.

## Running on a cluster

1. On the machine with the catalogues, select the events and write the self-contained inputs:

   ```bash
   python scripts/select_pe_events.py --scenario ET --n-per-family 15 --out results/pe/tasks_ET.txt
   PY=python scripts/prepare_gwem_inputs.sh results/pe/tasks_ET.txt inputs/gwem
   ```

   Each JSON file (~3 kB) holds the catalogue row, all resolved image positions, the EM
   measurements and the truths. `prepare` replays the catalogue chunk, checks it against the
   stored row and fails loudly if the reproduction is not exact.

2. Copy the repository and `inputs/gwem/` to the cluster, install `requirements.txt` (tested
   with bilby 2.8.2, dynesty 3.1.0, lenstronomy 1.14.2, lalsuite 7.26, numba 0.68) and submit:

   ```bash
   ls inputs/gwem/*.json > gwem_events.txt
   sbatch --array=1-$(wc -l < gwem_events.txt)%20 scripts/slurm_gwem.sh gwem_events.txt results/gwem_full \
          --lens-model EPL_shear --cosmology free --nlive 300
   ```

   Runs checkpoint every 10 minutes and resume if a task is requeued. Recommended sampler
   settings are nlive 300–500 and naccept 30. The strain-only `joint_pe` runs can be submitted
   the same way:

   ```bash
   python -m gwlens.joint_pe --catalogue data/pe/SIS.parquet --scenario ET --event-ids ... --npool 32
   ```

3. Outputs per event: `<label>_posterior.parquet` (samples with derived `mu_a`, `mu_b`, Morse
   indices, `kappa_ext`, `theta_e`, image assignment and, for `free`, the corrected source
   distance) and `<label>_summary.json` (truth, median, 90 per cent interval, PIT, run time).

## Two-stage runs on finished `joint_pe` posteriors

```bash
PY=python scripts/run_gwem_posthoc.sh inputs/gwem results/pe results/gwem_posthoc 8
```

By default this runs four configurations: the true family and EPL+shear without a cosmology,
EPL+shear with ΛCDM and `sigma_kext = 0.025`, and EPL+shear without a cosmology but with
`sigma_kext = 0.025`.
Each entry of `CONFIGS` is `cosmology:sigma_kext:lens_model[:sigma_img_gw]`; the paper also uses
`free:0:same:0.03 free:0:same:0.1` for the astrometry test. Known-sky GW posteriors
(`*_knownsky_posterior.parquet`) are written to directories with the suffix `_ks`. To split the
task list over several hosts, give each host a disjoint set of residues, e.g. `SHARD=0,1/10` on
one host and `SHARD=2,3/10` on the next; a final run without `SHARD` on the host that holds all
outputs fills any gaps, because finished tasks are skipped.

## Analysis

```bash
python analysis/pe_analysis.py --pe-dir results/pe --out-dir results/pe_analysis \
       --flow-dir results/v2_review_npe2 --flow-name Geometry_GWratio_FU --figure results/pe_analysis/fig_pe.pdf
python analysis/gwem_analysis.py --figure results/gwem_analysis/fig_gwem.pdf
```

`pe_analysis.py` compares the joint-PE posteriors with the summary model (coverage, PIT, width
ratios, Morse-index identification) for free- and known-sky runs, and feeds the PE relative
magnification into the scale-free NPE. `gwem_analysis.py` summarises the full and two-stage
GW+EM runs per configuration and compares them with each other (matched noise) and with the NPE.

## Relative-binning fallback

Where the relative-binning log-likelihood exceeds the fiducial (injection) log-likelihood by
more than 50, the binning approximation has failed and `joint_pe` evaluates the exact
likelihood for that point. The injection and relative-binning generators need separate
`waveform_arguments` dicts, because relative binning writes its bin edges into its own.

## Reproducibility note

`gwlens.joint_pe.interferometers` seeds bilby's own random generator (bilby >= 2 does not use
numpy's global generator for detector noise) since commit "Seed bilby noise generator". The
first 14 events of the 2026-10-16 batch (listed in `results/pe/events_started_before_seed_fix.txt`
on the analysis server) were run before this fix. Their posteriors are valid, because each run
generates its data once and analyses it consistently, but their noise realisations cannot be
regenerated from the seed.
