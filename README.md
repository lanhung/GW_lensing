# Magnification inference for strongly lensed gravitational waves

Code for simulating strongly lensed binary-black-hole (BBH) image pairs and inferring the
absolute magnification of a lensed gravitational-wave (GW) image from GW and electromagnetic
observables, together with the real-data calibration and validation used in the accompanying
MNRAS manuscript.

The repository has two generations of analysis:

* **v2 (current, `gwlens/`, `analysis/`, `realdata/`)** — realistic lensed-BBH populations
  drawn from the GWTC-4.0 population model, four galaxy-lens families, O5-like and Einstein
  Telescope networks, GW measurement errors calibrated on GWTC-1 to GWTC-4.0 events and validated
  by injections into public O4a detector noise, conformalized quantile regression and neural
  posterior estimation.
* **v1 (`core_experiments/`, `experiments/`, root scripts)** — the original selected-SIS and
  four-family benchmarks, kept for reproducibility. Two input leaks found during review are fixed
  (see [docs/CHANGES.md](docs/CHANGES.md)).

## Layout

| Path | Purpose |
| --- | --- |
| `gwlens/population.py` | GWTC-4.0 BP2P masses, spins, Madau-Dickinson-shaped rate, optical-depth-weighted source redshifts, lens redshifts, SDSS velocity-dispersion function |
| `gwlens/lenses.py` | SIS, SIE, SIE+shear and EPL+shear image solutions (lenstronomy), magnifications, Morse indices, delays |
| `gwlens/detectors.py` | O5-like (A+ H1/L1, AdV) and ET-D networks; optimal SNRs with Earth-rotation-dependent antenna patterns |
| `gwlens/catalogue.py` | End-to-end catalogue generator (selection, GW and EM measurement models) |
| `analysis/features.py` | Translation/rotation-invariant observable features and input sets |
| `analysis/benchmark.py` | HistGB + conformalized quantile regression; input-set, mixed-family, transfer, sky and learning-curve experiments |
| `analysis/npe.py` | Neural posterior estimation (conditional neural spline flow, zuko) with PIT/coverage checks |
| `analysis/make_figures.py` | Manuscript figures and LaTeX tables |
| `realdata/calibrate_distance_errors.py` | Calibrates distance errors versus SNR on GWTC-1–4.0 BBHs from the GWOSC event API |
| `realdata/fetch_gwtc4_population.py` | Extracts the GWTC-4.0 fiducial BBH population hyperparameters |
| `realdata/o4a_injections.py` | Injects lensed image pairs into public O4a H1/L1 strain and validates the relative-amplitude error model |
| `configs/gwtc4_bp2p.json` | Population hyperparameters used by the generator |
| `core_experiments/`, `experiments/`, `scripts/` | v1 benchmarks (leak-fixed) |
| `docs/` | Methods, results summary, data description and change log |

## Quick start

Python 3.10–3.12. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Real-data calibration (downloads public GWOSC metadata, a few MB):

```bash
python realdata/calibrate_distance_errors.py --out-dir realdata/outputs
python realdata/fetch_gwtc4_population.py
```

Generate catalogues (each family: 250 000 lens systems; about 2 min for SIS and 20–40 min for
the lenstronomy families on 60 cores):

```bash
for f in SIS SIE SIE_shear EPL_shear; do
  python -m gwlens.catalogue --family $f --n-systems 250000 --seed 20261016 --processes 60 --out data/v2/$f.parquet
done
```

Benchmark, neural posterior estimation and figures:

```bash
# HistGB/CQR benchmark, conditional calibration, NPE for four input sets (3 GPUs) and figures
PY=python scripts/run_analysis_v2.sh
# or individual steps, e.g.
python analysis/benchmark.py --out-dir results/v2 --jobs 16 --threads 4
python analysis/npe.py --out-dir results/v2_npe_conf --scenario ET --input-set "Geometry+GW(rel)+FU"
```

O4a noise validation (downloads ten 4096-s O4a strain files, about 1.3 GB):

```bash
python realdata/o4a_injections.py --pairs 400 --out-dir realdata/outputs
```

See [docs/METHODS.md](docs/METHODS.md) for the simulation and inference details,
[docs/RESULTS.md](docs/RESULTS.md) for the headline numbers and [docs/DATA.md](docs/DATA.md) for
data products.

## Licence and citation

See `NOTICE.md` and `CITATION.cff`.
