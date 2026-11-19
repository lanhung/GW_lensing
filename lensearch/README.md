# Search for strongly lensed GW pairs with known lenses

Real-data search of all public LVK events (O1–O4b with PE, the O4c event GW250207, O4c public alerts, and marginal
O4a/O4b candidates) for pairs of images of one source, conditioned on the catalogue of known strong lenses.

## Data (run where Zenodo/GWOSC are fast; outputs are small)

| Step | Command | Output |
|---|---|---|
| PE samples, PSDs, settings | `python -m lensearch.fetch_pe --out data/pe --workers 1` | `<event>.{parquet,npz,json}` |
| Marginal candidates | `python -m lensearch.fetch_candidates GWTC-5.0 --pe-files data/pe/files.csv --out data/cand` (also GWTC-4.1) | sky maps + template parameters |
| O4c alerts | `python -m lensearch.fetch_gracedb --out data/o4c_alerts` | nside-128 sky maps + distance layers |
| Lens catalogue | `python -m lensearch.lenscat_merge --catalog-dir <downloads> --out data/lenses_merged.parquet` | 48,739 systems |
| Strain segments | `python -m lensearch.fetch_strain EVENTS --pe-dir data/pe --out data/strain --tmp tmp` | analysis segments |

Zenodo allows 133 requests per minute; `fetch_pe` reads only the needed datasets with HTTP range requests.

## Stage 1: ranking all pairs (minutes to an hour)

```bash
python -m lensearch.rank --pe-dir data/pe --cand-dirs data/cand/GWTC-5.0 data/cand/GWTC-4.1 \
    --alerts-dir data/o4c_alerts --lenses data/lenses_merged.parquet --out results/rank --n-rot 200 --n-shuffle 50
```

`ln B_cat` (lensed by a catalogued lens, SIS delay and apparent-distance consistency) with a background from
right-ascension rotations of the catalogue; `ln B_pop` (lensed by any galaxy) with a background from time shuffles.

## Stage 2: joint PE of the top pairs on real strain

```bash
python -m lensearch.stage2_tasks --ranked results/rank/pairs_ranked.csv --out tasks_stage2.txt --events-out events.txt
PY=python PE_DIR=data/pe CACHE=data/strain lensearch/run_queue.sh tasks_stage2.txt SLOTS NPOOL results/stage2
```

Each pair: single A, single B (shared prior ranges) and joint A+B; coherence ratio ln C = ln Z_joint − ln Z_A − ln Z_B.
The likelihood uses the LVK PSDs and waveform flags (IMRPhenomXPHM, SpinTaylor precession) and reproduces the
LVK log-likelihood to 1–2 (no calibration marginalisation).

## Efficiency: lensed pairs in real O4 noise

```bash
python -m lensearch.inject library --family EPL_shear --n 30000 --out lib.parquet         # scale-free lens solutions
python -m lensearch.inject draw --run O4a --n 100 --psd-dir data/psd --library lib.parquet --out sys.csv
python -m lensearch.inject place --systems sys.csv --blocks blocks_all.csv --out placed.csv   # sidereal-day shifts
python -m lensearch.inject cut --placed placed.csv --out noise --tmp tmp                    # local (GWOSC)
lensearch/run_queue.sh tasks_inj.txt ... --injections placed.csv --noise-dir noise
```
