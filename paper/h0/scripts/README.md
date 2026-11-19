Scripts that produce the numbers, table and figures of Paper II (H0 from lensed GWs with identified hosts).

- `p2_numbers.py OUT` : combined and per-family H0, correction stages, delay-only comparison, slope test, strain-level
  check; writes `fig_corrections.pdf`, `fig_delay.pdf`, `fig_slope.pdf`, `table_h0.tex`, `numbers.json`.
- `p2lib.py`, `kde1d.py` : shared loading of the per-event h samples, selection fractions and summaries; KDE on a grid.
- `alpha_h.py` : selection fraction alpha(h) per lens family and scenario (run on the catalogs).
- `catalog_dists.py OUT.npz` : parent and detected populations from the data/v2 catalogs.
- `sample.py` : table of the analysed systems from the GW+EM input files (`sample.csv`).
- `fig_sample.py OUT` : Fig. 2 and `table_sample.tex`;  `fig_example.py OUT` : Fig. 3 (one ET EPL+shear system);
  `fig_persystem.py OUT` : Fig. 5;  `fig_pit.py OUT` : Fig. 8.
- `../figures/fig_schematic.tex` : Fig. 1 (TikZ, compiled with pdflatex).

Inputs (per-event posterior samples, selection fractions, summaries, GW+EM inputs) will be archived at Zenodo.
- `fig_fullpost.py OUT_DIR [SCENARIO EVENT_ID SUFFIX]` — Appendix A: corner plots of all parameters of the worked example
  (default ET EPL_shear-7777-0006095): the joint parameter estimation of both GW images (`fig_corner_gw{SUFFIX}.pdf`) and the
  GW+EM posterior of the lens model, kappa_ext and h (`fig_corner_lens{SUFFIX}.pdf`), plus `table_fullpost{SUFFIX}.tex` and
  `numbers_fullpost{SUFFIX}.json`. Needs `corner`. Supplemental Material S2: `fig_fullpost.py OUT O5 EPL_shear-7777-0025278 _O5`.
  Inputs: `p2/example/ET_EPL_shear-7777-0006095_knownsky_posterior.parquet`, `..._posthoc_lcdm_EPL_shear_h0_sn_posterior.parquet`
  and the GW+EM input `p2/inputs/inputs/gwem/ET_EPL_shear-7777-0006095.json` (true values).
- `table_persystem.py OUT` — Supplemental Material: `persystem.csv` (true values and 5/50/95% quantiles of every system),
  `table_persystem.tex` (Table S3), `table_coverage.tex` (Table S1) and `numbers_persystem.json`. Reads the joint-PE posteriors
  and summaries (`p2/pe/results/pe/{ET,O5}`) and `sample.csv`. The supplement is built with `../supplement/build.sh`.
- `fix0017024/` — O5 EPL_shear-7777-0017024 had the labels of its two GW images swapped between the joint parameter estimation
  and the GW+EM input; `README.md` describes the corrected input and the reruns (`run_fix.sh`, GPU server), and `collect_fix.sh`
  fetches them into `p2/fix`, from where `p2lib.py`, `p2_numbers.py` and `sample.py` read them on top of the archived results.
