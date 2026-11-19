# Final results of the lensed-pair search (2026-11-19)

Compact outputs of the search, copied from the analysis server's `results/` directory. The bulky
outputs are not kept here: posterior samples, dynesty result files, the stage-1 caches, and the
GWOSC strain and LVK posteriors, which are public and are fetched with `lensearch.fetch_*`.

| Path | Contents |
|---|---|
| `rank/summary.json`, `rank/pairs_ranked.csv.gz` | Stage 1: ln B_cat and ln B_pop for every event pair (`lensearch.rank`) |
| `stage2_tasks.txt`, `stage2_events.txt`, `stage2_tasks.pairs.csv` | Pairs and events selected for stage 2 |
| `marginal_pairs.csv`, `marginal_records.csv` | Sub-threshold search candidates in stage 2, with their search SNR |
| `inject/` | Lensed injection set: lens library, systems, noise placement, task list |
| `bg/` | Unlensed background: twins and selection-matched unrelated pairs (`lensearch.twins`, `lensearch.null_pairs`) |
| `analysis/` | Stage 2 (`lensearch.stage2_analysis`). `real_pairs.csv` holds ln C, the data-quality and sampling flags and the FAPs; `summary.json` holds the headline numbers |
| `paper_figures/` | Figures and the top-pairs table (`lensearch.paper_figures`) |

Headline result: there are 42 clean pairs. The largest ln C is 13.4, for GW250109_010541+GW250116_015318
(per-pair p = 1/17, global FAP 0.92). The pair GW240615_023849+GW240615_055016 has ln C 25.7 and is
excluded as a data artefact, because the candidate data-quality test flags it. The injection efficiency at
the loudest clean value is 0.22 ± 0.06, which bounds the number of lensed pairs to fewer than 10 at 90%
confidence.

