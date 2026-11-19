"""Collect stage-2 results: coherence ratio and lensing observables per pair.

For each joint run in --results with both matching single runs (same prior ranges): ln C = ln Z_joint - ln Z_A - ln Z_B
(with its sampling error), the posterior probabilities of the Morse-index difference n_B - n_A, the median and 90%
interval of ln(mu_B/mu_A), and the run times. Writes <results>/stage2_pairs.csv.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', type=Path, required=True)
    a = p.parse_args()
    summ = {f.name[:-len('_summary.json')]: json.loads(f.read_text()) for f in a.results.glob('*_summary.json')}
    rows = []
    for label, s in summ.items():
        if s['kind'] != 'joint':
            continue
        A, B = s['events']
        approx = s['approximant']
        tag = f'_for_{A}__{B}_{approx}'
        sa, sb = summ.get(f'single_{A}{tag}'), summ.get(f'single_{B}{tag}')
        row = dict(a=A, b=B, approximant=approx, ln_z_joint=s['log_evidence'], ln_z_joint_err=s['log_evidence_err'],
                   hours_joint=s['seconds'] / 3600)
        post = pd.read_parquet(a.results / f'{label}_posterior.parquet')
        if {'morse_a_n', 'morse_b_n'} <= set(post):
            dn = (post.morse_b_n - post.morse_a_n).round(2)
            for v in (-1.0, -0.5, 0.0, 0.5, 1.0):
                row[f'p_dn_{v:+.1f}'] = float(np.mean(dn == v))
        if 'ln_mu_rel' in post:
            q = np.quantile(post.ln_mu_rel, [0.05, 0.5, 0.95])
            row.update(ln_mu_rel_q05=q[0], ln_mu_rel_med=q[1], ln_mu_rel_q95=q[2])
        if sa and sb:
            row.update(ln_z_a=sa['log_evidence'], ln_z_b=sb['log_evidence'], hours_a=sa['seconds'] / 3600,
                       hours_b=sb['seconds'] / 3600)
            row['ln_coherence'] = s['log_evidence'] - sa['log_evidence'] - sb['log_evidence']
            row['ln_coherence_err'] = float(np.sqrt(s['log_evidence_err'] ** 2 + sa['log_evidence_err'] ** 2 +
                                                    sb['log_evidence_err'] ** 2))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(a.results / 'stage2_pairs.csv', index=False)
    with pd.option_context('display.width', 200):
        print(df.round(3).to_string())


if __name__ == '__main__':
    main()
