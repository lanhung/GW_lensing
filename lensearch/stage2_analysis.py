"""Stage-2 results of the lensed-pair search: coherence ratios of the real pairs, the unlensed-twin background,
the lensed-injection efficiency, and the data-quality and sampling diagnostics.

    python -m lensearch.stage2_analysis --collected results/collected --pe-dir data/pe --pairs results/stage2_tasks.txt \
        --marginal results/marginal_pairs.csv --twins results/bg/twins_s1.csv --placed results/inject/placed_s1.csv \
        --ranked results/rank/pairs_ranked.csv --out results/analysis

For every pair with a joint run and both single runs (same prior ranges):
    ln C = ln Z_joint - ln Z_A - ln Z_B                       (coherence ratio)
Diagnostics per event: our single-run maximum log-likelihood minus the LVK maximum-likelihood sample's value. A large
excess means the public strain contains power that the LVK analysis did not see (glitch subtraction that is not public);
such events are flagged (dq). Per pair: joint maximum log-likelihood minus the sum of the single maxima; a positive
excess means a single run did not find a mode that the joint run found (sampling flag).
Background: the selection-matched unlensed pairs of lensearch.null_pairs (--null-pairs) give the null distribution of
ln C; the false-alarm probability of a real value counts the null pairs at or above it, and the global one includes the
number of clean real pairs as trials. The pair-matched twins (lensearch.twins) are kept as a diagnostic only: they are
not selection-matched and sit systematically below the real pairs. The lensed injections give the efficiency.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DQ_EXCESS = dict(O4=10.0, O3=20.0)   # single max logL (ours) - LVK maxl above which an event is flagged
DQ_EXCESS_CAND = 10.0                # sub-threshold candidates (no LVK PE): single max logL - search SNR^2/2
CAND_SNR = {}                        # candidate name -> search network SNR (--candidates)
SAMPLING_EXCESS = 5.0                # joint max logL - (max A + max B) above which the singles are flagged
APPROX = 'IMRPhenomXPHM'


def load_runs(collected):
    runs = {}
    for d in sorted(Path(collected).glob('*')):
        if not d.is_dir():
            continue
        for f in d.glob('*_summary.json'):
            lab = f.name[:-len('_summary.json')]
            s = json.loads(f.read_text())
            post = d / f'{lab}_posterior.parquet'
            s['max_logl'] = float(pd.read_parquet(post, columns=['log_likelihood']).log_likelihood.max()) if post.exists() else np.nan
            s['dir'], s['posterior'] = d.name, str(post)
            runs[(f'{d.name}:' if d.name.endswith('_n1000') else '') + lab] = s   # repeats at other settings keep their own key
    return runs


def lvk_info(pe_dir, ev):
    f = Path(pe_dir) / f'{ev}.json'
    if not f.exists():
        return dict(lvk_maxl=np.nan, lvk_snr=np.nan, run='-', bayeswave=False)
    m = json.loads(f.read_text())
    cfg = m['config'].get('config', m['config'])
    cat = str(m.get('catalog', ''))
    post = Path(pe_dir) / f'{ev}.parquet'
    snr = np.nan
    if post.exists():
        p = pd.read_parquet(post)
        for c in ('network_matched_filter_snr', 'network_optimal_snr'):
            if c in p:
                snr = float(p[c].median())
                break
    return dict(lvk_maxl=float(m['maxl'].get('log_likelihood', np.nan)), lvk_snr=snr,
                run='O4' if ('4' in cat or '5' in cat) else 'O3', bayeswave='BAYESWAVE' in str(cfg.get('channel-dict', '')),
                gps=float(cfg.get('trigger-time', cfg.get('trigger_time', np.nan))))


def pair_row(runs, a, b, tag_events=None, prefix=''):
    """ln C and diagnostics of one pair from its joint and two single runs."""
    pa, pb = tag_events or (a, b)
    j = runs.get(f'{prefix}joint_{a}__{b}_{APPROX}')
    sa = runs.get(f'{prefix}single_{a}_for_{pa}__{pb}_{APPROX}')
    sb = runs.get(f'{prefix}single_{b}_for_{pa}__{pb}_{APPROX}')
    row = dict(a=a, b=b, has_joint=j is not None, has_a=sa is not None, has_b=sb is not None)
    if j:
        row.update(ln_z_joint=j['log_evidence'], joint_max_logl=j['max_logl'], hours_joint=j['seconds'] / 3600,
                   nlive=j.get('nlive', np.nan))
        p = pd.read_parquet(j['posterior'])
        if 'ln_mu_rel' in p:
            q = np.quantile(p.ln_mu_rel, [0.05, 0.5, 0.95])
            row.update(ln_mu_rel_q05=q[0], ln_mu_rel_med=q[1], ln_mu_rel_q95=q[2])
        if {'morse_a_n', 'morse_b_n'} <= set(p):
            dn = (p.morse_b_n - p.morse_a_n).round(2)
            row['p_dn_0'] = float(np.mean(dn == 0.0))
            row['p_dn_half'] = float(np.mean(np.abs(dn) == 0.5))
            row['p_dn_1'] = float(np.mean(np.abs(dn) == 1.0))
    for k, s in (('a', sa), ('b', sb)):
        if s:
            row.update({f'ln_z_{k}': s['log_evidence'], f'max_logl_{k}': s['max_logl'], f'hours_{k}': s['seconds'] / 3600})
    if j and sa and sb:
        row['ln_c'] = j['log_evidence'] - sa['log_evidence'] - sb['log_evidence']
        row['ln_c_err'] = float(np.sqrt(j['log_evidence_err'] ** 2 + sa['log_evidence_err'] ** 2 + sb['log_evidence_err'] ** 2))
        row['joint_excess'] = j['max_logl'] - sa['max_logl'] - sb['max_logl']
        row['sampling_flag'] = bool(row['joint_excess'] > SAMPLING_EXCESS)
    return row


def real_pairs(runs, pairs, pe_dir, kind):
    rows = []
    for a, b in pairs:
        row = pair_row(runs, a, b)
        row['kind'] = kind
        for k, ev in (('a', a), ('b', b)):
            info = lvk_info(pe_dir, ev)
            row[f'lvk_maxl_{k}'], row[f'snr_{k}'], row[f'bayeswave_{k}'] = info['lvk_maxl'], info['lvk_snr'], info['bayeswave']
            if np.isfinite(info['lvk_maxl']) or ev not in CAND_SNR:
                exc = row.get(f'max_logl_{k}', np.nan) - info['lvk_maxl']
                thr = DQ_EXCESS.get(info['run'], np.inf)
            else:   # a candidate without LVK PE: the search SNR is the reference (max logL ~ SNR^2/2 for the best template)
                exc = row.get(f'max_logl_{k}', np.nan) - CAND_SNR[ev] ** 2 / 2
                thr = DQ_EXCESS_CAND
                row[f'snr_{k}'] = CAND_SNR[ev]
            row[f'dq_excess_{k}'] = exc
            row[f'dq_flag_{k}'] = bool(np.isfinite(exc) and exc > thr)
            row[f'gps_{k}'] = info.get('gps', np.nan)
        row['dq_flag'] = row['dq_flag_a'] or row['dq_flag_b']
        row['dt_days'] = (row['gps_b'] - row['gps_a']) / 86400 if np.isfinite(row.get('gps_a', np.nan)) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def twin_pairs(runs, twins, prefix=''):
    rows = []
    for r in twins.itertuples():
        row = pair_row(runs, f'{r.system}A', f'{r.system}B', prefix=prefix)
        row.update(system=r.system, pair=r.pair, src_a=r.src_a, src_b=r.src_b, snr_sample_a=r.snr_sample_a,
                   snr_sample_b=r.snr_sample_b)
        rows.append(row)
    return pd.DataFrame(rows)


def injection_pairs(runs, placed):
    rows = []
    for r in placed.itertuples():
        row = pair_row(runs, f'{r.system}A', f'{r.system}B')
        row.update(system=r.system, run=r.run, snr_a=r.snr_a, snr_b=r.snr_b, delta_t_days=r.delta_t / 86400,
                   mu_a=r.mu_a, mu_b=r.mu_b, morse_a=r.morse_a, morse_b=r.morse_b, z_s=r.z_s)
        rows.append(row)
    return pd.DataFrame(rows)


def null_pair_rows(runs, null):
    """ln C of the selection-matched unlensed pairs of lensearch.null_pairs. An original twin pair uses its own runs;
    a new pairing uses its joint run (images in data-time order, either order is looked up) and the single-image
    evidences rescaled to its prior (columns ln_z_a, ln_z_b), or its own single runs when it needed new ones."""
    rows = []
    for r in null[null.selected].itertuples():
        if r.original_twin:
            row = pair_row(runs, f'{r.a[:-1]}A', f'{r.a[:-1]}B')
        elif not r.posterior_inside_prior:
            row = pair_row(runs, r.a, r.b)
            if not row.get('has_joint'):
                row = pair_row(runs, r.b, r.a)
        else:
            j = runs.get(f'joint_{r.a}__{r.b}_{APPROX}') or runs.get(f'joint_{r.b}__{r.a}_{APPROX}')
            row = dict(a=r.a, b=r.b, has_joint=j is not None)
            if j:
                row.update(ln_z_joint=j['log_evidence'], ln_c=j['log_evidence'] - r.ln_z_a - r.ln_z_b,
                           ln_c_err=float(j['log_evidence_err']), joint_max_logl=j['max_logl'], nlive=j.get('nlive', np.nan))
        row.update(src_a=r.src_a, src_b=r.src_b, s1_ln_bcat=r.ln_bcat, s1_ln_bpop=r.ln_bpop, original_twin=r.original_twin)
        rows.append(row)
    return pd.DataFrame(rows)


def false_alarm(x, background):
    """Fraction of background values at or above x, with the +1 of a finite sample."""
    bg = np.asarray(background)
    return (np.sum(bg >= x) + 1) / (len(bg) + 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--collected', type=Path, required=True)
    p.add_argument('--pe-dir', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True, help='stage-2 task list (joint lines)')
    p.add_argument('--marginal', type=Path, help='marginal_pairs.csv (a,b,kind)')
    p.add_argument('--twins', type=Path, nargs='*', default=[], help='twin tables (twins_s1.csv twins_x1.csv ...)')
    p.add_argument('--placed', type=Path, help='placed lensed injections')
    p.add_argument('--ranked', type=Path, help='stage-1 pairs_ranked.csv (adds the stage-1 statistics)')
    p.add_argument('--lvk-hanabi', type=Path, help='LVK GWTC-4.0 lensing release: hanabi Bayes factors / coherence ratios (O4a)')
    p.add_argument('--null-pairs', type=Path, help='lensearch.null_pairs table: the selection-matched background (replaces '
                                                    'the pair-matched twins as the null; twins are kept as a diagnostic)')
    p.add_argument('--candidates', type=Path, help='marginal_records.csv: search SNRs of the sub-threshold candidates '
                                                    '(data-quality test of events without LVK PE)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.candidates and a.candidates.exists():
        CAND_SNR.update({r.name: float(r.snr) for r in pd.read_csv(a.candidates).itertuples()})
    runs = load_runs(a.collected)
    print(f'{len(runs)} finished runs')

    pairs = [l.split()[1:3] for l in a.pairs.read_text().splitlines() if l.startswith('joint ')]
    real = real_pairs(runs, pairs, a.pe_dir, 'pe')
    if a.marginal and a.marginal.exists():
        m = pd.read_csv(a.marginal)
        real = pd.concat([real, real_pairs(runs, list(zip(m.a, m.b)), a.pe_dir, 'marginal')], ignore_index=True)
    if a.ranked and a.ranked.exists():
        rk = pd.read_csv(a.ranked)
        key = {c for c in rk.columns}
        ca, cb = ('a', 'b') if {'a', 'b'} <= key else ('event_a', 'event_b')
        stats = [c for c in rk.columns if c.startswith('ln_') or c.startswith('p_chance') or c in ('rank', 'dt_days', 'kind')]
        rk = rk[[ca, cb] + stats].rename(columns={ca: 'a', cb: 'b', **{c: f's1_{c}' for c in stats}}).drop_duplicates(['a', 'b'])
        real = real.merge(rk, on=['a', 'b'], how='left')

    if a.lvk_hanabi and a.lvk_hanabi.exists():
        h = pd.read_csv(a.lvk_hanabi)
        lv = {}
        for r in h.itertuples():
            lv[(r.event1_full_name, r.event2_full_name)] = lv[(r.event2_full_name, r.event1_full_name)] = (r.ln_coherence_ratio, r.ln_Bayes_factor)
        real['lvk_ln_c'] = [lv.get((r.a, r.b), (np.nan, np.nan))[0] for r in real.itertuples()]
        real['lvk_ln_b'] = [lv.get((r.a, r.b), (np.nan, np.nan))[1] for r in real.itertuples()]
    tw_tabs = [pd.read_csv(t) for t in a.twins if t.exists()]
    tw = pd.concat([twin_pairs(runs, t) for t in tw_tabs], ignore_index=True) if tw_tabs else pd.DataFrame()
    # the same twins analysed again with the settings of the real pairs (results/collected/bg_pe_n1000)
    rep = pd.concat([twin_pairs(runs, t, prefix='bg_pe_n1000:') for t in tw_tabs], ignore_index=True) if tw_tabs else pd.DataFrame()
    rep = rep[rep.has_joint | rep.has_a | rep.has_b] if len(rep) else rep
    inj = injection_pairs(runs, pd.read_csv(a.placed)) if a.placed and a.placed.exists() else pd.DataFrame()

    # background and false-alarm probabilities
    bg = tw.ln_c.dropna().values if len(tw) and 'ln_c' in tw else np.array([])
    nul = null_pair_rows(runs, pd.read_csv(a.null_pairs)) if a.null_pairs and a.null_pairs.exists() else pd.DataFrame()
    if len(nul):
        nul.to_csv(a.out / 'null_pairs_lnc.csv', index=False)
    bg_null = nul.ln_c.dropna().values if len(nul) and 'ln_c' in nul else np.array([])
    if len(bg_null):
        # per-pair false-alarm probability against the null pairs, and the probability that the largest of the clean
        # real pairs reaches that value by chance (trials = number of clean real pairs with ln C)
        n_trials = int((real.ln_c.notna() & ~real.dq_flag & ~real.get('sampling_flag', False).fillna(False).astype(bool)).sum())
        real['fap_pair'] = [false_alarm(x, bg_null) if np.isfinite(x) else np.nan for x in real.ln_c]
        real['fap_global'] = 1 - (1 - real.fap_pair) ** n_trials
        real['null_n'] = len(bg_null)
    if len(bg):
        if not len(bg_null):
            real['fap_global'] = [false_alarm(x, bg) if np.isfinite(x) else np.nan for x in real.get('ln_c', np.nan)]
        own = tw.dropna(subset=['ln_c']).groupby('pair').ln_c.agg(['mean', 'max', 'count'])
        real['twin_mean'] = [own['mean'].get(f'{r.a}+{r.b}', np.nan) for r in real.itertuples()]
        real['twin_max'] = [own['max'].get(f'{r.a}+{r.b}', np.nan) for r in real.itertuples()]
        real['twin_n'] = [int(own['count'].get(f'{r.a}+{r.b}', 0)) for r in real.itertuples()]
        real['ln_c_minus_twin'] = real.ln_c - real.twin_mean
    clean = real[~real.dq_flag & ~real.get('sampling_flag', False).fillna(False).astype(bool)] if len(real) else real

    summary = dict(n_runs=len(runs), n_pairs=len(real), n_pairs_with_lnc=int(real.ln_c.notna().sum()) if 'ln_c' in real else 0,
                   n_dq_flagged=int(real.dq_flag.sum()), n_sampling_flagged=int(real.get('sampling_flag', pd.Series(dtype=bool)).fillna(False).sum()),
                   n_twins_with_lnc=int(len(bg)), twin_lnc_median=float(np.median(bg)) if len(bg) else None,
                   twin_lnc_max=float(bg.max()) if len(bg) else None, twin_lnc_q90=float(np.quantile(bg, 0.9)) if len(bg) else None,
                   n_injections_with_lnc=int(inj.ln_c.notna().sum()) if len(inj) and 'ln_c' in inj else 0,
                   n_null_selected=int(len(nul)), n_null_with_lnc=int(len(bg_null)),
                   null_lnc_median=float(np.median(bg_null)) if len(bg_null) else None,
                   null_lnc_max=float(bg_null.max()) if len(bg_null) else None)
    if len(clean) and 'ln_c' in clean and clean.ln_c.notna().any():
        top = clean.loc[clean.ln_c.idxmax()]
        summary.update(top_clean_pair=f'{top.a}+{top.b}', top_clean_lnc=float(top.ln_c),
                       top_clean_fap=float(top.get('fap_global', np.nan)), top_clean_fap_pair=float(top.get('fap_pair', np.nan)),
                       clean_lnc_median=float(clean.ln_c.median()))
    ref = bg_null if len(bg_null) else bg
    if len(inj) and 'ln_c' in inj and inj.ln_c.notna().any() and len(ref):
        bg = ref
        thr = float(bg.max())
        summary.update(threshold_max_background=thr,
                       efficiency_at_max_background=float(np.mean(inj.ln_c.dropna() >= thr)),
                       efficiency_at_bg_q90=float(np.mean(inj.ln_c.dropna() >= np.quantile(bg, 0.9))))
    real.sort_values('ln_c', ascending=False).to_csv(a.out / 'real_pairs.csv', index=False)
    if len(tw):
        tw.to_csv(a.out / 'twins.csv', index=False)
        if 'nlive' in tw:
            for nl, g in tw.dropna(subset=['ln_c']).groupby('nlive'):
                summary[f'twins_nlive_{int(nl)}'] = dict(n=int(len(g)), median=float(g.ln_c.median()), max=float(g.ln_c.max()))
    if len(rep):
        rep.to_csv(a.out / 'twins_n1000_repeats.csv', index=False)
        both = rep.merge(tw[['system', 'ln_c']], on='system', suffixes=('_n1000', '_n500')) if 'ln_c' in rep else pd.DataFrame()
        if len(both):
            summary['repeat_lnc_diff_n1000_minus_n500'] = [round(float(x), 2) for x in (both.ln_c_n1000 - both.ln_c_n500).dropna()]
    if len(inj):
        inj.to_csv(a.out / 'injections.csv', index=False)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    cols = [c for c in ['a', 'b', 'kind', 'ln_c', 'ln_c_err', 'fap_global', 'twin_mean', 'twin_n', 'dq_flag', 'sampling_flag',
                        'joint_excess', 'dq_excess_a', 'dq_excess_b', 'ln_mu_rel_med', 'dt_days', 'lvk_ln_c'] if c in real]
    with pd.option_context('display.width', 250, 'display.max_rows', 200):
        print(real.sort_values('ln_c', ascending=False)[cols].round(2).to_string())
        if len(tw):
            print('\ntwins:')
            print(tw[[c for c in ['system', 'pair', 'ln_c', 'ln_c_err', 'joint_excess', 'ln_mu_rel_med'] if c in tw]].round(2).to_string())
        if len(inj) and 'ln_c' in inj:
            print('\ninjections with ln C:')
            print(inj.dropna(subset=['ln_c'])[['system', 'snr_a', 'snr_b', 'delta_t_days', 'mu_a', 'mu_b', 'ln_c', 'ln_c_err', 'ln_mu_rel_med']].round(2).to_string())
    print(json.dumps(summary, indent=1, default=float))


if __name__ == '__main__':
    main()
