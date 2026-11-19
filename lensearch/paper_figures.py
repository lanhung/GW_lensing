"""Figures and tables of the lensed-pair search paper from the stage-1 ranking and the stage-2 analysis.

    python -m lensearch.paper_figures --rank results/rank --analysis results/analysis --out paper_figures

fig_stage1.pdf  distributions of ln B_cat and ln B_pop over all pairs with their chance backgrounds
fig_dq.pdf      excess of our single-image maximum log-likelihood over the LVK maximum-likelihood sample, per event
fig_lnc.pdf     coherence ratios of the real pairs against the selection-matched unrelated pairs and the lensed injections
table_top.tex   the highest clean pairs
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DQ_EXCESS = dict(O4=10.0, O3=20.0)


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'figure.dpi': 150, 'pdf.fonttype': 42})
    return plt


def fig_stage1(rank_dir, analysis_dir, out, plt):
    r = pd.read_csv(rank_dir / 'pairs_ranked.csv')
    bg_cat = np.load(rank_dir / 'background_ln_bcat.npy', allow_pickle=True)
    bg_pop = np.load(rank_dir / 'background_ln_bpop_shuffle.npy', allow_pickle=True)
    real = pd.read_csv(analysis_dir / 'real_pairs.csv') if (analysis_dir / 'real_pairs.csv').exists() else None
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for a, col, bg, name in ((ax[0], 'ln_bcat', bg_cat, r'$\ln B_{\rm cat}$ (lensed by a catalogued lens)'),
                             (ax[1], 'ln_bpop', bg_pop, r'$\ln B_{\rm pop}$ (lensed by any galaxy)')):
        x = r[col].replace([-np.inf, np.inf], np.nan).dropna()
        a.hist(x, bins=np.linspace(-10, 15, 101), color='0.3', log=True, label='all pairs')
        bg = np.asarray(bg, float)
        if bg.ndim == 2:                       # value of every pair for every rotation -> largest per rotation
            bg = np.where(np.isfinite(bg), bg, -np.inf).max(axis=1)
        bg = bg[np.isfinite(bg)]
        if len(bg):
            a.hist(bg, bins=np.linspace(-10, 15, 51), color='C1', alpha=0.6, label='largest value per background trial')
        a.axvline(x.max(), color='k', ls='--', lw=1, label='observed largest value')
        if real is not None and f's1_{col}' in real:
            sel = real[f's1_{col}'].dropna()
            a.plot(sel, np.full(len(sel), 1.5), '|', color='C0', ms=8, label='stage-2 pairs')
        a.set_xlabel(name)
        a.set_ylabel('pairs')
        a.set_xlim(-10, 15)
    ax[0].legend(loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'fig_stage1.pdf')
    plt.close(fig)


def fig_dq(analysis_dir, out, plt):
    real = pd.read_csv(analysis_dir / 'real_pairs.csv')
    rows = {}
    for k in 'ab':
        for r in real.itertuples():
            ev = getattr(r, k)
            exc = getattr(r, f'dq_excess_{k}')
            if np.isfinite(exc):
                rows[ev] = dict(excess=exc, bayeswave=bool(getattr(r, f'bayeswave_{k}')), o3=ev < 'GW2021',
                                cand=not np.isfinite(getattr(r, f'lvk_maxl_{k}')))
    d = pd.DataFrame(rows).T.sort_values('excess')
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    x = np.arange(len(d))
    ax.bar(x, d.excess.astype(float), color=['C3' if b else ('C0' if c else '0.4') for b, c in zip(d.bayeswave, d.cand)])
    ax.axhline(DQ_EXCESS['O4'], color='k', ls='--', lw=0.8)
    ax.axhline(DQ_EXCESS['O3'], color='k', ls=':', lw=0.8)
    ax.set_ylabel(r'max $\ln\mathcal{L}$ (this work) $-$ reference')
    ax.set_xlabel('events, ordered by excess')
    ax.set_yscale('symlog', linthresh=10)
    flagged = d[d.excess > np.where(d.o3, DQ_EXCESS['O3'], DQ_EXCESS['O4'])]
    lines = [f'{ev[2:]}  {r.excess:+.0f}' for ev, r in flagged.sort_values('excess', ascending=False).iterrows()]
    ax.text(0.03, 0.97, 'flagged:\n' + '\n'.join(lines), transform=ax.transAxes, fontsize=5.5, family='monospace',
            va='top', ha='left')
    ax.plot([], [], color='C3', lw=6, label='LVK used glitch-subtracted strain')
    ax.plot([], [], color='0.4', lw=6, label='other events (ref.: LVK max. likelihood)')
    ax.plot([], [], color='C0', lw=6, label=r'sub-threshold (ref.: search SNR$^2/2$)')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=5.5, ncol=1)
    fig.savefig(out / 'fig_dq.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_lnc(analysis_dir, out, plt):
    real = pd.read_csv(analysis_dir / 'real_pairs.csv').dropna(subset=['ln_c'])
    def load(name):
        f = analysis_dir / name
        d = pd.read_csv(f) if f.exists() else pd.DataFrame()
        return d.dropna(subset=['ln_c']) if 'ln_c' in d else pd.DataFrame(columns=['ln_c', 'pair'])
    tw, inj = load('null_pairs_lnc.csv'), load('injections.csv')     # background: selection-matched unrelated pairs
    clean = real[~real.dq_flag.astype(bool) & ~real.sampling_flag.fillna(False).astype(bool)].sort_values('ln_c', ascending=False)
    flagged = real[real.dq_flag.astype(bool) | real.sampling_flag.fillna(False).astype(bool)]
    fig, (ax, ah) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw=dict(width_ratios=[3, 1], wspace=0.05), sharey=True)
    x = np.arange(len(clean))
    if len(tw):
        lo, med, hi = np.quantile(tw.ln_c, [0.05, 0.5, 0.95])
        ax.axhspan(lo, hi, color='0.85', label='unrelated selected pairs, 5–95%')
        ax.axhline(med, color='0.5', lw=1, label='background, median')
        ax.axhline(tw.ln_c.max(), color='0.5', lw=0.8, ls='--', label='background, largest')
    pe = clean[clean.kind == 'pe']
    mg = clean[clean.kind != 'pe']
    ax.errorbar(x[clean.kind.values == 'pe'], pe.ln_c, pe.ln_c_err, fmt='o', color='C0', ms=4, label='real pairs (LVK events)')
    ax.errorbar(x[clean.kind.values != 'pe'], mg.ln_c, mg.ln_c_err, fmt='o', mfc='none', color='C0', ms=4, label='real pairs (sub-threshold candidates)')
    ymax_plot, ymin_plot = 22.0, -35.0
    if len(flagged):
        fl = flagged.sort_values('ln_c', ascending=False)
        xf = len(clean) + 1 + np.arange(len(fl))
        ax.plot(xf, np.clip(fl.ln_c, ymin_plot + 2, ymax_plot - 2), 'x', color='C3', ms=5, label='excluded (data quality)')
        n_clip = 0
        for xi, v in zip(xf, fl.ln_c):
            if v > ymax_plot - 2 or v < ymin_plot + 2:
                y = ymax_plot - 2 if v > 0 else ymin_plot + 2
                dy = (-8 - 6 * (n_clip % 2)) if v > 0 else 7     # stagger labels of neighbouring clipped points
                ax.annotate(f'{v:+.0f}', xy=(xi, y), xytext=(0, dy), textcoords='offset points', fontsize=5,
                            ha='center', va='center', color='C3')
                n_clip += 1
        ax.axvline(len(clean) + 0.5, color='0.8', lw=0.8)
    ax.set_xlabel('real pairs, ordered by $\\ln\\mathcal{C}$')
    ax.set_ylabel(r'$\ln\mathcal{C}=\ln Z_{\rm joint}-\ln Z_A-\ln Z_B$')
    ax.legend(frameon=False, loc='lower left', ncol=1, fontsize=6)
    ax.set_xticks([])
    bins = np.linspace(ymin_plot, ymax_plot, 58)
    clip = lambda v: np.clip(v, ymin_plot + 0.1, ymax_plot - 0.1)       # values beyond the axes go to the edge bins
    if len(tw):
        ah.hist(clip(tw.ln_c), bins=bins, orientation='horizontal', color='0.7', label='unrelated pairs')
    if len(inj):
        ah.hist(clip(inj.ln_c), bins=bins, orientation='horizontal', color='C2', alpha=0.6, label='lensed injections')
    ah.hist(clip(clean.ln_c), bins=bins, orientation='horizontal', histtype='step', color='C0', label='real pairs (clean)')
    ah.set_xlabel('pairs')
    ah.legend(frameon=False, fontsize=6)
    ax.set_ylim(ymin_plot, ymax_plot)
    fig.savefig(out / 'fig_lnc.pdf', bbox_inches='tight')
    plt.close(fig)


def table_top(analysis_dir, out, n=10):
    real = pd.read_csv(analysis_dir / 'real_pairs.csv').dropna(subset=['ln_c'])
    clean = real[~real.dq_flag.astype(bool) & ~real.sampling_flag.fillna(False).astype(bool)].sort_values('ln_c', ascending=False).head(n)
    lines = [r'\begin{tabular}{llrrrrrr}', r'\toprule',
             r'Event $A$ & Event $B$ & $\Delta t$ [d] & $\ln B_{\rm cat}$ & $\ln B_{\rm pop}$ & $\ln\mathcal C$ & $p$ & FAP \\', r'\midrule']
    for r in clean.itertuples():
        f = lambda v, d=1: '--' if not np.isfinite(v) else f'{v:.{d}f}'
        tw = f(getattr(r, 'fap_pair', np.nan), 2)
        ea, eb = r.a.replace('_', r'\_'), r.b.replace('_', r'\_')
        dt = r.dt_days if np.isfinite(r.dt_days) else getattr(r, 's1_dt_days', np.nan)
        lines.append(f'{ea} & {eb} & {f(dt)} & {f(getattr(r, "s1_ln_bcat", np.nan))} & {f(getattr(r, "s1_ln_bpop", np.nan))} & '
                     f'${r.ln_c:.1f}\\pm{r.ln_c_err:.1f}$ & {tw} & {f(getattr(r, "fap_global", np.nan), 2)} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (out / 'table_top.tex').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rank', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plt = setup()
    fig_stage1(a.rank, a.analysis, a.out, plt)
    fig_dq(a.analysis, a.out, plt)
    fig_lnc(a.analysis, a.out, plt)
    table_top(a.analysis, a.out)
    print('written', sorted(f.name for f in a.out.iterdir()))


if __name__ == '__main__':
    main()
