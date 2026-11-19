"""Numbers, table and figures of Paper II (H0 from lensed GWs with identified hosts).

Inputs (scratchpad): per-event h_rel samples of the two-stage runs (pdet/scratch_pdet_*.npz: normalised source prior;
h0cmp/h_old.npz: flat source prior; h0cmp/h_ki.npz: drawn kappa_ext; h0cmp/h_nod.npz: delay only), selection
fractions alpha(h) (alpha/alpha_*.npz), per-event summaries (p2/all.jsonl), and the magnification summaries of the
full (strain + EM) and two-stage analyses (sn/full_lcdm, sn/posthoc, sn/out2).
usage: p2_numbers.py OUT_FIG_DIR
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{S}/h0cmp')
from kde1d import logpost, stats, GRID as grid  # noqa: E402

OUT = sys.argv[1]
FAMS = ['SIS', 'SIE', 'SIE_shear', 'EPL_shear']
FLAB = {'SIS': 'SIS', 'SIE': 'SIE', 'SIE_shear': 'SIE+shear', 'EPL_shear': 'EPL+shear'}
KS = {'ET': '_ks', 'O5': ''}
fam_of = lambda b: b.split('_', 1)[1].rsplit('-', 2)[0]  # noqa: E731
N = {}

# ---------------------------------------------------------------- samples
H = {}
for f in glob.glob(f'{S}/pdet/scratch_pdet_*.npz'):
    H.update({k[:-2]: v for k, v in np.load(f).items() if k.endswith('|h')})
for f in ('h_ki.npz', 'h_nod.npz'):
    H.update({k[:-2]: v for k, v in np.load(f'{S}/h0cmp/{f}').items() if k.endswith('|h')})
H.update(dict(np.load(f'{S}/h0cmp/h_old.npz')))  # flat source prior: keys dir|event
# O5 EPL_shear-7777-0017024 was rerun with the image labels of its GW injection (see p2/fix/README.md)
H.update({k[:-2]: v for k, v in np.load(f'{S}/p2/fix/h_fix.npz').items() if k.endswith('|h')})
KEXT = {k[:-2]: float(v) for k, v in np.load(f'{S}/h0cmp/h_ki.npz').items() if k.endswith('|k')}
KEXT.update({k[:-2]: float(v) for k, v in np.load(f'{S}/p2/fix/h_fix.npz').items() if k.endswith('|k')})
A = {}
for f in glob.glob(f'{S}/alpha/alpha_*.npz'):
    d = np.load(f)
    for sc in ('ET', 'O5'):
        a = d[f'sel_{sc}'].mean(0)
        A[(os.path.basename(f)[6:-4], sc)] = (d['h'], np.log(a))
LA = {k: np.interp(grid, *v) for k, v in A.items()}
SUM = {}
for line in open(f'{S}/p2/all.jsonl'):
    r = json.loads(line)
    SUM[f"{r['dir']}|{r['scenario']}_{r['event_id']}"] = r
for line in open(f'{S}/p2/fix/all_fix.jsonl'):  # corrected runs of O5 EPL_shear-7777-0017024
    r = json.loads(line)
    SUM[f"{r['dir']}|{r['scenario']}_{r['event_id']}"] = r
_L = {}


def L(key):
    if key not in _L:
        _L[key] = logpost(H[key])
    return _L[key]


def keys(sc, tag, model):
    """Events of scenario sc analysed with lens model `model` ('own' = the simulated family)."""
    out = []
    for k in H:
        d, b = k.split('|')
        if not b.startswith(sc + '_'):
            continue
        m = fam_of(b) if model == 'own' else model
        if d == f'lcdm_k0.025_{m}{tag}{KS[sc]}':
            out.append(k)
    return sorted(out)


def joint(ks, sel=True):
    tot = sum(L(k) for k in ks) - (sum(LA[(fam_of(k.split('|')[1]), k.split('|')[1][:2])] for k in ks) if sel else 0)
    m, lo, hi = stats(tot)
    return m, lo, hi, tot


def per_event(ks, sel=True):
    """Per-event 90% intervals (posterior with flat prior on h, divided by alpha if sel) and standard deviations."""
    cov, sd = [], []
    for k in ks:
        sc = k.split('|')[1][:2]
        lp = L(k) - (LA[(fam_of(k.split('|')[1]), sc)] if sel else 0)
        p = np.exp(lp - lp.max()); p /= p.sum()
        c = np.cumsum(p)
        q05, q95 = grid[np.searchsorted(c, 0.05)], grid[np.searchsorted(c, 0.95)]
        cov.append(q05 <= 1 <= q95)
        mu = p @ grid
        sd.append(np.sqrt(p @ (grid - mu) ** 2))
    return np.array(cov), np.array(sd)


def fmt(m, lo, hi):
    return f'{m:.3f}^{{+{hi - m:.3f}}}_{{-{m - lo:.3f}}}'


# ---------------------------------------------------------------- selection fraction
for sc in ('ET', 'O5'):
    for fam in FAMS:
        h, la = A[(fam, sc)]
        i = np.argmin(abs(h - 1))
        sl = np.polyfit(np.log(h[i - 10:i + 11]), la[i - 10:i + 11], 1)[0]
        N[f'alpha1_{sc}_{fam}'] = float(np.exp(la[i]))
        N[f'dlnalpha_{sc}_{fam}'] = float(sl)
    print(sc, 'alpha(1)', ' '.join(f"{f}={N[f'alpha1_{sc}_{f}']:.3f}" for f in FAMS),
          '| dln alpha/dln h', ' '.join(f"{N[f'dlnalpha_{sc}_{f}']:.2f}" for f in FAMS))

# ---------------------------------------------------------------- joint H0 table
rows = []
for lab, tag in (('baseline', '_h0_sn'), ('kext', '_h0_ki_sn')):
    for model in ('own', 'EPL_shear'):
        r = dict(case=lab, model=model)
        for sc in ('ET', 'O5'):
            ks = keys(sc, tag, model)
            m, lo, hi, _ = joint(ks)
            cov, sd = per_event(ks)
            cov0, _ = per_event(ks, sel=False)
            r[sc] = dict(n=len(ks), m=m, lo=lo, hi=hi, sd=float(np.median(sd)), cov=float(cov.mean()),
                         cov_nosel=float(cov0.mean()))
            if lab == 'kext':
                r[sc]['kext_sd'] = float(np.std([KEXT[k] for k in ks]))
        rows.append(r)
        print(f"{lab:8s} {model:9s}", '  '.join(
            f"{sc}: n={r[sc]['n']} {fmt(r[sc]['m'], r[sc]['lo'], r[sc]['hi'])} ({(r[sc]['m'] - 1) / ((r[sc]['hi'] - r[sc]['lo']) / 2):+.1f}sd)"
            f" sd_ev={r[sc]['sd']:.3f} cov={r[sc]['cov']:.2f} (no sel {r[sc]['cov_nosel']:.2f})" for sc in ('ET', 'O5')))
N['table'] = rows

# per-system coverage versus the magnification of the brighter GW image (simulated family, baseline, with selection)
for sc in ('ET', 'O5'):
    ks = keys(sc, '_h0_sn', 'own')
    cov, _ = per_event(ks)
    mu = np.array([max(SUM[k]['true_mu_a'], SUM[k]['true_mu_b']) for k in ks])
    for lab_, m in (('lt5', mu < 5), ('ge5', mu >= 5)):
        N[f'cov_mu_{sc}_{lab_}'] = (int(m.sum()), float(cov[m].mean()) if m.any() else float('nan'))
    print(sc, 'coverage |mu|<5: n=%d %.2f; |mu|>=5: n=%d %.2f' % (*N[f'cov_mu_{sc}_lt5'], *N[f'cov_mu_{sc}_ge5']))

# flat source prior (EPL+shear and own model) and the stages of correction
stages = {}
for sc in ('ET', 'O5'):
    flat = sorted(k for k in H if k.split('|')[0] == f'lcdm_k0.025_EPL_shear_h0{KS[sc]}' and k.split('|')[1].startswith(sc))
    norm = keys(sc, '_h0_sn', 'EPL_shear')
    st = {'flat': joint(flat, sel=False), 'flat+sel': joint(flat), 'norm': joint(norm, sel=False), 'norm+sel': joint(norm)}
    stages[sc] = st
    N[f'stages_{sc}'] = {k: v[:3] for k, v in st.items()}
    print(sc, 'EPL+shear stages', '  '.join(f"{k}: {fmt(*v[:3])}" for k, v in st.items()), f'(n flat {len(flat)}, norm {len(norm)})')
    flat_own = sorted(k for k in H if k.split('|')[1].startswith(sc) and
                      k.split('|')[0] == f"lcdm_k0.025_{fam_of(k.split('|')[1])}_h0{KS[sc]}")
    N[f'flat_own_{sc}'] = joint(flat_own)[:3]
    print(sc, 'own model flat source prior + sel', fmt(*joint(flat_own)[:3]), 'no sel', fmt(*joint(flat_own, sel=False)[:3]), len(flat_own))

# per family
for sc in ('ET', 'O5'):
    out = []
    for fam in FAMS:
        for model in ('own', 'EPL_shear'):
            ks = [k for k in keys(sc, '_h0_sn', model) if fam_of(k.split('|')[1]) == fam]
            m, lo, hi, _ = joint(ks)
            N[f'fam_{sc}_{fam}_{model}'] = (m, lo, hi)
            out.append(f"{fam}/{model[:3]} {100 * (m - 1):+.1f}±{100 * (hi - lo) / 2:.1f}%")
    print(sc, 'per family:', ' | '.join(out))

# ---------------------------------------------------------------- delay only
ratio = {}
for sc in ('ET', 'O5'):
    base = {k.split('|')[1]: k for k in keys(sc, '_h0_sn', 'own')}
    nod = {k.split('|')[1]: k for k in keys(sc, '_h0_nod_sn', 'own')}
    common = sorted(set(base) & set(nod))
    sb = np.array([np.std(H[base[e]]) for e in common])
    sn = np.array([np.std(H[nod[e]]) for e in common])
    ratio[sc] = (sb, sn, [fam_of(e) for e in common])
    r = sn / sb
    N[f'nod_ratio_{sc}'] = dict(n=len(common), median=float(np.median(r)), q10=float(np.quantile(r, 0.1)),
                                q90=float(np.quantile(r, 0.9)))
    shift = np.array([(np.median(H[nod[e]]) - np.median(H[base[e]])) / np.std(H[base[e]]) for e in common])
    N[f'nod_ratio_{sc}']['ivar_joint'] = float((np.sum(sn ** -2) / np.sum(sb ** -2)) ** -0.5)
    N[f'nod_ratio_{sc}']['median_shift'] = float(np.median(abs(shift)))
    print(sc, 'inverse-variance joint width ratio', N[f'nod_ratio_{sc}']['ivar_joint'])
    print(sc, f'delay-only/full sd ratio n={len(common)} median {np.median(r):.3f} 10-90% {np.quantile(r, .1):.2f}-{np.quantile(r, .9):.2f};'
          f' median |shift| {np.median(abs(shift)):.2f} sd; joint sd ratio',
          f"{(lambda a, b: (a[2] - a[1]) / (b[2] - b[1]))(joint([nod[e] for e in common], sel=False), joint([base[e] for e in common], sel=False)):.3f}")

# ---------------------------------------------------------------- slope
slope = {}
for sc in ('ET', 'O5'):
    pts = []
    for k in keys(sc, '_h0_sn', 'EPL_shear'):
        s = SUM.get(k)
        if s is None:
            continue
        pts.append((s['gamma_median'] - s['true_gamma'], np.log(np.median(H[k])), s['gamma_sd'], np.std(np.log(H[k])), s['family'],
                    s['true_gamma']))
    slope[sc] = pts
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts]); g = np.array([p[5] for p in pts])
    pred = x / (g - 1)
    N[f'slope_{sc}'] = dict(n=len(pts), corr=float(np.corrcoef(x, y)[0, 1]), fit=float(np.polyfit(x, y, 1)[0]),
                            resid_sd=float(np.std(y - pred)), y_sd=float(np.std(y)))
    print(sc, f"slope test n={len(pts)} corr(dgamma, ln h)={N[f'slope_{sc}']['corr']:.2f} fitted slope {N[f'slope_{sc}']['fit']:.2f}"
          f" sd(ln h) {np.std(y):.3f} sd(ln h - dgamma/(gamma-1)) {np.std(y - pred):.3f}")
    for fam in FAMS:
        sel = np.array([p[4] == fam for p in pts])
        print(f'   {fam}: mean dgamma {x[sel].mean():+.3f}±{x[sel].std() / np.sqrt(sel.sum()):.3f}  mean ln h {y[sel].mean():+.3f}')

# ---------------------------------------------------------------- strain-level check (magnification, LCDM at fixed H0)
UNMATCHED = {('ET', 'SIE-7777-0023296'), ('ET', 'EPL_shear-7777-0028423'), ('ET', 'SIS-7777-0007321'),
             ('ET', 'SIE-7777-0020513'), ('ET', 'EPL_shear-7777-0005239'), ('ET', 'SIE_shear-7777-0017305'),
             ('O5', 'EPL_shear-7777-0017024')}  # its full runs inject the two images with the other labels (p2/fix/README.md)
full = {(s['scenario'], s['event_id']): s for s in (json.load(open(f)) for f in glob.glob(f'{S}/sn/full_lcdm/*_summary.json'))}
two = {(s['scenario'], s['event_id']): s for s in (json.load(open(f)) for f in glob.glob(f'{S}/sn/posthoc/lcdm_k0.025_EPL_shear_sn/*_summary.json'))}
two.update({(s['scenario'], s['event_id']): s for s in (json.load(open(f)) for f in glob.glob(f'{S}/p2/fix/posthoc/gwem_posthoc_sn/lcdm_k0.025_EPL_shear_sn/*_summary.json'))})
for sc in ('ET', 'O5', 'all'):
    ev = [e for e in full if e in two and e not in UNMATCHED and (sc == 'all' or e[0] == sc)]
    w = lambda s: (s['ln_mu_a_q95'] - s['ln_mu_a_q05'])  # noqa: E731
    sh = np.array([(two[e]['ln_mu_a_median'] - full[e]['ln_mu_a_median']) / (w(full[e]) / 3.29) for e in ev])
    wr = np.array([w(two[e]) / w(full[e]) for e in ev])
    cf = np.mean([full[e]['covered90'] for e in ev]); c2 = np.mean([two[e]['covered90'] for e in ev])
    ef = np.median([full[e]['rel_err'] for e in ev]); e2 = np.median([two[e]['rel_err'] for e in ev])
    N[f'lcdm_full_{sc}'] = dict(n=len(ev), shift=float(np.median(abs(sh))), gt1=float(np.mean(abs(sh) > 1)), wr=float(np.median(wr)),
                                cov_full=float(cf), cov_2s=float(c2), err_full=float(ef), err_2s=float(e2))
    print(f'LCDM full vs two-stage {sc}: n={len(ev)} median |shift| {np.median(abs(sh)):.2f} sd, >1sd {np.mean(abs(sh) > 1):.3f},'
          f' width ratio {np.median(wr):.2f}, cov full {cf:.2f} 2s {c2:.2f}, err full {ef:.3f} 2s {e2:.3f}')
print(open(f'{S}/sn/out2/gwem_full_vs_twostage.csv').read())

# ---------------------------------------------------------------- figures
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 1.1, 'mathtext.fontset': 'cm'})
FC = {'SIS': '#1b9e77', 'SIE': '#d95f02', 'SIE_shear': '#7570b3', 'EPL_shear': '#e7298a'}

# Fig. 1: selection fraction and the corrections
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2), constrained_layout=True)
for sc, ls in (('ET', '-'), ('O5', '--')):
    for fam in FAMS:
        h, la = A[(fam, sc)]
        ax[0].plot(h, np.exp(la - la[np.argmin(abs(h - 1))]), ls, color=FC[fam], lw=0.9,
                   label=FLAB[fam] if sc == 'ET' else None)
ax[0].set_yscale('log'); ax[0].set_xlim(0.7, 1.3); ax[0].set_ylim(0.3, 3.5)
ax[0].set_yticks([0.3, 0.5, 1, 2, 3]); ax[0].set_yticklabels(['0.3', '0.5', '1', '2', '3']); ax[0].minorticks_off()
ax[0].set_xlabel(r'$h=H_0/H_0^{\rm true}$'); ax[0].set_ylabel(r'$\alpha(h)/\alpha(1)$')
ax[0].plot([], [], 'k-', lw=0.9, label='ET'); ax[0].plot([], [], 'k--', lw=0.9, label='O5')
ax[0].legend(frameon=False, ncol=1, loc='upper left', handlelength=1.6)
ax[0].text(0.97, 0.05, '(a)', transform=ax[0].transAxes, ha='right')
styles = {'flat': ('#999999', ':', 'unnormalized'), 'norm': ('#377eb8', '--', 'normalized'),
          'norm+sel': ('k', '-', 'normalized + selection')}
for j, sc in enumerate(('ET', 'O5')):
    a = ax[j + 1]
    for k, (c, ls, lab) in styles.items():
        tot = stages[sc][k][3]
        p = np.exp(tot - tot.max())
        a.plot(grid, p / np.trapezoid(p, grid), ls, color=c, label=lab)
    a.axvline(1, color='0.6', lw=0.6)
    a.set_xlim(0.94, 1.16); a.set_xlabel(r'$h=H_0/H_0^{\rm true}$'); a.set_yticks([])
    a.set_ylabel('combined posterior'); a.set_ylim(0, a.get_ylim()[1] * 2.1)
    a.text(0.03, 0.92, f"({'bc'[j]}) {sc}, EPL+shear model", transform=a.transAxes)
ax[1].legend(frameon=False, loc='upper right', bbox_to_anchor=(1.0, 0.90), handlelength=2.0, title='source prior', title_fontsize=7)
fig.savefig(f'{OUT}/fig_corrections.pdf'); plt.close(fig)

# Fig. 2: per-event width with and without the GW apparent distance
fig, ax = plt.subplots(figsize=(3.4, 2.7), constrained_layout=True)
for sc, mk in (('ET', 'o'), ('O5', 's')):
    sb, sn, fm = ratio[sc]
    for fam in FAMS:
        sel = np.array([f == fam for f in fm])
        ax.scatter(sb[sel], sn[sel], s=12, marker=mk, facecolor=FC[fam] if sc == 'ET' else 'none', edgecolor=FC[fam], lw=0.7,
                   label=FLAB[fam] if sc == 'ET' else None)
lim = [0.02, 0.4]
ax.plot(lim, lim, 'k-', lw=0.6)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lim); ax.set_ylim(lim)
tk = [0.03, 0.05, 0.1, 0.2, 0.3]
for f_ in (ax.set_xticks, ax.set_yticks):
    f_(tk)
ax.set_xticklabels([f'{t:g}' for t in tk]); ax.set_yticklabels([f'{t:g}' for t in tk]); ax.minorticks_off()
ax.set_xlabel(r'$\sigma_h$, delay + GW apparent distance'); ax.set_ylabel(r'$\sigma_h$, delay only')
ax.scatter([], [], marker='o', facecolor='k', edgecolor='k', s=12, label='ET (filled)')
ax.scatter([], [], marker='s', facecolor='none', edgecolor='k', s=12, label='O5 (open)')
ax.legend(frameon=False, loc='upper left', ncol=1)
fig.savefig(f'{OUT}/fig_delay.pdf'); plt.close(fig)

# Fig. 3: per-family offsets of the combined h, and the offset expected from the mean slope error
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4), constrained_layout=True, sharey=True)
for j, sc in enumerate(('ET', 'O5')):
    for i, fam in enumerate(FAMS):
        for model, dx, mk, fc in (('own', -0.18, 'o', 'k'), ('EPL_shear', 0.0, 's', 'none')):
            if fam == 'EPL_shear' and model == 'EPL_shear':
                continue
            m, lo, hi = N[f'fam_{sc}_{fam}_{model}']
            ax[j].errorbar(i + dx, 100 * (m - 1), yerr=[[100 * (m - lo)], [100 * (hi - m)]], fmt=mk, ms=4, mfc=fc, mec='k',
                           ecolor='k', elinewidth=0.7, capsize=0,
                           label=('simulated family' if model == 'own' else 'EPL+shear model') if i == 0 else None)
        p = [q for q in slope[sc] if q[4] == fam]
        pr = np.array([q[0] / (q[5] - 1) for q in p])
        N[f'pred_{sc}_{fam}'] = (float(pr.mean()), float(pr.std() / np.sqrt(len(pr))))
        ax[j].errorbar(i + 0.18, 100 * pr.mean(), yerr=100 * pr.std() / np.sqrt(len(pr)), fmt='x', ms=5, color='#d95f02',
                       elinewidth=0.7, capsize=0, label=r"$\langle\delta\gamma'/(\gamma'-1)\rangle$, EPL+shear model" if i == 0 else None)
    ax[j].axhline(0, color='0.6', lw=0.6)
    ax[j].set_xticks(range(4)); ax[j].set_xticklabels([FLAB[f] for f in FAMS])
    ax[j].set_xlim(-0.5, 3.5); ax[j].set_ylim(-9, 12)
    ax[j].text(0.03, 0.92, f"({'ab'[j]}) {sc}", transform=ax[j].transAxes)
ax[0].set_ylabel(r'$h-1$ of the family (\%)')
ax[1].legend(frameon=False, loc='upper right')
fig.savefig(f'{OUT}/fig_slope.pdf'); plt.close(fig)
for sc in ('ET', 'O5'):
    print(sc, 'predicted from slope:', ' '.join(f"{f} {100 * N[f'pred_{sc}_{f}'][0]:+.1f}+-{100 * N[f'pred_{sc}_{f}'][1]:.1f}%" for f in FAMS))

# ---------------------------------------------------------------- table
lab = {('baseline', 'own'): r'Simulated family, $\kappa_{\rm ext}=0$', ('baseline', 'EPL_shear'): r'EPL+shear, $\kappa_{\rm ext}=0$',
       ('kext', 'own'): r'Simulated family, $\kappa_{\rm ext}$ drawn', ('kext', 'EPL_shear'): r'EPL+shear, $\kappa_{\rm ext}$ drawn'}
t = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccccc@{}}', r'\toprule',
     r' & \multicolumn{3}{c}{ET (60 systems)} & \multicolumn{3}{c}{O5 (59 systems)}\\',
     r'\cmidrule(lr){2-4}\cmidrule(l){5-7}',
     r'Lens model, data & $h$ & $\sigma_h$ & cov. & $h$ & $\sigma_h$ & cov.\\', r'\midrule']
for r in rows:
    c = []
    for sc in ('ET', 'O5'):
        q = r[sc]
        c += [f"${fmt(q['m'], q['lo'], q['hi'])}$", f"{q['sd']:.3f}", f"{q['cov']:.2f}"]
    t.append(lab[(r['case'], r['model'])] + ' & ' + ' & '.join(c) + r'\\')
t += [r'\bottomrule', r'\end{tabular*}']
open(f'{OUT}/table_h0.tex', 'w').write('\n'.join(t) + '\n')
json.dump(N, open(f'{OUT}/numbers.json', 'w'), indent=1, default=float)
print('written', OUT)
