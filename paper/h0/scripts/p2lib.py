"""Shared loading of the Paper II inputs: per-event h samples, selection fractions, per-event summaries.
Same conventions as p2_numbers.py (keys 'dir|SC_event')."""
import glob, json, os, sys
import numpy as np

S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{S}/h0cmp')
from kde1d import logpost, stats, GRID as grid  # noqa: E402,F401

FAMS = ['SIS', 'SIE', 'SIE_shear', 'EPL_shear']
FLAB = {'SIS': 'SIS', 'SIE': 'SIE', 'SIE_shear': 'SIE+shear', 'EPL_shear': 'EPL+shear'}
FC = {'SIS': '#1b9e77', 'SIE': '#d95f02', 'SIE_shear': '#7570b3', 'EPL_shear': '#e7298a'}
KS = {'ET': '_ks', 'O5': ''}
fam_of = lambda b: b.split('_', 1)[1].rsplit('-', 2)[0]  # noqa: E731

H = {}
for f in glob.glob(f'{S}/pdet/scratch_pdet_*.npz'):
    H.update({k[:-2]: v for k, v in np.load(f).items() if k.endswith('|h')})
for f in ('h_ki.npz', 'h_nod.npz'):
    H.update({k[:-2]: v for k, v in np.load(f'{S}/h0cmp/{f}').items() if k.endswith('|h')})
H.update(dict(np.load(f'{S}/h0cmp/h_old.npz')))
# O5 EPL_shear-7777-0017024 was rerun with the image labels of its GW injection (see p2/fix/README.md)
H.update({k[:-2]: v for k, v in np.load(f'{S}/p2/fix/h_fix.npz').items() if k.endswith('|h')})
A = {}
for f in glob.glob(f'{S}/alpha/alpha_*.npz'):
    d = np.load(f)
    for sc in ('ET', 'O5'):
        A[(os.path.basename(f)[6:-4], sc)] = (d['h'], np.log(d[f'sel_{sc}'].mean(0)))
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


def lalpha(key):
    b = key.split('|')[1]
    return LA[(fam_of(b), b[:2])]


def keys(sc, tag, model):
    out = []
    for k in H:
        d, b = k.split('|')
        if not b.startswith(sc + '_'):
            continue
        m = fam_of(b) if model == 'own' else model
        if d == f'lcdm_k0.025_{m}{tag}{KS[sc]}':
            out.append(k)
    return sorted(out)


def post(key, sel=True):
    """Normalised per-event posterior of h on the grid (flat prior), divided by alpha(h) if sel."""
    lp = L(key) - (lalpha(key) if sel else 0)
    p = np.exp(lp - lp.max())
    return p / np.trapezoid(p, grid)


def quantiles(p, qs=(0.05, 0.5, 0.95)):
    c = np.cumsum(p); c /= c[-1]
    return [grid[np.searchsorted(c, q)] for q in qs]


def joint(ks, sel=True):
    tot = sum(L(k) for k in ks) - (sum(lalpha(k) for k in ks) if sel else 0)
    m, lo, hi = stats(tot)
    return m, lo, hi, tot
