"""Neural posterior estimation of ln(mu) with a conditional neural spline flow.

Trains p(ln mu_A | x) on simulated observables x (default: geometry + GW relative magnification +
follow-up, i.e. no absolute GW amplitude and no assumed cosmology) using the zuko neural spline
flow, then evaluates point accuracy, interval coverage at several credibility levels and the
probability-integral-transform (PIT) calibration on held-out events, per lens family.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zuko

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.benchmark import metrics, split  # noqa: E402
from analysis.features import FAMILIES, FOLLOWUP, GEO, INPUT_SETS, check_inputs, load  # noqa: E402

LEVELS = [0.5, 0.68, 0.9, 0.95]


def standardise(x, mean=None, std=None):
    x = np.nan_to_num(x, nan=-1.0)
    if mean is None:
        mean, std = x.mean(0), x.std(0) + 1e-6
    return (x - mean) / std, mean, std


def train_flow(x, y, xv, yv, device, epochs, seed):
    torch.manual_seed(seed)
    flow = zuko.flows.NSF(features=1, context=x.shape[1], transforms=4, hidden_features=[256, 256, 256], bins=12).to(device)
    opt = torch.optim.AdamW(flow.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X, Y = torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(y[:, None], dtype=torch.float32, device=device)
    XV, YV = torch.tensor(xv, dtype=torch.float32, device=device), torch.tensor(yv[:, None], dtype=torch.float32, device=device)
    best, best_state, bs = np.inf, None, 1024
    for ep in range(epochs):
        flow.train()
        perm = torch.randperm(len(X), device=device)
        for i in range(0, len(X), bs):
            ix = perm[i:i + bs]
            loss = -flow(X[ix]).log_prob(Y[ix]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
        sched.step()
        flow.eval()
        with torch.no_grad():
            v = float(np.mean([-flow(XV[i:i + 8192]).log_prob(YV[i:i + 8192]).mean().item() for i in range(0, len(XV), 8192)]))
        if v < best:
            best, best_state = v, {k: t.detach().clone() for k, t in flow.state_dict().items()}
    flow.load_state_dict(best_state)
    return flow, best


@torch.no_grad()
def load_flow(path, device='cpu'):
    """Load a saved flow checkpoint written by this script."""
    ck = torch.load(path, weights_only=False, map_location=device)
    flow = zuko.flows.NSF(**ck['arch']).to(device)
    flow.load_state_dict(ck['state'])
    flow.eval()
    return flow, ck


def posterior_samples(flow, x, device, n=1000):
    out = []
    X = torch.tensor(x, dtype=torch.float32, device=device)
    for i in range(0, len(X), 2048):
        s = flow(X[i:i + 2048]).sample((n,))[..., 0]  # (n, batch)
        out.append(s.T.cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data/v2'))
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--scenario', default='ET')
    p.add_argument('--input-set', default='Geometry+GW(rel)+FU')
    p.add_argument('--max-per-family', type=int, default=60000)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    sets = dict(INPUT_SETS)
    sets['Geometry+GW(rel)+FU'] = INPUT_SETS['Geometry+GW(rel)'] + FOLLOWUP
    cols = check_inputs(sets[a.input_set])
    data = load(a.data_dir, a.scenario, max_per_family=a.max_per_family, seed=1)
    lab = split(data, a.seed)
    # Early stopping uses 10 per cent of the training set; the calibration set is reserved for
    # conformal recalibration of the posterior intervals.
    trall, ca, te = data[lab == 'train'], data[lab == 'cal'], data[lab == 'test']
    vmask = np.random.default_rng(a.seed + 1).random(len(trall)) < 0.1
    tr, va = trall[~vmask], trall[vmask]
    xtr, m, s = standardise(tr[cols].to_numpy(float))
    xva, _, _ = standardise(va[cols].to_numpy(float), m, s)
    xte, _, _ = standardise(te[cols].to_numpy(float), m, s)
    t0 = time.time()
    flow, val_nll = train_flow(xtr, np.log(tr.mu.to_numpy()), xva, np.log(va.mu.to_numpy()), a.device, a.epochs, a.seed)
    torch.save(dict(state=flow.state_dict(), cols=cols, mean=m, std=s, input_set=a.input_set, scenario=a.scenario,
                    arch=dict(features=1, context=len(cols), transforms=4, hidden_features=[256, 256, 256], bins=12),
                    qconf=None), a.out_dir / f'flow_{a.scenario}_{a.input_set.replace("+", "_").replace("(", "").replace(")", "")}.pt')
    samples = posterior_samples(flow, xte, a.device)
    y = np.log(te.mu.to_numpy())
    # Conformal recalibration (per family) of the central 90 per cent posterior interval.
    xca, _, _ = standardise(ca[cols].to_numpy(float), m, s)
    sca = posterior_samples(flow, xca, a.device)
    yca = np.log(ca.mu.to_numpy())
    sc = np.maximum(np.quantile(sca, 0.05, 1) - yca, yca - np.quantile(sca, 0.95, 1))
    qconf = {}
    for fam in FAMILIES:
        v = np.sort(sc[ca.family.to_numpy() == fam])
        k = int(np.ceil((len(v) + 1) * 0.9))
        qconf[fam] = float(v[min(k, len(v)) - 1])
    ck = a.out_dir / f'flow_{a.scenario}_{a.input_set.replace("+", "_").replace("(", "").replace(")", "")}.pt'
    saved = torch.load(ck, weights_only=False)
    saved['qconf'] = qconf
    torch.save(saved, ck)
    pit = (samples < y[:, None]).mean(1)
    med = np.exp(np.median(samples, 1))
    rows = []
    for fam in FAMILIES + ['all']:
        mask = np.ones(len(te), bool) if fam == 'all' else (te.family.to_numpy() == fam)
        lo, hi = np.exp(np.quantile(samples[mask], 0.05, 1)), np.exp(np.quantile(samples[mask], 0.95, 1))
        r = dict(scenario=a.scenario, input_set=a.input_set, family=fam, method='NPE (NSF)', **metrics(te.mu.to_numpy()[mask], med[mask], lo, hi, np.random.default_rng(0)))
        for lev in LEVELS:
            ql, qh = np.quantile(samples[mask], (1 - lev) / 2, 1), np.quantile(samples[mask], (1 + lev) / 2, 1)
            r[f'coverage_{int(lev * 100)}'] = float(np.mean((y[mask] >= ql) & (y[mask] <= qh)))
        h, _ = np.histogram(pit[mask], bins=10, range=(0, 1))
        r['pit_hist'] = json.dumps((h / max(1, mask.sum())).round(4).tolist())
        r['pit_ks'] = float(np.max(np.abs(np.sort(pit[mask]) - np.linspace(0, 1, mask.sum()))))
        qv = np.array([qconf[f] for f in te.family.to_numpy()[mask]])
        clo, chi = np.quantile(samples[mask], 0.05, 1) - qv, np.quantile(samples[mask], 0.95, 1) + qv
        r['coverage90_conformal'] = float(np.mean((y[mask] >= clo) & (y[mask] <= chi)))
        r['width_ln_conformal'] = float(np.mean(chi - clo))
        rows.append(r)
    tag = a.input_set.replace("+", "_").replace("(", "").replace(")", "")
    pd.DataFrame(rows).to_csv(a.out_dir / f'npe_{a.scenario}_{tag}.csv', index=False)
    pd.DataFrame(dict(event_id=te.event_id, family=te.family, mu=te.mu, pred=med, lo=np.exp(np.quantile(samples, 0.05, 1)),
                      hi=np.exp(np.quantile(samples, 0.95, 1)), pit=pit, dl_true=te.dl_true, ln_dapp_a=te.gw_ln_dapp_a,
                      y_true=te.y_true)).to_parquet(a.out_dir / f'npe_predictions_{a.scenario}_{tag}.parquet', index=False)
    print(json.dumps(dict(val_nll=val_nll, seconds=time.time() - t0, n_train=len(tr)), indent=1))
    print(pd.DataFrame(rows)[['family', 'mae', 'median_rel', 'coverage90', 'coverage90_conformal', 'coverage_68', 'pit_ks']].to_string())


if __name__ == '__main__':
    main()
