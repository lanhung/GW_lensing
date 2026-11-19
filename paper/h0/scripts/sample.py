"""Table of the 119 analysed systems (truth and observed quantities) from the GW+EM input files.
usage: sample.py  -> writes p2/sample.csv and prints summary statistics."""
import glob, json, os
import numpy as np
import pandas as pd

S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ana = set()
for line in open(f'{S}/p2/all.jsonl'):
    r = json.loads(line)
    if r['dir'] in ('lcdm_k0.025_EPL_shear_h0_sn_ks', 'lcdm_k0.025_EPL_shear_h0_sn') and r['dir'].endswith('_ks') == (r['scenario'] == 'ET'):
        ana.add((r['scenario'], r['event_id']))
print('analysed events', len(ana))
rows = []
for f in glob.glob(f'{S}/p2/inputs/inputs/gwem/*.json'):
    d = json.load(open(f))
    fx = f'{S}/p2/fix/inputs/{os.path.basename(f)}'  # image labels of the GW injection (p2/fix/README.md)
    swap = os.path.exists(fx)
    if swap:
        d = json.load(open(fx))
    sc, eid = d['scenario'], d['event_id']
    if (sc, eid) not in ana:
        continue
    t, e, r = d['truth'], d['em'], d['row']
    ia, ib = e['gw_a'], e['gw_b']
    rows.append(dict(scenario=sc, event_id=eid, family=d['family'], z_l=r['true_z_l'], z_s=r['true_z_s'], sigma_v=r['true_sigma_v'],
                     theta_E=t['theta_E'], gamma=t.get('gamma', 2.0), q=t['q'], n_images=len(t['mu']),
                     mu_a=abs(t['mu'][ia]), mu_b=abs(t['mu'][ib]), morse_a=t['morse'][ia], morse_b=t['morse'][ib],
                     dt_days=abs(t['t'][ib] - t['t'][ia]) / 86400, rho_a=r[f'{sc}__true_rho_' + 'ab'[swap]], rho_b=r[f'{sc}__true_rho_' + 'ba'[swap]],
                     d_l=t['d_l'], m1=r['true_m1'], m2=r['true_m2'],
                     sep=np.hypot(e['x'][ia] - e['x'][ib], e['y'][ia] - e['y'][ib])))
df = pd.DataFrame(rows).sort_values(['scenario', 'family', 'event_id'])
df.to_csv(f'{S}/p2/sample.csv', index=False)
print(df.groupby('scenario').size())
for sc in ('ET', 'O5'):
    q = df[df.scenario == sc]
    print(f'--- {sc}: n={len(q)}  quads={int((q.n_images == 4).sum())}  mu_a>=5: {int((q.mu_a >= 5).sum())}  '
          f'saddle A (morse .5): {int((q.morse_a == 0.5).sum())}')
    for c in ('z_l', 'z_s', 'sigma_v', 'theta_E', 'dt_days', 'mu_a', 'mu_b', 'rho_a', 'rho_b', 'sep', 'gamma'):
        v = q[c].to_numpy()
        print(f'  {c:8s} median {np.median(v):8.3f}  10-90% {np.quantile(v, .1):8.3f} {np.quantile(v, .9):8.3f}  min {v.min():8.3f} max {v.max():8.3f}')
    print('  dt<1 d:', int((q.dt_days < 1).sum()), ' dt>100 d:', int((q.dt_days > 100).sum()))
