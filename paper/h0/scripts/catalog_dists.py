"""Parent (all lensed systems) and detected populations of the data/v2 catalogs, for the sample figure of Paper II.
Run in the repository root: python catalog_dists.py OUT.npz"""
import sys
import numpy as np
import pandas as pd

out = {}
for fam in ['SIS', 'SIE', 'SIE_shear', 'EPL_shear']:
    d = pd.read_parquet(f'data/v2/{fam}.parquet')
    gamma = d.true_gamma.to_numpy() if 'true_gamma' in d else np.full(len(d), 2.0)
    out.update({f'{fam}|parent|z_l': d.true_z_l.to_numpy(), f'{fam}|parent|z_s': d.true_z_s.to_numpy(),
                f'{fam}|parent|theta_E': d.true_theta_e.to_numpy(), f'{fam}|parent|gamma': gamma,
                f'{fam}|parent|sigma_v': d.true_sigma_v.to_numpy(), f'{fam}|parent|n_images': d.true_n_images.to_numpy()})
    for sc in ('ET', 'O5'):
        q = d[d[f'{sc}__selected']]
        out.update({f'{fam}|{sc}|z_l': q.true_z_l.to_numpy(), f'{fam}|{sc}|z_s': q.true_z_s.to_numpy(),
                    f'{fam}|{sc}|theta_E': q.true_theta_e.to_numpy(),
                    f'{fam}|{sc}|gamma': q.true_gamma.to_numpy() if 'true_gamma' in q else np.full(len(q), 2.0),
                    f'{fam}|{sc}|sigma_v': q.true_sigma_v.to_numpy(), f'{fam}|{sc}|n_images': q.true_n_images.to_numpy(),
                    f'{fam}|{sc}|dt_days': np.abs(q[f'{sc}__true_dt_ab'].to_numpy()) / 86400,
                    f'{fam}|{sc}|mu_max': np.maximum(np.abs(q[f'{sc}__true_mu_a']), np.abs(q[f'{sc}__true_mu_b'])).to_numpy(),
                    f'{fam}|{sc}|mu_a': np.abs(q[f'{sc}__true_mu_a']).to_numpy(),
                    f'{fam}|{sc}|rho_min': np.minimum(q[f'{sc}__true_rho_a'], q[f'{sc}__true_rho_b']).to_numpy(),
                    f'{fam}|{sc}|rho_a': q[f'{sc}__true_rho_a'].to_numpy()})
np.savez_compressed(sys.argv[1], **out)
