"""
Feasibility of spatial pre-emphasis for lfpack: compress Z = P X with P = I - kappa L (L the second difference along
each NP1 column, reflecting ends), decode X = P^-1 Z_hat. The codec error, roughly flat in Z, is low-passed in depth
in X, where the CSD (a second difference) amplifies it least. Compared with plain compression at the same tiers on one
insertion (v05 Cadzow output): size, voltage SNR, raw and depth-smoothed CSD SNR.
"""
# %%
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg

from cadzow_codec_csd_sweep import csd, smooth_depth
from cadzow_compression import CACHE_DIR, H, compress_roundtrip, snr_db

KAPPAS = (0.0, 1.0, 3.0, 10.0)
TIERS = {'default': 7.0, 'fine': 2.5}


def column_laplacian():
    """Second difference along each probe column (384 x 384, sparse), reflecting (Neumann) ends."""
    rows, cols, vals = [], [], []
    for col in np.unique(H['col']):
        i = np.where(H['col'] == col)[0]
        i = i[np.argsort(H['row'][i])]
        for k, ch in enumerate(i):
            nb = [i[k - 1] if k > 0 else i[k + 1], i[k + 1] if k < i.size - 1 else i[k - 1]]
            rows += [ch, ch, ch]
            cols += [ch, nb[0], nb[1]]
            vals += [-2.0, 1.0, 1.0]
    return scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(384, 384))


# %%
if __name__ == '__main__':
    pid = next(f.name[:36] for f in CACHE_DIR.glob('*_chunk_orig.npy') if f.name.startswith(sys.argv[1]))
    wav = np.load(CACHE_DIR.joinpath(f'{pid}_exp_np1_missing.npy')).astype(np.float64)
    c0 = csd(wav)
    s0 = smooth_depth(c0)
    lap = column_laplacian()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for kappa in KAPPAS:
            p = (scipy.sparse.identity(384) - kappa * lap).tocsc()
            solve = scipy.sparse.linalg.factorized(p)
            for tier, alpha in TIERS.items():
                size, zrec = compress_roundtrip(p @ wav, alpha, tmp)
                rec = solve(zrec)
                c = csd(rec)
                rows.append({'kappa': kappa, 'tier': tier, 'size_kb': size / 1024, 'snr_db': snr_db(wav, rec),
                             'csd_snr_db': snr_db(c0, c), 'csd_smooth_snr_db': snr_db(s0, smooth_depth(c))})
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent.joinpath(f'2026-10-10_cadzow_codec_preemphasis_{pid[:8]}.csv'), index=False)
    print(df.round(1).to_string())
