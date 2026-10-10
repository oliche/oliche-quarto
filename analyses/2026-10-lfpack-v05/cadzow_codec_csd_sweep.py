"""
CSD fidelity of the lfpack codec against file size, on one insertion: sweep of epsilon (spatial SVD truncation) and
alpha (wavelet-packet threshold) on the v05 Cadzow output. CSD SNR is against the CSD of the uncompressed input, for
the raw CSD and for a CSD smoothed in depth (Gaussian, 2 rows = 40 µm sigma, along each column).
"""
# %%
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage
from ibldsp.voltage import current_source_density

from cadzow_compression import CACHE_DIR, H, compress_roundtrip, snr_db

EPSILONS, ALPHAS = (100.0, 30.0, 10.0, 3.0), (2.5, 1.0, 0.3)


def csd(wav):
    """CSD (384, ns) of LFP (384, ns) [V]."""
    return current_source_density(np.asarray(wav, dtype=np.float64), H)


def smooth_depth(c, sigma_rows=2.0):
    """Gaussian smoothing along depth within each NP1 column (every 4th channel shares a column pattern)."""
    out = np.zeros_like(c)
    for col in np.unique(H['col']):
        i = np.where(H['col'] == col)[0]
        i = i[np.argsort(H['row'][i])]
        out[i] = scipy.ndimage.gaussian_filter1d(c[i], sigma_rows, axis=0, mode='nearest')
    return out


# %%
if __name__ == '__main__':
    pid = next(f.name[:36] for f in CACHE_DIR.glob('*_chunk_orig.npy') if f.name.startswith(sys.argv[1]))
    wav = np.load(CACHE_DIR.joinpath(f'{pid}_exp_np1_missing.npy')).astype(np.float64)
    c0 = csd(wav)
    s0 = smooth_depth(c0)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for eps in EPSILONS:
            for alpha in ALPHAS:
                size, rec = compress_roundtrip(wav, alpha, tmp, epsilon=eps)
                c = csd(rec)
                rows.append({'epsilon': eps, 'alpha': alpha, 'size_kb': size / 1024, 'cr': wav.size * 4 / size,
                             'snr_db': snr_db(wav, rec), 'csd_snr_db': snr_db(c0, c),
                             'csd_smooth_snr_db': snr_db(s0, smooth_depth(c))})
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent.joinpath(f'2026-10-10_cadzow_codec_csd_sweep_{pid[:8]}.csv'), index=False)
    print(df.round(1).to_string())
