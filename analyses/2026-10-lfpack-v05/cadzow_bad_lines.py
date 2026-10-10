"""
Bad-channel lines in the CSD of the real chunks, uncompressed and fine-compressed, v04 (interpolation before and after
Cadzow) against v05 (bad channels as missing).

Line index of an insertion: median over its bad channels of the CSD RMS on the channels whose CSD involves the bad
channel (same column, up to 2 rows away) over the CSD RMS of the same-column channels 3 to 6 rows away. 1 means no
line; an interpolated channel that misses the column curvature shows as a dip or a peak.
"""
# %%
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from ibldsp.voltage import current_source_density

from cadzow_compression import CACHE_DIR, H
from cadzow_stagger import ARCHIVE_FINE

SIGNALS = {'v04 Cadzow': 'production', 'v05 Cadzow': 'np1_missing', 'v04 fine': 'v04_fine', 'v05 fine': 'v05_fine'}


def line_index(csd_rms, bad):
    """|log| of the near over far CSD RMS ratio, median over bad channels (0 = no line); also the signed median."""
    r = []
    for b in np.where(bad)[0]:
        near = [k for k in range(b - 8, b + 9, 4) if 0 <= k < 384]
        far = [k for k in list(range(b - 24, b - 11, 4)) + list(range(b + 12, b + 25, 4)) if 0 <= k < 384 and not bad[k]]
        if far:
            r.append(np.log(np.mean(csd_rms[near]) / np.mean(csd_rms[far])))
    r = np.array(r)
    return np.median(np.abs(r)), np.median(r)


# %%
if __name__ == '__main__':
    rows = []
    for pid in sorted(f.name[:36] for f in CACHE_DIR.glob('*_chunk_orig.npy')):
        with h5py.File(ARCHIVE_FINE, 'r') as f:
            labels = f[pid]['00/meta'].attrs['labels'][:]
        bad = np.isin(labels, (1, 2))
        row = {'pid': pid[:8], 'n_bad': int(bad.sum())}
        for k, v in SIGNALS.items():
            c = current_source_density(np.load(CACHE_DIR.joinpath(f'{pid}_exp_{v}.npy')).astype(np.float64), H)
            row[k] = line_index(np.sqrt(np.mean(c ** 2, axis=1)), bad)[0]
        rows.append(row)
    df = pd.DataFrame(rows).set_index('pid')
    df.loc['median'] = df.median()
    print(df.round(3).to_string())
    df.to_csv(Path(__file__).parent.joinpath('2026-10-10_cadzow_bad_lines.csv'))
