"""
NP2 (single shank, full 2 x 192 grid, 15 um rows) on the calibrated synthetic field: lfpack v04 settings against
Gavish-Donoho shrinkage, with the NP2 noise scale (1.22, white-noise calibration) and the NP1 full-grid one (2.25).
`fill_grid` is a no-op on NP2 (no empty positions).
"""
# %%
from pathlib import Path

import neuropixel
import numpy as np
import pandas as pd
from ibldsp import cadzow
from ibldsp.voltage import current_source_density

import cadzow_synthetic
from cadzow_synthetic import FS, PRODUCTION, synthetic_lfp_v2
from cadzow_synthetic_remedies import seam_ratio

H2 = neuropixel.trace_header(version=2)
cadzow_synthetic.H = H2  # forward model and CSD on the NP2 geometry
CONFIGS = {
    'v04 settings': PRODUCTION,
    'shrinkage 1.22 (NP2)': {**PRODUCTION, 'gap_threshold': None, 'shrinkage': 1.22},
    'shrinkage 2.25 (NP1 value)': {**PRODUCTION, 'gap_threshold': None, 'shrinkage': 2.25},
}

# %%
if __name__ == '__main__':
    rng = np.random.default_rng(42)
    truth = synthetic_lfp_v2(rng)
    csd_true = current_source_density(truth, H2)
    rows = []
    for noise in (0.0, 0.015, 0.03, 0.06):
        wav = truth + rng.standard_normal(truth.shape) * noise * 100e-6
        nocad = np.sqrt(np.mean(current_source_density(wav, H2) ** 2, axis=1))
        for name, kw in CONFIGS.items():
            out = cadzow.cadzow_denoiser(wav, h={'x': H2['x'], 'y': H2['y']}, fs=FS, n_jobs=8, **kw)
            e = cadzow_synthetic.csd_error(out, csd_true)
            kept = np.sqrt(np.mean(current_source_density(out.astype(np.float64), H2) ** 2, axis=1)) / nocad
            rows.append({'noise': noise, 'method': name, 'csd_err': np.median(e), 'seam_ratio': seam_ratio(e),
                         'csd_kept': np.median(kept)})
    df = pd.DataFrame(rows)
    print(df.pivot(index='method', columns='noise', values=['csd_err', 'seam_ratio', 'csd_kept']).round(2).to_string())
    df.to_csv(Path(__file__).parent.joinpath('2026-10-10_cadzow_synthetic_np2.csv'), index=False)
