"""
Does the lfpack codec remove CSD signal or noise? The calibrated synthetic field (known CSD) through Cadzow (v04
settings, v05 `cadzow_denoiser_np1`) and the lfpack tiers, at 0 and 3 % white noise: CSD error against the true CSD,
before and after compression.
"""
# %%
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from ibldsp import cadzow
from ibldsp.voltage import current_source_density

from cadzow_compression import compress_roundtrip
from cadzow_synthetic import FS, PRODUCTION, H, csd_error, synthetic_lfp_v2
from cadzow_synthetic_remedies import HXY, seam_ratio

TIERS = {'uncompressed': None, 'default': 7.0, 'fine': 2.5}
CADZOW = {'no Cadzow': None, 'v04': lambda w: cadzow.cadzow_denoiser(w, h=HXY, fs=FS, **PRODUCTION, n_jobs=8),
          'v05': lambda w: cadzow.cadzow_denoiser_np1(w, h=HXY, fs=FS, n_jobs=8)}

# %%
if __name__ == '__main__':
    rng = np.random.default_rng(42)
    truth = synthetic_lfp_v2(rng)
    csd_true = current_source_density(truth, H)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for noise in (0.0, 0.03):
            wav = truth + rng.standard_normal(truth.shape) * noise * 100e-6
            for name, fcn in CADZOW.items():
                den = wav if fcn is None else fcn(wav).astype(np.float64)
                for tier, alpha in TIERS.items():
                    out = den if alpha is None else compress_roundtrip(den, alpha, tmp)[1]
                    e = csd_error(out, csd_true)
                    rows.append({'noise': noise, 'cadzow': name, 'tier': tier, 'csd_err': np.median(e),
                                 'seam_ratio': seam_ratio(e)})
                    print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent.joinpath('2026-10-10_cadzow_synthetic_codec.csv'), index=False)
    print(df.pivot_table(index=['noise', 'cadzow'], columns='tier', values=['csd_err', 'seam_ratio'])
          .round(2).to_string())
