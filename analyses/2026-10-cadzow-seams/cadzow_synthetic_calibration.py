"""
Calibrate the synthetic field so that Cadzow behaves on it as on the real chunks.

The real chunks give, per method, the fraction of CSD RMS kept (median over the 11 insertions,
`2026-10-09_cadzow_mssa_real.csv`). The synthetic source sharpness (`width_min`) and white-noise level are scanned and
the setting whose CSD-kept fractions match the real ones (least squares on log ratios) is retained. A synthetic set
that ranks production, ovx=48 and MSSA like the real data is then used to test the seam remedies.
"""
# %%
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from cadzow_cycle_spin import csd_rms
from cadzow_mssa_real import MSSA
from cadzow_remedies import run
from cadzow_synthetic import PRODUCTION, synthetic_lfp_v2

METHODS = {'production': ('hankel2d', PRODUCTION), 'ovx48': ('hankel2d', {**PRODUCTION, 'ovx': 48}),
           **{k: v[:2] for k, v in MSSA.items()}}
WIDTH_MIN = (20.0, 50.0, 100.0)
NOISE = (0.01, 0.02, 0.05, 0.1)  # white-noise RMS relative to the 100 µV signal RMS

# %%
if __name__ == '__main__':
    real = pd.read_csv(Path(__file__).parent.joinpath('2026-10-09_cadzow_mssa_real.csv'))
    target = real.filter(like='csd_kept').median().rename(lambda c: c.replace('csd_kept_', ''))
    rows = []
    for wmin, noise in itertools.product(WIDTH_MIN, NOISE):
        rng = np.random.default_rng(42)
        truth = synthetic_lfp_v2(rng, width_min=wmin)
        wav = truth + rng.standard_normal(truth.shape) * noise * 100e-6
        r0 = csd_rms(wav)
        kept = {k: csd_rms(run(wav, variant, **kwargs)) / r0 for k, (variant, kwargs) in METHODS.items()}
        rows.append({'width_min': wmin, 'noise': noise, **kept,
                     'misfit': np.sum([np.log(kept[k] / target[k]) ** 2 for k in METHODS])})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    print('real', target.round(3).to_dict())
    print(df.round(3).sort_values('misfit').to_string())
    df.to_csv(Path(__file__).parent.joinpath('2026-10-09_cadzow_synthetic_calibration.csv'), index=False)
