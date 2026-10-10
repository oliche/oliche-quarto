"""
Window-blending gain and trajectory structure of `ibldsp.cadzow.cadzow_denoiser`.

1. Summed synthesis gain per channel: `_process_window` is replaced by the identity, so the denoiser output divided by
   its input is exactly the sum of the window weights each channel receives (should be 1 everywhere).
2. Trajectory matrix of one 64-channel NP1 window: which entries hold data, and how many entries each channel fills
   (`trcount`, the averaging count when the rank-reduced matrix is scattered back to channels).
"""
# %%
from pathlib import Path
from unittest import mock

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import neuropixel
import numpy as np
import seaborn as sns
from ibldsp import cadzow

sns.set_theme(context='notebook')
FIGURE_DIR = Path(__file__).parent.joinpath('figures')
H = neuropixel.trace_header(version=1)
NC = 384


def summed_gain(**kwargs):
    """
    Sum of the synthesis weights each channel receives in `cadzow_denoiser`.

    Parameters
    ----------
    **kwargs
        Window parameters passed to `cadzow_denoiser` (``nswx``, ``ovx``, ``npad``).

    Returns
    -------
    gain : ndarray (NC,)
        Output / input per channel with an identity window fit.
    """
    identity = lambda WAV_sl, *args: WAV_sl.copy()  # noqa: E731
    wav = np.random.default_rng(0).standard_normal((NC, 256))
    with mock.patch.object(cadzow, '_process_window', identity):
        out = cadzow.cadzow_denoiser(wav, h={'x': H['x'][:NC], 'y': H['y'][:NC]}, fmax=None, **kwargs)
    gain = np.sum(out * wav, axis=1) / np.sum(wav ** 2, axis=1)
    return gain


# %% summed gain for production (ovx=32) and ovx=48
configs = {'production: nswx=64, ovx=32': dict(nswx=64, ovx=32), 'nswx=64, ovx=48': dict(nswx=64, ovx=48)}
gains = {k: summed_gain(**v) for k, v in configs.items()}
for k, g in gains.items():
    print(f'{k}: summed gain min {g.min():.6f}, max {g.max():.6f}')

# %% trajectory structure of one 64-channel window
sl = slice(64, 128)
x, y = H['x'][sl], H['y'][sl]
T, it, ic, trcount = cadzow.trajectory(x, y)
filled = np.zeros(T.shape, dtype=bool)
filled[it] = True
print(f'unique x {np.unique(x)}, {np.unique(y).size} unique y; T {T.shape}, filled {filled.mean():.1%}')

# %% figure
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), gridspec_kw={'width_ratios': [2, 1, 1.2]})
for k, g in gains.items():
    axs[0].plot(g, label=k)
axs[0].set(xlabel='channel', ylabel='summed window gain', title='Synthesis gain per channel (identity fit)',
           ylim=[0.9, 1.1])
axs[0].legend(loc='lower right')
axs[1].plot(trcount, np.arange(trcount.size), '.-')
axs[1].set(xlabel='trajectory entries per channel', ylabel='channel in window', title='trcount, 64-ch window')
axs[1].invert_yaxis()
axs[2].imshow(filled, aspect='auto', cmap='Greys', interpolation='none')
axs[2].grid(False)
axs[2].set(title=f'Trajectory matrix {T.shape}: filled entries', xlabel='column', ylabel='row')
fig.tight_layout()
fig.savefig(FIGURE_DIR.joinpath('2026-10-09_cadzow_window_gain.webp'), dpi=120, pil_kwargs={'quality': 80})
