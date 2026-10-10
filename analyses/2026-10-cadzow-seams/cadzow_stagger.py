"""
Staggered Cadzow: remove the spatial-window seams of the lfpack Cadzow denoiser.

The production denoiser (nswx=64, ovx=32) leaves seams every 32 channels (window starts). Here Cadzow is run twice
with the window grid offset by 1/4 and 3/4 of the window step (8 and 24 channels), and the two runs are blended with
complementary raised-cosine weights that sum to 1 and are null on each run's own seams.

Compared on the cached production chunks of `plot_cadzow_csd.py` (no-Cadzow output of the production worker), against
the production parameters recomputed on the same input.
"""
# %%
from pathlib import Path

import addcopyfighandler  # noqa: F401
import h5py
import matplotlib.pyplot as plt
import neuropixel
import numpy as np
import pandas as pd
import seaborn as sns
from ibldsp import cadzow
from ibldsp.voltage import current_source_density, interpolate_bad_channels

sns.set_theme(context='notebook')

CACHE_DIR = Path('/Users/olivier/scratch/lfp/csd_lfpack_v04')
ARCHIVE_FINE = Path('/Users/olivier/Documents/datadisk/lfp-processing/lfpack/v04/lf_compressed_v04_a2p5_fine_all.h5')
FIGURE_DIR = Path(__file__).parent.joinpath('figures')
H = neuropixel.trace_header(version=1)
FS = 250.0
# lfpack v04 production settings
CADZOW_KWARGS = {'rank': 5, 'niter': 1, 'fmax': None, 'nswx': 64, 'ovx': 32, 'gap_threshold': 2.0, 'ppca_k': 2.0}
STEP = CADZOW_KWARGS['nswx'] - CADZOW_KWARGS['ovx']  # 32 channels between window starts = seam period
OFFSETS = (STEP // 4, 3 * STEP // 4)  # 8 and 24 channels


def cadzow_offset(wav, offset, h=H, **kwargs):
    """
    Cadzow-denoise with the spatial window grid starting at channel `offset` instead of 0.

    Channels [offset:] are denoised as a block, so its windows start at offset + k * STEP. The first `offset`
    channels are taken from a single window run on channels [0:nswx], whose gain is flat over them. The switch
    between the two runs happens at channel `offset`, which is a seam of this grid.

    Parameters
    ----------
    wav : numpy.ndarray
        LFP (n_channels, n_samples) [V].
    offset : int
        First channel of the window grid.
    h : dict
        Probe geometry with 'x' and 'y' [um].
    **kwargs
        Forwarded to `ibldsp.cadzow.cadzow_denoiser`.

    Returns
    -------
    numpy.ndarray
        Denoised LFP (n_channels, n_samples) [V].
    """
    out = np.zeros_like(wav)
    sl = slice(offset, None)
    out[sl] = cadzow.cadzow_denoiser(wav[sl], h={k: h[k][sl] for k in 'xy'}, fs=FS, **kwargs)
    if offset:
        sl = slice(0, kwargs['nswx'])
        out[:offset] = cadzow.cadzow_denoiser(wav[sl], h={k: h[k][sl] for k in 'xy'}, fs=FS, **kwargs)[:offset]
    return out


def stagger_weights(nc, offset_a=OFFSETS[0], step=STEP):
    """
    Complementary raised-cosine weights for two runs whose seams are half a step apart.

    Parameters
    ----------
    nc : int
        Number of channels.
    offset_a : int
        Seam offset of run A; run B's seams are at offset_a + step / 2.
    step : int
        Seam period [channels].

    Returns
    -------
    tuple of numpy.ndarray
        (w_a, w_b), each (nc,), with w_a + w_b = 1, w_a = 0 on A's seams and w_b = 0 on B's seams.
    """
    w_a = np.sin(np.pi * (np.arange(nc) - offset_a) / step) ** 2
    return w_a, 1 - w_a


def cadzow_staggered(wav, labels, h=H, **kwargs):
    """
    Staggered Cadzow: two runs offset by 1/4 and 3/4 of the window step, blended to cancel the seams.

    Parameters
    ----------
    wav : numpy.ndarray
        LFP (n_channels, n_samples) [V].
    labels : numpy.ndarray
        Bad-channel labels (0 = good), re-interpolated after blending as in production.
    h : dict
        Probe geometry with 'x' and 'y' [um].
    **kwargs
        Forwarded to `ibldsp.cadzow.cadzow_denoiser`.

    Returns
    -------
    numpy.ndarray
        Denoised LFP (n_channels, n_samples) [V].
    """
    a, b = (cadzow_offset(wav, o, h=h, **kwargs) for o in OFFSETS)
    w_a, w_b = stagger_weights(wav.shape[0])
    out = w_a[:, np.newaxis] * a + w_b[:, np.newaxis] * b
    return interpolate_bad_channels(out, labels, h['x'], h['y'], groupby_y=True)


def seam_profile(wav, labels):
    """
    Step in log CSD RMS between adjacent channels, averaged by position in the 32-channel cycle.

    Parameters
    ----------
    wav : numpy.ndarray
        LFP (n_channels, n_samples) [V].
    labels : numpy.ndarray
        Bad-channel labels; steps next to bad channels are excluded.

    Returns
    -------
    numpy.ndarray
        (STEP,) mean |d log RMS| for the step between channel i and i + 1, by (i + 1) mod STEP.
    """
    csd = current_source_density(wav.astype(np.float64), H)
    jump = np.abs(np.diff(np.log(np.sqrt(np.mean(csd ** 2, axis=1)))))
    bad = np.where(labels != 0)[0]
    ok = ~np.isin(np.arange(jump.size), np.r_[bad - 2, bad - 1, bad, bad + 1])
    phase = (np.arange(jump.size) + 1) % STEP
    return np.array([jump[ok & (phase == p)].mean() for p in range(STEP)])


# %%
if __name__ == '__main__':
    pids = sorted(f.name[:36] for f in CACHE_DIR.glob('*_chunk_orig.npy'))
    rows = []
    for pid in pids:
        with h5py.File(ARCHIVE_FINE, 'r') as f:
            labels = f[pid]['00/meta'].attrs['labels'][:]
        orig = np.load(CACHE_DIR.joinpath(f'{pid}_chunk_orig.npy')).T.astype(np.float64)
        files = {k: CACHE_DIR.joinpath(f'{pid}_exp_{k}.npy') for k in ('production', 'staggered')}
        if not files['production'].exists():
            y = cadzow.cadzow_denoiser(orig, h=H, fs=FS, **CADZOW_KWARGS, n_jobs=8)
            np.save(files['production'], interpolate_bad_channels(y, labels, H['x'], H['y'], groupby_y=True))
        if not files['staggered'].exists():
            np.save(files['staggered'], cadzow_staggered(orig, labels, **CADZOW_KWARGS, n_jobs=8))
        rms_orig = np.median(np.sqrt(np.mean(current_source_density(orig, H) ** 2, axis=1)))
        for k, file in files.items():
            y = np.load(file)
            prof = seam_profile(y, labels)
            rms = np.median(np.sqrt(np.mean(current_source_density(y, H) ** 2, axis=1)))
            rows.append({'pid': pid, 'method': k, 'csd_kept': rms / rms_orig,
                         **{f'phase_{p}': prof[p] for p in range(STEP)}})
        print(pid[:8], 'done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent.joinpath('2026-10-09_cadzow_stagger_seams.csv'), index=False)
    cols = [f'phase_{p}' for p in range(STEP)]
    prof = df.groupby('method')[cols].median()
    print(prof[[f'phase_{p}' for p in (0, 8, 16, 24)]].round(3))
    print(df.groupby('method')['csd_kept'].median().round(3))

    # seam profile figure
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, p in prof.iterrows():
        ax.plot(np.arange(STEP), p.values, marker='o', label=k)
    for x, ls in ((0, '-'), (OFFSETS[0], ':'), (OFFSETS[1], ':')):
        ax.axvline(x, color='k', ls=ls, lw=0.8)
    ax.set(xlabel='channel mod 32 (solid: production seams, dotted: staggered-run seams)',
           ylabel='|Δ log CSD RMS|, median over insertions')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR.joinpath('2026-10-09_cadzow_stagger_seam_profile.webp'), dpi=120, pil_kwargs={'quality': 80})
