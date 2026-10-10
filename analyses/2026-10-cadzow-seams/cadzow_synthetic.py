"""
Synthetic ground truth for the Cadzow seams: laminar CSD sources forward-modelled onto the NP1 geometry.

Each source is a Gaussian current-density profile in depth (some as sink/source dipoles) with its own band-limited
time course. The potential is the laminar disc model, phi(z) = 1 / (2 sigma) * int C(z') (sqrt((z - z')^2 + R^2) -
|z - z'|) dz', so channels at the same depth carry the same signal. The truth CSD is the CSD of the noise-free
potential computed with the same `current_source_density` as the real data.

Cadzow runs on the noise-free and on the noisy field; the CSD error against the truth, per channel, shows whether the
seams are an algorithmic bias (present without noise) and which parameter creates them.
"""
# %%
import time
from pathlib import Path

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import neuropixel
import numpy as np
import pandas as pd
import scipy.fft
import scipy.signal
import seaborn as sns
from ibldsp import cadzow
from ibldsp.voltage import current_source_density

sns.set_theme(context='notebook')
FIGURE_DIR = Path(__file__).parent.joinpath('figures')
H = neuropixel.trace_header(version=1)
FS = 250.0
NS = 4096  # 16.4 s
NOISE_RMS = 0.2  # spatially white noise RMS relative to the signal RMS
PRODUCTION = {'rank': 5, 'niter': 1, 'fmax': None, 'nswx': 64, 'ovx': 32, 'gap_threshold': 2.0, 'ppca_k': 2.0}
METHODS = {
    'production': PRODUCTION,
    'fixed rank 5': {**PRODUCTION, 'gap_threshold': None},
    'no PPCA': {**PRODUCTION, 'ppca_k': None},
    'fixed rank 5, no PPCA': {**PRODUCTION, 'gap_threshold': None, 'ppca_k': None},
    'ovx=48': {**PRODUCTION, 'ovx': 48},
}


def synthetic_lfp(rng, n_sources=8, radius=500.0, sigma=1 / 3):
    """
    Laminar CSD sources forward-modelled onto the NP1 channel depths.

    Parameters
    ----------
    rng : numpy.random.Generator
    n_sources : int
        Number of laminar sources; every other one is a sink/source dipole.
    radius : float
        Radius of the disc sources [µm].
    sigma : float
        Tissue conductivity [S/m].

    Returns
    -------
    lfp : ndarray (384, NS)
        Noise-free potential, scaled to an RMS of 100 µV [V].
    """
    z = np.arange(-1000, 4840, 5.0)  # fine source grid [µm], extends beyond the probe
    y = H['y'].astype(float)
    kernel = (np.sqrt((y[:, None] - z[None, :]) ** 2 + radius ** 2) - np.abs(y[:, None] - z[None, :])) / (2 * sigma)
    sos = scipy.signal.butter(4, [1, 30], btype='bandpass', fs=FS, output='sos')
    lfp = np.zeros((y.size, NS))
    for k in range(n_sources):
        z0, width = rng.uniform(0, 3840), rng.uniform(60, 250)
        profile = np.exp(-0.5 * ((z - z0) / width) ** 2)
        if k % 2:  # dipole: sink above, source below
            profile -= np.exp(-0.5 * ((z - z0 - 2.5 * width) / width) ** 2)
        tc = scipy.signal.sosfiltfilt(sos, rng.standard_normal(NS))
        lfp += np.outer(kernel @ profile, tc)
    return lfp / np.sqrt(np.mean(lfp ** 2)) * 100e-6


def synthetic_lfp_v2(rng, n_sources=40, width_min=50.0, width_max=300.0, alpha=1.4, radius=500.0, sigma=1 / 3):
    """
    Laminar CSD sources with log-uniform widths and 1/f^alpha time courses, closer to the real chunks than
    `synthetic_lfp`.

    Parameters
    ----------
    rng : numpy.random.Generator
    n_sources : int
        Number of laminar sources; every other one is a sink/source dipole.
    width_min, width_max : float
        Range of the Gaussian source widths [µm], sampled log-uniformly.
    alpha : float
        Power-spectrum exponent of the time courses above 2 Hz (flat below); the real chunks fall as ~f^-1.4.
    radius : float
        Radius of the disc sources [µm].
    sigma : float
        Tissue conductivity [S/m].

    Returns
    -------
    lfp : ndarray (384, NS)
        Noise-free potential, scaled to an RMS of 100 µV [V].
    """
    z = np.arange(-1000, 4840, 2.5)
    y = H['y'].astype(float)
    kernel = (np.sqrt((y[:, None] - z[None, :]) ** 2 + radius ** 2) - np.abs(y[:, None] - z[None, :])) / (2 * sigma)
    f = scipy.fft.rfftfreq(NS, d=1 / FS)
    amp = np.maximum(f, 2.0) ** (-alpha / 2)
    amp[0] = 0
    lfp = np.zeros((y.size, NS))
    for k in range(n_sources):
        z0, width = rng.uniform(-200, 4040), np.exp(rng.uniform(np.log(width_min), np.log(width_max)))
        profile = np.exp(-0.5 * ((z - z0) / width) ** 2)
        if k % 2:
            profile -= np.exp(-0.5 * ((z - z0 - 2.5 * width) / width) ** 2)
        tc = scipy.fft.irfft(amp * scipy.fft.rfft(rng.standard_normal(NS)), n=NS)
        lfp += np.outer(kernel @ profile, tc / np.std(tc))
    return lfp / np.sqrt(np.mean(lfp ** 2)) * 100e-6


def csd_error(denoised, csd_true):
    """Per-channel RMS of the CSD error, relative to the median per-channel truth CSD RMS."""
    err = current_source_density(denoised.astype(np.float64), H) - csd_true
    return np.sqrt(np.mean(err ** 2, axis=1)) / np.median(np.sqrt(np.mean(csd_true ** 2, axis=1)))


# %%
if __name__ == '__main__':
    rng = np.random.default_rng(42)
    truth = synthetic_lfp(rng)
    csd_true = current_source_density(truth, H)
    noisy = truth + rng.standard_normal(truth.shape) * NOISE_RMS * 100e-6
    inputs = {'noise-free': truth, f'noise {NOISE_RMS:.0%}': noisy}

    errors, outputs, rows = {}, {}, []
    for kin, wav in inputs.items():
        errors[kin] = {'no Cadzow': csd_error(wav, csd_true)}
        for km, kwargs in METHODS.items():
            t0 = time.time()
            out = cadzow.cadzow_denoiser(wav, h=H, fs=FS, n_jobs=8, **kwargs)
            dt = time.time() - t0
            outputs[(kin, km)] = out
            errors[kin][km] = e = csd_error(out, csd_true)
            rows.append({'input': kin, 'method': km, 'csd_err_median': np.median(e), 'csd_err_p95': np.percentile(e, 95),
                         'time_s': dt})
    df = pd.DataFrame(rows)
    print(df.round(3).to_string())
    df.to_csv(Path(__file__).parent.joinpath('2026-10-09_cadzow_synthetic.csv'), index=False)

    # %% CSD error per channel, seams of the production grid marked
    fig, axs = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    for ax, (kin, err) in zip(axs, errors.items()):
        for km, e in err.items():
            ax.plot(e, np.arange(e.size), lw=1, label=km, color='k' if km == 'no Cadzow' else None)
        for s in range(32, 384, 32):
            ax.axhline(s, color='grey', lw=0.5, ls=':')
        ax.set(title=f'{kin}: CSD error per channel', xlabel='RMS(CSD error) / median RMS(true CSD)', xscale='log')
    axs[0].set(ylabel='channel (dotted: production window boundaries)')
    axs[0].invert_yaxis()
    axs[1].legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(FIGURE_DIR.joinpath('2026-10-09_cadzow_synthetic_error.webp'), dpi=110, pil_kwargs={'quality': 80})

    # %% CSD images: truth, production, error (noise-free and noisy), 4 s
    sl = slice(0, int(4 * FS))
    panels = {'true CSD': csd_true}
    for kin, wav in inputs.items():
        c = current_source_density(outputs[(kin, 'production')].astype(np.float64), H)
        panels[f'production, {kin}'] = c
        panels[f'error, {kin}'] = c - csd_true
    clim = np.percentile(np.abs(csd_true[:, sl]), 99)
    fig, axs = plt.subplots(1, len(panels), figsize=(18, 8), sharey=True)
    for ax, (k, c) in zip(axs, panels.items()):
        ax.imshow(c[:, sl], aspect='auto', cmap='berlin', vmin=-clim, vmax=clim, interpolation='none',
                  extent=[0, 4, 383.5, -0.5])
        ax.grid(False)
        ax.set(title=k, xlabel='time [s]')
        ax.plot(np.zeros(11) - 0.05, np.arange(32, 384, 32), '>', color='crimson', ms=5, clip_on=False)
    axs[0].set(ylabel='channel')
    fig.tight_layout()
    fig.savefig(FIGURE_DIR.joinpath('2026-10-09_cadzow_synthetic_csd.webp'), dpi=110, pil_kwargs={'quality': 80})
