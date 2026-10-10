"""
Smooth adaptive rank for Cadzow: Gavish-Donoho optimal singular-value shrinkage instead of a hard rank cut.

The hard adaptive rank (largest singular-value gap) makes neighbouring windows keep different numbers of components,
which shows as seams in the CSD. Here each singular value s of a trajectory matrix (m x n, beta = min/max) is shrunk
continuously against a noise level sigma (Gavish & Donoho 2017, Frobenius loss):

    y = s / (sigma sqrt(max(m, n))),   eta(y) = sqrt((y^2 - beta - 1)^2 - 4 beta) / y  if y > 1 + sqrt(beta), else 0

Two noise estimates, both keeping the windows independent (same parallel structure as production):

- 'window': sigma from the median singular value of the same window and bin (Marchenko-Pastur median).
- 'chunk': sigma per frequency bin from the whole chunk, the robust spread of the same-column second difference of
  the real channels' spectra (one cheap pass before the window map).

The trajectory entries are not i.i.d. (Hankel duplicates, interpolated virtual channels), so each estimate is
rescaled by a factor calibrated on pure white noise: the 99th percentile of the largest noise singular value lands
on the shrinker's bulk edge. Runs on the full grid (`cadzow_fill.py`) and keeps PPCA, with the shrinker in both SVDs.
"""
from unittest import mock

import numpy as np
import scipy.fft
import scipy.integrate
import scipy.optimize
from ibldsp import cadzow

from cadzow_fill import cadzow_full_grid, fill_virtual, full_grid

CONFIG = {'mode': 'window', 'scale': 1.0, 'sigma_bin': None}  # set by `cadzow_shrink` for the patched worker


def mp_median(beta):
    """Median of the Marchenko-Pastur distribution with aspect ratio beta (variance 1)."""
    lo, hi = (1 - np.sqrt(beta)) ** 2, (1 + np.sqrt(beta)) ** 2
    pdf = lambda t: np.sqrt(max((hi - t) * (t - lo), 0)) / (2 * np.pi * beta * t)
    return scipy.optimize.brentq(lambda x: scipy.integrate.quad(pdf, lo, x)[0] - 0.5, lo, hi)


_MP = {}


def mp_median_cached(beta):
    """`mp_median` memoised on beta."""
    if beta not in _MP:
        _MP[beta] = mp_median(beta)
    return _MP[beta]


def gd_shrink(s, sigma, shape):
    """
    Gavish-Donoho optimal shrinker (Frobenius loss) applied to singular values.

    Parameters
    ----------
    s : ndarray (nbins, k)
        Singular values per bin.
    sigma : ndarray (nbins,)
        Noise level per entry, per bin.
    shape : tuple
        Trajectory matrix shape (m, n).

    Returns
    -------
    ndarray (nbins, k)
        Shrunk singular values.
    """
    beta = min(shape) / max(shape)
    scale = sigma[:, None] * np.sqrt(max(shape))
    y = s / np.maximum(scale, 1e-30)
    eta = np.sqrt(np.maximum((y ** 2 - beta - 1) ** 2 - 4 * beta, 0)) / np.maximum(y, 1e-12)
    return np.where(y > 1 + np.sqrt(beta), eta * scale, 0.0)


def _sigma(s, shape):
    """Noise level per bin used by the shrinker, from `CONFIG`."""
    if CONFIG['mode'] == 'window':
        beta = min(shape) / max(shape)
        sig = np.median(s, axis=1) / np.sqrt(max(shape) * mp_median_cached(beta))
    else:
        sig = CONFIG['sigma_bin'][:s.shape[0]]
    return sig * CONFIG['scale']


def process_window_shrink(WAV_sl, it, ic, T_shape, scatter, r, imax, niter, gap_threshold, ppca_k, shrink_scale=None):
    """
    `ibldsp.cadzow._process_window` with Gavish-Donoho shrinkage (capped at rank r) instead of the hard rank cut.

    Same signature; ``gap_threshold`` is ignored.
    """
    def lowrank(T_batch):
        U, s, Vh = cadzow._safe_svd(T_batch)
        s_ = gd_shrink(s, _sigma(s, T_shape), T_shape)
        s_[:, r:] = 0.0
        return (U * s_[:, np.newaxis, :]) @ Vh

    WAV_ = WAV_sl.copy()
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for _ in range(niter):
            T_batch = np.zeros((imax, *T_shape), dtype=complex)
            T_batch[:, it[0], it[1]] = WAV_[ic, :imax].T
            T_batch_ = lowrank(T_batch)
            if ppca_k is not None:
                WAV_rec = (T_batch_[:, it[0], it[1]] @ scatter).T
                residual = np.abs(WAV_[:, :imax] - WAV_rec)
                med = np.median(residual, axis=0)
                mad = np.median(np.abs(residual - med[np.newaxis, :]), axis=0)
                mask = residual > (med + ppca_k * mad)
                WAV_clean = WAV_[:, :imax].copy()
                WAV_clean[mask] = WAV_rec[mask]
                T_batch[:, it[0], it[1]] = WAV_clean[ic, :].T
                T_batch_ = lowrank(T_batch)
            WAV_new = WAV_.copy()
            WAV_new[:, :imax] = (T_batch_[:, it[0], it[1]] @ scatter).T
            WAV_ = WAV_new
    return WAV_


def noise_per_bin(wav, h):
    """
    Per-channel white-noise level per frequency bin, from the same-column second difference of the real channels.

    For spatially white noise of level sigma, the second difference x[i-1] - 2 x[i] + x[i+1] along a column has
    variance 6 sigma^2; the smooth LFP is mostly cancelled. |d|^2 is exponential for complex Gaussian noise, so
    sigma^2 = median(|d|^2) / (6 ln 2).

    Parameters
    ----------
    wav : ndarray (nc, ns)
    h : dict
        Probe geometry with 'x' and 'y' [µm].

    Returns
    -------
    ndarray (ns // 2 + 1,)
    """
    W = scipy.fft.rfft(wav)
    d = []
    for xc in np.unique(h['x']):
        i = np.where(h['x'] == xc)[0]
        i = i[np.argsort(h['y'][i])]
        d.append(W[i[:-2]] - 2 * W[i[1:-1]] + W[i[2:]])
    d = np.concatenate(d)
    return np.sqrt(np.median(np.abs(d) ** 2, axis=0) / (6 * np.log(2)))


def cadzow_shrink(wav, h, mode='window', scale=1.0, **kwargs):
    """
    Full-grid Cadzow with Gavish-Donoho shrinkage.

    Parameters
    ----------
    wav : ndarray (nc, ns)
        LFP [V].
    h : dict
        Probe geometry with 'x' and 'y' [µm].
    mode : str
        'window' (noise from each window's median singular value) or 'chunk' (noise per bin from the whole chunk).
    scale : float
        Factor on the noise estimate (calibrated on white noise by `calibrate_scale`).
    **kwargs
        Forwarded to `cadzow_full_grid` / `cadzow_denoiser`.

    Returns
    -------
    ndarray (nc, ns)
    """
    CONFIG.update(mode=mode, scale=scale, sigma_bin=noise_per_bin(wav, h) if mode == 'chunk' else None)
    with mock.patch.object(cadzow, '_process_window', process_window_shrink):
        return cadzow_full_grid(wav, h, **kwargs)


def calibrate_scale(h, mode, nswx=64, ns=2048, q=99, seed=0):
    """
    Noise-scale factor so that the q-th percentile of the largest singular value of pure white noise sits on the
    shrinker's bulk edge sigma sqrt(max(m, n)) (1 + sqrt(beta)).

    Parameters
    ----------
    h : dict
        Probe geometry with 'x' and 'y' [µm].
    mode : str
        'window' or 'chunk'.
    nswx : int
        Window width in real channels.
    ns : int
        Number of samples of the noise realisation.
    q : float
        Percentile of the largest noise singular value.
    seed : int

    Returns
    -------
    float
    """
    wav = np.random.default_rng(seed).standard_normal((h['x'].size, ns))
    hg, ireal = full_grid(h)
    WAV = scipy.fft.rfft(fill_virtual(wav, hg, ireal))[:, 1:]  # drop DC
    sigma_chunk = noise_per_bin(wav, h)[1:]
    ratios = []
    for first in range(0, hg['x'].size - 2 * nswx + 1, nswx):
        sl = slice(first, first + 2 * nswx)
        T, it, ic, _ = cadzow.trajectory(hg['x'][sl], hg['y'][sl])
        T_batch = np.zeros((WAV.shape[1], *T.shape), dtype=complex)
        T_batch[:, it[0], it[1]] = WAV[sl][ic].T
        s = np.linalg.svd(T_batch, compute_uv=False)
        CONFIG.update(mode=mode, scale=1.0, sigma_bin=sigma_chunk)
        edge = _sigma(s, T.shape) * np.sqrt(max(T.shape)) * (1 + np.sqrt(min(T.shape) / max(T.shape)))
        ratios.append(s[:, 0] / edge)
    return float(np.percentile(np.concatenate(ratios), q))
