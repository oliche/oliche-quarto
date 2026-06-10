"""
Benchmark 6 LFP compression methods on Cadzow-denoised snippets.

Methods: SVD, wavelet packets, DCT, DCT-delta, STFT, SVD+wavelet packets.
Saves per-PID metric caches and one figure per PID.
"""
# %%
import sys
from pathlib import Path
sys.path.insert(0, '/Users/olivier/Documents/oliche-quarto/analyses/2026-06-lfp-compression')

import numpy as np
import scipy.fft
import scipy.signal
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pywt
import seaborn as sns
import addcopyfighandler  # noqa: F401
from one.api import ONE

from compress_fcns import load_pid

sns.set_theme(context='notebook')

ROOT_OUTPUT = Path('/Users/olivier/scratch/lfp')
FIGURE_DIR = Path('/Users/olivier/scratch/lfp/compression')
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

Q = 10
FS_RS = 250.0  # sr.fs / Q

RANKS = [1, 2, 4, 8, 16, 32, 64, 96]
KEEP_FRACS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
R_FIXED = 16

NAMES = ['SVD', 'Wavelet packets', 'DCT', 'DCT-δ', 'STFT', 'SVD+wavelets']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

pids = [
    '1a276285-8b0e-4cc9-9f0a-a3a002978724',
    '1e104bf4-7a24-4624-a5b2-c2c8289c0de7',
    '6638cfb3-3831-4fc2-9327-194b76cf22e1',
    '749cb2b7-e57e-4453-a794-f6230e4d0226',
    'd7ec0892-0a6c-4f4f-9d8f-72083692af5c',
    'da8dfec1-d265-44e8-84ce-6ae9c109b8bd',
    'dab512bd-a02d-4c1f-8dbc-9155a163efc0',
    'dc7e9403-19f7-409f-9240-05ee57cb7aea',
    'e8f9fba4-d151-4b00-bee7-447f0f3e752c',
    'eebcaf65-7fa4-4118-869d-a084e84530e2',
    'fe380793-8035-414e-b000-09bfe5ece92a',
]


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------

def compute_rmse_snr(snippets, reconstructions):
    """
    Compute per-channel RMSE and SNR across compression levels.

    Parameters
    ----------
    snippets : list of ndarray, shape (nc, ns), float32
        Original denoised snippets.
    reconstructions : list of list of ndarray
        ``reconstructions[k][s]`` shape ``(nc, ns)`` for level k, snippet s.

    Returns
    -------
    rmse : ndarray (nc, n_levels), float32
        RMSE averaged across snippets [µV].
    snr : ndarray (nc, n_levels), float32
        20·log10(RMS / RMSE) [dB].
    """
    n_levels = len(reconstructions)
    n_snip = len(snippets)
    nc = snippets[0].shape[0]

    rms = np.mean(
        [np.sqrt(np.mean(s.astype(np.float64) ** 2, axis=-1)) for s in snippets], axis=0
    )  # (nc,)

    rmse = np.zeros((nc, n_levels), dtype=np.float64)
    for k, recon_k in enumerate(reconstructions):
        rmse[:, k] = np.mean(
            [
                np.sqrt(np.mean(
                    (snippets[i].astype(np.float64) - recon_k[i].astype(np.float64)) ** 2,
                    axis=-1,
                ))
                for i in range(n_snip)
            ],
            axis=0,
        )

    snr = 20.0 * np.log10(rms[:, np.newaxis] / np.maximum(rmse, 1e-12))
    return rmse.astype(np.float32), snr.astype(np.float32)


# ---------------------------------------------------------------------------
# Compression functions
# ---------------------------------------------------------------------------

def compress_svd(snippets, ranks=RANKS):
    """
    SVD compression: X = U diag(s) Vh, keep top-r components.

    Parameters
    ----------
    snippets : list of ndarray, each (nc, ns), float32
    ranks : list of int

    Returns
    -------
    rmse : ndarray (nc, n_levels), float32
    snr : ndarray (nc, n_levels), float32
    cr : ndarray (n_levels,), float64
    """
    nc, ns = snippets[0].shape
    svds = [np.linalg.svd(s.astype(np.float64), full_matrices=False) for s in snippets]

    reconstructions = []
    for r in ranks:
        recon_r = [(U[:, :r] * sv[:r]) @ Vh[:r, :] for U, sv, Vh in svds]
        reconstructions.append([x.astype(np.float32) for x in recon_r])

    rmse, snr = compute_rmse_snr(snippets, reconstructions)
    cr = np.array([(nc * ns) / (r * (nc + ns)) for r in ranks])
    return rmse, snr, cr


def compress_wavelet_packets(snippets, keep_fracs=KEEP_FRACS):
    """
    Wavelet packet compression with per-channel hard thresholding.

    Decomposes each channel to level 5 (db4), hard-thresholds leaf coefficients
    by the top-f fraction, then reconstructs.

    Parameters
    ----------
    snippets : list of ndarray, each (nc, ns), float32
    keep_fracs : list of float
        Fraction of coefficients to retain; CR ≈ 1/f.

    Returns
    -------
    rmse : ndarray (nc, n_levels), float32
    snr : ndarray (nc, n_levels), float32
    cr : ndarray (n_levels,), float64
    """
    nc, ns = snippets[0].shape
    reconstructions = [
        [np.zeros((nc, ns), dtype=np.float32) for _ in snippets]
        for _ in keep_fracs
    ]

    for s_idx, snip in enumerate(snippets):
        # Build WP trees once per channel; save leaf data and sorted magnitudes
        channel_wp = []
        for c in range(nc):
            wp = pywt.WaveletPacket(data=snip[c].astype(np.float64), wavelet='db4', maxlevel=5)
            nodes = wp.get_level(5, 'natural')
            orig_data = [node.data.copy() for node in nodes]
            flat_sorted = np.sort(np.abs(np.concatenate(orig_data)))[::-1]
            channel_wp.append((wp, nodes, orig_data, flat_sorted))

        for f_idx, f in enumerate(keep_fracs):
            for c in range(nc):
                wp, nodes, orig_data, flat_sorted = channel_wp[c]
                n_keep = max(1, int(f * len(flat_sorted)))
                thresh = flat_sorted[n_keep - 1]
                for node, orig in zip(nodes, orig_data):
                    node.data = orig * (np.abs(orig) >= thresh)
                reconstructions[f_idx][s_idx][c] = (
                    wp.reconstruct(update=True)[:ns].astype(np.float32)
                )

    rmse, snr = compute_rmse_snr(snippets, reconstructions)
    cr = np.array([1.0 / f for f in keep_fracs])
    return rmse, snr, cr


def compress_dct(snippets, keep_fracs=KEEP_FRACS, delta=False):
    """
    DCT (or DCT-delta) compression with per-channel hard thresholding.

    Parameters
    ----------
    snippets : list of ndarray, each (nc, ns), float32
    keep_fracs : list of float
    delta : bool
        If True, pre-difference before DCT and integrate after IDCT (DCT-δ).

    Returns
    -------
    rmse : ndarray (nc, n_levels), float32
    snr : ndarray (nc, n_levels), float32
    cr : ndarray (n_levels,), float64
    """
    nc, ns = snippets[0].shape
    reconstructions = [[None] * len(snippets) for _ in keep_fracs]

    for s_idx, snip in enumerate(snippets):
        x = snip.astype(np.float64)
        if delta:
            x = np.diff(x, prepend=x[:, :1], axis=-1)

        coefs = scipy.fft.dct(x, type=2, norm='ortho', axis=-1)  # (nc, ns)
        sorted_abs = np.sort(np.abs(coefs), axis=-1)[:, ::-1]    # descending per channel

        for f_idx, f in enumerate(keep_fracs):
            n_keep = max(1, int(f * ns))
            thresholds = sorted_abs[:, n_keep - 1]  # (nc,)
            coefs_thresh = coefs * (np.abs(coefs) >= thresholds[:, np.newaxis])
            x_hat = scipy.fft.idct(coefs_thresh, type=2, norm='ortho', axis=-1)
            if delta:
                x_hat = np.cumsum(x_hat, axis=-1)
            reconstructions[f_idx][s_idx] = x_hat.astype(np.float32)

    rmse, snr = compute_rmse_snr(snippets, reconstructions)
    cr = np.array([1.0 / f for f in keep_fracs])
    return rmse, snr, cr


def compress_stft(snippets, keep_fracs=KEEP_FRACS, fs=FS_RS):
    """
    STFT thresholding: hard-threshold time-frequency bins per channel.

    Parameters
    ----------
    snippets : list of ndarray, each (nc, ns), float32
    keep_fracs : list of float
    fs : float
        Sampling rate [Hz].

    Returns
    -------
    rmse : ndarray (nc, n_levels), float32
    snr : ndarray (nc, n_levels), float32
    cr : ndarray (n_levels,), float64
    """
    nc, ns = snippets[0].shape
    reconstructions = [
        [np.zeros((nc, ns), dtype=np.float32) for _ in snippets]
        for _ in keep_fracs
    ]

    for s_idx, snip in enumerate(snippets):
        for c in range(nc):
            _, _, Zxx = scipy.signal.stft(
                snip[c].astype(np.float64), fs=fs, nperseg=64, noverlap=48
            )
            sorted_abs = np.sort(np.abs(Zxx.ravel()))[::-1]

            for f_idx, f in enumerate(keep_fracs):
                n_keep = max(1, int(f * len(sorted_abs)))
                thresh = sorted_abs[n_keep - 1]
                Zxx_thresh = Zxx * (np.abs(Zxx) >= thresh)
                _, x_hat = scipy.signal.istft(Zxx_thresh, fs=fs, nperseg=64, noverlap=48)
                reconstructions[f_idx][s_idx][c] = x_hat[:ns].astype(np.float32)

    rmse, snr = compute_rmse_snr(snippets, reconstructions)
    cr = np.array([1.0 / f for f in keep_fracs])
    return rmse, snr, cr


def compress_svd_wavelets(snippets, keep_fracs=KEEP_FRACS, r_fixed=R_FIXED):
    """
    Two-stage SVD + wavelet packet compression.

    Stage 1: SVD at fixed rank r_fixed (spatial compression).
    Stage 2: wavelet packet thresholding on each of the r_fixed temporal components.

    Parameters
    ----------
    snippets : list of ndarray, each (nc, ns), float32
    keep_fracs : list of float
        Fraction of wavelet coefficients retained in stage 2.
    r_fixed : int
        Fixed SVD rank for stage 1.

    Returns
    -------
    rmse : ndarray (nc, n_levels), float32
    snr : ndarray (nc, n_levels), float32
    cr : ndarray (n_levels,), float64
    """
    nc, ns = snippets[0].shape
    stage1_cr = (nc * ns) / (r_fixed * (nc + ns))
    reconstructions = [[None] * len(snippets) for _ in keep_fracs]

    for s_idx, snip in enumerate(snippets):
        U, s_vals, Vh = np.linalg.svd(snip.astype(np.float64), full_matrices=False)
        U_r = U[:, :r_fixed]    # (nc, r_fixed)
        s_r = s_vals[:r_fixed]  # (r_fixed,)
        Vh_r = Vh[:r_fixed, :]  # (r_fixed, ns)

        # Build WP trees once per temporal component
        comp_wp = []
        for k in range(r_fixed):
            wp = pywt.WaveletPacket(data=Vh_r[k], wavelet='db4', maxlevel=5)
            nodes = wp.get_level(5, 'natural')
            orig_data = [node.data.copy() for node in nodes]
            flat_sorted = np.sort(np.abs(np.concatenate(orig_data)))[::-1]
            comp_wp.append((wp, nodes, orig_data, flat_sorted))

        for f_idx, f in enumerate(keep_fracs):
            Vh_hat = np.zeros_like(Vh_r)
            for k in range(r_fixed):
                wp, nodes, orig_data, flat_sorted = comp_wp[k]
                n_keep = max(1, int(f * len(flat_sorted)))
                thresh = flat_sorted[n_keep - 1]
                for node, orig in zip(nodes, orig_data):
                    node.data = orig * (np.abs(orig) >= thresh)
                Vh_hat[k] = wp.reconstruct(update=True)[:ns]
            x_hat = (U_r * s_r) @ Vh_hat
            reconstructions[f_idx][s_idx] = x_hat.astype(np.float32)

    rmse, snr = compute_rmse_snr(snippets, reconstructions)
    cr = np.array([stage1_cr / f for f in keep_fracs])
    return rmse, snr, cr


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

# Column layout: (col_idx -> list of (method_idx, linestyle))
_PANEL_CURVES = [
    [(0, '-')],              # col 0: SVD
    [(1, '-'), (5, '--')],   # col 1: Wavelet pkts + SVD+wavelets
    [(2, '-'), (3, '--')],   # col 2: DCT + DCT-δ
    [(4, '-')],              # col 3: STFT
]
_COL_TITLES = ['SVD', 'Wavelet pkts', 'DCT / DCT-δ', 'STFT']


def plot_compression_figure(rmse_all, snr_all, cr_all, names, pid):
    """
    Two-row × four-column compression metrics figure.

    Parameters
    ----------
    rmse_all : ndarray (6,) of object, each element (nc, n_levels)
    snr_all : ndarray (6,) of object, each element (nc, n_levels)
    cr_all : ndarray (6,) of object, each element (n_levels,)
    names : list of str
    pid : str
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey='row')
    fig.suptitle(pid, fontsize=9)

    for row, metric_all in enumerate([rmse_all, snr_all]):
        scale = 1e6 if row == 0 else 1.0       # Volts → µV for RMSE row
        ylabel = 'RMSE (µV)' if row == 0 else 'SNR (dB)'
        for col, curves in enumerate(_PANEL_CURVES):
            ax = axes[row, col]
            for method_idx, ls in curves:
                data = metric_all[method_idx] * scale   # (nc, n_levels_i)
                cr = cr_all[method_idx]                 # (n_levels_i,)
                color = COLORS[method_idx]
                nc = data.shape[0]
                for c in range(nc):
                    ax.plot(cr, data[c], color=color, alpha=0.15, lw=0.5)
                ax.plot(
                    cr, np.median(data, axis=0),
                    color=color, lw=2, linestyle=ls, label=names[method_idx],
                )
            ax.set_xscale('log')
            ax.set_xlim([1.5, 1200])
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel('Compression ratio')
            if row == 0:
                ax.set_title(_COL_TITLES[col])

    # Legend in top-right panel only
    legend_handles = [
        mlines.Line2D(
            [], [], color=COLORS[i], lw=2,
            linestyle='--' if i in [3, 5] else '-',
            label=NAMES[i],
        )
        for i in range(6)
    ]
    axes[0, 3].legend(handles=legend_handles, fontsize=8, loc='upper right')

    fig.tight_layout()
    fig_path = FIGURE_DIR.joinpath(f'2026-06-06_compression_metrics_{pid}.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'{pid}: figure saved → {fig_path}')


# ---------------------------------------------------------------------------
# Gallery helpers
# ---------------------------------------------------------------------------

VMAX_GALLERY = 0.25e-3   # V — matches compress_lfp_main.py
DISPLAY_NS = 250          # 1 s at 250 Hz
TARGET_CRS = [1, 3, 10, 30, 100, 300, 1000]


def _lf_imshow(ax, data, vmax=VMAX_GALLERY):
    """Display a (nc, ns) LFP array as imshow with shared style."""
    ax.imshow(
        data, aspect='auto', cmap='RdBu_r',
        vmin=-vmax, vmax=vmax, interpolation='none', origin='lower',
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _achievable_crs(method_name, nc, ns, r_fixed=R_FIXED):
    """
    Return (params, crs) arrays for a given method.

    Parameters
    ----------
    method_name : str
    nc, ns : int
        Snippet dimensions.

    Returns
    -------
    params : ndarray
        Rank values (SVD) or keep_frac values (others).
    crs : ndarray
        Corresponding compression ratios.
    """
    if method_name == 'SVD':
        params = np.array(RANKS, dtype=float)
        crs = np.array([(nc * ns) / (r * (nc + ns)) for r in RANKS])
    elif method_name in ['Wavelet packets', 'DCT', 'DCT-δ', 'STFT']:
        params = np.array(KEEP_FRACS)
        crs = 1.0 / params
    else:  # SVD+wavelets
        stage1_cr = (nc * ns) / (R_FIXED * (nc + ns))
        params = np.array(KEEP_FRACS)
        crs = stage1_cr / params
    return params, crs


def _param_for_target_cr(method_name, cr_target, nc, ns, r_fixed=R_FIXED):
    """
    Compute the compression parameter that exactly hits cr_target.

    For SVD the rank is rounded to the nearest integer and clamped to [1, max(RANKS)].
    For transform methods keep_frac = 1 / cr_target.
    For SVD+wavelets keep_frac = stage1_cr / cr_target.

    Returns
    -------
    param : float
    achieved_cr : float
    """
    if method_name == 'SVD':
        r = nc * ns / (cr_target * (nc + ns))
        r = int(max(1, min(max(RANKS), round(r))))
        achieved_cr = nc * ns / (r * (nc + ns))
        return float(r), achieved_cr
    elif method_name == 'SVD+wavelets':
        stage1_cr = nc * ns / (r_fixed * (nc + ns))
        f = max(1e-6, min(1.0, stage1_cr / cr_target))
        return f, stage1_cr / f
    else:  # Wavelet packets, DCT, DCT-δ, STFT
        f = max(1e-6, min(1.0, 1.0 / cr_target))
        return f, 1.0 / f


def reconstruct_snippet(snip, method_name, param, fs=FS_RS, r_fixed=R_FIXED):
    """
    Reconstruct a single LFP snippet at a given compression parameter.

    Parameters
    ----------
    snip : ndarray (nc, ns), float32
    method_name : str
        One of NAMES.
    param : float
        Rank (SVD) or keep_frac (all other methods).
    fs : float
        Sampling rate [Hz].
    r_fixed : int
        SVD rank for the SVD+wavelets stage-1 step.

    Returns
    -------
    x_hat : ndarray (nc, ns), float64
    """
    nc, ns = snip.shape
    x = snip.astype(np.float64)

    if method_name == 'SVD':
        r = int(param)
        U, s, Vh = np.linalg.svd(x, full_matrices=False)
        return (U[:, :r] * s[:r]) @ Vh[:r, :]

    elif method_name == 'Wavelet packets':
        x_hat = np.zeros((nc, ns))
        for c in range(nc):
            wp = pywt.WaveletPacket(data=x[c], wavelet='db4', maxlevel=5)
            nodes = wp.get_level(5, 'natural')
            all_coefs = np.concatenate([node.data for node in nodes])
            n_keep = max(1, int(param * len(all_coefs)))
            thresh = np.sort(np.abs(all_coefs))[::-1][n_keep - 1]
            for node in nodes:
                node.data = node.data * (np.abs(node.data) >= thresh)
            x_hat[c] = wp.reconstruct(update=True)[:ns]
        return x_hat

    elif method_name == 'DCT':
        coefs = scipy.fft.dct(x, type=2, norm='ortho', axis=-1)
        n_keep = max(1, int(param * ns))
        thresholds = np.sort(np.abs(coefs), axis=-1)[:, ::-1][:, n_keep - 1]
        coefs_thresh = coefs * (np.abs(coefs) >= thresholds[:, np.newaxis])
        return scipy.fft.idct(coefs_thresh, type=2, norm='ortho', axis=-1)

    elif method_name == 'DCT-δ':
        dx = np.diff(x, prepend=x[:, :1], axis=-1)
        coefs = scipy.fft.dct(dx, type=2, norm='ortho', axis=-1)
        n_keep = max(1, int(param * ns))
        thresholds = np.sort(np.abs(coefs), axis=-1)[:, ::-1][:, n_keep - 1]
        coefs_thresh = coefs * (np.abs(coefs) >= thresholds[:, np.newaxis])
        return np.cumsum(scipy.fft.idct(coefs_thresh, type=2, norm='ortho', axis=-1), axis=-1)

    elif method_name == 'STFT':
        x_hat = np.zeros((nc, ns))
        for c in range(nc):
            _, _, Zxx = scipy.signal.stft(x[c], fs=fs, nperseg=64, noverlap=48)
            sorted_abs = np.sort(np.abs(Zxx.ravel()))[::-1]
            n_keep = max(1, int(param * len(sorted_abs)))
            thresh = sorted_abs[n_keep - 1]
            _, xc = scipy.signal.istft(Zxx * (np.abs(Zxx) >= thresh), fs=fs, nperseg=64, noverlap=48)
            x_hat[c] = xc[:ns]
        return x_hat

    elif method_name == 'SVD+wavelets':
        U, s_vals, Vh = np.linalg.svd(x, full_matrices=False)
        U_r = U[:, :r_fixed]
        s_r = s_vals[:r_fixed]
        Vh_r = Vh[:r_fixed, :]
        Vh_hat = np.zeros_like(Vh_r)
        for k in range(r_fixed):
            wp = pywt.WaveletPacket(data=Vh_r[k], wavelet='db4', maxlevel=5)
            nodes = wp.get_level(5, 'natural')
            all_coefs = np.concatenate([node.data for node in nodes])
            n_keep = max(1, int(param * len(all_coefs)))
            thresh = np.sort(np.abs(all_coefs))[::-1][n_keep - 1]
            for node in nodes:
                node.data = node.data * (np.abs(node.data) >= thresh)
            Vh_hat[k] = wp.reconstruct(update=True)[:ns]
        return (U_r * s_r) @ Vh_hat

    return x.copy()


def plot_compression_gallery(snippets, pid, residual=False):
    """
    LFP compression gallery: methods as rows, target CRs as columns.

    Shows the first snippet, first second of data. CR=1 column shows the original.
    Rows: SVD | Wavelet packets | DCT | DCT-δ | STFT | SVD+wavelets
    Cols: TARGET_CRS = [1, 3, 10, 30, 100, 300, 1000]

    Parameters
    ----------
    snippets : list of ndarray, each (nc, ns), float32
    pid : str
    residual : bool
        If True, display ``original − reconstruction`` with the same colorscale as
        the signal to reveal error structure.
    """
    snip = snippets[0]
    nc, ns = snip.shape
    orig = snip[:, :DISPLAY_NS]

    vmax = VMAX_GALLERY
    kind_str = f'residuals  (vmax={vmax * 1e6:.0f} µV, same gain as signal)' if residual else 'reconstruction'

    n_rows, n_cols = len(NAMES), len(TARGET_CRS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 16))
    fig.suptitle(f'{pid}  —  {kind_str}', fontsize=12)

    # Column headers: CR targets
    for j, cr_target in enumerate(TARGET_CRS):
        axes[0, j].set_title(f'CR = {cr_target}', fontsize=13, pad=4)

    for i, name in enumerate(NAMES):
        # Row label: method name
        axes[i, 0].set_ylabel(name, fontsize=13, rotation=90, labelpad=6,
                               va='center', color=COLORS[i])

        for j, cr_target in enumerate(TARGET_CRS):
            ax = axes[i, j]

            if cr_target == 1:
                if residual:
                    _lf_imshow(ax, np.zeros_like(orig), vmax=vmax)
                    ax.text(
                        0.5, 0.5, 'residual = 0',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, color='#888888',
                    )
                else:
                    _lf_imshow(ax, orig)
                continue

            param, achieved_cr = _param_for_target_cr(name, cr_target, nc, ns)

            if achieved_cr < cr_target / 2:
                _, max_cr = _param_for_target_cr(name, 2.0, nc, ns)
                ax.set_facecolor('#ebebeb')
                ax.text(
                    0.5, 0.5, f'N/A\n(max CR≈{int(max_cr)})',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color='#777777',
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            x_hat = reconstruct_snippet(snip, name, param).astype(np.float32)
            panel = (orig - x_hat[:, :DISPLAY_NS]) if residual else x_hat[:, :DISPLAY_NS]
            _lf_imshow(ax, panel, vmax=vmax)
            ax.text(
                0.02, 0.97, f'CR={achieved_cr:.0f}',
                transform=ax.transAxes, fontsize=9, va='top', ha='left', color='k',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, lw=0),
            )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fname = f'2026-06-06_compression_gallery_{pid}{"_residual" if residual else ""}.png'
    fig_path = FIGURE_DIR.joinpath(fname)
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f'{pid}: {"residuals" if residual else "gallery"} saved → {fig_path}')


def plot_aggregate_figure():
    """
    Single aggregate figure: one curve per method, averaged across all PIDs.

    Per PID: median RMSE / SNR over channels.  Across PIDs: mean.
    Produces a 1×2 figure (RMSE + SNR) saved to FIGURE_DIR.
    """
    from collections import defaultdict

    rmse_acc = defaultdict(list)
    snr_acc = defaultdict(list)
    cr_ref = {}

    for pid in pids:
        cache_file = ROOT_OUTPUT.joinpath(pid, f'metrics_{pid}.npz')
        if not cache_file.exists():
            continue
        d = np.load(cache_file, allow_pickle=True)
        for i, name in enumerate(d['names']):
            rmse_acc[name].append(np.median(d['rmse'][i], axis=0))  # (n_levels,)
            snr_acc[name].append(np.median(d['snr'][i], axis=0))
            if name not in cr_ref:
                cr_ref[name] = d['cr'][i]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Aggregate — median over channels, mean over PIDs  '
        f'(n={len([p for p in pids if ROOT_OUTPUT.joinpath(p, f"metrics_{p}.npz").exists()])})',
        fontsize=10,
    )

    for i, name in enumerate(NAMES):
        color = COLORS[i]
        ls = '--' if name in ['DCT-δ', 'SVD+wavelets'] else '-'
        cr = cr_ref[name]
        rmse_curve = np.mean(rmse_acc[name], axis=0) * 1e6   # V → µV
        snr_curve = np.mean(snr_acc[name], axis=0)
        axes[0].plot(cr, rmse_curve, color=color, lw=2.5, linestyle=ls, label=name)
        axes[1].plot(cr, snr_curve, color=color, lw=2.5, linestyle=ls, label=name)

    for ax, ylabel, title in zip(axes, ['RMSE (µV)', 'SNR (dB)'], ['RMSE', 'SNR']):
        ax.set_xscale('log')
        ax.set_xlim([1.5, 1200])
        ax.set_xlabel('Compression ratio')
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    axes[0].legend(fontsize=9, loc='upper left')

    fig.tight_layout()
    fig_path = FIGURE_DIR.joinpath('2026-06-06_compression_aggregate.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Aggregate figure saved → {fig_path}')


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

one = ONE(mode='remote', base_url='https://alyx.internationalbrainlab.org')

# %%
for pid in pids:
    ssl, sr, output_path, fs_rs = load_pid(pid, one, ROOT_OUTPUT, stream=False, q=Q)
    nc = sr.nc - sr.nsync

    snippet_files = [output_path.joinpath(f'cadzow_denoised_{i}.npy') for i in range(3)]
    if not all(f.exists() for f in snippet_files):
        print(f'{pid}: missing snippet files, skipping')
        continue

    snippets = [np.load(f) for f in snippet_files]

    cache_file = output_path.joinpath(f'metrics_{pid}.npz')
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        rmse_all = data['rmse']
        snr_all = data['snr']
        cr_all = data['cr']
        names = list(data['names'])
        print(f'{pid}: loaded from cache')
    else:
        print(f'{pid}: SVD...')
        rmse_svd, snr_svd, cr_svd = compress_svd(snippets, ranks=RANKS)
        print(f'{pid}: wavelet packets...')
        rmse_wp, snr_wp, cr_wp = compress_wavelet_packets(snippets, keep_fracs=KEEP_FRACS)
        print(f'{pid}: DCT...')
        rmse_dct, snr_dct, cr_dct = compress_dct(snippets, keep_fracs=KEEP_FRACS, delta=False)
        print(f'{pid}: DCT-δ...')
        rmse_dctd, snr_dctd, cr_dctd = compress_dct(snippets, keep_fracs=KEEP_FRACS, delta=True)
        print(f'{pid}: STFT...')
        rmse_stft, snr_stft, cr_stft = compress_stft(snippets, keep_fracs=KEEP_FRACS, fs=fs_rs)
        print(f'{pid}: SVD+wavelets...')
        rmse_sw, snr_sw, cr_sw = compress_svd_wavelets(
            snippets, keep_fracs=KEEP_FRACS, r_fixed=R_FIXED
        )

        rmse_all = np.empty(6, dtype=object)
        snr_all = np.empty(6, dtype=object)
        cr_all = np.empty(6, dtype=object)
        for i, (rm, sn, cr) in enumerate(zip(
            [rmse_svd, rmse_wp, rmse_dct, rmse_dctd, rmse_stft, rmse_sw],
            [snr_svd, snr_wp, snr_dct, snr_dctd, snr_stft, snr_sw],
            [cr_svd, cr_wp, cr_dct, cr_dctd, cr_stft, cr_sw],
        )):
            rmse_all[i] = rm
            snr_all[i] = sn
            cr_all[i] = cr

        names = NAMES
        np.savez(cache_file, rmse=rmse_all, snr=snr_all, cr=cr_all, names=np.array(names))
        print(f'{pid}: cache saved → {cache_file}')

    plot_compression_figure(rmse_all, snr_all, cr_all, names, pid)
    plot_compression_gallery(snippets, pid)
    plot_compression_gallery(snippets, pid, residual=True)

# %%
plot_aggregate_figure()
