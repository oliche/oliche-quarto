"""
Benchmark 6 LFP compression methods on Cadzow-denoised snippets.

Methods: SVD, SVD-adapt, SVD-adapt-CR12/16/24/32-WP.
Saves per-PID metric caches (metrics_v3_{pid}.npz) and one figure set per PID.
"""
# %%
import sys
from pathlib import Path
sys.path.insert(0, '/Users/olivier/Documents/oliche-quarto/analyses/2026-06-lfp-compression')

import numpy as np
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
# Epsilon sweep for SVD-adapt: high epsilon → high threshold → few components → high CR.
# Noise floor is computed from non-trivial SVs only (sv > 0.01% of max); on Cadzow-denoised
# snippets the effective signal SVs span ~4579× the noise floor, so eps must reach ~10^4 to
# collapse to rank=1.  30 log-spaced points from 10^4 to 10^-1 cover rank 1→full rank.
EPSILONS = np.logspace(4, -1, 30)
# Gap threshold sweep: rank = last position where sv[k]/sv[k+1] >= g, fallback to 1.
# Two-segment grid: coarse for dominant gap region (>1.5), fine for secondary gaps (1.001–1.5).
# High g → rank=1 (CR≈254); low g → keeps SVs past secondary gaps (rank up to noise floor).
GAP_THRESHOLDS = np.unique(np.concatenate([
    np.logspace(3, np.log10(1.5), 25),
    np.logspace(np.log10(1.5), np.log10(1.001), 35),
]))[::-1]  # descending order (high → low threshold)
SVD_ADAPT_CRS = [24, 32, 48]

QUARTO_FIGURE_DIR = Path('/Users/olivier/Documents/oliche-quarto/analyses/2026-06-lfp-compression-svd/figures')

NAMES = [
    'SVD', 'SVD-adapt',
    *[f'SVD-adapt-CR{cr}-WP' for cr in SVD_ADAPT_CRS],
]
COLORS = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']

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


def align_to_cr_grid(rmse, snr, cr_src, cr_tgt):
    """
    Resample per-channel metrics from cr_src onto cr_tgt via log-CR interpolation.

    Points in cr_tgt outside the range of cr_src are set to NaN (e.g. CRs
    that the adaptive method cannot reach because the natural gap rank is too low).

    Parameters
    ----------
    rmse : ndarray (nc, n_src)
    snr : ndarray (nc, n_src)
    cr_src : ndarray (n_src,)
        Source CR values; may be unsorted or contain duplicates from plateau behaviour.
    cr_tgt : ndarray (n_tgt,)
        Target CR grid (vanilla SVD's CRs, ascending).

    Returns
    -------
    rmse_out : ndarray (nc, n_tgt), float32
    snr_out : ndarray (nc, n_tgt), float32
    """
    order = np.argsort(cr_src)
    log_src = np.log(cr_src[order])
    log_tgt = np.log(cr_tgt)
    in_range = (log_tgt >= log_src[0]) & (log_tgt <= log_src[-1])
    nc = rmse.shape[0]

    rmse_out = np.full((nc, len(cr_tgt)), np.nan, dtype=np.float32)
    snr_out = np.full((nc, len(cr_tgt)), np.nan, dtype=np.float32)

    if in_range.any():
        for c in range(nc):
            rmse_out[c, in_range] = np.interp(log_tgt[in_range], log_src, rmse[c, order])
            snr_out[c, in_range] = np.interp(log_tgt[in_range], log_src, snr[c, order])

    return rmse_out, snr_out


# ---------------------------------------------------------------------------
# Wavelet-packet helper (used by two methods)
# ---------------------------------------------------------------------------

def _wp_compress_rows(rows, keep_frac, ns):
    """
    Wavelet-packet compress a set of 1-D temporal rows (in-place reconstruction).

    Parameters
    ----------
    rows : ndarray (r, ns), float64
        Temporal components to compress.
    keep_frac : float
        Fraction of wavelet coefficients to retain.
    ns : int
        Original signal length (for truncation after reconstruction).

    Returns
    -------
    out : ndarray (r, ns), float64
    """
    r = rows.shape[0]
    out = np.zeros((r, ns), dtype=np.float64)
    for k in range(r):
        wp = pywt.WaveletPacket(data=rows[k], wavelet='db4', maxlevel=5)
        nodes = wp.get_level(5, 'natural')
        all_coefs = np.concatenate([node.data for node in nodes])
        n_keep = max(1, int(keep_frac * len(all_coefs)))
        thresh = np.sort(np.abs(all_coefs))[::-1][n_keep - 1]
        for node in nodes:
            node.data = node.data * (np.abs(node.data) >= thresh)
        out[k] = wp.reconstruct(update=True)[:ns]
    return out


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


def compress_svd_adapt(snippets, epsilons=EPSILONS):
    """
    Adaptive-threshold SVD: rank determined by epsilon * noise_floor.

    For each epsilon, threshold = epsilon × median(lower half of singular values);
    components above threshold are kept (minimum 1).  The CR axis uses the mean
    rank across snippets.  Results are returned sorted by ascending CR.

    Parameters
    ----------
    snippets : list of ndarray, each (nc, ns), float32
    epsilons : array-like
        Threshold multipliers; larger values keep fewer components (higher CR).

    Returns
    -------
    rmse : ndarray (nc, n_levels), float32
    snr : ndarray (nc, n_levels), float32
    cr : ndarray (n_levels,), float64
    epsilons_sorted : ndarray (n_levels,), float64
        Epsilon value for each level, in the same ascending-CR order as cr.
    """
    nc, ns = snippets[0].shape
    svds = [np.linalg.svd(s.astype(np.float64), full_matrices=False) for s in snippets]

    reconstructions, crs = [], np.zeros(len(epsilons))
    for k, eps in enumerate(epsilons):
        recon_eps, ranks_eps = [], []
        for U, sv, Vh in svds:
            # Exclude near-zero SVs from dead/zeroed channels before estimating the noise floor.
            # Using median(lower half) of all SVs returns ~0 when many channels are zeroed out,
            # making the threshold meaningless.  Restricting to sv > 0.01% of max SV gives a
            # noise floor that reflects actual signal components, so epsilon maps cleanly to rank.
            sv_nz = sv[sv > sv[0] * 1e-4]
            tail = sv_nz[sv_nz.size // 2:] if sv_nz.size else sv
            noise_floor = float(np.nanmedian(tail)) if tail.size else float(sv[0])
            r = int(max(1, np.sum(sv > eps * noise_floor)))
            ranks_eps.append(r)
            recon_eps.append(((U[:, :r] * sv[:r]) @ Vh[:r, :]).astype(np.float32))
        reconstructions.append(recon_eps)
        crs[k] = (nc * ns) / (np.mean(ranks_eps) * (nc + ns))

    rmse, snr = compute_rmse_snr(snippets, reconstructions)
    order = np.argsort(crs)
    return rmse[:, order], snr[:, order], crs[order], np.array(epsilons)[order]


def compress_svd_adapt_gap(snippets, gap_thresholds=None):
    """
    Gap-threshold SVD sweep: rank = last position where sv[k]/sv[k+1] >= g, plus 1.

    Sweeps the minimum gap ratio ``g`` from high (selects rank=1) to low (selects rank
    near noise floor).  For each threshold, every snippet independently selects its rank
    as the last SV index where the ratio to the next SV exceeds ``g``; falls back to
    rank=1 if no ratio qualifies.  Results are returned sorted by ascending CR.

    This replaces the old cap-sweep variant, which plateaued once the cap exceeded the
    single dominant gap rank (~1–3 on Cadzow-denoised data) and never explored higher ranks.

    Parameters
    ----------
    snippets : list of ndarray, each (nc, ns), float32
    gap_thresholds : array-like or None
        Minimum ratio sv[k]/sv[k+1] to count as a gap; defaults to ``GAP_THRESHOLDS``.

    Returns
    -------
    rmse : ndarray (nc, n_levels), float32
    snr : ndarray (nc, n_levels), float32
    cr : ndarray (n_levels,), float64
        Sorted ascending (high CR / low rank first).
    gap_thresholds_sorted : ndarray (n_levels,), float64
        Gap threshold for each level, in the same ascending-CR order as cr.
    """
    if gap_thresholds is None:
        gap_thresholds = GAP_THRESHOLDS
    nc, ns = snippets[0].shape
    svds = [np.linalg.svd(s.astype(np.float64), full_matrices=False) for s in snippets]

    reconstructions, crs = [], np.zeros(len(gap_thresholds))
    for k, g in enumerate(gap_thresholds):
        recon_k, ranks_k = [], []
        for U, sv, Vh in svds:
            sv_nz = sv[sv > sv[0] * 1e-4]
            if sv_nz.size > 1:
                ratios = sv_nz[:-1] / (sv_nz[1:] + 1e-10)
                qualified = np.where(ratios >= g)[0]
                # last position with ratio >= g → everything up to and including it is "signal"
                r = int(qualified[-1]) + 1 if qualified.size > 0 else 1
            else:
                r = 1
            ranks_k.append(r)
            recon_k.append(((U[:, :r] * sv[:r]) @ Vh[:r, :]).astype(np.float32))
        reconstructions.append(recon_k)
        crs[k] = (nc * ns) / (float(np.mean(ranks_k)) * (nc + ns))

    rmse, snr = compute_rmse_snr(snippets, reconstructions)
    order = np.argsort(crs)
    return rmse[:, order], snr[:, order], crs[order], np.array(gap_thresholds)[order]


def compress_svd_adapt_wp(snippets, initial_cr, keep_fracs=KEEP_FRACS):
    """
    Two-stage SVD + wavelet-packet compression.

    Stage 1: SVD at rank r_fixed derived from initial_cr (spatial compression).
    Stage 2: db4 wavelet-packet thresholding on the r_fixed temporal components.

    Parameters
    ----------
    snippets : list of ndarray, each (nc, ns), float32
    initial_cr : float
        Target CR for the SVD stage; r_fixed = round(nc*ns / (initial_cr*(nc+ns))).
    keep_fracs : list of float
        Fraction of wavelet coefficients retained in stage 2.

    Returns
    -------
    rmse : ndarray (nc, n_levels), float32
    snr : ndarray (nc, n_levels), float32
    cr : ndarray (n_levels,), float64
    """
    nc, ns = snippets[0].shape
    r_fixed = max(1, round(nc * ns / (initial_cr * (nc + ns))))
    stage1_cr = (nc * ns) / (r_fixed * (nc + ns))
    reconstructions = [[None] * len(snippets) for _ in keep_fracs]

    for s_idx, snip in enumerate(snippets):
        U, sv, Vh = np.linalg.svd(snip.astype(np.float64), full_matrices=False)
        Vh_r = Vh[:r_fixed, :]  # (r_fixed, ns)

        # pre-build WP trees and sorted coefs once per temporal component
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
            x_hat = (U[:, :r_fixed] * sv[:r_fixed]) @ Vh_hat
            reconstructions[f_idx][s_idx] = x_hat.astype(np.float32)

    rmse, snr = compute_rmse_snr(snippets, reconstructions)
    cr = np.array([stage1_cr / f for f in keep_fracs])
    return rmse, snr, cr


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Gallery helpers
# ---------------------------------------------------------------------------

VMAX_GALLERY = 0.25e-3   # V
DISPLAY_NS = 250          # 1 s at 250 Hz
TARGET_CRS = [1, 16, 32, 64, 128, 256, 512, 1024]


def _lf_imshow(ax, data, vmax=VMAX_GALLERY):
    """Display a (nc, ns) LFP array as imshow with shared style."""
    im = ax.imshow(
        data, aspect='auto', cmap='RdBu_r',
        vmin=-vmax, vmax=vmax, interpolation='none', origin='lower',
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def reconstruct_snippet(snip, method_name, cr_target):
    """
    Reconstruct a single LFP snippet at a given target CR.

    Parameters
    ----------
    snip : ndarray (nc, ns), float32
    method_name : str
        One of NAMES.
    cr_target : float
        Target compression ratio.

    Returns
    -------
    x_hat : ndarray (nc, ns), float64
    """
    nc, ns = snip.shape
    x = snip.astype(np.float64)
    U, sv, Vh = np.linalg.svd(x, full_matrices=False)

    if method_name == 'SVD':
        r = max(1, min(len(sv), round(nc * ns / (cr_target * (nc + ns)))))
        return (U[:, :r] * sv[:r]) @ Vh[:r, :]

    elif method_name == 'SVD-adapt':
        # epsilon-sweep variant: treat like fixed-rank SVD at the target CR
        r = max(1, min(len(sv), round(nc * ns / (cr_target * (nc + ns)))))
        return (U[:, :r] * sv[:r]) @ Vh[:r, :]

    elif method_name == 'SVD-adapt-gap':
        sv_nz = sv[sv > sv[0] * 1e-4]
        if sv_nz.size > 1:
            ratios = sv_nz[:-1] / (sv_nz[1:] + 1e-10)
            # threshold = ratio at the rank boundary for cr_target; last position >= that gives r
            r_target = max(1, min(len(ratios), round(nc * ns / (cr_target * (nc + ns)))))
            g = float(ratios[r_target - 1])
            qualified = np.where(ratios >= g)[0]
            r = int(qualified[-1]) + 1 if qualified.size > 0 else 1
        else:
            r = 1
        return (U[:, :r] * sv[:r]) @ Vh[:r, :]

    else:
        # 'SVD-adapt-CRxx-WP' — extract initial CR from name
        initial_cr = float(method_name.split('CR')[1].split('-')[0])
        r_fixed = max(1, round(nc * ns / (initial_cr * (nc + ns))))
        stage1_cr = (nc * ns) / (r_fixed * (nc + ns))
        if cr_target <= stage1_cr:
            # target is at or below stage-1 CR; show pure SVD reconstruction
            return (U[:, :r_fixed] * sv[:r_fixed]) @ Vh[:r_fixed, :]
        keep_frac = max(1e-6, min(1.0, stage1_cr / cr_target))
        Vh_r = Vh[:r_fixed, :]
        Vh_hat = _wp_compress_rows(Vh_r.copy(), keep_frac, ns)
        return (U[:, :r_fixed] * sv[:r_fixed]) @ Vh_hat


def plot_compression_gallery(snippets, pid, residual=False):
    """
    LFP compression gallery: methods as rows, target CRs as columns.

    Shows the first snippet, first second of data.  CR=1 column shows the original.
    Rows: 6 methods.  Cols: TARGET_CRS = [1, 16, 32, 64, 128, 256, 512, 1024].

    Parameters
    ----------
    snippets : list of ndarray, each (nc, ns), float32
    pid : str
    residual : bool
        If True, display ``original − reconstruction`` with the same colour scale.
    """
    snip = snippets[0]
    nc, ns = snip.shape
    orig = snip[:, :DISPLAY_NS]
    vmax = VMAX_GALLERY
    kind_str = f'residuals  (vmax={vmax * 1e6:.0f} µV)' if residual else 'reconstruction'
    # Physical max CR for the pure SVD family: rank-1 limit.
    max_svd_cr = (nc * ns) / (1 * (nc + ns))

    n_rows, n_cols = len(NAMES), len(TARGET_CRS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(32, 14))
    fig.suptitle(f'{pid}  —  {kind_str}', fontsize=13)

    for j, cr_target in enumerate(TARGET_CRS):
        axes[0, j].set_title(f'CR = {cr_target}', fontsize=14, pad=4)

    for i, name in enumerate(NAMES):
        axes[i, 0].set_ylabel(name, fontsize=13, rotation=90, labelpad=6,
                              va='center', color=COLORS[i])

        # precompute stage1_cr for WP methods so we can grey out unreachable CRs
        if 'WP' in name:
            initial_cr = float(name.split('CR')[1].split('-')[0])
            r_fixed = max(1, round(nc * ns / (initial_cr * (nc + ns))))
            stage1_cr = (nc * ns) / (r_fixed * (nc + ns))
        else:
            stage1_cr = None

        # precompute gap ratios for SVD-adapt-gap row
        if name == 'SVD-adapt-gap':
            _sv_gap = np.linalg.svd(snip.astype(np.float64), compute_uv=False)
            _sv_gap_nz = _sv_gap[_sv_gap > _sv_gap[0] * 1e-4]
            _gap_ratios = _sv_gap_nz[:-1] / (_sv_gap_nz[1:] + 1e-10) if _sv_gap_nz.size > 1 else np.array([])
        else:
            _gap_ratios = None

        # precompute noise floor for SVD-adapt row so each cell can label its ε
        if name == 'SVD-adapt':
            _sv_sa = np.linalg.svd(snip.astype(np.float64), compute_uv=False)
            _sv_sa_nz = _sv_sa[_sv_sa > _sv_sa[0] * 1e-4]
            _tail_sa = _sv_sa_nz[_sv_sa_nz.size // 2:]
            _noise_floor_sa = float(np.nanmedian(_tail_sa)) if _tail_sa.size else float(_sv_sa[0])
        else:
            _sv_sa, _noise_floor_sa = None, None

        for j, cr_target in enumerate(TARGET_CRS):
            ax = axes[i, j]

            if cr_target == 1:
                if residual:
                    _lf_imshow(ax, np.zeros_like(orig), vmax=vmax)
                    ax.text(0.5, 0.5, 'residual = 0', transform=ax.transAxes,
                            ha='center', va='center', fontsize=11, color='#888888')
                else:
                    _lf_imshow(ax, orig)
                continue

            # grey out columns where the combined CR target is below the SVD stage alone
            if stage1_cr is not None and cr_target <= stage1_cr:
                ax.set_facecolor('#ebebeb')
                ax.text(0.5, 0.5, f'N/A\n(min CR≈{stage1_cr:.0f})',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, color='#777777')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # grey out columns beyond the rank-1 limit for pure SVD family rows
            if stage1_cr is None and cr_target > max_svd_cr:
                ax.set_facecolor('#ebebeb')
                ax.text(0.5, 0.5, f'N/A\n(max CR≈{max_svd_cr:.0f})',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, color='#777777')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # compute achieved CRs for the label
            if stage1_cr is None:
                if name == 'SVD-adapt-gap':
                    r_target = max(1, min(nc - 1, round(nc * ns / (cr_target * (nc + ns)))))
                    if _gap_ratios is not None and r_target <= len(_gap_ratios):
                        g_used = float(_gap_ratios[r_target - 1])
                        qualified = np.where(_gap_ratios >= g_used)[0]
                        r_used = int(qualified[-1]) + 1 if qualified.size > 0 else 1
                    else:
                        g_used, r_used = 1.0, r_target
                    achieved_cr = nc * ns / (r_used * (nc + ns))
                    cell_label = f'g={g_used:.1f} r={r_used}\nCR={achieved_cr:.0f}'
                elif name == 'SVD-adapt':
                    r = max(1, min(nc - 1, round(nc * ns / (cr_target * (nc + ns)))))
                    achieved_cr = nc * ns / (r * (nc + ns))
                    eps_cell = float(_sv_sa[r] / _noise_floor_sa) if r < len(_sv_sa) else 0.0
                    cell_label = f'CR={achieved_cr:.0f}\nε≈{eps_cell:.1e}'
                else:  # plain SVD
                    r = max(1, min(nc - 1, round(nc * ns / (cr_target * (nc + ns)))))
                    achieved_cr = nc * ns / (r * (nc + ns))
                    cell_label = f'CR={achieved_cr:.0f}'
            else:  # WP method: combined CR = stage1_cr / keep_frac = cr_target (exact)
                wp_cr = cr_target / stage1_cr
                cell_label = f'CR={cr_target:.0f}\nSVD×{stage1_cr:.0f} · WP×{wp_cr:.0f}'

            x_hat = reconstruct_snippet(snip, name, cr_target).astype(np.float32)
            panel = (orig - x_hat[:, :DISPLAY_NS]) if residual else x_hat[:, :DISPLAY_NS]
            _lf_imshow(ax, panel, vmax=vmax)
            ax.text(0.02, 0.97, cell_label, transform=ax.transAxes,
                    fontsize=9, va='top', ha='left', color='k',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, lw=0))

    # thin horizontal colorbar — fixed scale, shared across all panels
    import matplotlib.colors as mcolors
    import matplotlib.cm as mplcm
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    sm = mplcm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    fig.subplots_adjust(top=0.95, bottom=0.06)
    cbar_ax = fig.add_axes([0.25, 0.015, 0.50, 0.012])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    ticks = np.linspace(-vmax, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t * 1e6:.0f}' for t in ticks])
    cbar.set_label('µV', labelpad=2)

    fname = f'2026-06-09_compression_gallery_{pid}{"_residual" if residual else ""}.png'
    fig_path = FIGURE_DIR.joinpath(fname)
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f'{pid}: {"residuals" if residual else "gallery"} saved → {fig_path}')


EPSILONS_GALLERY  = [150, 300, 450, 750]   # SVD-adapt ε: rank = #{sv > ε·σ_noise}
ALPHAS_GALLERY    = [0, 12, 24, 32, 48, 64, 96]
WP_WAVELET_GALLERY = 'db4'
WP_MAXLEVEL_GALLERY = 5


def _svd_noise_floor(sv):
    """Median of the lower half of non-trivial singular values."""
    sv_nz = sv[sv > sv[0] * 1e-4]
    tail = sv_nz[sv_nz.size // 2:] if sv_nz.size else sv
    return float(np.nanmedian(tail)) if tail.size else float(sv[0])


def _count_wp_slots_gallery(ns):
    """Total leaf WP coefficients for a signal of length ns (db4, level 5)."""
    wp = pywt.WaveletPacket(data=np.zeros(ns), wavelet=WP_WAVELET_GALLERY,
                            maxlevel=WP_MAXLEVEL_GALLERY)
    return sum(len(node.data) for node in wp.get_level(WP_MAXLEVEL_GALLERY, 'natural'))


def reconstruct_adaptive_wp(snip, cr_svd_target, alpha):
    """
    Reconstruct snippet: SVD at rank from cr_svd_target, then adaptive WP at threshold alpha.

    Parameters
    ----------
    snip : ndarray (nc, ns), float32
    cr_svd_target : float
        Target SVD-stage compression ratio; rank is rounded to nearest integer.
    alpha : float
        WP threshold multiplier; 0 → pure SVD (no WP).

    Returns
    -------
    x_hat : ndarray (nc, ns), float32
    cr_svd : float   Actual SVD-stage CR.
    cr_wp  : float   WP-stage CR (1.0 when alpha=0).
    cr_total : float Combined CR.
    """
    nc, ns = snip.shape
    r = max(1, round(nc * ns / (cr_svd_target * (nc + ns))))
    x = snip.astype(np.float64)
    U, sv, Vh = np.linalg.svd(x, full_matrices=False)

    cr_svd_actual = nc * ns / (r * (nc + ns))

    if alpha == 0:
        x_hat = ((U[:, :r] * sv[:r]) @ Vh[:r, :]).astype(np.float32)
        return x_hat, cr_svd_actual, 1.0, cr_svd_actual

    sigma_noise = _svd_noise_floor(sv)
    n_wp_slots  = _count_wp_slots_gallery(ns)

    Vh_hat = np.zeros((r, ns))
    n_kept = 0
    for k in range(r):
        tau_k = alpha * sigma_noise / (sv[k] + 1e-40)
        wp = pywt.WaveletPacket(data=Vh[k], wavelet=WP_WAVELET_GALLERY,
                                maxlevel=WP_MAXLEVEL_GALLERY)
        nodes = wp.get_level(WP_MAXLEVEL_GALLERY, 'natural')
        for node in nodes:
            mask = np.abs(node.data) >= tau_k
            n_kept += int(mask.sum())
            node.data = node.data * mask
        Vh_hat[k] = wp.reconstruct(update=True)[:ns]

    x_hat = np.nan_to_num(
        ((U[:, :r] * sv[:r]) @ Vh_hat).astype(np.float32),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    cr_wp    = (r * n_wp_slots) / max(n_kept, 1)
    cr_total = cr_svd_actual * cr_wp
    return x_hat, cr_svd_actual, cr_wp, cr_total


def reconstruct_adaptive_wp_eps(snip, eps, alpha):
    """
    Reconstruct snippet: rank selected by eps × noise_floor threshold, then WP at alpha.

    Parameters
    ----------
    snip : ndarray (nc, ns), float32
    eps : float
        SVD-adapt threshold multiplier; rank = #{k : sv[k] > eps · σ_noise}.
    alpha : float
        WP threshold multiplier; 0 → pure SVD (no WP).

    Returns
    -------
    x_hat : ndarray (nc, ns), float32
    r : int             Rank selected by eps.
    cr_svd : float      Actual SVD-stage CR.
    cr_wp : float       WP-stage CR (1.0 when alpha=0).
    cr_total : float    Combined CR.
    """
    nc, ns = snip.shape
    x = snip.astype(np.float64)
    U, sv, Vh = np.linalg.svd(x, full_matrices=False)
    sigma_noise = _svd_noise_floor(sv)
    r = max(1, int(np.sum(sv > eps * sigma_noise)))
    cr_svd_actual = nc * ns / (r * (nc + ns))

    if alpha == 0:
        x_hat = ((U[:, :r] * sv[:r]) @ Vh[:r, :]).astype(np.float32)
        return x_hat, r, cr_svd_actual, 1.0, cr_svd_actual

    n_wp_slots = _count_wp_slots_gallery(ns)
    Vh_hat = np.zeros((r, ns))
    n_kept = 0
    for k in range(r):
        tau_k = alpha * sigma_noise / (sv[k] + 1e-40)
        wp = pywt.WaveletPacket(data=Vh[k], wavelet=WP_WAVELET_GALLERY,
                                maxlevel=WP_MAXLEVEL_GALLERY)
        nodes = wp.get_level(WP_MAXLEVEL_GALLERY, 'natural')
        for node in nodes:
            mask = np.abs(node.data) >= tau_k
            n_kept += int(mask.sum())
            node.data = node.data * mask
        Vh_hat[k] = wp.reconstruct(update=True)[:ns]

    x_hat = np.nan_to_num(
        ((U[:, :r] * sv[:r]) @ Vh_hat).astype(np.float32),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    cr_wp    = (r * n_wp_slots) / max(n_kept, 1)
    cr_total = cr_svd_actual * cr_wp
    return x_hat, r, cr_svd_actual, cr_wp, cr_total


def plot_adaptive_wp_gallery(snippets, pid, residual=False):
    """
    LFP gallery on a SVD-ε × WP-α grid.

    Rows: EPSILONS_GALLERY — SVD-adapt ε values; rank r is derived per snippet
          from r = #{sv > ε · σ_noise}, so the actual CR_SVD is labelled in each row.
    Cols: ALPHAS_GALLERY threshold multipliers (α=0 = pure SVD reference).
    Each cell is labelled with r, achieved CRs, RMSE (µV), and SNR (dB).

    Parameters
    ----------
    snippets : list of ndarray, each (nc, ns), float32
    pid : str
    residual : bool
        If True display original − reconstruction instead of reconstruction.
    """
    snip = snippets[0]
    nc, ns = snip.shape
    orig = snip[:, :DISPLAY_NS]
    rms_ch = np.sqrt(np.mean(snip.astype(np.float64) ** 2, axis=-1))
    vmax = VMAX_GALLERY
    kind_str = 'residual' if residual else 'reconstruction'

    n_rows, n_cols = len(EPSILONS_GALLERY), len(ALPHAS_GALLERY)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.2, n_rows * 3.2))
    fig.suptitle(f'{pid}  —  SVD-ε × WP-α {kind_str}', fontsize=12)

    for j, alpha in enumerate(ALPHAS_GALLERY):
        axes[0, j].set_title('SVD only  (α = 0)' if alpha == 0 else f'α = {alpha}', fontsize=11)

    for i, eps in enumerate(EPSILONS_GALLERY):
        row_ylabel_set = False
        for j, alpha in enumerate(ALPHAS_GALLERY):
            ax = axes[i, j]
            x_hat, r, cr_svd_actual, cr_wp, cr_total = reconstruct_adaptive_wp_eps(
                snip, eps, alpha,
            )
            panel = (orig - x_hat[:, :DISPLAY_NS]) if residual else x_hat[:, :DISPLAY_NS]
            _lf_imshow(ax, panel, vmax=vmax)

            err = snip.astype(np.float64) - x_hat.astype(np.float64)
            rmse_ch = np.sqrt(np.mean(err ** 2, axis=-1))
            rmse_uv = float(np.median(rmse_ch)) * 1e6
            snr_db  = float(20.0 * np.log10(np.median(rms_ch) / max(np.median(rmse_ch), 1e-12)))

            if alpha == 0:
                label = (f'r={r}  CR_SVD={cr_svd_actual:.0f}\n'
                         f'{rmse_uv:.1f} µV  {snr_db:.1f} dB')
            else:
                label = (f'r={r}  SVD×{cr_svd_actual:.0f}\n'
                         f'WP×{cr_wp:.1f}  Tot={cr_total:.0f}\n'
                         f'{rmse_uv:.1f} µV  {snr_db:.1f} dB')
            ax.text(0.03, 0.97, label, transform=ax.transAxes,
                    fontsize=8, va='top', ha='left', color='k',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, lw=0))

            if not row_ylabel_set:
                axes[i, 0].set_ylabel(
                    f'ε = {eps}',
                    fontsize=10, rotation=90, labelpad=4, va='center',
                )
                row_ylabel_set = True

    # Shared colorbar
    import matplotlib.colors as mcolors
    import matplotlib.cm as mplcm
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    sm = mplcm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    fig.subplots_adjust(top=0.93, bottom=0.06)
    cbar_ax = fig.add_axes([0.25, 0.015, 0.50, 0.012])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    ticks = np.linspace(-vmax, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t * 1e6:.0f}' for t in ticks])
    cbar.set_label('µV', labelpad=2)

    suffix = '_residual' if residual else ''
    fname = f'2026-06-12_adaptive_wp_gallery_{pid}{suffix}.png'
    for d in [FIGURE_DIR, QUARTO_FIGURE_DIR]:
        if d.exists():
            fig.savefig(d.joinpath(fname), dpi=120)
    plt.close(fig)
    print(f'{pid}: adaptive-WP {kind_str} gallery saved → {fname}')


FLAGSHIP_PID = 'dab512bd-a02d-4c1f-8dbc-9155a163efc0'


def plot_adaptive_wp_gallery_csd(snippets, pid, residual=False):
    """
    CSD gallery on a SVD-ε × WP-α grid — same layout as plot_adaptive_wp_gallery.

    Displays current source density (ibldsp.voltage.current_source_density) of the
    reconstructed LFP.  Called only for the flagship PID (notable CSD structure).
    The vmax is set from the 99th percentile of the original CSD so all panels share
    a consistent scale.

    Parameters
    ----------
    snippets : list of ndarray, each (nc, ns), float32
    pid : str
    residual : bool
        If True display CSD(original) − CSD(reconstruction); otherwise CSD(reconstruction).
    """
    from ibldsp.voltage import current_source_density
    import neuropixel

    snip = snippets[0]
    nc, ns = snip.shape
    orig = snip[:, :DISPLAY_NS]

    h = neuropixel.trace_header(version=1)
    h = {k: v[:nc] for k, v in h.items()}

    csd_orig_full = current_source_density(orig, h)
    csd_orig = np.nan_to_num(csd_orig_full, nan=0.0)
    vmax = float(np.nanpercentile(np.abs(csd_orig), 99))
    rms_csd_ch = np.sqrt(np.mean(csd_orig ** 2, axis=-1))  # per-channel CSD RMS for SNR

    kind_str = 'CSD residual' if residual else 'CSD'

    n_rows, n_cols = len(EPSILONS_GALLERY), len(ALPHAS_GALLERY)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.2, n_rows * 3.2))
    fig.suptitle(f'{pid}  —  SVD-ε × WP-α {kind_str}', fontsize=12)

    for j, alpha in enumerate(ALPHAS_GALLERY):
        axes[0, j].set_title('SVD only  (α = 0)' if alpha == 0 else f'α = {alpha}', fontsize=11)

    for i, eps in enumerate(EPSILONS_GALLERY):
        row_ylabel_set = False
        for j, alpha in enumerate(ALPHAS_GALLERY):
            ax = axes[i, j]
            x_hat, r, cr_svd_actual, cr_wp, cr_total = reconstruct_adaptive_wp_eps(
                snip, eps, alpha,
            )
            csd_hat = np.nan_to_num(
                current_source_density(x_hat[:, :DISPLAY_NS], h), nan=0.0,
            )
            panel = (csd_orig - csd_hat) if residual else csd_hat
            _lf_imshow(ax, panel, vmax=vmax)

            csd_err_ch = np.sqrt(np.mean((csd_orig - csd_hat) ** 2, axis=-1))
            rmse_csd = float(np.nanmedian(csd_err_ch))
            rms_csd  = float(np.nanmedian(rms_csd_ch))
            snr_db   = float(20.0 * np.log10(rms_csd / max(rmse_csd, 1e-40)))
            rmse_str = f'{rmse_csd:.2e}'

            if alpha == 0:
                label = (f'r={r}  CR_SVD={cr_svd_actual:.0f}\n'
                         f'RMSE={rmse_str}  {snr_db:.1f} dB')
            else:
                label = (f'r={r}  SVD×{cr_svd_actual:.0f}\n'
                         f'WP×{cr_wp:.1f}  Tot={cr_total:.0f}\n'
                         f'RMSE={rmse_str}  {snr_db:.1f} dB')
            ax.text(0.03, 0.97, label, transform=ax.transAxes,
                    fontsize=8, va='top', ha='left', color='k',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, lw=0))

            if not row_ylabel_set:
                axes[i, 0].set_ylabel(
                    f'ε = {eps}',
                    fontsize=10, rotation=90, labelpad=4, va='center',
                )
                row_ylabel_set = True

    import matplotlib.colors as mcolors
    import matplotlib.cm as mplcm
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    sm = mplcm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    fig.subplots_adjust(top=0.93, bottom=0.06)
    cbar_ax = fig.add_axes([0.25, 0.015, 0.50, 0.012])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    ticks = np.linspace(-vmax, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t:.2e}' for t in ticks])
    cbar.set_label('A/m$^3$', labelpad=2)

    suffix = '_residual' if residual else ''
    fname = f'2026-06-12_adaptive_wp_gallery_csd_{pid}{suffix}.png'
    for d in [FIGURE_DIR, QUARTO_FIGURE_DIR]:
        if d.exists():
            fig.savefig(d.joinpath(fname), dpi=120)
    plt.close(fig)
    print(f'{pid}: adaptive-WP CSD {kind_str} gallery saved → {fname}')


def plot_aggregate_figure():
    """
    Single aggregate figure: all 6 methods, averaged across PIDs.

    Per PID: median RMSE / SNR over channels.  Across PIDs: mean.
    Produces a 1×2 figure (RMSE + SNR) saved to FIGURE_DIR.
    """
    from collections import defaultdict

    rmse_acc = defaultdict(list)
    snr_acc = defaultdict(list)
    cr_ref = {}

    for pid in pids:
        cache_file = ROOT_OUTPUT.joinpath(pid, f'metrics_v5_{pid}.npz')
        if not cache_file.exists():
            continue
        d = np.load(cache_file, allow_pickle=True)
        for i, name in enumerate(d['names']):
            rmse_acc[name].append(np.nanmedian(d['rmse'][i], axis=0))
            snr_acc[name].append(np.nanmedian(d['snr'][i], axis=0))
            if name not in cr_ref:
                cr_ref[name] = d['cr'][i]

    n_pids = len([p for p in pids if ROOT_OUTPUT.joinpath(p, f'metrics_v5_{p}.npz').exists()])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Aggregate — median over channels, mean over PIDs  (n={n_pids})',
        fontsize=12,
    )

    for i, name in enumerate(NAMES):
        if name not in cr_ref:
            continue
        color = COLORS[i]
        cr = cr_ref[name]
        rmse_curve = np.nanmean(rmse_acc[name], axis=0) * 1e6
        snr_curve = np.nanmean(snr_acc[name], axis=0)
        axes[0].plot(cr, rmse_curve, color=color, lw=2.5, label=name)
        axes[1].plot(cr, snr_curve, color=color, lw=2.5, label=name)

    for ax, ylabel, title in zip(axes, ['RMSE (µV)', 'SNR (dB)'], ['RMSE', 'SNR']):
        ax.set_xscale('log')
        ax.set_xlim([1.5, 3000])
        ax.set_xlabel('Compression ratio', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.tick_params(labelsize=10)

    axes[0].legend(fontsize=10, loc='upper left')

    fig.tight_layout()
    fig_path = FIGURE_DIR.joinpath('2026-06-09_compression_aggregate.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Aggregate figure saved → {fig_path}')


def plot_adaptive_aggregate_figure():
    """
    Aggregate RMSE/SNR vs total CR for the adaptive SVD-ε + WP-α parameter sweep.

    One curve per ε in EPSILONS_GALLERY; rank r is derived per snippet from
    r = #{sv > ε · σ_noise}.  Each curve sweeps ALPHAS_GALLERY.
    Saves PNG to FIGURE_DIR and QUARTO_FIGURE_DIR.
    """
    import warnings
    palette = sns.color_palette('viridis', n_colors=len(EPSILONS_GALLERY))

    n_eps   = len(EPSILONS_GALLERY)
    n_alpha = len(ALPHAS_GALLERY)
    acc_cr   = [[[] for _ in range(n_alpha)] for _ in range(n_eps)]
    acc_rmse = [[[] for _ in range(n_alpha)] for _ in range(n_eps)]
    acc_snr  = [[[] for _ in range(n_alpha)] for _ in range(n_eps)]
    acc_r    = [[] for _ in range(n_eps)]    # rank per PID, for legend label

    n_wp_slots_cache = {}
    n_loaded = 0

    for pid in pids:
        snippet_files = [ROOT_OUTPUT.joinpath(pid, f'cadzow_denoised_{i}.npy') for i in range(3)]
        if not all(f.exists() for f in snippet_files):
            continue
        snippets = [np.load(f) for f in snippet_files]
        n_loaded += 1
        nc, ns = snippets[0].shape

        if ns not in n_wp_slots_cache:
            n_wp_slots_cache[ns] = _count_wp_slots_gallery(ns)
        n_wp_slots = n_wp_slots_cache[ns]

        svds = [np.linalg.svd(s.astype(np.float64), full_matrices=False) for s in snippets]
        rms_ch = np.mean([np.sqrt(np.mean(s.astype(np.float64) ** 2, axis=-1))
                          for s in snippets], axis=0)

        for i, eps in enumerate(EPSILONS_GALLERY):
            # rank derived from ε per snippet, averaged across snippets
            r_vals = [max(1, int(np.sum(sv > eps * _svd_noise_floor(sv)))) for _, sv, _ in svds]
            r = max(1, int(np.round(np.mean(r_vals))))
            cr_svd_actual = nc * ns / (r * (nc + ns))
            acc_r[i].append(r)

            for j, alpha in enumerate(ALPHAS_GALLERY):
                rmse_sum = np.zeros(nc)
                n_kept_total = 0

                for (U, sv, Vh), snip in zip(svds, snippets):
                    sigma_noise = _svd_noise_floor(sv)

                    if alpha == 0:
                        x_hat = ((U[:, :r] * sv[:r]) @ Vh[:r, :]).astype(np.float32)
                        n_kept_total += r * n_wp_slots
                    else:
                        Vh_hat = np.zeros((r, ns))
                        n_kept = 0
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', RuntimeWarning)
                            for k in range(r):
                                tau_k = alpha * sigma_noise / (sv[k] + 1e-40)
                                wp = pywt.WaveletPacket(
                                    data=Vh[k], wavelet=WP_WAVELET_GALLERY,
                                    maxlevel=WP_MAXLEVEL_GALLERY,
                                )
                                nodes = wp.get_level(WP_MAXLEVEL_GALLERY, 'natural')
                                for node in nodes:
                                    mask = np.abs(node.data) >= tau_k
                                    n_kept += int(mask.sum())
                                    node.data = node.data * mask
                                Vh_hat[k] = wp.reconstruct(update=True)[:ns]
                        x_hat = np.nan_to_num(
                            ((U[:, :r] * sv[:r]) @ Vh_hat).astype(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0,
                        )
                        n_kept_total += n_kept

                    err = snip.astype(np.float64) - x_hat.astype(np.float64)
                    rmse_sum += np.sqrt(np.mean(err ** 2, axis=-1))

                rmse_ch = rmse_sum / len(snippets)
                n_kept_avg = n_kept_total / len(snippets)
                cr_wp    = (r * n_wp_slots) / max(n_kept_avg, 1) if alpha > 0 else 1.0
                cr_total = cr_svd_actual * cr_wp

                rmse_med = float(np.median(rmse_ch))
                snr_med  = float(20.0 * np.log10(np.median(rms_ch) / max(rmse_med, 1e-12)))

                acc_cr[i][j].append(cr_total)
                acc_rmse[i][j].append(rmse_med)
                acc_snr[i][j].append(snr_med)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f'Adaptive SVD-ε + WP-α — aggregate RMSE/SNR  (n={n_loaded} PIDs, median over channels)',
        fontsize=12,
    )

    y_offsets = [10, -15, 10, -15]

    for i, (eps, color, dy) in enumerate(zip(EPSILONS_GALLERY, palette, y_offsets)):
        cr_vals   = np.array([float(np.mean(acc_cr[i][j]))   for j in range(n_alpha)])
        rmse_vals = np.array([float(np.mean(acc_rmse[i][j])) for j in range(n_alpha)]) * 1e6
        snr_vals  = np.array([float(np.mean(acc_snr[i][j]))  for j in range(n_alpha)])
        r_med     = int(np.median(acc_r[i]))
        cr_med    = float(np.mean(acc_cr[i][0]))   # α=0 point = pure SVD CR

        lbl = f'ε = {eps}  (r ≈ {r_med}, CR_SVD ≈ {cr_med:.0f})'
        axes[0].loglog(cr_vals, rmse_vals, '-o', color=color, lw=2, ms=6, label=lbl)
        axes[1].semilogx(cr_vals, snr_vals, '-o', color=color, lw=2, ms=6, label=lbl)

        for cr, rmse, snr, alpha in zip(cr_vals, rmse_vals, snr_vals, ALPHAS_GALLERY):
            a_lbl = 'SVD' if alpha == 0 else f'α={alpha}'
            axes[0].annotate(a_lbl, (cr, rmse), textcoords='offset points',
                             xytext=(4, dy), fontsize=7, color=color, alpha=0.9)
            axes[1].annotate(a_lbl, (cr, snr), textcoords='offset points',
                             xytext=(4, dy), fontsize=7, color=color, alpha=0.9)

    for ax, ylabel, title in zip(axes, ['RMSE (µV)', 'SNR (dB)'], ['RMSE', 'SNR']):
        ax.set_xscale('log')
        ax.set_xlabel('Compression ratio', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(True, which='both', alpha=0.3)

    axes[0].legend(fontsize=10, loc='upper left')
    axes[1].legend(fontsize=10, loc='upper right')

    fig.tight_layout()
    fname = '2026-06-12_adaptive_wp_aggregate.png'
    for d in [FIGURE_DIR, QUARTO_FIGURE_DIR]:
        if d.exists():
            fig.savefig(d.joinpath(fname), dpi=150)
    plt.close(fig)
    print(f'Adaptive aggregate figure saved → {fname}')


def plot_epsilon_vs_rank():
    """
    Two-panel figure: ε vs rank (left) and SV noise floor vs rank (right).

    Left: dimensionless threshold multiplier ε = threshold / noise_floor.
    Right: singular value at each rank point (µV), with the per-PID noise floor
    (median of lower half of SVs) as a dashed horizontal line.
    Top x-axis on both panels shows the corresponding compression ratio.
    """
    ranks = np.array(RANKS)  # [1, 2, 4, 8, 16, 32, 64, 96]
    palette = sns.color_palette('tab10', n_colors=len(pids))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cr_ref = None  # CR ticks for secondary x-axis (from first available PID)

    for pid, color in zip(pids, palette):
        cache_file = ROOT_OUTPUT.joinpath(pid, f'metrics_v5_{pid}.npz')
        if not cache_file.exists():
            continue
        d = np.load(cache_file, allow_pickle=True)
        if 'eps_adapt' not in d:
            continue
        # eps_adapt stored ascending-CR = descending-rank; reverse to ascending rank
        eps = d['eps_adapt'][::-1]
        if cr_ref is None and 'cr_svd' in d:
            cr_ref = d['cr_svd'][::-1]  # same reversal: now cr_ref[0] = CR at rank 1

        # recompute SV spectrum from first snippet (fast: no U/Vh needed)
        snippet_file = ROOT_OUTPUT.joinpath(pid, 'cadzow_denoised_0.npy')
        if not snippet_file.exists():
            continue
        sv = np.linalg.svd(np.load(snippet_file).astype(np.float64), compute_uv=False)
        sv_nz = sv[sv > sv[0] * 1e-4]
        tail = sv_nz[sv_nz.size // 2:]
        noise_floor = float(np.nanmedian(tail)) if tail.size else float(sv[0])

        sv_at_ranks = sv[ranks - 1] * 1e6   # µV, 0-indexed: rank r → sv[r-1]

        axes[0].plot(ranks, eps, lw=1.5, marker='o', ms=4, color=color, label=pid[:8])
        axes[1].plot(ranks, sv_at_ranks, lw=1.5, marker='o', ms=4, color=color, label=pid[:8])
        axes[1].axhline(noise_floor * 1e6, color=color, lw=0.8, ls='--', alpha=0.6)

    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(ranks)
        ax.set_xticklabels(ranks)
        ax.tick_params(labelsize=10)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlabel('Number of singular values kept (rank)', fontsize=12)

        # secondary x-axis: compression ratio
        if cr_ref is not None:
            ax2 = ax.twiny()
            ax2.set_xscale('log')
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(ranks)
            ax2.set_xticklabels([f'{cr:.0f}' for cr in cr_ref], fontsize=8)
            ax2.set_xlabel('Compression ratio', fontsize=10)

    axes[0].set_ylabel('ε  (dimensionless: threshold / noise floor)', fontsize=12)
    axes[0].set_title('SVD-adapt: ε vs rank', fontsize=12)
    axes[0].legend(fontsize=8, ncol=2, loc='upper right')

    axes[1].set_ylabel('Singular value at rank r  (µV)', fontsize=12)
    axes[1].set_title('SV spectrum — noise floor = dashed line', fontsize=12)
    axes[1].legend(fontsize=8, ncol=2, loc='upper right')

    fig.tight_layout()
    fig_path = FIGURE_DIR.joinpath('2026-06-10_epsilon_vs_rank.png')
    fig.savefig(fig_path, dpi=150)
    if QUARTO_FIGURE_DIR.exists():
        fig.savefig(QUARTO_FIGURE_DIR.joinpath('2026-06-10_epsilon_vs_rank.png'), dpi=150)
    plt.close(fig)
    print(f'ε vs rank figure saved → {fig_path}')


def plot_gap_threshold_vs_rank():
    """
    Two-panel calibration figure for the gap-threshold SVD criterion.

    Left: gap ratio sv[k]/sv[k+1] at each rank boundary vs rank, one line per PID.
          Horizontal reference lines at g=2, 5, 10.  Top x-axis shows CR.
          Reading off: "to get rank r, use g between this curve's value at r and r-1."
    Right: SV magnitude at rank r (µV) vs rank, with per-PID noise floor (dashed).
           Identical layout to the right panel of plot_epsilon_vs_rank.

    Analogous to plot_epsilon_vs_rank but for the gap-ratio criterion.
    """
    palette = sns.color_palette('tab10', n_colors=len(pids))
    max_rank_plot = 32   # beyond this ratios approach 1 (noise floor territory)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cr_ref = None

    for pid, color in zip(pids, palette):
        snippet_file = ROOT_OUTPUT.joinpath(pid, 'cadzow_denoised_0.npy')
        if not snippet_file.exists():
            continue
        snip = np.load(snippet_file).astype(np.float64)
        nc_s, ns_s = snip.shape
        sv = np.linalg.svd(snip, compute_uv=False)
        sv_nz = sv[sv > sv[0] * 1e-4]
        if sv_nz.size < 2:
            continue

        # ratios[k] = sv[k]/sv[k+1]: gap between rank k+1 and k+2 (1-indexed rank = k+1)
        ratios = sv_nz[:-1] / (sv_nz[1:] + 1e-10)
        n_plot = min(len(ratios), max_rank_plot)
        rank_vals = np.arange(1, n_plot + 1)
        ratio_vals = ratios[:n_plot]
        sv_at_ranks = sv[rank_vals - 1] * 1e6  # µV, rank k → sv[k-1]

        if cr_ref is None:
            cr_ref = nc_s * ns_s / (rank_vals * (nc_s + ns_s))

        sv_nz_tail = sv_nz[sv_nz.size // 2:]
        noise_floor_uv = float(np.nanmedian(sv_nz_tail)) * 1e6 if sv_nz_tail.size else sv_nz[0] * 1e6

        axes[0].plot(rank_vals, ratio_vals, lw=1.5, marker='o', ms=3, color=color, label=pid[:8])
        axes[1].plot(rank_vals, sv_at_ranks, lw=1.5, marker='o', ms=3, color=color, label=pid[:8])
        axes[1].axhline(noise_floor_uv, color=color, lw=0.8, ls='--', alpha=0.6)

    # reference threshold lines: helps read off what g to pick for a target rank/CR
    for g_ref, ls, lbl in [(2, '--', 'g=2'), (5, ':', 'g=5'), (10, '-.', 'g=10')]:
        axes[0].axhline(g_ref, color='gray', lw=1.2, ls=ls, alpha=0.7, label=lbl)

    for ax in axes:
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xticks(RANKS[:6])    # 1,2,4,8,16,32
        ax.set_xticklabels(RANKS[:6])
        ax.tick_params(labelsize=10)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlabel('Number of singular values kept (rank)', fontsize=12)

        if cr_ref is not None:
            ax2 = ax.twiny()
            ax2.set_xscale('log')
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(RANKS[:6])
            ax2.set_xticklabels([f'{cr:.0f}' for cr in cr_ref[:6]], fontsize=8)
            ax2.set_xlabel('Compression ratio', fontsize=10)

    axes[0].set_ylabel('Gap ratio  sv[k] / sv[k+1]  (dimensionless)', fontsize=12)
    axes[0].set_title('Gap spectrum per PID — choose g threshold from this', fontsize=12)
    axes[0].legend(fontsize=8, ncol=2, loc='upper right')

    axes[1].set_ylabel('Singular value at rank r  (µV)', fontsize=12)
    axes[1].set_title('SV spectrum — noise floor = dashed line', fontsize=12)
    axes[1].legend(fontsize=8, ncol=2, loc='upper right')

    fig.tight_layout()
    fig_path = FIGURE_DIR.joinpath('2026-06-10_gap_vs_rank.png')
    fig.savefig(fig_path, dpi=150)
    if QUARTO_FIGURE_DIR.exists():
        fig.savefig(QUARTO_FIGURE_DIR.joinpath('2026-06-10_gap_vs_rank.png'), dpi=150)
    plt.close(fig)
    print(f'Gap vs rank figure saved → {fig_path}')


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

    cache_file = output_path.joinpath(f'metrics_v5_{pid}.npz')
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        rmse_all = data['rmse']
        snr_all = data['snr']
        cr_all = data['cr']
        names = list(data['names'])
        eps_adapt = data['eps_adapt'] if 'eps_adapt' in data else None
        print(f'{pid}: loaded from cache')
    else:
        print(f'{pid}: SVD...')
        rmse_svd, snr_svd, cr_svd = compress_svd(snippets, ranks=RANKS)
        # Sort ascending by CR so cr_svd is the common alignment target for adaptive methods.
        svd_ord = np.argsort(cr_svd)
        cr_svd, rmse_svd, snr_svd = cr_svd[svd_ord], rmse_svd[:, svd_ord], snr_svd[:, svd_ord]

        print(f'{pid}: SVD-adapt (ε-sweep)...')
        _rmse_sa, _snr_sa, _cr_sa, _eps_sa = compress_svd_adapt(snippets, epsilons=EPSILONS)
        rmse_sa_eps, snr_sa_eps = align_to_cr_grid(_rmse_sa, _snr_sa, _cr_sa, cr_svd)
        cr_sa_eps = cr_svd.copy()
        # Nearest source ε for each aligned CR target (on log scale).
        log_cr_sa = np.log(_cr_sa)
        eps_adapt = np.array([_eps_sa[np.argmin(np.abs(log_cr_sa - np.log(ct)))] for ct in cr_svd])

        rmse_wp_list, snr_wp_list, cr_wp_list = [], [], []
        for initial_cr in SVD_ADAPT_CRS:
            print(f'{pid}: SVD-adapt-CR{initial_cr}-WP...')
            rm, sn, cr = compress_svd_adapt_wp(snippets, initial_cr, keep_fracs=KEEP_FRACS)
            rmse_wp_list.append(rm)
            snr_wp_list.append(sn)
            cr_wp_list.append(cr)

        n_methods = len(NAMES)
        rmse_all = np.empty(n_methods, dtype=object)
        snr_all = np.empty(n_methods, dtype=object)
        cr_all = np.empty(n_methods, dtype=object)
        for i, (rm, sn, cr) in enumerate(zip(
            [rmse_svd, rmse_sa_eps, *rmse_wp_list],
            [snr_svd, snr_sa_eps, *snr_wp_list],
            [cr_svd, cr_sa_eps, *cr_wp_list],
        )):
            rmse_all[i] = rm
            snr_all[i] = sn
            cr_all[i] = cr

        names = NAMES
        np.savez(cache_file, rmse=rmse_all, snr=snr_all, cr=cr_all, names=np.array(names),
                 cr_svd=cr_svd, eps_adapt=eps_adapt)
        print(f'{pid}: cache saved → {cache_file}')

    plot_adaptive_wp_gallery(snippets, pid)
    plot_adaptive_wp_gallery(snippets, pid, residual=True)
    if pid == FLAGSHIP_PID:
        plot_adaptive_wp_gallery_csd(snippets, pid)
        plot_adaptive_wp_gallery_csd(snippets, pid, residual=True)

# %%
plot_adaptive_aggregate_figure()
plot_epsilon_vs_rank()
plot_gap_threshold_vs_rank()