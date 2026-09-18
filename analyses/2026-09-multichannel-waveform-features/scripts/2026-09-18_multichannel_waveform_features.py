"""Prototype for ibleatools #123: multi-channel waveform features.

https://github.com/int-brain-lab/ibleatools/issues/123

Goal: spatial spread (already in ibldsp.waveforms.compute_spatial_spread) and a
new "apparent velocity" feature computed as a signed slowness (dt/dx, s/m) fit
across channels around the peak, with per-channel peak picking narrowed by a
Hanning window in time around the max-channel peak sample.

Stage 1: load one benchmark snippet, run ap+waveforms feature extraction
(ibleatools branch waveform-amplitude-volts, PR #124, so amplitudes are Volts),
time it, and eyeball a handful of multi-channel waveforms.

Run cell-by-cell (# %% cells).

Environment note (2026-09-18): the monorepo .venv's `packages/spikeinterface`
was on Olivier's own `olive-saturation` fork/branch (uncommitted work in
progress for a different project, at spikeinterface 0.104.1), which is
incompatible with dartsort 0.5.24 (two separate private-API breaks:
`recording._recording_segments` and
`get_chunk_with_margin(channel_indices=...)`). Official PyPI
`spikeinterface==0.104.9` + `dartsort==0.5.24` is confirmed compatible; the
ibleatools[full] pyproject.toml floor was bumped to reflect this
(`spikeinterface>=0.104.9`, see PR for #123). The int-brain-lab/spikeinterface
GitHub fork's `main` branch was fast-forwarded to upstream `main` (now
0.105.0), but `packages/spikeinterface` in this checkout is still on
`olive-saturation` (left untouched: it has an uncommitted change) -- so the
monorepo .venv itself is NOT yet fixed. This script was developed against a
throwaway venv (copies of ONE/iblutil/ibllib/ibl-neuropixel/iblatlas/ibleatools
installed outside the workspace so uv's `spikeinterface = {workspace = true}`
source override didn't pull in the broken fork, constrained to
spikeinterface==0.104.9 + dartsort==0.5.24), since deleted. To rerun: either
rebuild that kind of throwaway venv, or update `packages/spikeinterface` past
`olive-saturation` (e.g. onto the now-synced `main`) and resync the monorepo
.venv -- Olivier's call given the uncommitted saturation work on that branch.
"""

# %% Imports and configuration
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import addcopyfighandler  # noqa: F401
except ImportError:
    pass
from one.api import ONE

from ephysatlas.feature_calculators import (
    FeatureComputationOptions,
    IBLPIDFeatureCalculator,
    SnippetWindow,
)
from ibldsp import waveforms as wf

sns.set_theme(context="notebook")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATE = "2026-09-18"
FIG_DIR = Path.home().joinpath("Documents", "figures")
SCRATCH_DIR = Path.home().joinpath("scratch", "ephysatlas_waveform_proto")
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

from BENCHMARKS import pids  # noqa: E402

one = ONE()

# %% 1. Load one benchmark snippet and compute ap + waveforms features
pid = pids[0]
output_dir = SCRATCH_DIR.joinpath(pid)
window = SnippetWindow(t_start=300.0, duration_ap=2.0, duration_lf=2.0)

calc = IBLPIDFeatureCalculator(pid=pid, one=one)
options = FeatureComputationOptions(
    features_to_compute=["ap", "waveforms"],
    output_dir=output_dir,
    save_waveforms=True,
    # First run (fresh output_dir): ~105-111s for 2s of AP / 384 channels / 5404
    # spikes, CPU-only (n_jobs=0). Skip recompute on reruns while iterating on
    # stage 2 below -- the saved waveform arrays/spikes.pqt are read from disk
    # either way and don't depend on this flag.
    skip_saved_computation=True,
)

t0 = time.time()
result = calc.compute_snippet(window, options)
dt_compute = time.time() - t0
logger.info(
    "pid=%s duration_ap=%.1fs -> feature computation took %.1fs (%s channels)",
    pid,
    window.duration_ap,
    dt_compute,
    result.features.shape,
)

# %% 2. Load the saved waveform arrays + spikes dataframe
waveforms_dir = result.snippet_level_dir.joinpath("waveforms")
raw = np.load(waveforms_dir.joinpath("raw.npy")).astype(np.float64)
denoised = np.load(waveforms_dir.joinpath("denoised.npy")).astype(np.float64)
channel_index = np.load(waveforms_dir.joinpath("waveform_channels.npy"))
df_spikes = pd.read_parquet(waveforms_dir.joinpath("spikes.pqt"))
n_spikes, nsw, ncw = denoised.shape

logger.info(
    "raw=%s denoised=%s channel_index=%s n_spikes=%d",
    raw.shape,
    denoised.shape,
    channel_index.shape,
    df_spikes.shape[0],
)
logger.info("df_spikes columns: %s", list(df_spikes.columns))
logger.info("df_spikes head:\n%s", df_spikes.head())

# %% 3. Channel geometry for this snippet (for spatial spread / slowness)
snippet = calc.get_destriped_snippet(window)
geometry = snippet.geometry  # dict with 'x', 'y' (and maybe more), per real channel
logger.info(
    "geometry keys=%s, n_channels=%d", list(geometry.keys()), len(geometry["x"])
)

# %% 4. Eyeball a handful of multi-channel waveforms: raw (top row) vs denoised (bottom row)
# Amplitudes are real Volts (PR #124), typically tens of uV: plot_wiggle's
# scale=0.3 default assumes dimensionless/normalized units and produces a flat
# blank plot here, so scale it to this snippet's amplitudes instead. Raw and
# denoised share the same scale (from the denoised amplitudes) so the before/
# after comparison is fair -- raw's noise floor is clipped/cluttered at this
# gain, which is the honest picture of how much denoising actually does.
n_show = 6
rng = np.random.default_rng(0)
idx_show = rng.choice(df_spikes.shape[0], size=n_show, replace=False)
wiggle_scale = np.nanpercentile(np.abs(denoised[idx_show].astype(np.float64)), 99)
logger.info("wiggle scale (shared, raw and denoised) = %.3g V (p99 of denoised)", wiggle_scale)

fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 11), sharey=True, sharex=True)
for col, i in enumerate(idx_show):
    wav_raw = raw[i].T.astype(np.float64)
    wav_denoised = denoised[i].T.astype(np.float64)
    wf.double_wiggle(wav_raw, fs=30_000, ax=axes[0, col], title=f"spike {i} (raw)", scale=wiggle_scale)
    wf.double_wiggle(wav_denoised, fs=30_000, ax=axes[1, col], title=f"spike {i} (denoised)", scale=wiggle_scale)
fig.suptitle(f"pid={pid} - multi-channel waveforms (random sample)")
fig.tight_layout()
fig.savefig(FIG_DIR.joinpath(f"{DATE}_waveform_proto_wiggle_sample.png"), dpi=150)
# No plt.show(): blocks forever when this file is run headless (e.g. from a
# shell, no GUI event loop). Running cell-by-cell in PyCharm/Jupyter displays
# the figure automatically; otherwise inspect the saved PNG.

logger.info("Stage 1 done. dt_compute=%.1fs for %.1fs of AP data", dt_compute, window.duration_ap)

# %% Stage 2: spatial spread (reuse ibldsp) + slowness (new, prototype-only) -----------
# Per-spike neighbourhood channel geometry: channel_index maps each spike's detection
# channel -> its neighbour real-channel ids (padded with sentinel == n_channels_real for
# unused slots, per ephysatlas.features.dart_subtraction_numpy).
xy_um = np.c_[geometry["x"], geometry["y"]]  # (n_channels_real, 2), micrometres
xy_padded = np.vstack([xy_um, [np.nan, np.nan]])
neighbor_real_idx = channel_index[df_spikes["channel"].to_numpy()]
neighbor_xy_um = xy_padded[neighbor_real_idx]  # (n_spikes, ncw, 2)

# Reference (max-deflection) channel/time per spike, same convention as
# ibldsp.waveforms.compute_spike_features.
df_peak = wf.find_peak(denoised)

# %% Spatial spread: ibldsp.waveforms.compute_spatial_spread bypassed on purpose.
# It calls dist_chanel_from_peak(channel_geometry, df) but that function's signature
# expects the peak_trace_idx array directly, not the whole dataframe. Also,
# weights_spk_ch(weight_type="peak") returns the *signed* peak value (77% negative on
# this snippet), so spatial_spread_weighted's weighted mean divides by a near-zero or
# negative weight sum for some spikes -> nonsensical (even negative) spread. Both look
# like pre-existing bugs in ibldsp.waveforms, not fixed here (repo untouched per
# instructions) -- calling the lower-level helpers directly with abs(weights) instead.
channel_geometry_3d = np.dstack(
    [neighbor_xy_um[:, :, 0], neighbor_xy_um[:, :, 1], np.zeros_like(neighbor_xy_um[:, :, 0])]
)
t0 = time.time()
eu_dist_um = wf.dist_chanel_from_peak(channel_geometry_3d, df_peak["peak_trace_idx"].to_numpy())
weights_abs = np.abs(wf.weights_spk_ch(denoised, weight_type="peak"))
spatial_spread_um = wf.spatial_spread_weighted(eu_dist_um, weights_abs)
dt_spread = time.time() - t0
logger.info(
    "spatial_spread: %.3fs for %d spikes, p5/50/95=%s um",
    dt_spread,
    n_spikes,
    np.round(np.nanpercentile(spatial_spread_um, [5, 50, 95]), 1),
)


# %% Slowness (new, v3): Hanning-windowed per-channel cross-correlation pick, weighted
# 1D fit of dt vs axial (dy, along the probe) offset from the reference channel.
# v2 also fit a lateral (dx) component; dropped (see 2026-09-18 distributions figure,
# 3rd panel, from that version): on this single-shank NP1.0 snippet the ~48um lateral
# extent (4 columns) vs ~300um axial extent in a typical neighbourhood makes the
# lateral slowness estimate noise-dominated (|slowness_x| systematically larger and
# flatter-distributed than |slowness_y|), not a real fast lateral signal worth reporting
# without a fundamentally more careful estimator (larger radius, pooled across spikes
# per unit, ...).
from ibldsp.utils import parabolic_max  # noqa: E402

HALF_WINDOW_MS = 0.5
MIN_CHANNELS = 4
MIN_WEIGHT_FRAC = 0.1
half_win_samples = int(round(HALF_WINDOW_MS * 1e-3 * 30_000.0))
hann_full = np.hanning(2 * half_win_samples + 1)


def _windowed_segment(arr, i, pt, fs, half_win, hann):
    t0_, t1_ = pt - half_win, pt + half_win + 1
    w0, w1 = max(0, -t0_), len(hann) - max(0, t1_ - arr.shape[1])
    t0c, t1c = max(0, t0_), min(arr.shape[1], t1_)
    if t1c <= t0c:
        return None, t0c, t1c
    return arr[i, t0c:t1c, :] * hann[w0:w1, np.newaxis], t0c, t1c


def picks_xcorr(seg, ref_row):
    """Cross-correlate every channel's windowed snippet against the reference
    (peak) channel's own windowed snippet; sub-sample lag via parabolic
    interpolation (ibldsp.utils.parabolic_max) on |corr| so a phase-inverted
    channel (negative correlation peak) is still matched on shape, not just
    amplitude sign. Weight = normalized cross-correlation coefficient magnitude
    (0-1) -- a fit-quality score, robust to gradual non-stationarity in waveform
    shape across channels.

    Sub-sample accuracy: validated against a frequency-domain phase-slope
    regression alternative (ibldsp.waveforms.get_apf_from2spikes/get_phase_slope,
    the basis of wave_shift_phase) in 2026-09-18_waveform_pick_accuracy_validation.py.
    Parabolic interpolation is not locking to the nearest sample (~6% of real picks
    land within 0.01 samples of an integer). Phase regression is exact in the
    noiseless case, but under this snippet's actual denoised-channel residual noise
    (what we pick against here) xcorr+parabolic is *more* accurate (RMSE 0.099 vs
    0.147 samples) -- matching wave_shift_phase's own documented caveat that it
    "does not work well with raw data sampled at 30kHz" without a per-template
    calibration step (get_spike_slopeparams) too slow to run per spike per channel.
    Kept xcorr+parabolic."""
    win = seg.shape[0]
    ref = seg[:, ref_row]
    n_fft = 2 * win
    R = np.fft.rfft(ref, n=n_fft)
    S = np.fft.rfft(seg, n=n_fft, axis=0)
    corr_full = np.fft.fftshift(np.fft.irfft(np.conj(R)[:, np.newaxis] * S, n=n_fft, axis=0), axes=0)
    lags = np.arange(n_fft) - n_fft // 2
    keep_lag = np.abs(lags) <= (win - 1)
    corr, lags_c = corr_full[keep_lag, :], lags[keep_lag]
    ipeak, cpeak = parabolic_max(np.abs(corr).T)  # per-channel (row) sub-sample peak
    lag_at_peak = lags_c[0] + ipeak
    norm = np.sqrt(np.sum(ref**2) * np.sum(seg**2, axis=0))
    weight = np.where(norm > 0, cpeak / norm, np.nan)
    return lag_at_peak, weight


def _weighted_lstsq(X, y, w):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, np.newaxis], y * sw, rcond=None)
    return beta


def compute_windowed_slowness(
    arr,
    df_peak,
    xy_um,
    fs=30_000.0,
    half_window_ms=HALF_WINDOW_MS,
    min_channels=MIN_CHANNELS,
    min_weight_frac=MIN_WEIGHT_FRAC,
):
    """Signed slowness (s/m) per spike, from a weighted linear fit of per-channel
    cross-correlation pick time vs axial (dy, along the probe) offset from the
    reference (max-deflection) channel: dt = slowness_y * dy + t0.

    Per issue #123: computes the *inverse* of velocity (slowness, proportional to
    delta T) rather than fitting velocity directly, since velocity = 1/slowness
    blows up whenever the fit's dt/dy slope is near zero (e.g. near-simultaneous
    arrival across the local neighbourhood) -- confirmed on real data below (a
    ~50/50 split of huge +-velocities right around slowness_y=0).

    Picks come from picks_xcorr (see above): sub-sample cross-correlation lag of
    each channel's windowed snippet against the reference channel's own windowed
    snippet, with the peak picked on |corr| so a phase-inverted channel is still
    matched on shape (not just amplitude sign), weighted by the normalized
    correlation coefficient -- robust to gradual non-stationarity in waveform shape
    across channels, and cheap (~0.45s for 5404 spikes, negligible next to the
    ~110s waveform-extraction step).

    Sign convention (channel geometry y increases away from the probe tip, i.e.
    towards the brain surface, confirmed on this snippet: y in [20, 3840] um):
    slowness_y > 0 means later pick times at larger y (up the probe) -> wave moving
    "up". velocity_m_s = 1 / slowness_y carries the same sign.

    A lateral (dx, across shank columns) component was tried and dropped -- see the
    module docstring comment above this function.

    Parameters
    ----------
    arr : np.ndarray
        Multi-channel waveform snippets, shape (n_spikes, nsw, ncw), Volts.
    df_peak : pd.DataFrame
        Output of ibldsp.waveforms.find_peak(arr): peak_trace_idx, peak_time_idx,
        peak_val.
    xy_um : np.ndarray
        Per-spike neighbour channel (x, y) coordinates, shape (n_spikes, ncw, 2),
        micrometres, NaN for padded/unused channel slots.
    fs : float
        Sampling frequency (Hz).
    half_window_ms : float
        Half-width of the Hanning window in ms, centred on the reference peak time.
    min_channels : int
        Minimum number of channels kept after thresholding for a fit to be attempted.
    min_weight_frac : float
        Channels with a weight below this fraction of the neighbourhood's max are
        dropped before fitting (keeps the fit local to channels with a real pick,
        not noise floor).

    Returns
    -------
    pd.DataFrame
        slowness_y_s_per_m (signed) and n_channels_used, one row per spike.
    """
    n_spikes, nsw, ncw = arr.shape
    half_win = int(round(half_window_ms * 1e-3 * fs))
    hann = np.hanning(2 * half_win + 1)

    peak_time = df_peak["peak_time_idx"].to_numpy()
    peak_trace = df_peak["peak_trace_idx"].to_numpy()

    sy = np.full(n_spikes, np.nan)
    n_used = np.zeros(n_spikes, dtype=int)
    for i in range(n_spikes):
        seg, t0c, t1c = _windowed_segment(arr, i, peak_time[i], fs, half_win, hann)
        if seg is None:
            continue
        lag, weight = picks_xcorr(seg, peak_trace[i])
        dt = lag / fs

        dy = xy_um[i, :, 1] - xy_um[i, peak_trace[i], 1]
        valid = np.isfinite(weight) & np.isfinite(dy) & (weight > 0)
        if valid.sum() < min_channels:
            continue
        w = weight[valid]
        keep = w >= min_weight_frac * w.max()
        if keep.sum() < min_channels:
            continue
        X = np.c_[dy[valid][keep] * 1e-6, np.ones(keep.sum())]
        beta = _weighted_lstsq(X, dt[valid][keep], w[keep])
        sy[i] = beta[0]
        n_used[i] = keep.sum()
    return pd.DataFrame({"slowness_y_s_per_m": sy, "n_channels_used": n_used})


t0 = time.time()
df_slowness = compute_windowed_slowness(denoised, df_peak, neighbor_xy_um)
dt_slowness = time.time() - t0
n_valid = int(df_slowness["slowness_y_s_per_m"].notna().sum())
logger.info(
    "compute_windowed_slowness: %.2fs for %d spikes (%.3f ms/spike), %d/%d valid, %.2f%% upgoing",
    dt_slowness,
    n_spikes,
    dt_slowness / n_spikes * 1e3,
    n_valid,
    n_spikes,
    float((df_slowness["slowness_y_s_per_m"] > 0).mean()) * 100,
)

# %% 5. Diagnostic panel: same 6 spikes as the stage-1 wiggle plot. Top row: wiggle with
# the Hanning window shaded and each channel's pick overlaid. Bottom row: the weighted
# dt-vs-dy fit behind slowness_y, dt x-axis shared across all 6 spikes for comparability.
dt_ms_by_spike = {}
dy_um_by_spike = {}
w_by_spike = {}
keep_by_spike = {}
for i in idx_show:
    seg, t0c, t1c = _windowed_segment(denoised, i, int(df_peak["peak_time_idx"].iloc[i]), 30_000.0, half_win_samples, hann_full)
    trace = int(df_peak["peak_trace_idx"].iloc[i])
    lag, weight = picks_xcorr(seg, trace)
    dt_ms = lag / 30_000.0 * 1e3
    dy_um = neighbor_xy_um[i, :, 1] - neighbor_xy_um[i, trace, 1]
    valid = np.isfinite(weight) & np.isfinite(dy_um) & (weight > 0)
    w_all = np.where(valid, weight, 0.0)
    keep = valid & (w_all >= MIN_WEIGHT_FRAC * w_all.max())
    dt_ms_by_spike[i], dy_um_by_spike[i], w_by_spike[i], keep_by_spike[i] = dt_ms, dy_um, w_all, keep

dt_xlim = 1.15 * max(np.abs(dt_ms_by_spike[i][keep_by_spike[i]]).max() for i in idx_show)

fig, axes = plt.subplots(2, n_show, figsize=(3.2 * n_show, 8))
for col, i in enumerate(idx_show):
    pt = int(df_peak["peak_time_idx"].iloc[i])
    trace = int(df_peak["peak_trace_idx"].iloc[i])
    seg, t0c, t1c = _windowed_segment(denoised, i, pt, 30_000.0, half_win_samples, hann_full)
    dt_ms, dy_um, w_all, keep = dt_ms_by_spike[i], dy_um_by_spike[i], w_by_spike[i], keep_by_spike[i]
    sy_i = df_slowness["slowness_y_s_per_m"].iloc[i]
    velocity = 1 / sy_i if np.isfinite(sy_i) and sy_i != 0 else np.nan
    direction = "up" if sy_i > 0 else "down" if sy_i < 0 else "?"

    ax = axes[0, col]
    wav = denoised[i].T
    wf.double_wiggle(wav, fs=30_000, ax=ax, scale=wiggle_scale)
    ax.axvspan(t0c / 30_000.0 * 1e3, t1c / 30_000.0 * 1e3, color="green", alpha=0.15)
    ax.axhline(trace + 1, color="red", lw=0.5, ls="--")
    # overlay each channel's xcorr pick (absolute time = ref peak time + lag)
    pick_ms = pt / 30_000.0 * 1e3 + dt_ms
    ch_rows = np.arange(ncw) + 1
    ax.scatter(pick_ms[keep], ch_rows[keep], c=w_all[keep], cmap="viridis", s=18, zorder=3, edgecolors="none")
    ax.scatter(pick_ms[~keep], ch_rows[~keep], color="lightgrey", s=8, zorder=2)
    ax.set(title=f"spike {i}: ref ch={trace}")

    ax = axes[1, col]
    ax.scatter(dt_ms[keep], dy_um[keep], c=w_all[keep], cmap="viridis", s=25)
    ax.scatter(dt_ms[~keep], dy_um[~keep], color="lightgrey", s=10)
    ax.axvline(0, color="k", lw=0.5)
    ax.axhline(0, color="k", lw=0.5)
    if np.isfinite(sy_i) and keep.sum() >= 2:
        dy_mean = np.average(dy_um[keep] * 1e-6, weights=w_all[keep])
        dt_mean = np.average(dt_ms[keep] * 1e-3, weights=w_all[keep])
        yy = np.array([dy_um[keep].min(), dy_um[keep].max()])
        tt_ms = (dt_mean + sy_i * (yy * 1e-6 - dy_mean)) * 1e3
        ax.plot(tt_ms, yy, color="crimson", lw=1.5)
    ax.set(
        xlim=(-dt_xlim, dt_xlim),
        xlabel="dt (ms)",
        ylabel="dy from ref (um)",
        title=f"slowness={sy_i:.3g} s/m\nv={velocity:.3g} m/s ({direction})",
    )
fig.suptitle(
    "xcorr-pick slowness fit, 0.5ms half-window (green=window, red dashed=ref channel, "
    "dots=picks colour-coded by weight, dt axis shared across spikes)"
)
fig.tight_layout()
fig.savefig(FIG_DIR.joinpath(f"{DATE}_waveform_proto_slowness_diagnostic.png"), dpi=150)

# %% 6. Global distributions across all spikes in the snippet
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
finite_spread = spatial_spread_um[np.isfinite(spatial_spread_um)]
axes2[0].hist(finite_spread, bins=60, color="steelblue")
axes2[0].set(xlabel="spatial spread (um)", ylabel="count", title=f"n={finite_spread.size}")

sy_all = df_slowness["slowness_y_s_per_m"].to_numpy()
sy_all = sy_all[np.isfinite(sy_all)]
axes2[1].hist(np.clip(sy_all, -2, 2), bins=80, color="darkorange")
axes2[1].axvline(0, color="k", lw=0.5)
axes2[1].set(
    xlabel="signed slowness (s/m), clipped to +-2",
    ylabel="count",
    title=f"n={sy_all.size}, {np.mean(sy_all > 0) * 100:.0f}% upgoing",
)
fig2.suptitle(f"pid={pid}: {n_spikes} spikes, spread={dt_spread:.2f}s, slowness={dt_slowness:.2f}s")
fig2.tight_layout()
fig2.savefig(FIG_DIR.joinpath(f"{DATE}_waveform_proto_slowness_distributions.png"), dpi=150)

logger.info("Stage 2 done.")
