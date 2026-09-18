"""Consistency check across a few more benchmark PIDs for ibleatools #123.

Reruns the spatial-spread + xcorr-pick slowness pipeline from
2026-09-18_multichannel_waveform_features.py on additional BENCHMARKS.py PIDs to
check the feature distributions and timing hold up outside the single PID used to
develop it. Same window (t_start=300s, 2s AP) and same method (final version:
xcorr+parabolic picks, axial-only slowness, no lateral component -- see that
script's history for why).

Run cell-by-cell (# %% cells). Needs the same environment as the main script --
see its docstring for the spikeinterface/dartsort compatibility note.
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
from ibldsp.utils import parabolic_max

sns.set_theme(context="notebook")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATE = "2026-09-18"
FIG_DIR = Path.home().joinpath("Documents", "figures")
SCRATCH_DIR = Path.home().joinpath("scratch", "ephysatlas_waveform_proto")
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

from BENCHMARKS import pids  # noqa: E402

one = ONE()

HALF_WINDOW_MS = 0.5
MIN_CHANNELS = 4
MIN_WEIGHT_FRAC = 0.1


# %% Core pipeline, copied verbatim from 2026-09-18_multichannel_waveform_features.py
# (picks_xcorr / _weighted_lstsq / compute_windowed_slowness / spatial spread call) --
# kept identical there so any drift between the two scripts would be a bug, not a
# feature. See that script for the full derivation/validation notes.
def _windowed_segment(arr, i, pt, fs, half_win, hann):
    t0_, t1_ = pt - half_win, pt + half_win + 1
    w0, w1 = max(0, -t0_), len(hann) - max(0, t1_ - arr.shape[1])
    t0c, t1c = max(0, t0_), min(arr.shape[1], t1_)
    if t1c <= t0c:
        return None, t0c, t1c
    return arr[i, t0c:t1c, :] * hann[w0:w1, np.newaxis], t0c, t1c


def picks_xcorr(seg, ref_row):
    win = seg.shape[0]
    ref = seg[:, ref_row]
    n_fft = 2 * win
    R = np.fft.rfft(ref, n=n_fft)
    S = np.fft.rfft(seg, n=n_fft, axis=0)
    corr_full = np.fft.fftshift(np.fft.irfft(np.conj(R)[:, np.newaxis] * S, n=n_fft, axis=0), axes=0)
    lags = np.arange(n_fft) - n_fft // 2
    keep_lag = np.abs(lags) <= (win - 1)
    corr, lags_c = corr_full[keep_lag, :], lags[keep_lag]
    ipeak, cpeak = parabolic_max(np.abs(corr).T)
    lag_at_peak = lags_c[0] + ipeak
    norm = np.sqrt(np.sum(ref**2) * np.sum(seg**2, axis=0))
    weight = np.where(norm > 0, cpeak / norm, np.nan)
    return lag_at_peak, weight


def _weighted_lstsq(X, y, w):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, np.newaxis], y * sw, rcond=None)
    return beta


def compute_windowed_slowness(
    arr, df_peak, xy_um, fs=30_000.0, half_window_ms=HALF_WINDOW_MS,
    min_channels=MIN_CHANNELS, min_weight_frac=MIN_WEIGHT_FRAC,
):
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


def spatial_spread(denoised, df_peak, neighbor_xy_um):
    channel_geometry_3d = np.dstack(
        [neighbor_xy_um[:, :, 0], neighbor_xy_um[:, :, 1], np.zeros_like(neighbor_xy_um[:, :, 0])]
    )
    eu_dist_um = wf.dist_chanel_from_peak(channel_geometry_3d, df_peak["peak_trace_idx"].to_numpy())
    weights_abs = np.abs(wf.weights_spk_ch(denoised, weight_type="peak"))
    return wf.spatial_spread_weighted(eu_dist_um, weights_abs)


def run_pid(pid, window, one, scratch_dir):
    """Run the ap+waveforms pipeline for one PID and return per-spike results."""
    output_dir = scratch_dir.joinpath(pid)
    calc = IBLPIDFeatureCalculator(pid=pid, one=one)
    options = FeatureComputationOptions(
        features_to_compute=["ap", "waveforms"],
        output_dir=output_dir,
        save_waveforms=True,
        skip_saved_computation=True,
    )
    t0 = time.time()
    result = calc.compute_snippet(window, options)
    dt_compute = time.time() - t0

    waveforms_dir = result.snippet_level_dir.joinpath("waveforms")
    denoised = np.load(waveforms_dir.joinpath("denoised.npy")).astype(np.float64)
    channel_index = np.load(waveforms_dir.joinpath("waveform_channels.npy"))
    df_spikes = pd.read_parquet(waveforms_dir.joinpath("spikes.pqt"))
    n_spikes = denoised.shape[0]

    snippet = calc.get_destriped_snippet(window)
    geometry = snippet.geometry
    xy_um = np.c_[geometry["x"], geometry["y"]]
    xy_padded = np.vstack([xy_um, [np.nan, np.nan]])
    neighbor_real_idx = channel_index[df_spikes["channel"].to_numpy()]
    neighbor_xy_um = xy_padded[neighbor_real_idx]

    df_peak = wf.find_peak(denoised)

    t0 = time.time()
    spread_um = spatial_spread(denoised, df_peak, neighbor_xy_um)
    dt_spread = time.time() - t0

    t0 = time.time()
    df_slowness = compute_windowed_slowness(denoised, df_peak, neighbor_xy_um)
    dt_slowness = time.time() - t0

    return dict(
        pid=pid,
        n_spikes=n_spikes,
        n_channels=denoised.shape[2],
        dt_compute=dt_compute,
        dt_spread=dt_spread,
        dt_slowness=dt_slowness,
        spread_um=spread_um,
        slowness_y_s_per_m=df_slowness["slowness_y_s_per_m"].to_numpy(),
    )


# %% Run on a few more benchmark PIDs (pid[0] already covered in the main script).
# pid[7] (dc7e9403..., "static noise: positive spikes") was tried as a deliberate
# stress test and dropped: dartsort.subtract() at the default detection_threshold=4.0
# was still running after >2h on it (vs ~110s normal) when killed -- almost certainly
# because the static/positive-noise character triggers far more spurious threshold
# crossings, blowing up the number of "spikes" fit/subtracted per chunk. A genuine
# robustness finding for #123 (very noisy snippets could make waveform-feature
# computation pathologically slow), but not something to block this consistency
# check on -- swapped in pids[3] instead.
pids_to_run = [pids[0], pids[1], pids[2], pids[3]]
window = SnippetWindow(t_start=300.0, duration_ap=2.0, duration_lf=2.0)

results = []
for pid in pids_to_run:
    logger.info("=== Running pid=%s ===", pid)
    try:
        results.append(run_pid(pid, window, one, SCRATCH_DIR))
    except Exception:
        logger.exception("pid=%s failed, skipping", pid)

# %% Summary table
rows = []
for r in results:
    spread = r["spread_um"]
    slow = r["slowness_y_s_per_m"]
    spread_f = spread[np.isfinite(spread)]
    slow_f = slow[np.isfinite(slow)]
    rows.append(dict(
        pid=r["pid"],
        n_spikes=r["n_spikes"],
        dt_compute_s=round(r["dt_compute"], 1),
        dt_spread_s=round(r["dt_spread"], 3),
        dt_slowness_s=round(r["dt_slowness"], 3),
        spread_p50_um=round(np.median(spread_f), 1),
        spread_p5_um=round(np.percentile(spread_f, 5), 1),
        spread_p95_um=round(np.percentile(spread_f, 95), 1),
        slowness_p50=round(np.median(slow_f), 4),
        slowness_iqr=round(np.percentile(slow_f, 75) - np.percentile(slow_f, 25), 4),
        frac_upgoing=round(float(np.mean(slow_f > 0)), 3),
    ))
df_summary = pd.DataFrame(rows)
logger.info("Summary across PIDs:\n%s", df_summary.to_string(index=False))
df_summary.to_csv(SCRATCH_DIR.joinpath(f"{DATE}_multi_pid_summary.csv"), index=False)

# %% Comparison figure: spatial spread + slowness distributions, one row per PID
n_pid = len(results)
fig, axes = plt.subplots(n_pid, 2, figsize=(11, 3 * n_pid), sharex="col", squeeze=False)
for row, r in enumerate(results):
    spread_f = r["spread_um"][np.isfinite(r["spread_um"])]
    slow_f = r["slowness_y_s_per_m"][np.isfinite(r["slowness_y_s_per_m"])]

    ax = axes[row, 0]
    ax.hist(spread_f, bins=60, color="steelblue")
    ax.set(ylabel=f"{r['pid'][:8]}\ncount")
    if row == 0:
        ax.set_title("spatial spread (um)")
    if row == n_pid - 1:
        ax.set_xlabel("spatial spread (um)")

    ax = axes[row, 1]
    ax.hist(np.clip(slow_f, -2, 2), bins=80, color="darkorange")
    ax.axvline(0, color="k", lw=0.5)
    frac_up = np.mean(slow_f > 0) * 100
    if row == 0:
        ax.set_title("signed slowness (s/m), clipped to +-2")
    ax.set(ylabel=f"n={slow_f.size}, {frac_up:.0f}% up")
    if row == n_pid - 1:
        ax.set_xlabel("slowness (s/m)")
fig.suptitle("Consistency check across benchmark PIDs (t_start=300s, 2s AP)")
fig.tight_layout()
fig.savefig(FIG_DIR.joinpath(f"{DATE}_waveform_proto_multi_pid_consistency.png"), dpi=150)

logger.info("Multi-PID consistency check done.")
