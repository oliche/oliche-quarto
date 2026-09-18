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


# %% Slowness (v4): now just calls the real ibldsp.waveforms.compute_slowness (moved
# there from this script -- see int-brain-lab/ibl-neuropixel#94). Two changes from v3,
# both from looking at the v3 diagnostic panel on real data:
#
# 1. Picks now come from wf.chained_xcorr_pick, walking outward from the reference
#    (peak) channel by axial distance and cross-correlating only ADJACENT channels,
#    instead of wf.xcorr_pick correlating every channel against one fixed distant
#    reference. Spike 4592 (rightmost panel below) showed why: below channel 10 the
#    v3 picks jumped to a different phase of the wavelet -- a cycle skip, where the
#    waveform shape had drifted enough from the reference that a distant, unrelated
#    lag of the correlogram became the taller peak. Adjacent channels are always
#    similar enough that this can't happen; see chained_xcorr_pick's docstring and
#    test_chained_xcorr_avoids_cycle_skip in ibl-neuropixel for a clean synthetic case.
# 2. The fit is now a full 3D (dx, dy) plane, but only the axial term is kept as
#    slowness_s_per_m. Not equivalent to ignoring dx: dx and dy are mildly correlated
#    by the probe's channel layout (staggered columns), so including dx as a
#    covariate controls for that and gives a less biased axial estimate than the v3
#    1D-in-dy-only fit -- even though dx's own coefficient (the lateral slowness) is
#    still too noise-dominated to report on its own, per the v2/v3 history above.
HALF_WINDOW_MS = 0.5
MIN_WEIGHT_FRAC = 0.1
half_win_samples = int(round(HALF_WINDOW_MS * 1e-3 * 30_000.0))
hann_full = np.hanning(2 * half_win_samples + 1)

t0 = time.time()
df_peak_slowness = wf.compute_slowness(
    denoised, df_peak.copy(), channel_geometry_3d, half_window_ms=HALF_WINDOW_MS
)
dt_slowness = time.time() - t0
slowness_s_per_m = df_peak_slowness["slowness_s_per_m"].to_numpy()
n_valid = int(np.isfinite(slowness_s_per_m).sum())
logger.info(
    "wf.compute_slowness: %.2fs for %d spikes (%.3f ms/spike), %d/%d valid, %.2f%% upgoing",
    dt_slowness,
    n_spikes,
    dt_slowness / n_spikes * 1e3,
    n_valid,
    n_spikes,
    float(np.mean(slowness_s_per_m[np.isfinite(slowness_s_per_m)] > 0)) * 100,
)


# %% 5. Diagnostic panels: chained-pick walk order, Hanning window, and picks overlaid
# on the wiggle plot (top row); the weighted dt-vs-dy fit behind slowness_s_per_m
# (bottom row), dt axis shared within each panel for comparability. Three sets of 6
# spikes for variety -- the first repeats stage 1's random sample (includes spike 4592,
# the cycle-skip example above), the other two are fresh random draws.
def plot_slowness_diagnostic(idx_set, fname_suffix, seed_label):
    dt_ms_by_spike, dy_um_by_spike, w_by_spike, keep_by_spike, order_by_spike = {}, {}, {}, {}, {}
    for i in idx_set:
        trace = int(df_peak["peak_trace_idx"].iloc[i])
        seg, t0c, t1c = wf.hanning_window_segment(
            denoised, i, int(df_peak["peak_time_idx"].iloc[i]), half_win_samples, hann_full
        )
        x_i, y_i = neighbor_xy_um[i, :, 0], neighbor_xy_um[i, :, 1]
        valid_ch = np.isfinite(x_i) & np.isfinite(y_i)
        order = np.flatnonzero(valid_ch)
        order = order[np.argsort(y_i[order])]
        lag, weight = wf.chained_xcorr_pick(seg, trace, order)
        dt_ms = lag / 30_000.0 * 1e3
        dy_um = y_i - y_i[trace]
        valid = valid_ch & np.isfinite(weight) & np.isfinite(dt_ms) & (weight > 0)
        w_all = np.where(valid, weight, 0.0)
        keep = valid & (w_all >= MIN_WEIGHT_FRAC * w_all.max())
        dt_ms_by_spike[i], dy_um_by_spike[i] = dt_ms, dy_um
        w_by_spike[i], keep_by_spike[i], order_by_spike[i] = w_all, keep, order

    dt_xlim = 1.15 * max(np.abs(dt_ms_by_spike[i][keep_by_spike[i]]).max() for i in idx_set)

    fig, axes = plt.subplots(2, len(idx_set), figsize=(3.2 * len(idx_set), 8))
    for col, i in enumerate(idx_set):
        pt = int(df_peak["peak_time_idx"].iloc[i])
        trace = int(df_peak["peak_trace_idx"].iloc[i])
        seg, t0c, t1c = wf.hanning_window_segment(denoised, i, pt, half_win_samples, hann_full)
        dt_ms, dy_um = dt_ms_by_spike[i], dy_um_by_spike[i]
        w_all, keep = w_by_spike[i], keep_by_spike[i]
        sy_i = slowness_s_per_m[i]
        velocity = 1 / sy_i if np.isfinite(sy_i) and sy_i != 0 else np.nan
        direction = "up" if sy_i > 0 else "down" if sy_i < 0 else "?"

        ax = axes[0, col]
        wf.double_wiggle(denoised[i].T, fs=30_000, ax=ax, scale=wiggle_scale)
        ax.axvspan(t0c / 30_000.0 * 1e3, t1c / 30_000.0 * 1e3, color="green", alpha=0.15)
        ax.axhline(trace + 1, color="red", lw=0.5, ls="--")
        # overlay each channel's chained pick (absolute time = ref peak time + dt)
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
            title=f"slowness={sy_i:.3g} s/m (axial term of 3D fit)\nv={velocity:.3g} m/s ({direction})",
        )
    fig.suptitle(
        f"Chained-pick slowness fit ({seed_label}), 0.5ms half-window (green=window, "
        "red dashed=ref channel, dots=picks colour-coded by weight, dt axis shared "
        "within this panel)"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR.joinpath(f"{DATE}_waveform_proto_slowness_diagnostic{fname_suffix}.png"), dpi=150)
    plt.close(fig)


plot_slowness_diagnostic(idx_show, "", "same 6 spikes as Figure 1")
rng2 = np.random.default_rng(1)
plot_slowness_diagnostic(rng2.choice(n_spikes, size=n_show, replace=False), "_2", "random set 2")
rng3 = np.random.default_rng(2)
plot_slowness_diagnostic(rng3.choice(n_spikes, size=n_show, replace=False), "_3", "random set 3")

# %% 6. Global distributions across all spikes in the snippet
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
finite_spread = spatial_spread_um[np.isfinite(spatial_spread_um)]
axes2[0].hist(finite_spread, bins=60, color="steelblue")
axes2[0].set(xlabel="spatial spread (um)", ylabel="count", title=f"n={finite_spread.size}")

sy_all = slowness_s_per_m[np.isfinite(slowness_s_per_m)]
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
