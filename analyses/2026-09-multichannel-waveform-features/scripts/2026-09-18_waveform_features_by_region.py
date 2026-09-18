"""Cross-plot spatial_spread_um vs slowness_s_per_m, coloured by Allen/Cosmos brain
region, across all BENCHMARKS.py PIDs -- is there a visible distinction per region?

Uses the real production path (ephysatlas.features.spikes() via
IBLPIDFeatureCalculator.compute_snippet), not a local reimplementation, so
`result.features` already has spatial_spread_um/slowness_s_per_m AND the merged
channel metadata (acronym, atlas_id) for free.

Run cell-by-cell (# %% cells). Needs the same environment as the main prototype
script -- see its docstring for the spikeinterface/dartsort compatibility note.
Forces a fresh recompute (skip_saved_computation=False): the scratch cache from
earlier in the session predates spatial_spread_um/slowness_s_per_m.
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
from iblatlas.regions import BrainRegions

from ephysatlas.feature_calculators import (
    FeatureComputationOptions,
    IBLPIDFeatureCalculator,
    SnippetWindow,
)

sns.set_theme(context="notebook")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATE = "2026-09-18"
FIG_DIR = Path.home().joinpath("Documents", "figures")
SCRATCH_DIR = Path.home().joinpath("scratch", "ephysatlas_waveform_proto")
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

from BENCHMARKS import pids  # noqa: E402

one = ONE()
br = BrainRegions()

# dc7e9403 ("static noise: positive spikes") dropped: dartsort.subtract() hangs for
# 2+ hours on it (see 2026-09-18_multichannel_waveform_features_multi_pid.py) -- a
# real robustness gap, not something to work around here.
pids_to_run = [p for p in pids if p != "dc7e9403-19f7-409f-9240-05ee57cb7aea"]
window = SnippetWindow(t_start=300.0, duration_ap=2.0, duration_lf=2.0)


# %% Run every benchmark PID through the real production feature-computation path
def run_pid(pid):
    output_dir = SCRATCH_DIR.joinpath(pid)
    calc = IBLPIDFeatureCalculator(pid=pid, one=one)
    options = FeatureComputationOptions(
        features_to_compute=["ap", "waveforms"],
        output_dir=output_dir,
        save_waveforms=False,
        # Force fresh: earlier-cached waveforms_features.pqt in this scratch dir
        # predates spatial_spread_um/slowness_s_per_m.
        skip_saved_computation=False,
    )
    t0 = time.time()
    result = calc.compute_snippet(window, options)
    dt_compute = time.time() - t0
    df = result.features.copy()
    df["pid"] = pid
    return df, dt_compute


frames = []
for pid in pids_to_run:
    logger.info("=== Running pid=%s ===", pid)
    try:
        df, dt_compute = run_pid(pid)
        logger.info("pid=%s: %.1fs, %d channels", pid, dt_compute, len(df))
        frames.append(df)
    except Exception:
        logger.exception("pid=%s failed, skipping", pid)

df_all = pd.concat(frames, ignore_index=True)
df_all.to_parquet(SCRATCH_DIR.joinpath(f"{DATE}_waveform_features_by_region.pqt"))
logger.info("Total channels across %d PIDs: %d", len(frames), len(df_all))

# %% Attach Cosmos region + its standard IBL colour to every channel
valid_acronym = df_all["acronym"].notna() & (df_all["acronym"] != "void") & (df_all["acronym"] != "root")
df_valid = df_all[valid_acronym].copy()
cosmos_acronym = br.id2acronym(df_valid["atlas_id"].to_numpy(), mapping="Cosmos")
df_valid["cosmos_acronym"] = cosmos_acronym
# Cosmos has no leaf for white matter: anything it can't place falls back to "root".
# Checked (2026-09-18): every such channel here is a real fiber tract (arb, int, fp,
# dhc, opt, ...) -- relabel for a meaningful, distinctly-coloured category instead of
# a misleading "root".
is_fiber = df_valid["cosmos_acronym"] == "root"
df_valid.loc[is_fiber, "cosmos_acronym"] = "fiber tracts"

cosmos_rgb = br.get(br.acronym2id(df_valid["cosmos_acronym"].where(~is_fiber, "root"))).rgb.copy()
cosmos_rgb[is_fiber.to_numpy()] = [120, 120, 120]  # grey, distinct from any Cosmos leaf colour
df_valid["cosmos_hex"] = ["#%02x%02x%02x" % tuple(c) for c in cosmos_rgb]

df_valid = df_valid.dropna(subset=["spatial_spread_um", "slowness_s_per_m"])
logger.info(
    "%d/%d channels have region + both features (%d PIDs, %d Cosmos regions)",
    len(df_valid), len(df_all), df_valid["pid"].nunique(), df_valid["cosmos_acronym"].nunique(),
)

# %% Cross-plot: spatial_spread_um vs slowness_s_per_m, coloured by Cosmos region
fig, ax = plt.subplots(figsize=(8, 6.5))
region_order = (
    df_valid.groupby("cosmos_acronym")["spatial_spread_um"].median().sort_values().index
)
for region in region_order:
    sub = df_valid[df_valid["cosmos_acronym"] == region]
    ax.scatter(
        sub["slowness_s_per_m"], sub["spatial_spread_um"],
        color=sub["cosmos_hex"].iloc[0], label=f"{region} (n={len(sub)})", s=18, alpha=0.7,
        edgecolors="none",
    )
ax.set(
    xlim=(-2, 2),
    xlabel="signed slowness (s/m)",
    ylabel="spatial spread (um)",
    title=f"Per-channel waveform features by Cosmos region\n"
    f"({df_valid['pid'].nunique()} PIDs, {len(df_valid)} channels, {window.duration_ap}s AP @ t={window.t_start}s)",
)
ax.axvline(0, color="k", lw=0.5, zorder=0)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, markerscale=1.5, frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR.joinpath(f"{DATE}_waveform_features_by_region_scatter.png"), dpi=150)

# %% Per-region summary (median + IQR), same Cosmos colours, sorted by spread
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
summary = (
    df_valid.groupby("cosmos_acronym")
    .agg(
        n=("pid", "size"),
        spread_p50=("spatial_spread_um", "median"),
        spread_p25=("spatial_spread_um", lambda x: np.percentile(x, 25)),
        spread_p75=("spatial_spread_um", lambda x: np.percentile(x, 75)),
        slowness_p50=("slowness_s_per_m", "median"),
        slowness_p25=("slowness_s_per_m", lambda x: np.percentile(x, 25)),
        slowness_p75=("slowness_s_per_m", lambda x: np.percentile(x, 75)),
        hex=("cosmos_hex", "first"),
    )
    .sort_values("spread_p50")
)
y = np.arange(len(summary))
axes2[0].barh(
    y, summary["spread_p75"] - summary["spread_p25"], left=summary["spread_p25"],
    color=summary["hex"], alpha=0.7,
)
axes2[0].scatter(summary["spread_p50"], y, color="k", s=15, zorder=3)
axes2[0].set(yticks=y, yticklabels=[f"{a} (n={n})" for a, n in zip(summary.index, summary["n"])])
axes2[0].set_xlabel("spatial spread (um), IQR")

axes2[1].barh(
    y, summary["slowness_p75"] - summary["slowness_p25"], left=summary["slowness_p25"],
    color=summary["hex"], alpha=0.7,
)
axes2[1].scatter(summary["slowness_p50"], y, color="k", s=15, zorder=3)
axes2[1].axvline(0, color="k", lw=0.5, zorder=0)
axes2[1].set_xlabel("signed slowness (s/m), IQR")
axes2[1].set_xlim(-1, 1)

fig2.suptitle("Per-Cosmos-region median + IQR, sorted by spatial spread")
fig2.tight_layout()
fig2.savefig(FIG_DIR.joinpath(f"{DATE}_waveform_features_by_region_summary.png"), dpi=150)

logger.info("Done.")
