"""Swanson-flatmap + region-heatmap significance display, one row per band.

Mirrors the brainwide-map convention (Swanson panels on top, a region-by-band
heatmap below with anatomically-sorted, region-coloured x-tick labels) but for
the LFP-encoder significance fraction per band instead of decoding scores.
"""

from __future__ import annotations

from pathlib import Path

import addcopyfighandler  # noqa: F401
import iblatlas.atlas as atlas
import iblatlas.plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from iblutil.numerical import ismember
from matplotlib.gridspec import GridSpec

sns.set_theme(context="notebook", style="whitegrid")

FIG_DIR = Path.home().joinpath("Documents", "figures")
DATE = "2026-07-06"
CMAP = "RdBu_r"
BANDS = ["delta", "theta", "beta", "gamma"]

# Compression tiers are an ordered variable (none -> default -> aggressive), so
# colour them with a fixed sequential ramp rather than arbitrary categorical
# hues -- darker = more compression -- reused across every tier-comparison figure.
TIER_ORDER = ["uncompressed", "default", "aggressive"]
TIER_COLORS = dict(zip(TIER_ORDER, sns.color_palette("Blues", 3)))


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR.joinpath(f"{DATE}_lfpenc_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def swanson_significance_grid(
    df_region: pd.DataFrame, tier: str, value_col: str = "median_r2_cv",
    label: str = "median held-out R²", vmin: float = -0.1, vmax: float = 0.1,
    bands: list[str] | None = None, kind: str = "real", cmap: str = CMAP,
) -> Path:
    """One Swanson flatmap per band plus an anatomically-sorted heatmap below.

    Parameters
    ----------
    df_region : DataFrame
        Output of ``region_aggregate.region_significance`` (one tier only).
    tier : str
        Compression tier label, used in the title only.
    value_col : str, default "median_r2_cv"
        Column to colour the maps by (e.g. ``"median_r2_cv"`` for the real
        fit, ``"median_null_p95"`` for the null noise floor).
    bands : list of str, optional
        Which bands to show as columns, defaults to delta/theta/beta/gamma.
    kind : {"real", "null", "fracsig", "fracsig_delta_vs_<ref>"}, default "real"
        Tags the title and filename; does not affect the data shown.
    cmap : str, default "RdBu_r"
        Diverging by default (for R²-like quantities centred on zero); pass a
        sequential map (e.g. ``"mako"``) for a strictly-positive quantity
        such as a significant-fraction, or keep ``"RdBu_r"`` for a signed delta.

    Returns
    -------
    Path
        Saved PNG path.
    """
    bands = bands or BANDS
    br = atlas.BrainRegions()

    pivot = df_region.pivot(index="beryl_acronym", columns="band", values=value_col)
    pivot = pivot.reindex(columns=bands)

    fig = plt.figure(figsize=(3.2 * len(bands), 9))
    gs_top = GridSpec(1, len(bands), figure=fig)
    gs_top.update(top=0.95, bottom=0.55, left=0.06, right=0.92, wspace=0.05)
    gs_cbar = GridSpec(1, 1, figure=fig)
    gs_cbar.update(top=0.95, bottom=0.55, left=0.94, right=0.96)
    gs_low = GridSpec(1, 1, figure=fig)
    gs_low.update(top=0.45, bottom=0.05, left=0.06, right=0.96)

    for i, band in enumerate(bands):
        ax = fig.add_subplot(gs_top[i])
        vals = pivot[band].dropna()
        iblatlas.plots.plot_swanson_vector(
            vals.index.to_numpy(), vals.to_numpy(), ax=ax, br=br,
            orientation="portrait", linewidth=0.1, cmap=cmap, vmin=vmin, vmax=vmax,
        )
        ax.set(title=band)
        ax.axis("off")

    cax = fig.add_subplot(gs_cbar[0])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=label)

    ax_heat = fig.add_subplot(gs_low[0])
    _region_heatmap(ax_heat, pivot, br, value_col, vmin, vmax, cmap=cmap)

    fig.suptitle(f"LFP-encoder {label} by band and region  ·  {kind}  ·  {tier} compression", y=0.99)
    return _save(fig, f"swanson_{kind}_{tier}")


def make_real_null_grids(tier_dirs: dict[str, Path], pids: set[str] | None = None) -> list[Path]:
    """Build the real-R²/null-floor Swanson grid for every compression tier.

    Parameters
    ----------
    tier_dirs : dict[str, Path]
        Tier name (``"default"``/``"aggressive"``/``"uncompressed"``) -> its
        ``results_bwm_cluster/<tier>`` directory.
    pids : set of str, optional
        Restrict to these PIDs (e.g. ``pid_qc.KEEP_PIDS``); see
        ``region_aggregate.load_channel_scores``.

    Returns
    -------
    list of Path
        Six saved PNGs: one (real, null) pair per tier.
    """
    import region_aggregate as ra

    paths = []
    for tier, path in tier_dirs.items():
        df_region = ra.region_significance(ra.load_channel_scores(path, pids=pids))
        paths.append(swanson_significance_grid(df_region, tier, value_col="median_r2_cv", kind="real"))
        paths.append(swanson_significance_grid(
            df_region, tier, value_col="median_null_p95", label="median null-floor R² (95th pctile)", kind="null",
        ))
    return paths


def make_fracsig_grids(tier_dirs: dict[str, Path], pids: set[str] | None = None) -> list[Path]:
    """Significant-channel-fraction Swanson grid for every compression tier.

    Unlike raw R² -- not comparable across tiers because each tier's null and
    real R² are normalised against that tier's own compressed ``Y`` variance,
    see ``PLAN.md`` -- ``frac_sig`` is a fair cross-tier yardstick: same
    design, same circular-shift null procedure, only the LFP source changes.
    This is the direct region-resolved read on whether compression costs
    detectable behaviour-LFP coupling.

    Parameters
    ----------
    tier_dirs : dict[str, Path]
        Tier name -> its ``results_bwm_cluster/<tier>`` directory.
    pids : set of str, optional
        Restrict to these PIDs (e.g. ``pid_qc.KEEP_PIDS``); see
        ``region_aggregate.load_channel_scores``.

    Returns
    -------
    list of Path
        One saved PNG per tier.
    """
    import region_aggregate as ra

    paths = []
    for tier, path in tier_dirs.items():
        df_region = ra.region_significance(ra.load_channel_scores(path, pids=pids))
        paths.append(swanson_significance_grid(
            df_region, tier, value_col="frac_sig",
            label="fraction of channels significant (p<0.05)",
            vmin=0.0, vmax=1.0, cmap="mako", kind="fracsig",
        ))
    return paths


def swanson_compression_delta_grid(
    df_ref: pd.DataFrame, df_tier: pd.DataFrame, tier: str, ref: str,
    value_col: str = "frac_sig", vmin: float = -0.4, vmax: float = 0.4,
    bands: list[str] | None = None,
) -> Path:
    """Region x band map of ``tier - ref``, to localise where compression costs sensitivity.

    Parameters
    ----------
    df_ref, df_tier : DataFrame
        ``region_aggregate.region_significance`` output for the reference
        tier (normally ``"uncompressed"``) and the tier being evaluated.
    tier, ref : str
        Tier labels, used in the title/filename.
    value_col : str, default "frac_sig"
        Column to difference; ``frac_sig`` is the cross-tier-comparable one.
    bands : list of str, optional
        Defaults to delta/theta/beta/gamma.

    Returns
    -------
    Path
        Saved PNG path.
    """
    merged = df_tier.merge(df_ref, on=["beryl_acronym", "band"], suffixes=("", "_ref"))
    merged["delta_col"] = merged[value_col] - merged[f"{value_col}_ref"]
    return swanson_significance_grid(
        merged, tier, value_col="delta_col",
        label=f"Δ {value_col} ({tier} − {ref})",
        vmin=vmin, vmax=vmax, bands=bands, cmap="RdBu_r", kind=f"fracsigdelta_vs_{ref}",
    )


def make_compression_delta_grids(
    tier_dirs: dict[str, Path], ref: str = "uncompressed", pids: set[str] | None = None,
) -> list[Path]:
    """Build the compression-impact delta grid for every non-reference tier.

    Parameters
    ----------
    tier_dirs : dict[str, Path]
        Tier name -> its ``results_bwm_cluster/<tier>`` directory. Must
        include ``ref``.
    ref : str, default "uncompressed"
        Reference tier (the pre-compression Cadzow checkpoint).
    pids : set of str, optional
        Restrict to these PIDs (e.g. ``pid_qc.KEEP_PIDS``); see
        ``region_aggregate.load_channel_scores``.

    Returns
    -------
    list of Path
        One saved PNG per non-reference tier (2, for default/aggressive).
    """
    import region_aggregate as ra

    dfs = {tier: ra.region_significance(ra.load_channel_scores(path, pids=pids)) for tier, path in tier_dirs.items()}
    df_ref = dfs[ref]
    return [
        swanson_compression_delta_grid(df_ref, df, tier, ref)
        for tier, df in dfs.items() if tier != ref
    ]


def tier_headline_barplot(tier_dirs: dict[str, Path]) -> Path:
    """Brain-wide significance-fraction summary, one bar group per band.

    The single-number companion to the per-region grids above: does the
    tier ranking (uncompressed >= default >= aggressive, or not) hold up
    consistently across bands. Deliberately plots ``frac_sig`` only, not
    R² -- R² is not comparable across tiers (see ``PLAN.md``).

    Parameters
    ----------
    tier_dirs : dict[str, Path]
        Tier name -> its ``results_bwm_cluster/<tier>`` directory.

    Returns
    -------
    Path
        Saved PNG path.
    """
    import region_aggregate as ra

    df = ra.tier_headline(tier_dirs)
    bands = [b for b in BANDS if b in df["band"].unique()]
    df = df[df["band"].isin(bands)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(bands))
    width = 0.8 / len(TIER_ORDER)
    for i, tier in enumerate(TIER_ORDER):
        sub = df[df["tier"] == tier].set_index("band").reindex(bands)
        ax.bar(x + (i - 1) * width, sub["frac_sig"].to_numpy(), width, label=tier, color=TIER_COLORS[tier])
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel("fraction of channels significant (p<0.05)")
    ax.set_ylim(0, 1)
    ax.legend(title="compression tier", frameon=False)
    ax.set_title("Brain-wide significance fraction by band and compression tier")
    sns.despine(ax=ax)
    fig.tight_layout()
    return _save(fig, "tier_headline_fracsig")


def collapse_rate_barplot(tier_dirs: dict[str, Path], threshold: float = -0.5) -> Path:
    """Fraction of PIDs with a collapsed fit (median held-out R² < threshold in >=1 band).

    The metric neither :func:`tier_headline_barplot` (frac_sig) nor
    :func:`retention_barplot` can see: a whole insertion's per-band model
    fit can catastrophically overfit (see ``region_aggregate.collapse_rate``
    docstring) while still reading as "significant" because its
    permutation null collapses in lockstep. Bars show total collapse rate;
    the annotation breaks it down by band.

    Parameters
    ----------
    tier_dirs : dict[str, Path]
        Tier name -> its ``results_bwm_cluster/<tier>`` directory.
    threshold : float, default -0.5

    Returns
    -------
    Path
        Saved PNG path.
    """
    import region_aggregate as ra

    rate = ra.collapse_rate(tier_dirs, threshold=threshold).set_index("tier")
    tiers = [t for t in TIER_ORDER if t in rate.index]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    y = rate.loc[tiers, "frac_pids_collapsed"].to_numpy() * 100
    bars = ax.bar(tiers, y, color=[TIER_COLORS[t] for t in tiers])
    for bar, tier in zip(bars, tiers):
        n = int(rate.loc[tier, "n_pids_collapsed"])
        ax.annotate(f"{n} PIDs", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"% of PIDs with a collapsed fit\n(median held-out R² < {threshold:.1f} in >=1 band)")
    ax.set_title("Catastrophic-overfit rate by compression tier")
    sns.despine(ax=ax)
    fig.tight_layout()
    return _save(fig, "tier_collapse_rate")


def retention_barplot(tier_dirs: dict[str, Path], ref: str = "uncompressed") -> Path:
    """Channel-level significance-retention rate, one bar group per band.

    Sharper than :func:`tier_headline_barplot`'s marginal ``frac_sig``: this
    is the fraction of channels significant in ``ref`` that *remain*
    significant after compression, i.e. whether compression adds uniform
    noise (high retention) or reshuffles which channels look interesting
    (low retention despite a similar marginal fraction) -- the latter is
    the stronger argument against shipping compressed data for
    channel-resolved analyses.

    Parameters
    ----------
    tier_dirs : dict[str, Path]
        Tier name -> its ``results_bwm_cluster/<tier>`` directory.
    ref : str, default "uncompressed"

    Returns
    -------
    Path
        Saved PNG path.
    """
    import region_aggregate as ra

    agreement = ra.channel_agreement(tier_dirs, ref=ref)
    bands = [b for b in BANDS if b in agreement["band"].unique()]
    tiers = [t for t in TIER_ORDER if t in agreement["tier"].unique()]
    agreement = agreement[agreement["band"].isin(bands)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(bands))
    width = 0.8 / len(tiers)
    offset0 = -(len(tiers) - 1) / 2
    for i, tier in enumerate(tiers):
        sub = agreement[agreement["tier"] == tier].set_index("band").reindex(bands)
        ax.bar(x + (offset0 + i) * width, sub["retention_rate"].to_numpy(), width, label=tier, color=TIER_COLORS[tier])
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel("retention rate")
    ax.set_ylim(0, 1)
    ax.legend(title="compression tier", frameon=False)
    ax.set_title(f"Channel-level significance retention under compression\n(fraction of {ref}-significant channels still significant)")
    sns.despine(ax=ax)
    fig.tight_layout()
    return _save(fig, "tier_retention_rate")


def _region_heatmap(
    ax, pivot: pd.DataFrame, br: atlas.BrainRegions, value_col: str, vmin: float, vmax: float, cmap: str = CMAP,
) -> None:
    """Anatomically-sorted region x band heatmap with region-coloured x-labels."""
    regions = pivot.index.to_numpy()
    _, rind = ismember(regions, br.acronym)
    order = np.argsort(br.order[rind])
    pivot_sorted = pivot.iloc[order]
    rind_sorted = rind[order]

    sns.heatmap(
        pivot_sorted.T, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar=False,
        linewidths=0.2, linecolor="w",
        xticklabels=pivot_sorted.index.to_numpy(), yticklabels=pivot_sorted.columns.to_numpy(),
    )
    ax.tick_params(left=False, bottom=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for tick, rid in zip(ax.get_xticklabels(), rind_sorted):
        color = br.rgba[rid].astype(float) / 255
        if color[:3].sum() > 2.4:  # very light region colours are unreadable as text
            color = color[:3] * 0.35
        tick.set_color(color)
        tick.set_rotation(90)
        tick.set_fontsize(6)
