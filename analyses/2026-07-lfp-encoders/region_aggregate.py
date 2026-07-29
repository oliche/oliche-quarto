"""Region-level pooling of the brain-wide LFP-encoder scores.

Loads the per-``(pid, channel)`` score shards written by the cluster run
(``results_bwm_cluster/<tier>``), remaps each channel to its Beryl-atlas
region and pools across insertions to give, for every (Beryl region, band)
pair, a significance fraction and the median effect sizes needed for the
Swanson/heatmap display.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/olivier/PycharmProjects/ephys-atlas/sdsc-slurms/2026-07_lfp-encoders")
import results_io as rio  # noqa: E402

import iblatlas.atlas as atlas

ALPHA = 0.05
MIN_CHANNELS = 20
MIN_PIDS = 3
BANDS = ["delta", "theta", "beta", "gamma", "raw"]


def load_channel_scores(tier_dir: Path, pids: set[str] | None = None) -> pd.DataFrame:
    """Load one tier's per-channel scores and attach the Beryl region.

    Parameters
    ----------
    tier_dir : Path
        e.g. ``results_bwm_cluster/default``.
    pids : set of str, optional
        If given, restrict to these PIDs (e.g. ``pid_qc.KEEP_PIDS`` to drop the
        data-quality-excluded insertions in ``DATA_ISSUES.md``). Diagnostics
        that characterise collapse itself (``collapse_rate``/``pid_band_r2``)
        should be called with ``pids=None`` -- filtering would hide the thing
        being measured.

    Returns
    -------
    DataFrame
        ``rio.load_scores`` output plus ``beryl_id``/``beryl_acronym``,
        with root/void channels dropped.
    """
    df = rio.load_scores(tier_dir)
    if pids is not None:
        df = df[df["pid"].isin(pids)]
    br = atlas.BrainRegions()
    df["r2_cv"] = df["r2_cv"].clip(-1.0, 1.0)  # near-zero-variance channels blow up unclipped CV R²
    df["beryl_id"] = br.remap(df["atlas_id"].to_numpy(), source_map="Allen", target_map="Beryl")
    df["beryl_acronym"] = br.id2acronym(df["beryl_id"].to_numpy())
    return df[~df["beryl_acronym"].isin(["void", "root"])].reset_index(drop=True)


def region_significance(df: pd.DataFrame) -> pd.DataFrame:
    """Pool channel scores to one row per (beryl_acronym, band).

    Parameters
    ----------
    df : DataFrame
        Output of :func:`load_channel_scores` (or a concatenation across
        tiers with a ``tier`` column added beforehand).

    Returns
    -------
    DataFrame
        ``beryl_acronym``, ``band``, ``n_channels``, ``n_pids``,
        ``frac_sig`` (fraction of channels with ``p_value < ALPHA``),
        ``median_r2_cv``, ``median_r2_full``, ``median_null_p95`` (the
        per-channel null-95th-percentile R², i.e. the noise floor) and one
        ``median_dr2_<group>`` per drop-R² group present. Regions below
        ``MIN_CHANNELS``/``MIN_PIDS`` coverage are dropped.
    """
    dr2_cols = [c for c in df.columns if c.startswith("dr2_")]
    grp = df.groupby(["beryl_acronym", "band"], observed=True)
    out = grp.agg(
        n_channels=("channel", "size"),
        n_pids=("pid", "nunique"),
        frac_sig=("p_value", lambda s: float(np.mean(s < ALPHA))),
        median_r2_cv=("r2_cv", "median"),
        median_r2_full=("r2_full", "median"),
        median_null_p95=("null_p95", "median"),
        **{f"median_{c}": (c, "median") for c in dr2_cols},
    ).reset_index()
    return out[(out["n_channels"] >= MIN_CHANNELS) & (out["n_pids"] >= MIN_PIDS)].reset_index(drop=True)


def pid_band_r2(tier_dirs: dict[str, Path]) -> pd.DataFrame:
    """Median held-out R² per (pid, band, tier), the unit the collapse check operates on.

    Uses the same ``r2_cv`` clip to ``[-1, 1]`` as :func:`load_channel_scores`
    (near-zero-variance channels otherwise blow up unclipped CV R² to
    implausible magnitudes and would dominate the median).

    Parameters
    ----------
    tier_dirs : dict[str, Path]
        Tier name -> its ``results_bwm_cluster/<tier>`` directory.

    Returns
    -------
    DataFrame
        ``pid``, ``band``, ``tier``, ``median_r2_cv``.
    """
    rows = []
    for tier, path in tier_dirs.items():
        df = load_channel_scores(path)
        df = df[df["band"] != "raw"]
        med = df.groupby(["pid", "band"], observed=True)["r2_cv"].median().reset_index(name="median_r2_cv")
        med["tier"] = tier
        rows.append(med)
    return pd.concat(rows, ignore_index=True)


def collapse_rate(tier_dirs: dict[str, Path], threshold: float = -0.5) -> pd.DataFrame:
    """Fraction of PIDs whose fit has collapsed (median held-out R² below ``threshold``) in >=1 band.

    A collapsed fit means the model predicts held-out data *worse than the
    fold mean* across at least half its channels -- the automatic per-PID
    lambda selection (:func:`solve.select_lambda`, maximises median CV-R²
    across ~288 targets) picked a value so under-regularised for this
    insertion that it overfits catastrophically. This is invisible to the
    permutation p-value (the null reuses the same bad lambda and collapses
    in lockstep, so the observed R² can still "beat" all null draws and get
    a floor p-value) and to the marginal ``frac_sig``/retention metrics in
    :func:`region_significance`/:func:`channel_agreement` -- it only shows
    up by looking at R² sign/magnitude directly.

    Parameters
    ----------
    tier_dirs : dict[str, Path]
        Tier name -> its ``results_bwm_cluster/<tier>`` directory.
    threshold : float, default -0.5

    Returns
    -------
    DataFrame
        One row per tier: ``n_pids_collapsed`` (>=1 band below threshold),
        ``frac_pids_collapsed``, ``n_pids_total``, plus one
        ``n_bands_collapsed_<band>`` column per band.
    """
    med = pid_band_r2(tier_dirs)
    bad = med[med["median_r2_cv"] < threshold]
    rows = []
    for tier, path in tier_dirs.items():
        n_total = med[med["tier"] == tier]["pid"].nunique()
        tier_bad = bad[bad["tier"] == tier]
        row = {
            "tier": tier,
            "n_pids_total": n_total,
            "n_pids_collapsed": tier_bad["pid"].nunique(),
            "frac_pids_collapsed": tier_bad["pid"].nunique() / n_total if n_total else np.nan,
        }
        for band, g in tier_bad.groupby("band", observed=True):
            row[f"n_bands_collapsed_{band}"] = g["pid"].nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def channel_agreement(tier_dirs: dict[str, Path], ref: str = "uncompressed", alpha: float = ALPHA) -> pd.DataFrame:
    """Per-band significant-channel agreement between each tier and the reference.

    ``frac_sig`` alone (see :func:`tier_headline`) is a marginal count: a
    tier could lose 10 % of channels here and gain a different 10 % there
    and still show the same marginal fraction, while quietly reshuffling
    *which* channels the analysis would flag. This answers the sharper
    question: of the channels significant in the uncompressed reference,
    how many survive compression (rather than just "how many are
    significant overall").

    Parameters
    ----------
    tier_dirs : dict[str, Path]
        Tier name -> its ``results_bwm_cluster/<tier>`` directory. Must
        include ``ref``.
    ref : str, default "uncompressed"
        Reference tier compared against.
    alpha : float, default 0.05

    Returns
    -------
    DataFrame
        One row per (tier, band) for every non-reference tier:
        ``n_ref_sig``, ``retained`` (significant in both), ``lost``
        (significant in ``ref`` only), ``gained`` (significant in ``tier``
        only), ``retention_rate = retained / n_ref_sig``.
    """
    scores = {tier: rio.load_scores(path)[["pid", "channel", "band", "p_value"]] for tier, path in tier_dirs.items()}
    ref_sig = scores[ref].assign(sig_ref=lambda d: d["p_value"] < alpha).drop(columns="p_value")

    rows = []
    for tier, df in scores.items():
        if tier == ref:
            continue
        merged = df.merge(ref_sig, on=["pid", "channel", "band"])
        merged["sig_tier"] = merged["p_value"] < alpha
        for band, g in merged.groupby("band", observed=True):
            n_ref_sig = int(g["sig_ref"].sum())
            retained = int((g["sig_ref"] & g["sig_tier"]).sum())
            lost = int((g["sig_ref"] & ~g["sig_tier"]).sum())
            gained = int((~g["sig_ref"] & g["sig_tier"]).sum())
            rows.append({
                "tier": tier, "band": band, "n_ref_sig": n_ref_sig,
                "retained": retained, "lost": lost, "gained": gained,
                "retention_rate": retained / n_ref_sig if n_ref_sig else np.nan,
            })
    return pd.DataFrame(rows)


def tier_headline(tier_dirs: dict[str, Path]) -> pd.DataFrame:
    """Median R² and significant-channel fraction per tier and band.

    Parameters
    ----------
    tier_dirs : dict[str, Path]
        Tier name -> its ``results_bwm_cluster/<tier>`` directory.

    Returns
    -------
    DataFrame
        One row per (tier, band): ``median_r2_cv``, ``frac_sig``, ``n_pids``.
    """
    rows = []
    for tier, path in tier_dirs.items():
        df = rio.load_scores(path)
        df["r2_cv"] = df["r2_cv"].clip(-1.0, 1.0)
        for band, sub in df.groupby("band", observed=True):
            rows.append({
                "tier": tier,
                "band": band,
                "median_r2_cv": sub["r2_cv"].median(),
                "frac_sig": float((sub["p_value"] < ALPHA).mean()),
                "n_pids": sub["pid"].nunique(),
            })
    return pd.DataFrame(rows)