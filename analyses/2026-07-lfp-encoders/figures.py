"""Figures for the LFP-encoding model.

Six displays covering PLAN.md's five plus the Step-4 validation overlay:

1. kernel depth x lag heatmaps (one panel per event, CSD-style);
2. drop-R² depth profiles per regressor group;
3. fitted-kernel vs event-triggered-average overlays (alignment validation);
4. raw-vs-band R² small multiples (the D1 comparison);
5. diagnostics -- regressor collinearity + held-out R²(lambda) curve;
6. observed vs circular-shift-null R² (significance).

All figures use the seaborn notebook theme and are written, date-prefixed, to
``~/Documents/figures``.
"""

from __future__ import annotations

from pathlib import Path

import addcopyfighandler  # noqa: F401  (enables ctrl-c copy of the active figure)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context="notebook", style="whitegrid")

FIG_DIR = Path.home().joinpath("Documents", "figures")
DATE = "2026-07-04"  # fixed at authoring time; do not bump on re-runs


def _save(fig: plt.Figure, name: str) -> Path:
    """Write ``fig`` to the figures directory with the fixed date prefix."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR.joinpath(f"{DATE}_lfpenc_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def _band_columns(meta: pd.DataFrame, band: str) -> np.ndarray:
    """Row indices of ``meta`` (i.e. target columns) belonging to ``band``, depth-sorted."""
    sub = meta.index[meta["band"] == band].to_numpy()
    return sub[np.argsort(meta.loc[sub, "axial_um"].to_numpy())]


# --- 1. kernel depth x lag heatmaps ---------------------------------------
def kernel_depth_lag(res, band: str, events: list[str] | None = None) -> Path:
    """Heatmaps of each event kernel over depth (rows) and lag (columns)."""
    events = events or ["stimOn_on", "move_on", "feedback_on"]
    meta = res.target_meta
    cols = _band_columns(meta, band)
    depths = meta.loc[cols, "axial_um"].to_numpy()

    fig, axes = plt.subplots(1, len(events), figsize=(4.2 * len(events), 5), sharey=True, squeeze=False)
    for ax, ev in zip(axes[0], events):
        K = res.kernels[ev][:, cols].T  # (n_depth, n_lag)
        vmax = np.abs(K).max() or 1.0
        im = ax.pcolormesh(res.taus, depths, K, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_title(ev)
        ax.set_xlabel("lag from event (s)")
        fig.colorbar(im, ax=ax, fraction=0.046, label="kernel weight")
    axes[0, 0].set_ylabel("depth along probe (µm)")
    fig.suptitle(f"{res.pid[:8]}  ·  {res.kind} / {band}  ·  event kernels")
    fig.tight_layout()
    return _save(fig, f"kernels_{res.kind}_{band}")


# --- 2. drop-R² depth profiles --------------------------------------------
def dr2_depth(res, band: str) -> Path:
    """Cross-validated drop-R² vs depth, one line per regressor group."""
    meta = res.target_meta
    cols = _band_columns(meta, band)
    depths = meta.loc[cols, "axial_um"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.5, 6))
    for group, dr in res.dr2.items():
        ax.plot(dr[cols], depths, marker="o", ms=3, label=group)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("cross-validated ΔR² (full − without group)")
    ax.set_ylabel("depth along probe (µm)")
    ax.set_title(f"{res.pid[:8]}  ·  {res.kind} / {band}  ·  variance attributed by group")
    ax.legend(title="group", fontsize=8)
    fig.tight_layout()
    return _save(fig, f"dr2depth_{res.kind}_{band}")


# --- 3. kernel vs event-triggered average ---------------------------------
def event_triggered_average(
    Y: np.ndarray, tvec: np.ndarray, fs: float, event_times: np.ndarray, taus: np.ndarray, col: int
) -> np.ndarray:
    """Average of target ``col`` in a lag window ``taus`` (s) around each event."""
    lags = np.round(taus * fs).astype(int)
    idx = np.searchsorted(tvec, event_times[np.isfinite(event_times)])
    span = int(np.abs(lags).max())
    idx = idx[(idx >= span) & (idx < len(tvec) - span)]
    return Y[idx[:, None] + lags[None, :], col].mean(0)


def kernel_vs_eta(res, targets, event_times: np.ndarray, event: str, col: int) -> Path:
    """Overlay the fitted onset kernel against the raw event-triggered average."""
    eta = event_triggered_average(targets.Y, targets.tvec, targets.fs, event_times, res.taus, col)
    K = res.kernels[event][:, col]
    row = res.target_meta.iloc[col]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(res.taus, eta - eta.mean(), label="event-triggered average", color="0.4", lw=2)
    ax.plot(res.taus, K, label="fitted kernel (deconvolved)", color="C3", lw=2)
    ax.axvline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel(f"lag from {event.split('_')[0]} (s)")
    ax.set_ylabel(f"{res.kind} response ({row['band']})")
    ax.set_title(f"{res.pid[:8]}  ·  ch {row['channel']} {row['acronym']} @ {row['axial_um']:.0f}µm")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, f"kernelvseta_{res.kind}_{row['band']}_ch{row['channel']}")


# --- 4. raw-vs-band R² small multiples ------------------------------------
def raw_vs_band(res_raw, res_band) -> Path:
    """R² depth profiles: raw voltage beside each band-power envelope."""
    panels = ["raw"] + list(pd.unique(res_band.target_meta["band"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.7 * len(panels), 5.5), sharey=True)
    for ax, panel in zip(axes, panels):
        res = res_raw if panel == "raw" else res_band
        cols = _band_columns(res.target_meta, panel)
        depths = res.target_meta.loc[cols, "axial_um"].to_numpy()
        ax.plot(res.r2_cv[cols], depths, marker="o", ms=3, color="C0")
        ax.set_title(panel)
        ax.set_xlabel("held-out R²")
    axes[0].set_ylabel("depth along probe (µm)")
    fig.suptitle(f"{res_band.pid[:8]}  ·  raw voltage vs band-power envelopes (D1)")
    fig.tight_layout()
    return _save(fig, "rawvsband_r2")


# --- 5. diagnostics: collinearity + R²(lambda) ----------------------------
def diagnostics(design, lam_curve: tuple[np.ndarray, np.ndarray]) -> Path:
    """Regressor collinearity matrix and the held-out R²(lambda) tuning curve."""
    corr = np.corrcoef(design.base, rowvar=False)
    lambdas, curve = lam_curve

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 5))
    im = a0.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    a0.set_xticks(range(len(design.base_names)))
    a0.set_yticks(range(len(design.base_names)))
    a0.set_xticklabels(design.base_names, rotation=90, fontsize=7)
    a0.set_yticklabels(design.base_names, fontsize=7)
    a0.set_title("base-regressor collinearity")
    fig.colorbar(im, ax=a0, fraction=0.046, label="Pearson r")

    a1.semilogx(lambdas, curve, marker="o")
    a1.set_xlabel("smoothness λ")
    a1.set_ylabel("median held-out R²")
    a1.set_title("regularisation tuning")
    fig.tight_layout()
    return _save(fig, "diagnostics")


# --- 6. observed vs null R² ------------------------------------------------
def null_comparison(res, null: np.ndarray, band: str) -> Path:
    """Observed held-out R² vs the circular-shift null, for one band."""
    cols = _band_columns(res.target_meta, band)
    obs = res.r2_cv[cols]
    null_band = null[:, cols].ravel()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(null_band, bins=40, density=True, color="0.7", label="circular-shift null")
    ax.hist(obs, bins=20, density=True, color="C3", alpha=0.6, label="observed (per channel)")
    q95 = np.quantile(null_band, 0.95)
    ax.axvline(q95, color="k", ls="--", lw=1, label=f"null 95th pct = {q95:.4f}")
    ax.set_xlabel("held-out R²")
    ax.set_ylabel("density")
    ax.set_title(f"{res.pid[:8]}  ·  {res.kind} / {band}  ·  significance vs null")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, f"null_{res.kind}_{band}")
