"""Cross-recording kernel-weight rasters, anatomically sorted.

Pools the fitted event kernels for one regressor (or one band) across every
insertion and channel into a single wide raster -- lag on the y-axis, one
column per (pid, channel) sorted by Beryl-atlas anatomical order, with a
region-colour strip -- in the style of the population rastermap-vs-region
comparison used for the ephys-atlas cell aggregates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import addcopyfighandler  # noqa: F401
import iblatlas.atlas as atlas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from iblutil.numerical import ismember

sns.set_theme(context="notebook", style="ticks", font_scale=1.0)

sys.path.insert(0, "/Users/olivier/PycharmProjects/ephys-atlas/sdsc-slurms/2026-07_lfp-encoders")
import results_io as rio  # noqa: E402

FIG_DIR = Path.home().joinpath("Documents", "figures")
DATE = "2026-07-06"
BANDS = ["delta", "theta", "beta", "gamma"]


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR.joinpath(f"{DATE}_lfpenc_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


class TierKernels:
    """Anatomically-ordered channel catalogue and lazy kernel-slice loader for one tier.

    Parameters
    ----------
    tier_dir : Path
        e.g. ``results_bwm_cluster/default``.
    pids : set of str, optional
        Restrict to these PIDs (e.g. ``pid_qc.KEEP_PIDS`` to drop the
        data-quality-excluded insertions in ``DATA_ISSUES.md``).
    """

    def __init__(self, tier_dir: Path, pids: set[str] | None = None) -> None:
        self.dir = Path(tier_dir)
        config = json.loads(self.dir.joinpath("model_config.json").read_text())
        self.base_names: list[str] = config["base_names"]
        self.groups: dict[str, list[int]] = config["groups"]
        basis = np.load(self.dir.joinpath("basis.npz"))
        self.B: np.ndarray = basis["B"]
        self.taus: np.ndarray = basis["taus"]
        self.n_basis = self.B.shape[1]

        scores = rio.load_scores(self.dir)
        if pids is not None:
            scores = scores[scores["pid"].isin(pids)]
        pvals = scores.pivot(index=["pid", "channel"], columns="band", values="p_value")
        pvals.columns = [f"p_{c}" for c in pvals.columns]

        meta = scores[scores["band"] == "delta"][["pid", "channel", "acronym", "atlas_id"]].reset_index(drop=True)
        br = atlas.BrainRegions()
        meta["beryl_id"] = br.remap(meta["atlas_id"].to_numpy(), source_map="Allen", target_map="Beryl")
        meta["beryl_acronym"] = br.id2acronym(meta["beryl_id"].to_numpy())
        meta = meta[~meta["beryl_acronym"].isin(["void", "root"])].copy()
        meta = meta.join(pvals, on=["pid", "channel"])
        _, rind = ismember(meta["beryl_id"].to_numpy(), br.id)
        meta["order"] = br.order[rind]
        meta["rind"] = rind
        self.meta = meta.sort_values("order", kind="stable").reset_index(drop=True)
        self.br = br

    def base_index(self, name: str) -> int:
        return self.base_names.index(name)

    def significant_mask(self, band: str, alpha: float = 0.05) -> np.ndarray:
        """Boolean mask over :attr:`meta` rows, True where that channel's ``band`` fit beat the null.

        Parameters
        ----------
        band : {"delta", "theta", "beta", "gamma", "raw"}
        alpha : float, default 0.05
            Permutation p-value threshold.
        """
        return (self.meta[f"p_{band}"] < alpha).to_numpy()

    def raster(self, base_name: str, band: str) -> np.ndarray:
        """Kernel-weight raster for one regressor and band, columns anatomically sorted.

        Parameters
        ----------
        base_name : str
            Entry of ``base_names`` (e.g. ``"stimOn_on"``).
        band : {"delta", "theta", "beta", "gamma", "raw"}

        Returns
        -------
        ndarray, shape (n_lags, n_channels)
            Reconstructed kernel, columns ordered as :attr:`meta`.
        """
        kind = "raw" if band == "raw" else "band"
        bi = self.base_index(base_name)
        block = slice(bi * self.n_basis, (bi + 1) * self.n_basis)
        by_pid = {pid: g["channel"].to_numpy() for pid, g in self.meta.groupby("pid", sort=False)}

        out_cols = []
        for pid, chans in by_pid.items():
            d = np.load(self.dir.joinpath("kernels", f"{pid}_{kind}.npz"))
            W = d["W"]
            n_targets = W.shape[1]
            n_ch_total = n_targets // (4 if kind == "band" else 1)
            if kind == "band":
                boff = BANDS.index(band) * n_ch_total
                cols = boff + chans
            else:
                cols = chans
            if block.stop > W.shape[0]:
                # Gated regressor (pupil) absent from this PID's design entirely --
                # not just zero-weighted -- so the column block doesn't exist.
                K = np.full((len(self.taus), len(cols)), np.nan, dtype=np.float32)
            else:
                K = (self.B @ W[block][:, cols]).astype(np.float32)  # (n_lags, n_chans)
            out_cols.append(K)
        return np.concatenate(out_cols, axis=1)

    def region_strip(self, meta: pd.DataFrame | None = None) -> np.ndarray:
        """RGB image row, one pixel per column of :meth:`raster`, shape (1, n_channels, 3).

        Parameters
        ----------
        meta : DataFrame, optional
            A (possibly significance-filtered) subset of :attr:`meta`, in the
            order matching the raster's columns. Defaults to the full catalogue.
        """
        meta = self.meta if meta is None else meta
        return self.br.rgb[meta["rind"].to_numpy()].astype(np.uint8)[np.newaxis, :, :]

    def region_boundaries(self, level: str = "cosmos", meta: pd.DataFrame | None = None) -> list[tuple[int, str]]:
        """Tick positions at the start of each contiguous coarse-region block.

        Parameters
        ----------
        level : {"cosmos", "beryl"}
            Which grouping to draw boundaries/labels for.
        meta : DataFrame, optional
            A (possibly significance-filtered) subset of :attr:`meta`, in the
            order matching the raster's columns. Defaults to the full catalogue.

        Returns
        -------
        list of (int, str)
            Column index and label for each block's midpoint.
        """
        meta = self.meta if meta is None else meta
        if level == "cosmos":
            ids = self.br.remap(meta["beryl_id"].to_numpy(), source_map="Beryl", target_map="Cosmos")
            labels = self.br.id2acronym(ids)
        else:
            labels = meta["beryl_id"].map(lambda i: self.br.id2acronym([i])[0]).to_numpy()
        change = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
        return [((change[i] + change[i + 1]) // 2, labels[change[i]]) for i in range(len(change) - 1)]


def _vmax(K: np.ndarray) -> float:
    return float(np.nanpercentile(np.abs(K), 99)) or 1.0


_CMAP = plt.get_cmap("RdBu_r").copy()
_CMAP.set_bad("0.85")  # gated regressor absent for this PID (e.g. no pupil coverage)


def plot_pid_channel_kernel_compare(
    pid: str, channel: int, band: str, base_name: str, tier_dirs: dict[str, Path],
) -> Path:
    """Overlay one channel's fitted kernel for one regressor across compression tiers.

    A spot-check for the kernel-outlier / catastrophic-overfit diagnosis: loads
    the raw ``W`` for ``pid`` from every tier, reconstructs the lag-domain
    kernel for one ``(base_name, band, channel)`` and overlays the traces,
    each labelled with its held-out R² and selected lambda -- so a collapsed
    tier (huge amplitude, deeply negative R², tiny lambda) is visible directly
    against a healthy one on the same axes.

    Parameters
    ----------
    pid : str
    channel : int
        Channel index within this PID (0-based, as stored in scores/kernels).
    band : {"delta", "theta", "beta", "gamma"}
    base_name : str
        Regressor to display (e.g. ``"feedback_on"``).
    tier_dirs : dict[str, Path]
        Tier name -> its ``results_bwm_cluster/<tier>`` directory.

    Returns
    -------
    Path
        Saved PNG path.
    """
    any_dir = next(iter(tier_dirs.values()))
    basis = np.load(any_dir.joinpath("basis.npz"))
    B, taus = basis["B"], basis["taus"]
    config = json.loads(any_dir.joinpath("model_config.json").read_text())
    base_names, n_basis = config["base_names"], config["n_basis"]
    r = base_names.index(base_name)
    block = slice(r * n_basis, (r + 1) * n_basis)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    acronym = "?"
    for tier, tier_dir in tier_dirs.items():
        W = np.load(tier_dir.joinpath("kernels", f"{pid}_band.npz"))["W"]
        n_ch = W.shape[1] // len(BANDS)
        bi = BANDS.index(band)
        col = bi * n_ch + channel
        if block.stop > W.shape[0]:
            continue  # gated regressor (e.g. pupil) absent from this PID's design entirely
        K = B @ W[block][:, col]

        scores = rio.load_scores(tier_dir)
        row = scores[(scores["pid"] == pid) & (scores["band"] == band) & (scores["channel"] == channel)]
        r2 = row["r2_cv"].to_numpy()[0] if len(row) else np.nan
        lam = row["lam"].to_numpy()[0] if len(row) else np.nan
        if len(row):
            acronym = row["acronym"].to_numpy()[0]
        ax.plot(taus, K, label=f"{tier}  (CV R²={r2:.2f}, λ={lam:g})")

    ax.axhline(0, color="0.7", lw=0.8)
    ax.axvline(0, color="0.7", lw=0.8, ls="--")
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("kernel weight")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{pid}  ·  ch{channel} ({acronym})  ·  {base_name}, {band}")
    sns.despine(ax=ax)
    fig.tight_layout()
    return _save(fig, f"kernel_pid_{pid[:8]}_ch{channel}_{band}_{base_name}")


def plot_regressor_all_bands(
    tk: TierKernels, base_name: str, tier: str, bands: list[str] | None = None,
    sig_only: bool = True, alpha: float = 0.05,
) -> Path:
    """One regressor, all bands stacked (+ raw), each restricted to its own significant channels.

    Each band has a different significance mask (the overall band fit's
    permutation p-value), so -- unlike :func:`plot_band_regressor_family` --
    every row gets its own anatomically-sorted x-axis and region strip.

    Parameters
    ----------
    tk : TierKernels
    base_name : str
        Regressor to display (e.g. ``"stimOn_on"``).
    tier : str
        Compression tier label, for the title.
    bands : list of str, optional
        Defaults to delta/theta/beta/gamma/raw.
    sig_only : bool, default True
        Restrict each band's columns to channels significant (``p_value < alpha``)
        for that band's overall encoding fit.
    alpha : float, default 0.05

    Returns
    -------
    Path
        Saved PNG path.
    """
    bands = bands or [*BANDS, "raw"]

    fig, axes = plt.subplots(
        2 * len(bands), 1, figsize=(16, 1.9 * len(bands)),
        gridspec_kw={"height_ratios": [1, 0.25] * len(bands)},
    )
    for i, band in enumerate(bands):
        ax, axr = axes[2 * i], axes[2 * i + 1]
        mask = tk.significant_mask(band, alpha) if sig_only else np.ones(len(tk.meta), dtype=bool)
        meta_sub = tk.meta[mask]
        n_ch = int(mask.sum())
        K = tk.raster(base_name, band)[:, mask]
        vmax = _vmax(K)
        im = ax.imshow(K, aspect="auto", cmap=_CMAP, vmin=-vmax, vmax=vmax,
                        extent=[0, n_ch, tk.taus[-1], tk.taus[0]])
        ax.set_ylabel(f"{band}\nlag (s)", fontsize=8)
        ax.tick_params(bottom=False, labelbottom=False)
        fig.colorbar(im, ax=ax, fraction=0.01, pad=0.005)

        labels = tk.region_boundaries("cosmos", meta_sub)
        axr.imshow(tk.region_strip(meta_sub), aspect="auto", extent=[0, n_ch, 0, 1])
        axr.set_yticks([])
        axr.set_xticks([c for c, _ in labels])
        axr.set_xticklabels([lab for _, lab in labels], rotation=90, fontsize=6)
        axr.tick_params(bottom=False)
        for spine in axr.spines.values():
            spine.set_visible(False)

    sig_note = f" (p<{alpha} per band)" if sig_only else ""
    fig.suptitle(f"{base_name}  ·  kernel weight, all bands{sig_note}  ·  {tk.meta['pid'].nunique()} PIDs  ·  {tier}")
    fig.tight_layout()
    tag = "sig" if sig_only else "all"
    return _save(fig, f"kernelraster_regressor_{base_name}_{tier}_{tag}")


def plot_band_regressor_family(
    tk: TierKernels, band: str, base_names: list[str], tier: str,
    sig_only: bool = True, alpha: float = 0.05,
    vmax_by_name: dict[str, float] | None = None, rasters: dict[str, np.ndarray] | None = None,
    mask: np.ndarray | None = None, mask_note: str | None = None,
) -> Path:
    """One band, a family of regressors stacked, same anatomically-sorted x-axis.

    All rows share one band, hence one significance mask, hence one shared
    x-axis and region strip (unlike :func:`plot_regressor_all_bands`).

    Parameters
    ----------
    tk : TierKernels
    band : {"delta", "theta", "beta", "gamma", "raw"}
    base_names : list of str
        Regressors to display as separate rows (e.g. the wheel family:
        ``["wheel_vel", "wheel_speed"]``).
    tier : str
        Compression tier label, for the title.
    sig_only : bool, default True
        Restrict columns to channels significant (``p_value < alpha``) for
        this band's overall encoding fit. Ignored if ``mask`` is given.
    alpha : float, default 0.05
    vmax_by_name : dict[str, float], optional
        Fixed colour-scale per regressor (e.g. pooled across compression
        tiers by :func:`make_all_tier_kernelrasters`), so the same weight
        magnitude renders as the same colour across tier figures. Defaults
        to this tier's own 99th-percentile (the old, tier-local behaviour).
    rasters : dict[str, ndarray], optional
        Precomputed, already-masked ``(n_lags, n_ch)`` raster per regressor
        (avoids recomputing when the caller already built it for
        ``vmax_by_name``). Defaults to calling :meth:`TierKernels.raster`.
    mask : ndarray, optional
        Explicit boolean mask over :attr:`TierKernels.meta` rows, e.g. a
        cross-tier intersection built by :func:`make_all_tier_kernelrasters`
        so every tier's figure shows the exact same channels. Overrides
        ``sig_only``/``alpha`` when given.
    mask_note : str, optional
        Text describing how ``mask`` was built, shown in the title in place
        of the default ``p<alpha`` note.

    Returns
    -------
    Path
        Saved PNG path.
    """
    if mask is None:
        mask = tk.significant_mask(band, alpha) if sig_only else np.ones(len(tk.meta), dtype=bool)
    meta_sub = tk.meta[mask]
    n_ch = int(mask.sum())
    labels = tk.region_boundaries("cosmos", meta_sub)
    strip = tk.region_strip(meta_sub)

    fig, axes = plt.subplots(
        len(base_names) + 1, 1, figsize=(16, 1.6 * len(base_names) + 0.6),
        gridspec_kw={"height_ratios": [1] * len(base_names) + [0.15]}, sharex=True,
    )
    for ax, name in zip(axes[:-1], base_names):
        K = rasters[name] if rasters is not None else tk.raster(name, band)[:, mask]
        vmax = vmax_by_name[name] if vmax_by_name is not None else _vmax(K)
        im = ax.imshow(K, aspect="auto", cmap=_CMAP, vmin=-vmax, vmax=vmax,
                        extent=[0, n_ch, tk.taus[-1], tk.taus[0]])
        ax.set_ylabel(f"{name}\nlag (s)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.01, pad=0.005)

    axr = axes[-1]
    axr.imshow(strip, aspect="auto", extent=[0, n_ch, 0, 1])
    axr.set_yticks([])
    axr.set_xticks([c for c, _ in labels])
    axr.set_xticklabels([lab for _, lab in labels], rotation=90, fontsize=6)
    axr.tick_params(bottom=False)

    if mask_note is not None:
        sig_note = f" ({mask_note})"
    else:
        sig_note = f" (p<{alpha})" if sig_only else ""
    fig.suptitle(f"{band}  ·  kernel weight, regressor family{sig_note}  ·  {n_ch} channels / {tk.meta['pid'].nunique()} PIDs  ·  {tier}")
    fig.tight_layout()
    tag = "sig" if (sig_only or mask is not None) else "all"
    return _save(fig, f"kernelraster_band_{band}_family_{tier}_{tag}")


def _shared_significant_mask(tks: dict[str, "TierKernels"], intersect_on: tuple[str, ...], band: str, alpha: float) -> dict[str, np.ndarray]:
    """Per-tier boolean mask selecting the (pid, channel) intersection significant in every ``intersect_on`` tier.

    Each tier sorts its own channel catalogue independently, so row order
    can't be assumed to match across tiers -- the intersection is built by
    an explicit ``(pid, channel)`` key rather than positionally, then mapped
    back onto each tier's own row order.

    Parameters
    ----------
    tks : dict[str, TierKernels]
    intersect_on : tuple of str
        Tier names whose significance must all hold (e.g. ``("uncompressed", "default")``).
    band : str
    alpha : float

    Returns
    -------
    dict[str, ndarray]
        One boolean mask per tier in ``tks``, aligned to that tier's own ``meta`` order.
    """
    keep_keys = None
    for tier in intersect_on:
        tk = tks[tier]
        sig = tk.significant_mask(band, alpha)
        keys = set(zip(tk.meta.loc[sig, "pid"], tk.meta.loc[sig, "channel"]))
        keep_keys = keys if keep_keys is None else (keep_keys & keys)
    return {
        tier: pd.MultiIndex.from_frame(tk.meta[["pid", "channel"]]).isin(keep_keys)
        for tier, tk in tks.items()
    }


def make_all_tier_kernelrasters(
    tier_dirs: dict[str, Path], bands: list[str] | None = None, sig_only: bool = True, alpha: float = 0.05,
    pids: set[str] | None = None, intersect_on: tuple[str, ...] = ("uncompressed", "default"),
) -> list[Path]:
    """Band-regressor-family kernel raster for every band, repeated across every compression tier.

    Same figure already produced for the ``default`` tier, generated for
    ``aggressive``/``uncompressed`` too so the fitted kernel *shape* -- not
    just the scalar R²/frac_sig -- can be flipped between tiers to check
    whether compression distorts the spatiotemporal structure of the fit
    (not only its significance). Each regressor's colour scale (vmax) is
    pooled **across the three tiers** before plotting, so a given colour
    means the same weight magnitude in every tier's figure -- otherwise
    each figure's independent 99th-percentile scaling hides exactly the
    tier-to-tier magnitude differences (e.g. compression outliers) this
    comparison exists to reveal.

    All tiers' rasters display the same set of ``(pid, channel)`` columns --
    those significant in **every** ``intersect_on`` tier -- rather than each
    tier's own independent significance mask, so a column-for-column
    comparison across tier figures is showing the same physical channels,
    not two different subsets that happen to both pass p<0.05.

    Parameters
    ----------
    tier_dirs : dict[str, Path]
        Tier name -> its ``results_bwm_cluster/<tier>`` directory.
    bands : list of str, optional
        Defaults to delta/theta/beta/gamma/raw.
    sig_only, alpha
        See :func:`plot_band_regressor_family`.
    pids : set of str, optional
        Restrict to these PIDs (e.g. ``pid_qc.KEEP_PIDS``).
    intersect_on : tuple of str, default ``("uncompressed", "default")``
        Tiers whose significance masks are intersected to pick the shared
        channel set applied to every tier's figure. Ignored if ``sig_only``
        is False.

    Returns
    -------
    list of Path
        ``len(tier_dirs) * len(bands)`` saved PNGs.
    """
    bands = bands or [*BANDS, "raw"]
    tks = {tier: TierKernels(tier_dir, pids=pids) for tier, tier_dir in tier_dirs.items()}
    base_names = next(iter(tks.values())).base_names
    mask_note = f"p<{alpha}, {' ∩ '.join(intersect_on)}" if sig_only else None

    paths = []
    for band in bands:
        if sig_only:
            masks = _shared_significant_mask(tks, intersect_on, band, alpha)
        else:
            masks = {tier: np.ones(len(tk.meta), dtype=bool) for tier, tk in tks.items()}
        rasters_by_tier = {
            tier: {name: tk.raster(name, band)[:, masks[tier]] for name in base_names}
            for tier, tk in tks.items()
        }
        vmax_by_name = {
            name: _vmax(np.concatenate([rasters_by_tier[tier][name].ravel() for tier in tks]))
            for name in base_names
        }
        for tier, tk in tks.items():
            paths.append(plot_band_regressor_family(
                tk, band, base_names, tier, vmax_by_name=vmax_by_name, rasters=rasters_by_tier[tier],
                mask=masks[tier], mask_note=mask_note,
            ))
    return paths