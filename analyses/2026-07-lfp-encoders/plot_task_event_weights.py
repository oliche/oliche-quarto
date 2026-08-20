"""Kernel-weight rasters for all 4 task events, every compression tier, v01 vs v01_smart.

24 figures (4 events x 3 tiers x 2 versions): one task event per regressor
(stimOn_on/move_on/feedback_on/wheel_speed), all bands stacked with a single shared
region-strip axis (`plot_feedback_move_weights.plot_regressor_shared_axis`). For a given
event, all 6 (tier x version) figures share **one** colour-scale vmax per band and the
**same fixed (pid, channel) column order** -- verified identical across all 6 (asserted,
not assumed) -- so any two of the 6 can be flipped between and a difference in colour or
column position is a real difference, not a scale/ordering artefact.
"""

import numpy as np

import figures_kernelraster as fkr
import plot_feedback_move_weights as pfw

EVENTS = ["stimOn_on", "move_on", "feedback_on", "wheel_speed"]
TIERS = ["default", "aggressive", "uncompressed"]


def _assert_fixed_insertions(tks: dict[str, dict[str, fkr.TierKernels]]) -> None:
    """Fail loudly rather than silently misalign columns across figures."""
    flat = [(f"{tier}/{version}", tk) for tier, by_version in tks.items() for version, tk in by_version.items()]
    ref_name, ref_tk = flat[0]
    ref = ref_tk.meta[["pid", "channel"]].reset_index(drop=True)
    for name, tk in flat[1:]:
        other = tk.meta[["pid", "channel"]].reset_index(drop=True)
        if not ref.equals(other):
            raise ValueError(f"insertion/channel set+order differs: {ref_name} vs {name}")


def main() -> None:
    fkr.DATE = "2026-07-26"
    bands = [*fkr.BANDS, "raw"]

    # build every tier's TierKernels once (6 total), reused across all 4 events
    tks = {tier: {v: fkr.TierKernels(d) for v, d in pfw.version_dirs(tier).items()} for tier in TIERS}
    _assert_fixed_insertions(tks)
    flat_tks = [(tier, version, tk) for tier, by_version in tks.items() for version, tk in by_version.items()]

    for event in EVENTS:
        # one vmax per band, pooled across all 6 (tier, version) rasters for this event --
        # not just within a tier pair -- so the colour scale is fixed across every figure
        # a user might flip between for this task variable.
        vmax_by_band = {
            band: fkr._vmax(np.concatenate([tk.raster(event, band).ravel() for _, _, tk in flat_tks]))
            for band in bands
        }
        for tier, version, tk in flat_tks:
            path = pfw.plot_regressor_shared_axis(
                tk, event, tier=f"{version}_{tier}", bands=bands, vmax_by_band=vmax_by_band,
            )
            print("saved", path)


if __name__ == "__main__":
    main()
