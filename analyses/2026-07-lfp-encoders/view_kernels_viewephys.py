"""Browse fitted encoding-model kernels in viewephys, interactively.

Treats a regressor's pooled kernel array (one channel per row, lag on the time axis)
as an "ephys" trace so viewephys's pan/zoom/gain image viewer can be used to browse
kernel weights across channels/depth, exactly like browsing a raw recording.

Usage -- run inside IPython so the Qt window stays responsive alongside the prompt
(per viewephys's own docs: use the `%gui qt` magic first if the window doesn't show):

    %gui qt
    %run -i view_kernels_viewephys.py
    ve = show_kernel("dab512bd-a02d-4c1f-8dbc-9155a163efc0", "band", "feedback_on", band="delta")
    ve2 = show_kernel("dab512bd-a02d-4c1f-8dbc-9155a163efc0", "raw", "move_on", title="move_on raw")

Or from a plain script (blocks until the window is closed):

    python view_kernels_viewephys.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/olivier/PycharmProjects/ephys-atlas/sdsc-slurms/2026-07_lfp-encoders")
import results_io as rio  # noqa: E402

TIER_DIR = Path(__file__).resolve().parent.joinpath("results_bwm_v01_smart", "default")
BANDS = ["delta", "theta", "beta", "gamma"]


def load_kernel_view(
    pid: str, kind: str, base_name: str, tier_dir: Path = TIER_DIR, band: str | None = None,
) -> tuple[np.ndarray, float, float]:
    """Build a ``(channel, lag)`` array + sample rate/t0 for one fitted kernel.

    Parameters
    ----------
    pid : str
        Probe insertion UUID.
    kind : {"raw", "band"}
    base_name : str
        Regressor to display (e.g. ``"feedback_on"``, ``"move_on"``).
    tier_dir : Path, default the v01_smart ``default`` tier
        Run directory holding ``model_config.json``/``basis.npz``/``kernels/``.
    band : str, optional
        Required when ``kind == "band"`` (one of :data:`BANDS`); ignored for ``"raw"``.

    Returns
    -------
    data : ndarray, shape (n_channels, n_lags)
        Kernel weight, channel order matching the stored ``W`` columns (raw
        electrode/binned-channel order, not anatomically re-sorted).
    fs : float
        Samples/second along the lag axis (``1 / (taus[1] - taus[0])``).
    t0 : float
        Time (s) of the first lag sample (negative -- the acausal window edge).
    """
    W, _ = rio.load_kernels(pid, kind, tier_dir)
    B, taus = rio.load_basis(tier_dir)
    config = json.loads(Path(tier_dir).joinpath("model_config.json").read_text())
    base_index = config["base_names"].index(base_name)
    K = rio.expand_kernel(W, base_index, B)  # (n_lags, n_targets)

    if kind == "band":
        if band is None:
            raise ValueError(f"band is required for kind='band' (one of {BANDS})")
        n_ch = K.shape[1] // len(BANDS)
        b = BANDS.index(band)
        K = K[:, b * n_ch:(b + 1) * n_ch]

    fs = 1.0 / (taus[1] - taus[0])
    t0 = float(taus[0])
    return K.T.astype(np.float32), fs, t0


def show_kernel(
    pid: str, kind: str, base_name: str, tier_dir: Path = TIER_DIR, band: str | None = None,
    title: str | None = None,
):
    """Load one kernel and open it in a viewephys window (channel x lag image)."""
    from viewephys.gui import viewephys

    data, fs, t0 = load_kernel_view(pid, kind, base_name, tier_dir, band=band)
    label = base_name if band is None else f"{base_name} {band}"
    return viewephys(data, fs=fs, t0=t0, title=title or f"{pid[:8]} {label} ({kind})")


if __name__ == "__main__":
    from viewephys.gui import create_app

    app = create_app()
    ve = show_kernel("dab512bd-a02d-4c1f-8dbc-9155a163efc0", "band", "feedback_on", band="delta")
    app.exec()
