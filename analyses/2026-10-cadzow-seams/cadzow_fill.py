"""
Cadzow on the full NP1 grid: virtual channels fill the empty checkerboard positions before the trajectory is built.

NP1 has 2 of the 4 lateral positions per 20 µm row, so the 2-D block-Hankel trajectory of a window is half zeros,
which the SVD treats as data. Here each empty grid position gets a virtual channel, the average of its 4-neighbours
(all real channels on a checkerboard), and `cadzow_denoiser` runs on the 768-position full grid with windows of the
same depth (nswx and ovx doubled). Only the real channels are returned. Optional imputation passes replace the virtual
channels by the denoised estimate and run again.
"""
import numpy as np
from ibldsp import cadzow


def full_grid(h):
    """
    Full 4-column grid of an NP1 probe and the grid index of each real channel.

    Parameters
    ----------
    h : dict
        Probe geometry with 'x' and 'y' [µm].

    Returns
    -------
    hg : dict
        Grid geometry {'x', 'y'}, ordered by row then column (same order as NP1 channels).
    ireal : ndarray (nc,)
        Grid index of each real channel.
    """
    xu, yu = np.unique(h['x']), np.unique(h['y'])
    gx, gy = np.meshgrid(xu, yu)  # (ny, nx)
    ireal = np.searchsorted(yu, h['y']) * xu.size + np.searchsorted(xu, h['x'])
    return {'x': gx.ravel(), 'y': gy.ravel()}, ireal


def fill_virtual(wav, h, ireal, good=None):
    """
    Put the real channels on the full grid and fill every other position (empty checkerboard positions and bad
    channels) with the mean of its known 4-neighbours.

    Known positions are the good real channels. Positions without a known neighbour (e.g. inside a cluster of bad
    channels) are filled in later passes from the positions filled before them.

    Parameters
    ----------
    wav : ndarray (nc, ns)
    h : dict
        Full-grid geometry from `full_grid`.
    ireal : ndarray (nc,)
    good : ndarray (nc,) of bool, optional
        Good real channels; bad ones are filled like empty positions. Default: all good.

    Returns
    -------
    ndarray (n_grid, ns)
    """
    nx, ny = np.unique(h['x']).size, np.unique(h['y']).size
    g = np.zeros((ny, nx, wav.shape[1]))
    known = np.zeros((ny, nx), dtype=bool)
    g.reshape(ny * nx, -1)[ireal] = wav
    known.ravel()[ireal] = True if good is None else good
    while not known.all():
        acc, cnt = np.zeros_like(g), np.zeros((ny, nx))
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            src = (slice(max(dy, 0), ny + min(dy, 0)), slice(max(dx, 0), nx + min(dx, 0)))
            dst = (slice(max(-dy, 0), ny + min(-dy, 0)), slice(max(-dx, 0), nx + min(-dx, 0)))
            acc[dst] += g[src] * known[src][..., None]
            cnt[dst] += known[src]
        new = ~known & (cnt > 0)
        g[new] = acc[new] / cnt[new][:, None]
        known |= new
    return g.reshape(ny * nx, -1)


def cadzow_full_grid(wav, h, n_impute=0, bad=None, **kwargs):
    """
    `ibldsp.cadzow.cadzow_denoiser` on the full NP1 grid with interpolated virtual channels.

    Parameters
    ----------
    wav : ndarray (nc, ns)
        LFP [V].
    h : dict
        Probe geometry with 'x' and 'y' [µm].
    n_impute : int
        Extra passes in which the virtual channels are replaced by the previous denoised estimate.
    bad : ndarray (nc,) of bool, optional
        Bad channels: their data is discarded, they are filled like the empty grid positions and their output is the
        Cadzow estimate (no interpolation before or after is needed).
    **kwargs
        Forwarded to `cadzow_denoiser`; ``nswx`` and ``ovx`` are given in real channels and doubled on the grid.

    Returns
    -------
    ndarray (nc, ns)
        Denoised real channels.
    """
    hg, ireal = full_grid(h)
    good = None if bad is None else ~np.asarray(bad, dtype=bool)
    igood = ireal if good is None else ireal[good]
    kwargs = {**kwargs, 'nswx': 2 * kwargs.get('nswx', 64), 'ovx': 2 * kwargs.get('ovx', 32)}
    g = fill_virtual(wav, hg, ireal, good=good)
    for _ in range(n_impute + 1):
        out = cadzow.cadzow_denoiser(g, h=hg, **kwargs)
        g = out.astype(np.float64)
        g[igood] = wav if good is None else wav[good]  # good channels back to the data, the rest keep the estimate
    return out[ireal]
