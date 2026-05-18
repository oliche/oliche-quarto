"""
Multi-channel LFP denoising pipeline for Neuropixel probes (NP1 and NP2).

Adds geometry-aware denoising functions and a modular evaluation framework
that sits alongside the existing ibldsp pipeline.

Usage (standalone script):
    python lfp_pipeline.py

Usage (import):
    from lfp_pipeline import cadzow_geometry, adaptive_svd_denoise, column_car
    from lfp_pipeline import run_pipeline, evaluate_xcorr_metrics, plot_pipeline_comparison
"""

import itertools
import numpy as np
import scipy.fft
import scipy.signal
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

import neuropixel
import ibldsp.voltage
import ibldsp.cadzow
import ibldsp.fourier


# =============================================================================
# New denoising techniques
# =============================================================================

def cadzow_geometry(wav, fs=2500, rank=4, niter=1, fmax=200, h=None, nswx=None, ovx=None):
    """
    Geometry-aware Cadzow rank denoiser for NP1 and NP2.

    Unlike cadzow_np1 (which slides a window over all channels in probe order),
    this version processes each probe column separately along the y-axis.
    This is better for NP2 whose two straight columns are spatially independent,
    and also valid for NP1's four columns.

    The 1-D trajectory matrix within each column uses only y-coordinates,
    which is the natural spatial axis for a single column.

    Parameters
    ----------
    wav : np.ndarray (nc, ns)
    fs : float
    rank : int  low-rank approximation order
    niter : int  Cadzow iterations
    fmax : float  only de-rank frequencies up to fmax Hz
    h : dict  trace header (neuropixel.trace_header); inferred from nc if None
    nswx : int  sliding window size along y (None → whole column at once)
    ovx : int   overlap between windows (None → nswx // 4)

    Returns
    -------
    np.ndarray (nc, ns)
    """
    nc, ns = wav.shape
    if h is None:
        version = 1 if nc == 384 else 2
        h = neuropixel.trace_header(version=version)

    fscale = scipy.fft.rfftfreq(ns, d=1 / fs)
    imax = int(np.searchsorted(fscale, fmax))
    WAV = scipy.fft.rfft(wav.astype(np.float64))
    WAV_out = WAV.copy()

    for col in np.unique(h['col']):
        idx = np.where(h['col'] == col)[0]
        isort = np.argsort(h['y'][idx])
        itr = idx[isort]          # channels in this column, sorted by y
        nc_col = len(itr)
        y_col = h['y'][itr].astype(float)
        x_col = np.zeros(nc_col)  # 1-D along y only

        _nswx = nswx if nswx is not None else nc_col
        _ovx = ovx if ovx is not None else max(1, _nswx // 4)
        _ovx = min(_ovx, _nswx // 2)

        if nc_col <= _nswx:
            WAV_out[itr, :] = ibldsp.cadzow.denoise(
                WAV[itr, :], x=x_col, y=y_col, r=rank, imax=imax, niter=niter
            )
        else:
            WAV_col = np.zeros_like(WAV[itr, :])
            gain = np.zeros(nc_col)
            # Hann overlap-add weights
            hann_ramp = scipy.signal.windows.hann(_ovx * 2)[:_ovx]
            stride = _nswx - _ovx
            nwin = int(np.ceil((nc_col - _ovx) / stride))
            for i in range(nwin):
                i0 = i * stride
                i1 = min(i0 + _nswx, nc_col)
                sl = slice(i0, i1)
                nw = i1 - i0
                gw = np.ones(nw)
                if i > 0:
                    gw[:_ovx] = hann_ramp
                if i < nwin - 1 and nw == _nswx:
                    gw[-_ovx:] = hann_ramp[::-1]
                block = ibldsp.cadzow.denoise(
                    WAV[itr[sl], :], x=x_col[sl], y=y_col[sl],
                    r=rank, imax=imax, niter=niter
                )
                WAV_col[sl, :] += block * gw[:, np.newaxis]
                gain[sl] += gw
            gain = np.maximum(gain, 1e-10)
            WAV_out[itr, :] = WAV_col / gain[:, np.newaxis]

    return scipy.fft.irfft(WAV_out, n=ns)


def adaptive_svd_denoise(wav, h=None, safety_factor=1.0):
    """
    SVD denoising with rank selection via the Gavish-Donoho (2014) optimal
    hard threshold for unknown noise level (median-based estimator).

    Operates per probe column so that spatially coherent signal components
    within each column are preserved.

    Parameters
    ----------
    wav : np.ndarray (nc, ns)
    h : dict  trace header; inferred from nc if None
    safety_factor : float  multiply threshold by this (>1 keeps more signal)

    Returns
    -------
    np.ndarray (nc, ns)  low-rank reconstruction
    """
    nc, ns = wav.shape
    if h is None:
        version = 1 if nc == 384 else 2
        h = neuropixel.trace_header(version=version)

    wav_out = np.zeros_like(wav, dtype=float)

    for col in np.unique(h['col']):
        idx = np.where(h['col'] == col)[0]
        block = wav[idx, :].astype(float)
        nc_col, ns_col = block.shape

        U, s, Vt = np.linalg.svd(block, full_matrices=False)

        # Gavish-Donoho universal threshold for unknown σ:
        # τ = ω(β) * median(s)   where β = min/max aspect ratio
        beta = nc_col / ns_col if nc_col < ns_col else ns_col / nc_col
        omega = _gavish_donoho_omega(beta)
        threshold = safety_factor * omega * np.median(s)
        rank = int(np.sum(s > threshold))
        rank = max(1, min(rank, nc_col - 1))

        wav_out[idx, :] = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]

    return wav_out


def _gavish_donoho_omega(beta):
    """
    ω(β) factor for the Gavish-Donoho (2014) universal threshold.
    beta = min(nc, ns) / max(nc, ns) ∈ (0, 1].
    Equation from Table 1 of Gavish & Donoho 2014.
    """
    lam_sq = (
        (2 * (beta + 1) + 8 * beta)
        / (beta + 1 + np.sqrt(beta**2 + 14 * beta + 1))
    )
    # μ_β ≈ median of Marchenko-Pastur: approximated via empirical formula
    # μ_β ≈ 1 + sqrt(β) for our range; refined numerical value:
    mp_median = _mp_median(beta)
    return np.sqrt(2 * (beta + 1) + (8 * beta) / (beta + 1 + np.sqrt(beta**2 + 14 * beta + 1))) / mp_median


def _mp_median(beta, n_samples=1000):
    """Estimate the median of the Marchenko-Pastur distribution numerically."""
    x = np.linspace((1 - np.sqrt(beta)) ** 2, (1 + np.sqrt(beta)) ** 2, n_samples)
    pdf = np.sqrt(np.maximum((1 + np.sqrt(beta)) ** 2 - x, 0) * np.maximum(x - (1 - np.sqrt(beta)) ** 2, 0))
    pdf /= (2 * np.pi * beta * x)
    pdf = np.nan_to_num(pdf)
    cdf = np.cumsum(pdf) / np.sum(pdf)
    return x[np.searchsorted(cdf, 0.5)]


def column_car(wav, h=None):
    """
    Common Average Reference applied per probe column.

    More spatially precise than global CAR. For NP2 the two columns sit at
    x=27 and x=59 μm and may have independent DC drift / slow fluctuations.
    For NP1 (4 columns at x=11,27,43,59) each column gets its own reference.

    Parameters
    ----------
    wav : np.ndarray (nc, ns)
    h : dict  trace header

    Returns
    -------
    np.ndarray (nc, ns)
    """
    nc = wav.shape[0]
    if h is None:
        version = 1 if nc == 384 else 2
        h = neuropixel.trace_header(version=version)

    wav_out = np.empty_like(wav, dtype=float)
    for col in np.unique(h['col']):
        idx = np.where(h['col'] == col)[0]
        wav_out[idx, :] = wav[idx, :] - np.median(wav[idx, :], axis=0)
    return wav_out


def global_pca_filter(wav, n_components=3):
    """
    Remove the top n_components global PCA components.

    Targets spatially broad artifacts (mains hum, breathing, heartbeat,
    common-mode pickup) that project onto the leading singular vectors.

    Parameters
    ----------
    wav : np.ndarray (nc, ns)
    n_components : int  number of global components to remove

    Returns
    -------
    np.ndarray (nc, ns)  residual after global components removed
    """
    wav_f = wav.astype(float)
    U, s, Vt = np.linalg.svd(wav_f, full_matrices=False)
    global_component = (U[:, :n_components] * s[:n_components]) @ Vt[:n_components, :]
    return wav_f - global_component


# =============================================================================
# Double-density interleaved CSD + Cadzow
# =============================================================================

def compute_dense_csd(wav, h):
    """
    Build a double-density CSD by interleaving two sets of spatial diffs per column:
      - adjacent diff:   wav[i+1] - wav[i]   at y-midpoint (y_i + y_{i+1}) / 2
      - skip-one diff:   wav[i+2] - wav[i]   at y-midpoint (y_i + y_{i+2}) / 2

    Both are normalised by their respective dy so units are consistent (V/µm).
    Sorting the two sets together by y position gives ~2× the spatial density of
    the standard CSD1, which gives Cadzow a richer trajectory matrix to work with.

    Parameters
    ----------
    wav : np.ndarray (nc, ns)
    h   : probe geometry dict with 'x', 'y', 'col'

    Returns
    -------
    dense  : np.ndarray (n_dense, ns)
    h_dense: dict with 'x', 'y', 'col' arrays matching the rows of dense
    """
    segments_d, segments_y, segments_x, segments_col = [], [], [], []

    for col in np.unique(h['col']):
        idx = np.where(h['col'] == col)[0]
        isort = np.argsort(h['y'][idx])
        itr = idx[isort]
        y = h['y'][itr].astype(float)
        x_col = float(h['x'][itr[0]])
        data = wav[itr, :].astype(float)

        # adjacent diff normalised by spacing
        dy_adj = y[1:] - y[:-1]                          # (nc_col-1,)
        d_adj = np.diff(data, axis=0) / dy_adj[:, None]  # (nc_col-1, ns)
        y_adj = (y[:-1] + y[1:]) / 2

        # skip-one diff normalised by spacing
        dy_skip = y[2:] - y[:-2]                              # (nc_col-2,)
        d_skip = (data[2:] - data[:-2]) / dy_skip[:, None]   # (nc_col-2, ns)
        y_skip = (y[:-2] + y[2:]) / 2

        # merge and sort by y
        all_y = np.concatenate([y_adj, y_skip])
        all_d = np.concatenate([d_adj, d_skip], axis=0)
        order = np.argsort(all_y)

        segments_d.append(all_d[order])
        segments_y.append(all_y[order])
        segments_x.append(np.full(order.size, x_col))
        segments_col.append(np.full(order.size, col))

    dense = np.concatenate(segments_d, axis=0)
    h_dense = {
        'x':   np.concatenate(segments_x),
        'y':   np.concatenate(segments_y),
        'col': np.concatenate(segments_col).astype(int),
    }
    return dense, h_dense


def cadzow_dense_csd(wav, fs, h, rank=4, fmax=200, nswx=None, ovx=None):
    """
    bandpass → dense interleaved CSD → Cadzow on the dense representation.

    The dense CSD has ~2× the spatial sampling of the standard CSD1.
    Cadzow denoises in this denser space, then the result is returned as-is
    (i.e. the output is a denoised dense CSD, not reconstructed voltage).

    Parameters
    ----------
    wav  : np.ndarray (nc, ns)  bandpassed voltage
    fs   : float
    h    : probe geometry
    rank : int
    fmax : float
    nswx, ovx : Cadzow window parameters (None → whole column at once)

    Returns
    -------
    dense_denoised : np.ndarray (n_dense, ns)
    h_dense        : geometry dict for the dense output
    """
    dense, h_dense = compute_dense_csd(wav, h)
    dense_denoised = cadzow_geometry(dense, fs=fs, rank=rank, fmax=fmax,
                                     h=h_dense, nswx=nswx, ovx=ovx)
    return dense_denoised, h_dense


def match_dense_channels(h_dense_np1, h_dense_np2, intercept, slope, n_pairs=80):
    """
    Match NP1 and NP2 dense CSD channel positions using the same linear y-transform
    used for the raw channel matching  (y_np1 = intercept + slope * y_np2).

    Returns
    -------
    idx_np1, idx_np2 : index arrays into h_dense_np1 / h_dense_np2
    """
    y1 = h_dense_np1['y']
    y2 = h_dense_np2['y']
    y2_mapped = intercept + slope * y2

    # keep NP1 positions that fall inside the NP2 mapped range
    in_range = (y1 >= y2_mapped.min()) & (y1 <= y2_mapped.max())
    idx_np1 = np.where(in_range)[0]
    idx_np2 = np.argmin(np.abs(y2_mapped[:, None] - y1[idx_np1][None, :]), axis=0)

    # subsample to n_pairs
    step = max(1, len(idx_np1) // n_pairs)
    return idx_np1[::step], idx_np2[::step]


def evaluate_dense_csd_xcorr(dense_np1, dense_np2, idx_np1, idx_np2, fs, max_lag_ms=200):
    """
    xcorr between matched dense CSD channels on NP1 and NP2.

    Returns
    -------
    dict with 'peak' and 'lag_ms' arrays
    """
    peaks, lags = [], []
    for ch1, ch2 in zip(idx_np1, idx_np2):
        peak, lag = xcorr_peak(dense_np1[ch1, :], dense_np2[ch2, :], fs, max_lag_ms)
        peaks.append(peak)
        lags.append(lag)
    return {'peak': np.array(peaks), 'lag_ms': np.array(lags)}


# =============================================================================
# Modular pipeline framework
# =============================================================================

def run_pipeline(raw, stages):
    """
    Run a preprocessing pipeline defined as a list of (name, callable, kwargs).

    Each function receives (data, **kwargs) where data is (nc, ns) float array.

    Parameters
    ----------
    raw : np.ndarray (nc, ns)
    stages : list of (str, callable, dict)

    Returns
    -------
    dict  {stage_name: np.ndarray (nc, ns)}
    """
    data = {}
    x = raw.copy().astype(float)
    for name, func, kwargs in stages:
        x = func(x, **kwargs)
        data[name] = x.copy()
    return data


def make_pipeline_np1(sr):
    """Default pipeline for NP1 (mirrors current `preprocessing` in the script)."""
    h = sr.geometry
    return [
        ('bandpass', ibldsp.voltage.destripe_lfp,
         dict(fs=sr.fs, h=h, neuropixel_version=1, k_filter=None, channel_labels=None)),
        ('car',      ibldsp.voltage.car, dict(collection=None)),
        ('cadzow',   ibldsp.cadzow.cadzow_np1,
         dict(fs=sr.fs, rank=4, fmax=200, h=h)),
    ]


def make_pipeline_np2(sr):
    """Column-aware pipeline for NP2."""
    h = sr.geometry
    return [
        ('bandpass', ibldsp.voltage.destripe_lfp,
         dict(fs=sr.fs, h=h, neuropixel_version=2, k_filter=None, channel_labels=None)),
        ('col_car',  column_car, dict(h=h)),
        ('cadzow',   cadzow_geometry,
         dict(fs=sr.fs, rank=4, fmax=200, h=h)),
    ]


# =============================================================================
# Parameter grid
# =============================================================================

def parameter_grid(**param_ranges):
    """
    Yield all combinations of parameter ranges.

    Example
    -------
    for params in parameter_grid(rank=[2, 4, 6], fmax=[100, 200]):
        ...  # params = {'rank': 2, 'fmax': 100}, then {'rank': 2, 'fmax': 200}, ...
    """
    keys = list(param_ranges.keys())
    for values in itertools.product(*param_ranges.values()):
        yield dict(zip(keys, values))


def sweep_cadzow_params(raw, sr, h, param_ranges, version=1):
    """
    Run Cadzow with all combinations from param_ranges and return
    {param_str: denoised_array} for downstream evaluation.

    param_ranges : dict with keys from {rank, fmax, nswx, ovx, niter}
    """
    bp = ibldsp.voltage.destripe_lfp(
        raw.copy().astype(float), fs=sr.fs, h=h,
        neuropixel_version=version, k_filter=None, channel_labels=None
    )
    results = {}
    cadzow_fn = ibldsp.cadzow.cadzow_np1 if version == 1 else cadzow_geometry
    for params in parameter_grid(**param_ranges):
        key = '_'.join(f'{k}{v}' for k, v in params.items())
        results[key] = cadzow_fn(bp.copy(), fs=sr.fs, h=h, **params)
    return results


# =============================================================================
# Evaluation: cross-correlation between matched NP1 / NP2 channels
# =============================================================================

def xcorr_peak(sig1, sig2, fs, max_lag_ms=200):
    """
    Normalised cross-correlation peak and its lag between two signals.

    Returns
    -------
    peak : float  peak normalised xcorr ∈ [-1, 1]
    lag_ms : float  lag at peak in milliseconds
    """
    norm = np.sqrt(np.sum(sig1 ** 2) * np.sum(sig2 ** 2))
    if norm < 1e-20:
        return 0.0, 0.0
    xcorr = correlate(sig1, sig2) / norm
    lags = correlation_lags(len(sig1), len(sig2))
    lag_max = int(max_lag_ms / 1e3 * fs)
    mask = np.abs(lags) <= lag_max
    peak_idx = np.argmax(np.abs(xcorr[mask]))
    return float(xcorr[mask][peak_idx]), float(lags[mask][peak_idx] / fs * 1e3)


def evaluate_xcorr_metrics(data_np1, data_np2, idx_np1, idx_np2, fs, max_lag_ms=200, n_pairs=50):
    """
    For each preprocessing stage, compute peak cross-correlation across
    spatially matched NP1 / NP2 channel pairs.

    Parameters
    ----------
    data_np1 : dict {stage: np.ndarray (nc, ns)}  from run_pipeline
    data_np2 : dict {stage: np.ndarray (nc, ns)}
    idx_np1  : array of NP1 channel indices (matched to idx_np2)
    idx_np2  : array of NP2 channel indices
    fs       : sampling frequency (Hz)
    max_lag_ms : float  maximum lag to search
    n_pairs  : int  subsample to this many pairs (speeds things up)

    Returns
    -------
    metrics : dict {stage: {'peak': array, 'lag_ms': array}}
    """
    stages = list(data_np1.keys())
    n_total = len(idx_np1)
    step = max(1, n_total // n_pairs)
    pairs = list(zip(idx_np1[::step], idx_np2[::step]))

    metrics = {s: {'peak': [], 'lag_ms': []} for s in stages}
    for ch1, ch2 in pairs:
        for s in stages:
            sig1 = data_np1[s][ch1, :]
            sig2 = data_np2[s][ch2, :]
            peak, lag = xcorr_peak(sig1, sig2, fs, max_lag_ms)
            metrics[s]['peak'].append(peak)
            metrics[s]['lag_ms'].append(lag)

    return {s: {k: np.array(v) for k, v in m.items()} for s, m in metrics.items()}


def evaluate_csd_xcorr(data_np1, data_np2, h1, h2, idx_np1, idx_np2, fs,
                        csd_orders=(1, 2), n_pairs=50, max_lag_ms=200):
    """
    Pipeline: preprocessed → CSD(n) → xcorr across matched NP1/NP2 pairs.

    CSD is a spatial derivative that removes global common-mode signals and
    emphasises locally generated currents. xcorr on CSD is therefore a much
    more sensitive metric for local signal preservation than plain LFP xcorr.

    Parameters
    ----------
    data_np1 / data_np2 : dict {stage: np.ndarray (nc, ns)}
    h1 / h2 : probe geometry dicts for NP1 and NP2
    idx_np1 / idx_np2 : matched channel index arrays
    fs : float  sampling frequency (Hz)
    csd_orders : tuple of int  CSD orders to evaluate (1 = first derivative, 2 = second)
    n_pairs : int  number of channel pairs to subsample
    max_lag_ms : float

    Returns
    -------
    metrics : dict keyed by '{stage}_csd{n}' → {'peak': array, 'lag_ms': array}
    """
    stages = list(data_np1.keys())
    step = max(1, len(idx_np1) // n_pairs)
    pairs = list(zip(idx_np1[::step], idx_np2[::step]))

    metrics = {}
    for n in csd_orders:
        for stage in stages:
            key = f'{stage}_csd{n}'
            csd_np1 = ibldsp.voltage.current_source_density(data_np1[stage], h=h1, n=n, scale=False)
            csd_np2 = ibldsp.voltage.current_source_density(data_np2[stage], h=h2, n=n, scale=False)
            peaks, lags = [], []
            for ch1, ch2 in pairs:
                peak, lag = xcorr_peak(csd_np1[ch1, :], csd_np2[ch2, :], fs, max_lag_ms)
                peaks.append(peak)
                lags.append(lag)
            metrics[key] = {'peak': np.array(peaks), 'lag_ms': np.array(lags)}
    return metrics


def sweep_csd_xcorr(sweep_np1, bp_np2, h1, h2, idx_np1, idx_np2, fs,
                     csd_orders=(1, 2), n_pairs=80, max_lag_ms=200):
    """
    Evaluate a parameter sweep dict (from sweep_cadzow_params) using CSD xcorr.

    Parameters
    ----------
    sweep_np1 : dict {param_key: denoised np.ndarray (nc, ns)}  NP1 sweep results
    bp_np2    : np.ndarray (nc, ns)  NP2 bandpass (reference)
    h1 / h2   : geometry dicts
    idx_np1 / idx_np2 : matched channel indices

    Returns
    -------
    dict {param_key: {f'csd{n}': float mean peak xcorr}}
    """
    step = max(1, len(idx_np1) // n_pairs)
    pairs = list(zip(idx_np1[::step], idx_np2[::step]))

    # Pre-compute NP2 CSD reference once per order
    ref_csd = {n: ibldsp.voltage.current_source_density(bp_np2, h=h2, n=n, scale=False)
               for n in csd_orders}

    results = {}
    for key, wav in sweep_np1.items():
        entry = {}
        for n in csd_orders:
            csd1 = ibldsp.voltage.current_source_density(wav, h=h1, n=n, scale=False)
            peaks = [xcorr_peak(csd1[ch1, :], ref_csd[n][ch2, :], fs, max_lag_ms)[0]
                     for ch1, ch2 in pairs]
            entry[f'csd{n}'] = float(np.mean(peaks))
        results[key] = entry
    return results


def summary_table(metrics):
    """Print a summary table of mean ± std peak xcorr per stage."""
    print(f"{'Stage':<20} {'Mean peak xcorr':>16} {'Std':>8} {'Median':>10}")
    print('-' * 58)
    for stage, m in metrics.items():
        peaks = m['peak']
        print(f"{stage:<20} {np.mean(peaks):16.4f} {np.std(peaks):8.4f} {np.median(peaks):10.4f}")


# =============================================================================
# Visualisation
# =============================================================================

def plot_pipeline_comparison(metrics, title=None, fig=None, ax=None):
    """
    Box plot of peak normalised cross-correlation per preprocessing stage.
    Higher = more signal preserved relative to the reference probe.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.2), 5))
    stages = list(metrics.keys())
    data = [metrics[s]['peak'] for s in stages]
    ax.boxplot(data, labels=stages, notch=False, patch_artist=True,
               medianprops=dict(color='k', linewidth=2))
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set(xlabel='Preprocessing stage', ylabel='Peak normalised xcorr (NP1 vs NP2)',
           title=title or 'Pipeline comparison — cross-probe correlation')
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    return fig, ax


def plot_xcorr_traces(data_np1, data_np2, ch_np1, ch_np2, fs, stages=None,
                      max_lag_ms=200, t_lim=(27, 32)):
    """
    Reproduces the time-trace + xcorr panel from 2026-03-24_NP1NP2.py
    for a given channel pair and a subset of stages.

    t_lim : (t_start, t_end) in seconds for the trace axis view window.
    """
    stages = stages or list(data_np1.keys())
    t = np.arange(data_np1[stages[0]].shape[1]) / fs
    lag_max = int(max_lag_ms / 1e3 * fs)

    fig, axes = plt.subplots(len(stages), 2, figsize=(22, 3 * len(stages)),
                             sharex='col', gridspec_kw={'width_ratios': [5, 1]})
    if len(stages) == 1:
        axes = axes[np.newaxis, :]

    for i, stage in enumerate(stages):
        sig1 = data_np1[stage][ch_np1, :]
        sig2 = data_np2[stage][ch_np2, :]
        axes[i, 0].plot(t, sig1, label=f'NP1 ch{ch_np1}')
        axes[i, 0].plot(t, sig2, label=f'NP2 ch{ch_np2}')
        axes[i, 0].set(ylabel=stage, xlim=t_lim)
        axes[i, 0].legend(loc='upper right', fontsize=7)

        lags = correlation_lags(len(sig1), len(sig2))
        norm = np.sqrt(np.sum(sig1 ** 2) * np.sum(sig2 ** 2))
        xcorr = correlate(sig1, sig2) / (norm + 1e-20)
        mask = np.abs(lags) <= lag_max
        axes[i, 1].plot(lags[mask] / fs * 1e3, xcorr[mask])
        axes[i, 1].axvline(0, color='k', lw=0.5)
        axes[i, 1].set(ylim=(-0.2, 1))

    axes[0, 1].set(title='Norm. xcorr')
    axes[-1, 0].set(xlabel='Time (s)')
    axes[-1, 1].set(xlabel='Lag (ms)')
    fig.tight_layout()
    return fig, axes


def plot_singular_values(wav, h=None, title=None):
    """
    Plot singular value spectra per probe column and the Gavish-Donoho threshold.
    Useful for tuning the adaptive SVD rank.
    """
    nc = wav.shape[0]
    if h is None:
        version = 1 if nc == 384 else 2
        h = neuropixel.trace_header(version=version)

    cols = np.unique(h['col'])
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4), sharey=True)
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        idx = np.where(h['col'] == col)[0]
        block = wav[idx, :].astype(float)
        nc_col = block.shape[0]
        ns_col = block.shape[1]
        _, s, _ = np.linalg.svd(block, full_matrices=False)

        beta = nc_col / ns_col if nc_col < ns_col else ns_col / nc_col
        omega = _gavish_donoho_omega(beta)
        threshold = omega * np.median(s)

        ax.semilogy(s, '.-', ms=4, label='singular values')
        ax.axhline(threshold, color='r', lw=1.5, ls='--', label=f'GD threshold (rank≈{int(np.sum(s > threshold))})')
        ax.set(xlabel='Component index', title=f'Column {col}')
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Singular value')
    fig.suptitle(title or 'Singular value spectrum per column')
    fig.tight_layout()
    return fig, axes


# =============================================================================
# Demo / standalone script
# =============================================================================

if __name__ == '__main__':
    import spikeglx
    from pathlib import Path

    root_path = Path('/datadisk/Data/2026/np1np2')

    def get_lf_file(root_path, number, probe_type):
        session_path = root_path / f"{number:03d}"
        suffix = 'imec1' if probe_type == 'NP1' else 'imec0c'
        for lf_file in session_path.rglob('*.lf.*bin'):
            if lf_file.parent.name.endswith(suffix):
                return lf_file

    number = 4
    sample_slice = slice(30_000, int(30_000 * 4))

    sr_np1 = spikeglx.Reader(get_lf_file(root_path, number, 'NP1'))
    sr_np2 = spikeglx.Reader(get_lf_file(root_path, number, 'NP2'))

    intercept, slope = [2601.3, 1.179] if number <= 4 else [79, 1.204]
    idx_np1 = np.where(
        (sr_np1.geometry['y'] >= intercept + slope * sr_np2.geometry['y'].min()) &
        (sr_np1.geometry['y'] <= intercept + slope * sr_np2.geometry['y'].max())
    )[0]
    idx_np2 = np.argmin(np.abs(
        (intercept + slope * sr_np2.geometry['y'])[:, None] - sr_np1.geometry['y'][idx_np1][None, :]
    ), axis=0)

    raw_np1 = sr_np1[sample_slice, :-1].T
    raw_np2 = sr_np2[sample_slice, :-1].T

    # ----- NP1 pipeline comparison -----
    h1 = sr_np1.geometry
    bp_np1 = ibldsp.voltage.destripe_lfp(raw_np1.copy().astype(float), fs=sr_np1.fs, h=h1,
                                          neuropixel_version=1, k_filter=None, channel_labels=None)
    data_np1 = {
        'bandpass':   bp_np1,
        'global_car': ibldsp.voltage.car(bp_np1.copy()),
        'col_car':    column_car(bp_np1.copy(), h=h1),
        'global_pca': global_pca_filter(bp_np1.copy(), n_components=3),
        'cadzow_orig': ibldsp.cadzow.cadzow_np1(bp_np1.copy(), fs=sr_np1.fs, rank=4, fmax=200, h=h1),
        'cadzow_geom': cadzow_geometry(bp_np1.copy(), fs=sr_np1.fs, rank=4, fmax=200, h=h1),
        'svd_adapt':  adaptive_svd_denoise(bp_np1.copy(), h=h1),
    }

    # ----- NP2 pipeline comparison -----
    h2 = sr_np2.geometry
    bp_np2 = ibldsp.voltage.destripe_lfp(raw_np2.copy().astype(float), fs=sr_np2.fs, h=h2,
                                          neuropixel_version=2, k_filter=None, channel_labels=None)
    data_np2 = {
        'bandpass':   bp_np2,
        'global_car': ibldsp.voltage.car(bp_np2.copy()),
        'col_car':    column_car(bp_np2.copy(), h=h2),
        'global_pca': global_pca_filter(bp_np2.copy(), n_components=3),
        'cadzow_orig': ibldsp.cadzow.cadzow_np1(bp_np2.copy(), fs=sr_np2.fs, rank=4, fmax=200, h=h2),
        'cadzow_geom': cadzow_geometry(bp_np2.copy(), fs=sr_np2.fs, rank=4, fmax=200, h=h2),
        'svd_adapt':  adaptive_svd_denoise(bp_np2.copy(), h=h2),
    }

    # ----- Evaluate -----
    metrics = evaluate_xcorr_metrics(data_np1, data_np2, idx_np1, idx_np2, fs=sr_np1.fs)
    summary_table(metrics)

    fig_box, _ = plot_pipeline_comparison(metrics, title=f'Dataset {number:03d}')

    # Trace panel for a single representative channel pair
    np_ch2 = 40
    ch_np1_ex = idx_np1[np.argmin(np.abs(
        sr_np1.geometry['y'][idx_np1] - (intercept + slope * sr_np2.geometry['y'][np_ch2])
    ))]
    fig_tr, _ = plot_xcorr_traces(data_np1, data_np2, ch_np1_ex, np_ch2, fs=sr_np1.fs)

    # Singular value spectrum for NP2
    fig_sv, _ = plot_singular_values(bp_np2, h=h2, title=f'NP2 singular values — dataset {number:03d}')

    # ----- Cadzow parameter sweep -----
    sweep = sweep_cadzow_params(
        raw_np1, sr_np1, h1,
        param_ranges=dict(rank=[2, 4, 6, 8], fmax=[100, 200, 300]),
        version=1
    )
    # Evaluate the sweep against NP2 bandpass reference
    ref_np2 = {'ref': bp_np2}
    sweep_with_ref = {k: {'ref': bp_np2} for k in sweep}
    # Build per-key metrics using the same matched channels
    print("\nCadzow parameter sweep (NP1 → NP2 xcorr):")
    print(f"{'Params':<20} {'Mean peak xcorr':>16}")
    print('-' * 38)
    for key, wav_denoised in sweep.items():
        peaks = []
        step = max(1, len(idx_np1) // 50)
        for ch1, ch2 in zip(idx_np1[::step], idx_np2[::step]):
            peak, _ = xcorr_peak(wav_denoised[ch1, :], bp_np2[ch2, :], sr_np1.fs)
            peaks.append(peak)
        print(f"{key:<20} {np.mean(peaks):16.4f}")

    plt.show()
