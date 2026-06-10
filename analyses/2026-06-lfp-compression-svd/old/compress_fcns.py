"""
Why it's well-motivated:
  The SVD at each frequency bin finds the dominant spatial modes. A channel with anomalous impedance will have an amplitude that doesn't fit the plane-wave model —
  its residual |WAV_original - WAV_reconstructed| will be large relative to the other channels at that frequency. The median+k·MAD threshold is robust to the handful
  of bad channels (they don't inflate the baseline), so you only correct the true outliers. Channels that are already consistent with the model are left untouched.

  What this adds vs. plain Cadzow:
  Plain Cadzow replaces everyone with the rank-r projection. With ppca_k, only the outlier channels get corrected — the others keep their original values going into
  the second SVD, which then produces a better reconstruction because the trajectory matrix is no longer contaminated by the bad channels.

  Caveats to watch for:
  - ppca_k=3 is a reasonable start; go higher (4–5) if you're worried about correcting real high-amplitude neural signals in narrow frequency bands
  - The window size nw=32 means each group has 32 input channels — with only 4 channels per output group, the MAD threshold is computed across just 4 values per
  frequency bin, so the statistics are noisy. You could try larger nw to get more channels in the residual distribution
  - Cost: roughly 2× the SVD budget when ppca_k is set
"""

from pathlib import Path
import numpy as np
import scipy.signal
import scipy.fft
import neuropixel
from brainbox.io.one import SpikeSortingLoader
from ibldsp.cadzow import trajectory


def _apply_rank_threshold(s, r, gap_threshold=None):
    """Zero singular values beyond the adaptive rank for each frequency bin.

    Parameters
    ----------
    s : np.ndarray (nbins, k)
        Singular values sorted descending per row, modified in-place.
    r : int
        Hard upper bound on rank; also the fallback when no gap qualifies.
    gap_threshold : float or None
        Minimum s[i]/s[i+1] ratio to count as a dominant spectral gap.
        The rank per bin is set to the position of the largest such ratio,
        clamped to [1, r].  If the maximum ratio is below gap_threshold the
        rank falls back to r.  None disables adaptive selection (fixed rank r).
    """
    if gap_threshold is None:
        s[:, r:] = 0.0
        return
    ratios = s[:, :-1] / (s[:, 1:] + 1e-10)    # (nbins, k-1)
    has_gap = ratios.max(axis=1) >= gap_threshold
    r_adapt = np.where(has_gap, np.argmax(ratios, axis=1) + 1, r)
    r_adapt = np.clip(r_adapt, 1, r)
    idx = np.arange(s.shape[1])[np.newaxis, :]  # (1, k)
    s[idx >= r_adapt[:, np.newaxis]] = 0.0


def load_pid(pid, one, root_output, stream=False, q=10):
    """
    Load spike sorting loader and raw electrophysiology for a given probe insertion.

    Parameters
    ----------
    pid : str
        Probe insertion ID.
    one : ONE
        Authenticated ONE instance.
    root_output : Path
        Root directory under which a per-pid output folder is created.
    stream : bool
        Stream data from server if True; download to cache if False.
    q : int
        Temporal decimation factor used to derive the resampled sampling rate.

    Returns
    -------
    ssl : SpikeSortingLoader
    sr : object
        Raw electrophysiology reader (LFP band).
    output_path : Path
        Per-pid output directory (created if absent).
    fs_rs : float
        Resampled sampling rate [Hz] = ``sr.fs / q``.
    """
    output_path = Path(root_output).joinpath(pid)
    output_path.mkdir(parents=True, exist_ok=True)
    ssl = SpikeSortingLoader(pid=pid, one=one)
    sr = ssl.raw_electrophysiology(band='lf', stream=stream)
    fs_rs = sr.fs / q
    return ssl, sr, output_path, fs_rs


def psth(w, fs, i_events, event_window=(-1, 1)):
    """
    Compute PSTH for a single channel.

    Parameters
    ----------
    w : np.ndarray (ns, nc)
        LFP array, samples × channels.
    fs : float
        Sampling rate [Hz].
    i_events : np.ndarray (nevents,)
        Event sample indices in the resampled signal.
    event_window : tuple of float
        Pre/post window in seconds.

    Returns
    -------
    p : np.ndarray (ntimes, nevents)
        Baseline-corrected PSTH at channel 250.
    """
    event_window_samples = (np.array(event_window) * fs).astype(int)
    sample_window = np.round(np.arange(event_window_samples[0], event_window_samples[1])).astype(int)
    idx_psth = np.tile(sample_window[:, np.newaxis], (1, i_events.size))
    idx_psth += i_events
    p = w[idx_psth, 250].astype(np.float32).T
    p = p - np.mean(p, axis=1)[:, np.newaxis]
    return p


def psth_all_channels(w, fs, i_events, event_window=(-1, 1)):
    """
    Compute PSTH, median across all channels.

    Parameters
    ----------
    w : np.ndarray (ns, nc)
        LFP array, samples × channels.
    fs : float
        Sampling rate [Hz].
    i_events : np.ndarray (nevents,)
        Event sample indices in the resampled signal.
    event_window : tuple of float
        Pre/post window in seconds.

    Returns
    -------
    p : np.ndarray (nevents_valid, ntimes)
        Baseline-corrected PSTH, median across channels.
    valid : np.ndarray (nevents,) bool
        Boolean mask of in-bounds events.
    """
    ns, nc = w.shape
    ew_samp = (np.array(event_window) * fs).astype(int)
    sample_window = np.arange(ew_samp[0], ew_samp[1])
    idx = sample_window[:, np.newaxis] + i_events[np.newaxis, :]   # (ntimes, nevents)
    valid = np.all((idx >= 0) & (idx < ns), axis=0)
    p = w[idx[:, valid], :].astype(np.float32)   # (ntimes, nevents_valid, nc)
    p = p.transpose(1, 0, 2)                      # (nevents_valid, ntimes, nc)
    return np.median(p, axis=2), valid             # (nevents_valid, ntimes)


def cadzow_reconstruct(WAV, x_in, y_in, x_out, y_out, r, imax=None, ppca_k=None, gap_threshold=None):
    """
    Reconstruct signal at new spatial positions using F-X Cadzow rank reduction.

    Extends the standard Cadzow denoiser to predict at positions absent from the
    input by including them in the block-Toeplitz trajectory matrix and reading off
    the de-ranked values.

    Speed: uses a single batched ``np.linalg.svd`` call over all frequency bins
    (shape ``(imax, nrows, ncols)``) instead of a Python loop, and a precomputed
    scatter matrix to replace per-frequency ``np.bincount`` calls.

    Parameters
    ----------
    WAV : np.ndarray (nc_in, nf), complex
        Input channels in the frequency domain (rfft output).
    x_in : np.ndarray (nc_in,)
        Lateral coordinates of input channels [µm].
    y_in : np.ndarray (nc_in,)
        Depth coordinates of input channels [µm].
    x_out : np.ndarray (nc_out,)
        Lateral coordinates of virtual output channels [µm].
    y_out : np.ndarray (nc_out,)
        Depth coordinates of virtual output channels [µm].
    r : int
        Maximum SVD rank (hard ceiling on the number of plane waves retained).
    imax : int, optional
        Maximum frequency-bin index to process; higher bins stay zero (low-pass).
    ppca_k : float, optional
        If set, enables a PPCA-style outlier correction before the final reconstruction.
        After an initial rank-r projection, input channels whose amplitude deviates from
        the model by more than ``median + ppca_k * MAD`` (per frequency bin) are replaced
        by their model prediction, then the SVD is repeated on the cleaned data.
        This suppresses impedance-mismatch artefacts (1/f amplitude outliers) without
        altering channels that are already consistent with the spatial model.
        Typical values: 3–5.  ``None`` disables (default, original behaviour).
    gap_threshold : float, optional
        If set, enables adaptive per-frequency-bin rank selection via the singular-value
        gap criterion.  The rank for each bin is the position of the largest ratio
        s[i]/s[i+1] in the singular value spectrum, clamped to [1, r].  When the
        maximum ratio is below ``gap_threshold`` the rank falls back to ``r``.
        A value around 1.5–2.0 is a reasonable starting point.  ``None`` uses fixed
        rank ``r`` (default, backward-compatible).

    Returns
    -------
    WAV_out : np.ndarray (nc_out, nf), complex
    """
    nc_in, nf = WAV.shape
    nc_out = len(x_out)
    imax = int(min(imax if imax is not None else nf, nf))

    x_all = np.r_[x_in.astype(float), x_out.astype(float)]
    y_all = np.r_[y_in.astype(float), y_out.astype(float)]
    T, it, ic, _ = trajectory(x_all, y_all)

    sel_in = ic < nc_in
    it_in = (it[0][sel_in], it[1][sel_in])
    ic_in = ic[sel_in]
    it_out = (it[0][~sel_in], it[1][~sel_in])
    ic_out = ic[~sel_in] - nc_in
    trcount_out = np.maximum(np.bincount(ic_out, minlength=nc_out).astype(float), 1.0)
    scatter = np.zeros((len(ic_out), nc_out))
    scatter[np.arange(len(ic_out)), ic_out] = 1.0 / trcount_out[ic_out]

    T_batch = np.zeros((imax, *T.shape), dtype=complex)
    T_batch[:, it_in[0], it_in[1]] = WAV[ic_in, :imax].T

    U, s, Vh = np.linalg.svd(T_batch, full_matrices=False)
    _apply_rank_threshold(s, r, gap_threshold)
    T_batch_ = (U * s[:, np.newaxis, :]) @ Vh

    if ppca_k is not None:
        trcount_in = np.maximum(np.bincount(ic_in, minlength=nc_in).astype(float), 1.0)
        scatter_in = np.zeros((len(ic_in), nc_in))
        scatter_in[np.arange(len(ic_in)), ic_in] = 1.0 / trcount_in[ic_in]

        vals_in = T_batch_[:, it_in[0], it_in[1]]
        WAV_rec_in = (vals_in @ scatter_in).T

        residual = np.abs(WAV[:, :imax] - WAV_rec_in)
        med = np.median(residual, axis=0)
        mad = np.median(np.abs(residual - med[None, :]), axis=0)
        outlier = residual > (med + ppca_k * mad)

        WAV_clean = WAV[:, :imax].copy()
        WAV_clean[outlier] = WAV_rec_in[outlier]

        T_batch[:, it_in[0], it_in[1]] = WAV_clean[ic_in, :].T
        U, s, Vh = np.linalg.svd(T_batch, full_matrices=False)
        _apply_rank_threshold(s, r, gap_threshold)
        T_batch_ = (U * s[:, np.newaxis, :]) @ Vh

    vals = T_batch_[:, it_out[0], it_out[1]]
    WAV_out = np.zeros((nc_out, nf), dtype=complex)
    WAV_out[:, :imax] = (vals @ scatter).T

    return WAV_out


def cadzow_merge_columns(wav, version=1, r=5, nw=32, fmax=100.0, fs=500.0, n_jobs=-1, ppca_k=None, gap_threshold=None):
    """
    Merge Neuropixel columns into a single virtual column via Cadzow reconstruction.

    Groups 4 consecutive channels (2 depth levels × all columns per level) and
    reconstructs the signal at a virtual channel placed at the lateral centroid
    and mean depth of each group.  A Hann-windowed overlap-add scheme blends
    adjacent processing windows.  Windows are processed in parallel via joblib
    threads (NumPy releases the GIL during SVD, so no pickling overhead).

    Parameters
    ----------
    wav : np.ndarray (nc, ns), float32
        Input LFP array, channels × samples.
    version : {1, 2}
        Neuropixel probe version.
    r : int
        Maximum SVD rank (hard ceiling), default 5.
    nw : int
        Input channels per processing window, default 32 (→ 8 output channels/win).
    fmax : float
        Maximum frequency processed by Cadzow [Hz]; higher bins pass through as zero.
    fs : float
        Sampling rate of ``wav`` [Hz], used to convert ``fmax`` to a bin index.
    n_jobs : int
        Joblib thread count; -1 uses all available cores.
    ppca_k : float, optional
        Outlier threshold for PPCA-style correction (see ``cadzow_reconstruct``).
        ``None`` disables (default).
    gap_threshold : float, optional
        Adaptive rank threshold (see ``cadzow_reconstruct``).  ``None`` uses fixed
        rank ``r`` (default).

    Returns
    -------
    wav_out : np.ndarray (nc_out, ns), float32
    h_out : dict
        Channel geometry dict with keys ``'x'`` and ``'y'`` [µm].
    """
    from joblib import Parallel, delayed

    h = neuropixel.trace_header(version=version)
    x_in_all = h['x'].astype(float)
    y_in_all = h['y'].astype(float)
    nc, ns = wav.shape

    n_per_group = 4
    nc_out = nc // n_per_group
    x_center = float(np.mean(np.unique(x_in_all)))
    y_out_all = np.array([y_in_all[i * n_per_group:(i + 1) * n_per_group].mean() for i in range(nc_out)])
    x_out_all = np.full(nc_out, x_center)

    WAV = scipy.fft.rfft(wav, axis=-1)
    nf = WAV.shape[-1]
    imax = int(np.searchsorted(scipy.fft.rfftfreq(wav.shape[-1], d=1.0 / fs), fmax))

    n_out_per_win = nw // n_per_group
    step = n_out_per_win // 2
    hann_win = scipy.signal.windows.hann(n_out_per_win)
    windows = []
    i_out = 0
    while i_out < nc_out:
        j_out = min(i_out + n_out_per_win, nc_out)
        n_out_w = j_out - i_out
        gw = hann_win[:n_out_w] if n_out_w == n_out_per_win else np.ones(n_out_w)
        windows.append((i_out, j_out, gw))
        i_out += step

    def _process(i_out, j_out, gw):
        i_in, j_in = i_out * n_per_group, j_out * n_per_group
        W = cadzow_reconstruct(
            WAV[i_in:j_in],
            x_in_all[i_in:j_in], y_in_all[i_in:j_in],
            x_out_all[i_out:j_out], y_out_all[i_out:j_out],
            r=r, imax=imax, ppca_k=ppca_k, gap_threshold=gap_threshold,
        )
        return i_out, j_out, W * gw[:, None], gw

    # prefer='threads': NumPy releases the GIL during LAPACK calls
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_process)(i, j, gw) for i, j, gw in windows
    )

    WAV_out = np.zeros((nc_out, nf), dtype=complex)
    gain_out = np.zeros(nc_out)
    for i_out, j_out, W_gw, gw in results:
        WAV_out[i_out:j_out] += W_gw
        gain_out[i_out:j_out] += gw

    WAV_out /= np.maximum(gain_out[:, None], 1e-9)
    wav_out = scipy.fft.irfft(WAV_out, n=ns, axis=-1).real.astype(np.float32)
    return wav_out, {'x': x_out_all, 'y': y_out_all}


def cadzow_denoise_probe(wav, version=1, r=5, nw=32, fmax=100.0, fs=500.0, n_jobs=-1, ppca_k=None, gap_threshold=None):
    """
    Denoise LFP in the F-X domain using Cadzow rank reduction, keeping the full probe geometry.

    Unlike ``cadzow_merge_columns``, output channels are co-located with the input —
    no spatial downsampling.  Intended for impedance-mismatch correction and coherent
    noise suppression while preserving individual-channel spatial resolution.

    On NP1 the default ``nw=32`` covers ≈ 160 µm depth (8 depth levels × 4 columns)
    with a 50 % overlap step of 16 channels (≈ 80 µm).

    Parameters
    ----------
    wav : np.ndarray (nc, ns), float32
        Input LFP, channels × samples.
    version : {1, 2}
        Neuropixel probe version.
    r : int
        Maximum SVD rank (hard ceiling), default 5.
    nw : int
        Channels per processing window, default 32.
    fmax : float
        Maximum frequency processed [Hz]; higher bins pass through unchanged.
    fs : float
        Sampling rate [Hz].
    n_jobs : int
        Joblib thread count; -1 uses all cores.
    ppca_k : float, optional
        Outlier threshold in MAD units for PPCA correction (see ``cadzow_reconstruct``).
        ``None`` disables (default).
    gap_threshold : float, optional
        Adaptive rank threshold (see ``cadzow_reconstruct``).  ``None`` uses fixed
        rank ``r`` (default).

    Returns
    -------
    wav_out : np.ndarray (nc, ns), float32
    """
    from joblib import Parallel, delayed

    h = neuropixel.trace_header(version=version)
    x_h = h['x'][:wav.shape[0]].astype(float)
    y_h = h['y'][:wav.shape[0]].astype(float)
    nc, ns = wav.shape

    WAV = scipy.fft.rfft(wav, axis=-1)
    nf = WAV.shape[-1]
    imax = int(np.searchsorted(scipy.fft.rfftfreq(ns, d=1.0 / fs), fmax))

    step = nw // 2
    hann_win = scipy.signal.windows.hann(nw)
    windows = []
    i = 0
    while i < nc:
        j = min(i + nw, nc)
        n_w = j - i
        gw = hann_win[:n_w] if n_w == nw else np.ones(n_w)
        windows.append((i, j, gw))
        i += step

    def _process(i, j, gw):
        nc_w = j - i
        T, it, ic, _ = trajectory(x_h[i:j], y_h[i:j])

        trcount = np.maximum(np.bincount(ic, minlength=nc_w).astype(float), 1.0)
        scatter = np.zeros((len(ic), nc_w))
        scatter[np.arange(len(ic)), ic] = 1.0 / trcount[ic]

        T_batch = np.zeros((imax, *T.shape), dtype=complex)
        T_batch[:, it[0], it[1]] = WAV[i + ic, :imax].T

        U, s, Vh = np.linalg.svd(T_batch, full_matrices=False)
        _apply_rank_threshold(s, r, gap_threshold)
        T_batch_ = (U * s[:, np.newaxis, :]) @ Vh

        if ppca_k is not None:
            # Identify per-frequency outlier channels, replace with model prediction, redo SVD
            WAV_rec = (T_batch_[:, it[0], it[1]] @ scatter).T
            residual = np.abs(WAV[i:j, :imax] - WAV_rec)
            med = np.median(residual, axis=0)
            mad = np.median(np.abs(residual - med[None, :]), axis=0)
            outlier = residual > (med + ppca_k * mad)
            WAV_clean = WAV[i:j, :imax].copy()
            WAV_clean[outlier] = WAV_rec[outlier]
            T_batch[:, it[0], it[1]] = WAV_clean[ic, :].T
            U, s, Vh = np.linalg.svd(T_batch, full_matrices=False)
            _apply_rank_threshold(s, r, gap_threshold)
            T_batch_ = (U * s[:, np.newaxis, :]) @ Vh

        WAV_out_w = np.zeros((nc_w, nf), dtype=complex)
        WAV_out_w[:, :imax] = (T_batch_[:, it[0], it[1]] @ scatter).T
        return i, j, WAV_out_w * gw[:, None], gw

    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_process)(i, j, gw) for i, j, gw in windows
    )

    WAV_out = np.zeros((nc, nf), dtype=complex)
    gain_out = np.zeros(nc)
    for i, j, W_gw, gw in results:
        WAV_out[i:j] += W_gw
        gain_out[i:j] += gw

    WAV_out /= np.maximum(gain_out[:, None], 1e-9)
    return scipy.fft.irfft(WAV_out, n=ns, axis=-1).real.astype(np.float32)