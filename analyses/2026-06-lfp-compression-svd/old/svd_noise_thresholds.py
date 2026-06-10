def compress_lfp_channel_average_psd(
    X,
    fs,
    *,
    channels_per_sample=16,
    window_s=0.4,
    step_s=0.2,
    bands=None,
    channel_xyz=None,
):
    """
    Spatial + temporal LFP compression.

    1. Average every `channels_per_sample` channels.
    2. Compute Welch PSD in sliding windows.
    3. Average PSD into frequency bands.

    Parameters
    ----------
    X : array, shape [channels, time]
        Preprocessed LFP.
    fs : float
        Sampling rate in Hz.
    channels_per_sample : int
        Number of neighboring channels to average spatially.
    window_s : float
        PSD window size in seconds.
    step_s : float
        PSD step size in seconds.
    bands : dict or None
        Example:
            {
                "delta": (1, 4),
                "theta": (4, 8),
                "alpha": (8, 12),
                "beta": (12, 30),
                "gamma": (30, 80),
            }
    channel_xyz : array or None, shape [channels, 3]
        Optional channel positions. If provided, sample xyz positions are averaged.

    Returns
    -------
    out : dict
        Contains:
            X_sample_t : spatially compressed signal [samples, time]
            psd_df     : bandpower table
            groups     : channel groups
            sample_xyz : averaged xyz positions, if provided
    """
    import numpy as np
    import pandas as pd
    import scipy.signal

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be [channels, time], got shape {X.shape}")

    if bands is None:
        bands = {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 12.0),
            "beta": (12.0, 30.0),
            "gamma": (30.0, min(80.0, 0.95 * fs / 2)),
        }

    n_channels, n_time = X.shape

    groups = [
        np.arange(i, min(i + channels_per_sample, n_channels), dtype=int)
        for i in range(0, n_channels, channels_per_sample)
    ]

    X_sample_t = np.vstack([
        np.nanmean(X[g, :], axis=0)
        for g in groups
    ]).astype(np.float32)

    sample_xyz = None
    if channel_xyz is not None:
        channel_xyz = np.asarray(channel_xyz, dtype=float)
        sample_xyz = np.vstack([
            np.nanmean(channel_xyz[g], axis=0)
            for g in groups
        ])

    win = int(round(window_s * fs))
    step = int(round(step_s * fs))

    starts = np.arange(0, n_time - win + 1, step, dtype=int)
    rows = []

    for wi, s0 in enumerate(starts):
        seg = X_sample_t[:, s0 : s0 + win]

        freqs, psd = scipy.signal.welch(
            seg,
            fs=fs,
            axis=1,
            nperseg=min(win, int(fs)),
            noverlap=0,
        )

        for sample in range(X_sample_t.shape[0]):
            row = {
                "window_ind": int(wi),
                "sample": int(sample),
                "t_center": float((s0 + win / 2) / fs),
                "window_s": float(window_s),
                "step_s": float(step_s),
                "chan_start": int(groups[sample][0]),
                "chan_stop": int(groups[sample][-1]),
                "n_channels": int(len(groups[sample])),
            }

            if sample_xyz is not None:
                row["x"] = float(sample_xyz[sample, 0])
                row["y"] = float(sample_xyz[sample, 1])
                row["z"] = float(sample_xyz[sample, 2])

            for band_name, (lo, hi) in bands.items():
                m = (freqs >= lo) & (freqs < hi)
                row[f"psd_{band_name}"] = (
                    float(np.nanmean(psd[sample, m])) if np.any(m) else np.nan
                )

            rows.append(row)

    psd_df = pd.DataFrame(rows)

    return {
        "type": "channel_average_psd",
        "X_sample_t": X_sample_t,
        "psd_df": psd_df,
        "groups": groups,
        "sample_xyz": sample_xyz,
        "fs": float(fs),
        "channels_per_sample": int(channels_per_sample),
        "window_s": float(window_s),
        "step_s": float(step_s),
        "bands": bands,
    }

def compress_lfp_svd_eps(
    X,
    *,
    epsilon=9.0,
    epsilon_mode="median_tail",  # "median_tail" or "absolute"
    quantize=True,
    min_rank=1,
    return_reconstruction=True,
):
    """
    SVD compression for LFP matrix.

    Parameters
    ----------
    X : array, shape [channels, time]
        Preprocessed LFP.
    epsilon : float
        If epsilon_mode="median_tail":
            threshold = epsilon * median(lower half of singular values)
        If epsilon_mode="absolute":
            threshold = epsilon
    epsilon_mode : {"median_tail", "absolute"}
        Method for choosing the singular-value cutoff.
    quantize : bool
        If True, quantizes temporal SVD scores using a noise-floor-derived step.
    min_rank : int
        Minimum number of singular components to keep.
    return_reconstruction : bool
        If True, also returns reconstructed X_hat.

    Returns
    -------
    compressed : dict
        Contains U, scores or scores_q, q_step, threshold, rank, etc.
    X_hat : array or None
        Reconstructed LFP, shape [channels, time].
    """
    import numpy as np

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be [channels, time], got shape {X.shape}")

    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    if epsilon_mode == "median_tail":
        tail = s[s.size // 2 :]
        noise_floor = float(np.nanmedian(tail)) if tail.size else float(np.nanmedian(s))
        if not np.isfinite(noise_floor) or noise_floor <= 0:
            noise_floor = float(np.nanmedian(s[s > 0])) if np.any(s > 0) else 0.0
        threshold = float(epsilon) * noise_floor

    elif epsilon_mode == "absolute":
        noise_floor = np.nan
        threshold = float(epsilon)

    else:
        raise ValueError("epsilon_mode must be 'median_tail' or 'absolute'")

    keep = s > threshold

    if keep.sum() < min_rank and s.size:
        keep[: min(min_rank, s.size)] = True

    U_k = U[:, keep].astype(np.float32)
    scores = (s[keep, None] * Vt[keep, :]).astype(np.float32)

    compressed = {
        "type": "svd_eps",
        "epsilon": float(epsilon),
        "epsilon_mode": str(epsilon_mode),
        "noise_floor": float(noise_floor) if np.isfinite(noise_floor) else None,
        "threshold": float(threshold),
        "rank": int(keep.sum()),
        "shape": tuple(X.shape),
        "U": U_k,
    }

    if quantize:
        q_step = max(float(threshold) / np.sqrt(max(X.shape[1], 1)), 1e-8)
        scores_q = np.round(scores / q_step).astype(np.int32)
        compressed["scores_q"] = scores_q
        compressed["q_step"] = np.float32(q_step)

        if return_reconstruction:
            scores_hat = scores_q.astype(np.float32) * np.float32(q_step)
            X_hat = (U_k @ scores_hat).astype(np.float32)
        else:
            X_hat = None

    else:
        compressed["scores"] = scores

        if return_reconstruction:
            X_hat = (U_k @ scores).astype(np.float32)
        else:
            X_hat = None

    return compressed, X_hat
