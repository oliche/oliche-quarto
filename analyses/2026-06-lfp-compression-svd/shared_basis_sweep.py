"""
Shared spatial basis for lfpack: basis compression ratio vs behaviour decoding at constant file size.

Codec per 8 s chunk X_j (nc × ns_ext, nc = 384, ns_ext = 2048 + 2 × 128 guard samples):

    current (m = inf):  X_j ≈ U_j[:, :r_j] · diag(s_j) · Vh_j        U_j stored per chunk, float32
    shared  (m < inf):  X_j ≈ B · C_j · diag(s_j) · Vh_j             B (nc × m) once per recording,
                                                                     C_j (m × r_j) unit columns, float16
    with  Y_j = Bᵀ X_j = C_j · diag(s_j) · Vh_j  (SVD of the projection), so the temporal rows are fitted
    inside span(B) and absorb the basis error.

The noise floor σ_j and the rank r_j = #{sv > ε σ_j} always come from the full SVD of X_j (projecting onto
B removes most noise dimensions, which would bias the "median of the lower half" estimate), so ε and α keep
their meaning and m = inf reproduces the current codec.  B = top-m eigenvectors of
G = Σ_j Us_j Us_jᵀ / ||Us_j||², Us_j = U_j[:, :r_j] · s_j (every chunk weighs the same).

Arms: m ∈ {inf (float32 U, current), inf-f16 (float16 U, no shared basis), 96, 64, 48, 32, 24} × ε ∈ {100, 150}.
Tracks:
  - size: α chosen so the total bytes equal the mild reference (m = inf, ε = 100, α = 14), then wheel R² and
    choice / feedback balanced accuracy are decoded straight from the in-memory reconstruction.
  - snr:  α chosen so the pooled snippet SNR equals the reference (σ_eff = σ · α / α_ref); bytes and SNR only.
α is calibrated on 40 evenly spaced snippet chunks per probe, pooled over the 11 probes, by interpolation on a
log-α grid.  Bytes = HDF5 storage of the format-2 codec datasets (gzip + shuffle) + float16 C / float32 s + B.
"""

# %%
import gc
import io
from pathlib import Path

import h5py
import lfpack
import numpy as np
import pandas as pd
import pynapple as nap
import pywt
import scipy.signal
from joblib import Parallel, delayed
from lfpack import _container
from lfpack._core import (
    _COMPRESS_CHUNK,
    _COMPRESS_OVERLAP,
    _WP_MAXLEVEL,
    _WP_WAVELET,
    _reconstruct_vh_from_wp,
    _svd_noise_floor,
)
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

nap.nap_config.suppress_conversion_warnings = True

SCRATCH = Path('/Users/olivier/scratch/lfp')
OUT_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
TODAY = '2026-09-25'
CALIB_CSV = OUT_DIR.joinpath(f'{TODAY}_shared-basis_calibration.csv')
ARMS_CSV = OUT_DIR.joinpath(f'{TODAY}_shared-basis_arms.csv')
WHEEL_CSV = OUT_DIR.joinpath(f'{TODAY}_shared-basis_wheel.csv')
CATEG_CSV = OUT_DIR.joinpath(f'{TODAY}_shared-basis_categorical.csv')

PIDS = [
    '1a276285-8b0e-4cc9-9f0a-a3a002978724',
    '1e104bf4-7a24-4624-a5b2-c2c8289c0de7',
    '6638cfb3-3831-4fc2-9327-194b76cf22e1',
    '749cb2b7-e57e-4453-a794-f6230e4d0226',
    'd7ec0892-0a6c-4f4f-9d8f-72083692af5c',
    'da8dfec1-d265-44e8-84ce-6ae9c109b8bd',
    'dab512bd-a02d-4c1f-8dbc-9155a163efc0',
    'dc7e9403-19f7-409f-9240-05ee57cb7aea',
    'e8f9fba4-d151-4b00-bee7-447f0f3e752c',
    'eebcaf65-7fa4-4118-869d-a084e84530e2',
    'fe380793-8035-414e-b000-09bfe5ece92a',
]
REF = dict(m='inf', epsilon=100.0, alpha=14.0)  # mild tier
MS = ['inf', 'inf-f16', 48, 24, 96, 64, 32]  # ordered by priority for the overnight run
EPSILONS = [100.0, 150.0]
ALPHA_GRID = np.logspace(np.log10(2.0), np.log10(600.0), 40)
N_SNIPPETS = 40
RMAX = 40  # columns of U·s cached per chunk in pass 1 (ranks are ~15-24 at ε = 100)
FLOOR_K = 64
N_JOBS = 8
FS = 250.0
SOS = {
    'delta': scipy.signal.butter(4, (1.0, 4.0), 'bandpass', fs=FS, output='sos'),
    'gamma': scipy.signal.butter(4, (30.0, 90.0), 'bandpass', fs=FS, output='sos'),
}

# behaviour decoding, copied from ibldevtools/olivier/2026-08-31_LFP_compression_decoding{,_categorical}.py
Q = 10
FS_LF_RAW = 2500.0
BIN_SIZE = 0.05
BANDS = {'delta': (1.0, 4.0), 'theta': (4.0, 8.0), 'beta': (15.0, 30.0), 'gamma': (30.0, 90.0)}
N_SHUFFLES = 20
TARGETS = {'choice': ('firstMovement_times', -0.2, 0.0), 'feedback': ('feedback_times', 0.0, 0.2)}


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------

def chunk_bounds(ns, chunk=_COMPRESS_CHUNK, overlap=_COMPRESS_OVERLAP):
    """(i0_read, i1_read, n_written, left_overlap) of every codec chunk, as in lfpack.compress_to_h5."""
    out = []
    for i0_w in range(0, ns, chunk):
        i1_w = min(i0_w + chunk, ns)
        i0_r, i1_r = max(0, i0_w - overlap), min(ns, i1_w + overlap)
        out.append((i0_r, i1_r, i1_w - i0_w, i0_w - i0_r))
    return out


def read_chunk(npy, bounds):
    """Load one extended chunk as (nc, ns_ext) float64."""
    i0_r, i1_r, _, _ = bounds
    return np.asarray(np.load(npy, mmap_mode='r')[i0_r:i1_r, :], dtype=np.float64).T


def decompose(x, B, m, epsilon, sv_full=None):
    """
    Spatial stage of one chunk.

    Parameters
    ----------
    x : ndarray (nc, ns_ext)
    B : ndarray (nc, m) or None
        Shared orthonormal basis; None for the per-chunk arms.
    m : int or str
        'inf' (float32 U), 'inf-f16' (float16 U), the shared-basis size (re-decomposition inside B) or 'p<size>[q8]'
        (projection: full-chunk SVD as in lfpack, only U is replaced by B · C, C = Bᵀ U float16 with a float32
        per-column scale; 'q8' stores C in int8 with the same per-column scale).
    epsilon : float
    sv_full : ndarray or None
        Singular values of x from pass 1 (avoids a second full SVD for shared arms).

    Returns
    -------
    dict with U_unit (nc|m, r) as stored (float16 unless m == 'inf'), s (r,), rows (r, ns_ext), sigma, spatial
    (nc, r) float64 = the decoded spatial factor without s.
    """
    if B is None:
        U, sv, Vh = np.linalg.svd(x, full_matrices=False)
        sigma = _svd_noise_floor(sv)
        r = max(1, int(np.sum(sv > epsilon * sigma)))
        U_unit, s, rows = U[:, :r], sv[:r], Vh[:r]
        if m == 'inf-f16':
            U_unit = U_unit.astype(np.float16)
        spatial = U_unit.astype(np.float64)
    elif str(m).startswith('p'):  # projection: per-chunk codec unchanged, only U_scaled is projected onto B
        U, sv, Vh = np.linalg.svd(x, full_matrices=False)
        sigma = _svd_noise_floor(sv)
        r = max(1, int(np.sum(sv > epsilon * sigma)))
        s, rows = sv[:r], Vh[:r]
        C = B.T @ U[:, :r]
        if str(m).endswith('q8'):
            scale = np.abs(C).max(axis=0) / 127 + 1e-40
            U_unit = np.round(C / scale).astype(np.int8)
        else:
            scale = np.linalg.norm(C, axis=0) + 1e-40  # float32 per-column scale (stored as s · scale)
            U_unit = (C / scale).astype(np.float16)
        spatial = B @ U_unit.astype(np.float64) * scale.astype(np.float32)
        return dict(U_unit=U_unit, s=s, rows=rows, sigma=sigma, spatial=spatial, s_stored=s * scale)
    else:
        sigma = _svd_noise_floor(sv_full)
        r = min(max(1, int(np.sum(sv_full > epsilon * sigma))), B.shape[1])
        Uy, sy, Vy = np.linalg.svd(B.T @ x, full_matrices=False)
        U_unit, s, rows = Uy[:, :r].astype(np.float16), sy[:r], Vy[:r]
        spatial = B @ U_unit.astype(np.float64)
    return dict(U_unit=U_unit, s=s, rows=rows, sigma=sigma, spatial=spatial)


def wp_coefficients(rows):
    """db4 level-5 wavelet-packet leaf coefficients of every row, concatenated in natural order (r, n_slots)."""
    out = []
    for row in rows:
        wp = pywt.WaveletPacket(data=row, wavelet=_WP_WAVELET, maxlevel=_WP_MAXLEVEL)
        out.append(np.concatenate([node.data for node in wp.get_level(_WP_MAXLEVEL, 'natural')]))
    return np.array(out)


def threshold(coeffs, s, sigma, alpha, floor_k=FLOOR_K):
    """lfpack's per-row hard threshold τ_k = α σ / s_k with the dominant-row survival floor."""
    Vh_hat = np.zeros_like(coeffs)
    for k in range(coeffs.shape[0]):
        mask = np.abs(coeffs[k]) >= alpha * sigma / (s[k] + 1e-40)
        if floor_k > 0 and k == 0 and mask.sum() < floor_k:
            mask[np.argpartition(np.abs(coeffs[k]), -floor_k)[-floor_k:]] = True
        Vh_hat[k] = coeffs[k] * mask
    return Vh_hat.astype(np.float32)


def chunk_result(dec, Vh_hat, bounds):
    """Per-chunk dict accepted by lfpack._container.write_codec (U_scaled unused for the byte count)."""
    _, _, n_w, lo = bounds
    flat = Vh_hat.ravel()
    idx = np.flatnonzero(flat)
    return dict(U_scaled=np.zeros((1, Vh_hat.shape[0]), np.float32), vh_indices=idx, vh_values=flat[idx],
                vh_shape=Vh_hat.shape, ns_original=n_w, ns_extended=dec['rows'].shape[1], left_overlap=lo,
                alpha=1.0, cr_svd=0.0, cr_wp=0.0, cr_total=0.0, rmse=0.0)


def storage_bytes(results, U_units, s_list, B, m, nc):
    """Bytes of the format-2 codec datasets (gzip + shuffle) for these chunks, spatial part per arm."""
    with h5py.File(io.BytesIO(), 'w') as f:
        scale = f.create_group('r/00')
        scale.create_group('meta')
        _container.write_codec(scale, results, _WP_WAVELET, _WP_MAXLEVEL)
        cg = scale['codec']
        del cg['u_values']
        n = {name: cg[name].id.get_storage_size() for name in ('chunk_table', 'vh_deltas', 'vh_values')}
        u = np.concatenate([np.asarray(uu).ravel() for uu in U_units])
        s = np.concatenate(s_list).astype(np.float32)
        if m == 'inf':  # current codec: U_scaled = U · s in float32
            us = np.concatenate([(np.asarray(uu, np.float64) * ss).astype(np.float32).ravel()
                                 for uu, ss in zip(U_units, s_list)])
            n['spatial'] = _h5_size(cg, 'us', us)
        else:
            n['spatial'] = _h5_size(cg, 'u', u if u.dtype == np.int8 else u.astype(np.float16)) + _h5_size(cg, 's', s)
            if B is not None:
                n['spatial'] += _h5_size(cg, 'basis', B.astype(np.float32).ravel())
        return n


def _h5_size(group, name, data):
    kw = dict(chunks=(min(_container._STORAGE_CHUNK, data.size),), compression='gzip', shuffle=True)
    return group.create_dataset(name, data=data, **kw).id.get_storage_size()


def energies(x, x_hat, lo, n_w):
    """Signal and error energy over the written samples, broadband and per band (filtered on the extended chunk)."""
    center = slice(lo, lo + n_w)
    err = x - x_hat
    out = {'sig': float(np.sum(x[:, center] ** 2)), 'err': float(np.sum(err[:, center] ** 2))}
    for band, sos in SOS.items():
        xf = scipy.signal.sosfiltfilt(sos, x, axis=1)[:, center]
        ef = scipy.signal.sosfiltfilt(sos, err, axis=1)[:, center]
        out[f'sig_{band}'], out[f'err_{band}'] = float(np.sum(xf ** 2)), float(np.sum(ef ** 2))
    return out


# ---------------------------------------------------------------------------
# Pass 1: per-chunk singular values and U·s, then the shared basis
# ---------------------------------------------------------------------------

def _pass1_chunk(npy, bounds):
    """Singular values (zero-padded to nc) and the first RMAX columns of U·s (zero-padded) of one chunk."""
    x = read_chunk(npy, bounds)
    U, sv, _ = np.linalg.svd(x, full_matrices=False)
    k = min(RMAX, sv.size)
    sv_pad, us = np.zeros(x.shape[0], np.float32), np.zeros((x.shape[0], RMAX), np.float32)
    sv_pad[: sv.size], us[:, :k] = sv, U[:, :k] * sv[:k]
    return sv_pad, us


def pass1(pid):
    """(n_chunks, nc) singular values and (n_chunks, nc, RMAX) U·s of every chunk, kept in memory (disk is full)."""
    npy = SCRATCH.joinpath(pid, 'lf_resampled_car_cadzow.npy')
    bounds = chunk_bounds(np.load(npy, mmap_mode='r').shape[0])
    res = Parallel(n_jobs=N_JOBS, max_nbytes=None)(delayed(_pass1_chunk)(npy, b) for b in bounds)
    return np.array([r[0] for r in res]), np.array([r[1] for r in res])


def shared_basis(sv, us, epsilon, m):
    """Top-m eigenvectors of Σ_j Us_j Us_jᵀ / ||Us_j||² with Us_j truncated at the chunk's rank."""
    G = np.zeros((us.shape[1], us.shape[1]))
    for sv_j, us_j in zip(sv, us):
        sigma = _svd_noise_floor(sv_j.astype(np.float64))
        r = min(max(1, int(np.sum(sv_j > epsilon * sigma))), RMAX)
        a = us_j[:, :r].astype(np.float64)
        G += a @ a.T / max(np.sum(a ** 2), 1e-300)
    w, V = np.linalg.eigh(G)
    return V[:, np.argsort(w)[::-1][:m]]


def arm_basis(sv, us, m, epsilon):
    return None if m in ('inf', 'inf-f16') else shared_basis(sv, us, epsilon, int(str(m).lstrip('p').removesuffix('q8')))


def prepare(pids):
    """Pass 1 over every probe; keep the singular values and the shared bases of every (ε, m) arm."""
    svs, bases = {}, {}
    for pid in pids:
        sv, us = pass1(pid)
        svs[pid] = sv
        for eps in EPSILONS:
            for m in MS:
                bases[(pid, str(m), eps)] = arm_basis(sv, us, m, eps)
        del us
        print(f'pass 1 {pid[:8]}', flush=True)
    return svs, bases


# ---------------------------------------------------------------------------
# Calibration of α on snippets
# ---------------------------------------------------------------------------

def calibrate_probe(pid, m, epsilon, sv, B):
    """Bytes and energies on N_SNIPPETS chunks of one probe, for every α of ALPHA_GRID."""
    npy = SCRATCH.joinpath(pid, 'lf_resampled_car_cadzow.npy')
    bounds = chunk_bounds(np.load(npy, mmap_mode='r').shape[0])
    picks = np.linspace(1, len(bounds) - 2, N_SNIPPETS).astype(int)
    decs, coeffs, xs = [], [], []
    for ci in picks:
        x = read_chunk(npy, bounds[ci])
        dec = decompose(x, B, m, epsilon, sv_full=sv[ci].astype(np.float64))
        decs.append(dec), coeffs.append(wp_coefficients(dec['rows'])), xs.append(x)
    nc = xs[0].shape[0]
    rows = []
    for alpha in ALPHA_GRID:
        results, e = [], dict(sig=0.0, err=0.0)
        for ci, dec, c, x in zip(picks, decs, coeffs, xs):
            Vh_hat = threshold(c, dec['s'], dec['sigma'], alpha)
            results.append(chunk_result(dec, Vh_hat, bounds[ci]))
            vt = _reconstruct_vh_from_wp(Vh_hat, x.shape[1], Vh_hat.shape[0])
            x_hat = (dec['spatial'] * dec['s']) @ vt
            _, _, n_w, lo = bounds[ci]
            e['sig'] += float(np.sum(x[:, lo:lo + n_w] ** 2))
            e['err'] += float(np.sum((x - x_hat)[:, lo:lo + n_w] ** 2))
        n = storage_bytes(results, [d['U_unit'] for d in decs], [d.get('s_stored', d['s']) for d in decs], None, m, nc)
        if B is not None:  # amortize the per-recording basis over the snippet share of the recording
            n['spatial'] += _h5_size_standalone(B) * N_SNIPPETS / len(bounds)
        rows.append(dict(pid=pid, m=str(m), epsilon=epsilon, alpha=alpha, bytes=sum(n.values()),
                         bytes_spatial=n['spatial'], **e))
    return rows


def _h5_size_standalone(B):
    with h5py.File(io.BytesIO(), 'w') as f:
        return _h5_size(f, 'basis', B.astype(np.float32).ravel())


def solve_alphas(calib):
    """α per arm for the size track (total bytes = reference) and the snr track (pooled SNR = reference)."""
    g = calib.groupby(['m', 'epsilon', 'alpha'])[['bytes', 'sig', 'err']].sum().reset_index()
    g['snr'] = 10 * np.log10(g['sig'] / g['err'])
    ref = g[(g.m == 'inf') & (g.epsilon == REF['epsilon'])].sort_values('alpha')
    la = np.log(ref.alpha.values)
    ref_bytes = np.exp(np.interp(np.log(REF['alpha']), la, np.log(ref.bytes.values)))
    ref_snr = np.interp(np.log(REF['alpha']), la, ref.snr.values)
    out = []
    for (m, eps), d in g.groupby(['m', 'epsilon']):
        d = d.sort_values('alpha')
        la = np.log(d.alpha.values)
        # bytes decrease and snr decreases with α: interpolate on the reversed (increasing) arrays
        a_size = np.exp(np.interp(np.log(ref_bytes), np.log(d.bytes.values[::-1]), la[::-1]))
        a_snr = np.exp(np.interp(ref_snr, d.snr.values[::-1], la[::-1]))
        out += [dict(m=m, epsilon=eps, track='size', alpha=a_size), dict(m=m, epsilon=eps, track='snr', alpha=a_snr)]
    out = pd.DataFrame(out)
    ref_rows = (out.m == 'inf') & (out.epsilon == REF['epsilon'])
    out.loc[ref_rows, 'alpha'] = REF['alpha']  # the reference is the mild tier exactly
    return out, ref_bytes, ref_snr


# ---------------------------------------------------------------------------
# Full encode of one probe for one arm
# ---------------------------------------------------------------------------

def _encode_chunk(npy, bounds, B, m, epsilon, alpha, sv_full):
    x = read_chunk(npy, bounds)
    dec = decompose(x, B, m, epsilon, sv_full=sv_full)
    Vh_hat = threshold(wp_coefficients(dec['rows']), dec['s'], dec['sigma'], alpha)
    vt = _reconstruct_vh_from_wp(Vh_hat, x.shape[1], Vh_hat.shape[0])
    x_hat = (dec['spatial'] * dec['s']) @ vt
    _, _, n_w, lo = bounds
    e = energies(x, x_hat, lo, n_w)
    center = x_hat[:, lo:lo + n_w].astype(np.float32)
    binned = center.reshape(center.shape[0] // 4, 4, n_w).sum(axis=1).T  # (n_w, nc / 4), as bin_channels=4
    return chunk_result(dec, Vh_hat, bounds), dec['U_unit'], dec.get('s_stored', dec['s']), e, binned


def encode_probe(pid, m, epsilon, alpha, keep_recon, sv, B):
    """Encode a whole probe; return bytes, per-chunk energies and (optionally) the binned reconstruction."""
    npy = SCRATCH.joinpath(pid, 'lf_resampled_car_cadzow.npy')
    ns, nc = np.load(npy, mmap_mode='r').shape
    bounds = chunk_bounds(ns)
    res = Parallel(n_jobs=N_JOBS, max_nbytes=None)(
        delayed(_encode_chunk)(npy, b, B, m, epsilon, alpha, sv[ci].astype(np.float64)) for ci, b in enumerate(bounds)
    )
    n = storage_bytes([r[0] for r in res], [r[1] for r in res], [r[2] for r in res], B, m, nc)
    e = pd.DataFrame([r[3] for r in res])
    recon = np.concatenate([r[4] for r in res], axis=0) if keep_recon else None
    ranks = np.array([r[2].size for r in res])
    return n, e, recon, ranks


def summarize(pid, m, epsilon, alpha, track, n, e, ranks):
    snr_chunk = 10 * np.log10(e.sig / e.err)
    row = dict(pid=pid, m=str(m), epsilon=epsilon, alpha=alpha, track=track, bytes=sum(n.values()),
               bytes_spatial=n['spatial'], bytes_vh=n['vh_deltas'] + n['vh_values'], rank_median=np.median(ranks),
               snr=10 * np.log10(e.sig.sum() / e.err.sum()), snr_chunk_median=np.median(snr_chunk),
               snr_chunk_p10=np.percentile(snr_chunk, 10))
    for band in SOS:
        row[f'snr_{band}'] = 10 * np.log10(e[f'sig_{band}'].sum() / e[f'err_{band}'].sum())
    return row


# ---------------------------------------------------------------------------
# Behaviour decoding (functions copied verbatim from the 2026-08-31 decoding scripts)
# ---------------------------------------------------------------------------

def band_envelope(sig, band):
    filtered = nap.apply_bandpass_filter(sig, cutoff=band, order=4)
    env = nap.compute_hilbert_envelope(filtered)
    floor = np.median(env.values, axis=0, keepdims=True) * 1e-6 + np.finfo(np.float32).tiny
    log_env = np.log(np.maximum(env.values, floor)).astype(np.float32)
    return nap.TsdFrame(t=env.t, d=log_env, time_support=env.time_support, columns=sig.columns)


def stack_envelopes(lfp, clean_ep):
    envelopes = {b: band_envelope(lfp, f).restrict(clean_ep) for b, f in BANDS.items()}
    band_names = list(envelopes)
    stacked = np.stack([envelopes[b].values for b in band_names], axis=-1)
    envelope = nap.TsdFrame(t=envelopes[band_names[0]].t, d=stacked.reshape(stacked.shape[0], -1),
                            time_support=envelopes[band_names[0]].time_support)
    del envelopes, stacked
    return envelope


def decode_wheel(envelope, wheel):
    common_ep = envelope.time_support.intersect(wheel.time_support)
    X = envelope.restrict(common_ep).bin_average(BIN_SIZE)
    y = wheel.restrict(common_ep).bin_average(BIN_SIZE)
    valid = np.isfinite(y.values) & np.all(np.isfinite(X.values), axis=1)
    Xv, yv = X.values[valid], y.values[valid]
    y_pred, r2_null = np.empty(len(yv)), []
    for train_idx, test_idx in KFold(n_splits=5, shuffle=False).split(Xv):
        model = RidgeCV(alphas=np.logspace(-2, 4, 20)).fit(Xv[train_idx], yv[train_idx])
        y_pred[test_idx] = model.predict(Xv[test_idx])
        r2_null.append(r2_score(np.random.permutation(yv[test_idx]), y_pred[test_idx]))
    return {'r2': r2_score(yv, y_pred), 'r2_null': float(np.mean(r2_null)), 'n_bins': int(valid.sum())}


def trial_features(envelope, event_times, t_pre, t_post):
    X = np.full((len(event_times), envelope.shape[1]), np.nan, dtype=np.float32)
    for i, t_evt in enumerate(event_times):
        if not np.isfinite(t_evt):
            continue
        window = envelope.restrict(nap.IntervalSet(start=t_evt + t_pre, end=t_evt + t_post))
        if len(window) > 0:
            X[i] = window.values.mean(axis=0)
    return X


def decode_categorical(X, y):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    y_pred, null_scores = np.empty(len(y), dtype=int), []
    for train_idx, test_idx in kfold.split(X, y):
        model = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-3, 3, 10), max_iter=2000))
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])
        for _ in range(N_SHUFFLES):
            null_scores.append(balanced_accuracy_score(np.random.permutation(y[test_idx]), y_pred[test_idx]))
    return {'balanced_accuracy': balanced_accuracy_score(y, y_pred), 'null': float(np.mean(null_scores)),
            'n_trials': len(y)}


_BEHAVIOUR = {}


def behaviour(pid, one):
    """Wheel speed, trials, session-clock sample times and clean epochs of one probe (cached)."""
    if pid not in _BEHAVIOUR:
        from brainbox.io.one import SessionLoader, SpikeSortingLoader

        ssl = SpikeSortingLoader(pid=pid, one=one)
        sl = SessionLoader(one=one, eid=ssl.eid)
        sl.load_wheel()
        sl.load_trials()
        reader = lfpack.LFPackReader(SCRATCH.joinpath(pid, 'lf_compressed.h5'))
        saturated = reader.saturation_times()
        times = ssl.timesprobe2times(np.arange(reader.ns) * Q / FS_LF_RAW)
        saturated_ep = nap.IntervalSet(start=saturated['start_time'], end=saturated['stop_time'])
        clean_ep = nap.IntervalSet(start=times[0], end=times[-1]).set_diff(saturated_ep)
        wheel = nap.Tsd(t=sl.wheel['times'].values, d=np.abs(sl.wheel['velocity'].values))
        _BEHAVIOUR[pid] = dict(wheel=wheel, trials=sl.trials, times=times, clean_ep=clean_ep)
    return _BEHAVIOUR[pid]


def decode_all(recon, bh):
    """Wheel R² and choice / feedback balanced accuracy from a (ns, 96) binned reconstruction."""
    lfp = nap.TsdFrame(t=bh['times'][: recon.shape[0]], d=recon)
    envelope = stack_envelopes(lfp, bh['clean_ep'])
    del lfp
    gc.collect()
    wheel = decode_wheel(envelope, bh['wheel'])
    categ = {}
    trials = bh['trials']
    for target, (event_col, t_pre, t_post) in TARGETS.items():
        X = trial_features(envelope, trials[event_col].values, t_pre, t_post)
        if target == 'choice':
            y_raw, keep = trials['choice'].values, trials['choice'].values != 0
        else:
            y_raw, keep = trials['feedbackType'].values, np.ones(len(trials), dtype=bool)
        keep &= np.all(np.isfinite(X), axis=1)
        categ[target] = decode_categorical(X[keep], (y_raw[keep] == 1).astype(int))
    del envelope
    gc.collect()
    return wheel, categ


def _append(csv, row):
    pd.DataFrame([row]).to_csv(csv, mode='a', header=not csv.exists(), index=False)


# %% Sanity check: the per-chunk arm with m = 'inf' reproduces lfpack.compress exactly
if __name__ == '__main__':
    npy = SCRATCH.joinpath(PIDS[0], 'lf_resampled_car_cadzow.npy')
    b = chunk_bounds(np.load(npy, mmap_mode='r').shape[0])[100]
    x = read_chunk(npy, b)
    dec = decompose(x, None, 'inf', 100.0)
    Vh_hat = threshold(wp_coefficients(dec['rows']), dec['s'], dec['sigma'], 14.0)
    c = lfpack.compress(x.astype(np.float32), epsilon=100.0, alpha=14.0)
    x_mine = (dec['spatial'] * dec['s']) @ _reconstruct_vh_from_wp(Vh_hat, x.shape[1], Vh_hat.shape[0])
    x_lfp = lfpack.decompress(c)
    rel = np.linalg.norm(x_mine - x_lfp) / np.linalg.norm(x_lfp)
    print(f'sanity: rank {Vh_hat.shape[0]} vs {c.U_scaled.shape[1]}, relative difference {rel:.2e}')
    assert Vh_hat.shape[0] == c.U_scaled.shape[1] and rel < 1e-4

# %% Pass 1 and α calibration
if __name__ == '__main__':
    svs, bases = prepare(PIDS)
    if CALIB_CSV.exists():
        calib = pd.read_csv(CALIB_CSV)
    else:
        jobs = [(pid, m, eps, svs[pid], bases[(pid, str(m), eps)]) for eps in EPSILONS for m in MS for pid in PIDS]
        calib = Parallel(n_jobs=N_JOBS, verbose=5, max_nbytes=None)(delayed(calibrate_probe)(*j) for j in jobs)
        calib = pd.DataFrame([row for rows in calib for row in rows])
        calib.to_csv(CALIB_CSV, index=False)
    alphas, ref_bytes, ref_snr = solve_alphas(calib)
    print(f'reference snippets: {ref_bytes / 1e6:.2f} MB, SNR {ref_snr:.2f} dB')
    print(alphas.pivot_table(index=['epsilon', 'm'], columns='track', values='alpha').round(1), flush=True)

# %% Full encodes: size track with decoding, then snr track (bytes + SNR only)
if __name__ == '__main__':
    from one.api import ONE

    one = ONE()
    done = pd.read_csv(ARMS_CSV) if ARMS_CSV.exists() else pd.DataFrame(columns=['pid', 'm', 'epsilon', 'track'])
    done_keys = set(zip(done.pid, done.m.astype(str), done.epsilon.astype(float), done.track))
    for track in ('size', 'snr'):
        for eps in EPSILONS:
            for m in MS:
                sel = (alphas.m == str(m)) & (alphas.epsilon == eps) & (alphas.track == track)
                alpha = float(alphas[sel].alpha.iloc[0])
                if track == 'snr' and str(m) == 'inf' and eps == REF['epsilon']:
                    continue  # identical to the size-track reference
                for pid in PIDS:
                    if (pid, str(m), eps, track) in done_keys:
                        continue
                    n, e, recon, ranks = encode_probe(pid, m, eps, alpha, track == 'size', svs[pid],
                                                      bases[(pid, str(m), eps)])
                    row = summarize(pid, m, eps, alpha, track, n, e, ranks)
                    if track == 'size':
                        wheel, categ = decode_all(recon, behaviour(pid, one))
                        del recon
                        _append(WHEEL_CSV, dict(pid=pid, m=str(m), epsilon=eps, alpha=alpha, **wheel))
                        for target, res in categ.items():
                            _append(CATEG_CSV, dict(pid=pid, m=str(m), epsilon=eps, alpha=alpha, target=target, **res))
                        row['r2'] = wheel['r2']
                    _append(ARMS_CSV, row)
                    print(f"{track} ε={eps:.0f} m={m} {pid[:8]} α={alpha:.1f}: {row['bytes'] / 1e6:.2f} MB "
                          f"(spatial {row['bytes_spatial'] / 1e6:.2f}), SNR {row['snr']:.2f} dB, "
                          f"p10 {row['snr_chunk_p10']:.2f}, R² {row.get('r2', np.nan):.3f}", flush=True)
                    gc.collect()
