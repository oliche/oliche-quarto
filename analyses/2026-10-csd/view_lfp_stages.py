"""
Inspect LFP preprocessing stages in viewephys, for any insertion and start time.

Streams a few seconds of the raw LF band (2500 Hz) from a pid and applies the lfpack production steps
(see `ibldsp.voltage._resample_lfp_chunk`) on a padded window, so filter edges stay out of the view:

  - raw:    LF band as recorded, no processing (2500 Hz),
  - lp:     NP1 sample shift (channel rephasing), 0.5 Hz highpass, FIR lowpass and decimation to 250 Hz,
  - cadzow: full lfpack chain: highpassed signal + bad-channel interpolation, CAR, FIR decimation to 250 Hz, Cadzow denoising
            (`ibldsp.cadzow.LFP_KWARGS_NP1`, bad channels as missing).

Each stage is shown as voltage (LFP), bipolar (BPR, `current_source_density` with n=1, unscaled, in V) and CSD,
one viewephys window per (stage, kind) pair listed in `views` (default: raw, lp, lp-BPR, cadzow, cadzow-BPR).

Usage
-----
Run cell by cell in an IPython shell (`%run -i view_lfp_stages.py` or an IDE): set `pid`, `t0`, `duration`,
and `views` in the last cell. `BENCHMARK_PIDS` lists the lfpack benchmark insertions with a
start time. The viewers need a Qt event loop (`%gui qt` in IPython).
"""
# %%
import numpy as np
import scipy.signal
from brainbox.io.one import SpikeSortingLoader
from iblatlas.regions import BrainRegions
from ibldsp import cadzow, fourier
from ibldsp.voltage import (
    current_source_density,
    detect_bad_channels,
    interpolate_bad_channels,
)
from one.api import ONE
from viewephys.gui import EphysViewer, viewephys

Q = 10  # decimation factor, 2500 -> 250 Hz
HIGHPASS = 0.5  # Hz, lfpack production
PAD = 10.0  # s of context on each side, discarded after filtering
KINDS = ('LFP', 'BPR', 'CSD')
STAGES = ('raw', 'lp', 'cadzow')


def load_stages(pid, t0, duration, one):
    """
    Stream a padded LF window and build the raw, lowpassed and Cadzow-denoised stages.

    Parameters
    ----------
    pid : str
        Probe insertion id.
    t0 : float
        Start of the displayed window in the recording [s].
    duration : float
        Length of the displayed window [s].
    one : one.api.One
        ONE instance, used to stream the LF band.

    Returns
    -------
    dict
        stage name -> (n_channels, n_samples) array [V], 'raw' at 2500 Hz, 'lp' and 'cadzow' at 250 Hz.
    dict
        Trace header with geometry and atlas ids when available.
    numpy.ndarray
        Channel labels (0 good, 1 dead, 2 noisy, 3 outside brain).
    """
    ssl = SpikeSortingLoader(pid=pid, one=one)
    sr = ssl.raw_electrophysiology(band='lf', stream=True)
    fs, nc = float(sr.fs), sr.nc - sr.nsync
    pad = int(PAD * fs // Q) * Q  # whole number of decimated samples
    first = max(0, int(t0 * fs) - pad)
    last = min(sr.ns, int((t0 + duration) * fs) + pad)
    raw = sr[first:last, :nc].T.astype(np.float32)
    h = {k: np.asarray(v) for k, v in sr.geometry.items()}
    h.setdefault('col', np.round((h['x'] - h['x'].min()) / 16).astype(int))
    h.setdefault('row', np.round(h['y'] / 20).astype(int))
    try:  # brain region colours in the viewer
        h['atlas_id'] = ssl.load_channels()['atlas_id']
    except Exception as e:  # noqa: BLE001 histology not always available
        print(f'no atlas ids: {e}')

    # lp: NP1 sample shift, highpass, then FIR lowpass + decimation to 250 Hz
    hp = raw if sr.major_version != 1 else fourier.fshift(raw, h['sample_shift'][:nc], axis=1)
    sos = scipy.signal.butter(3, HIGHPASS, btype='highpass', fs=fs, output='sos')
    hp = scipy.signal.sosfiltfilt(sos, hp, axis=-1)
    lp = scipy.signal.decimate(hp, Q, axis=1, ftype='fir', n=256)
    # cadzow: full lfpack chain, in the order of the production worker
    labels, _ = detect_bad_channels(hp, fs=fs, psd_hf_threshold=1.4)
    cz = interpolate_bad_channels(hp, labels, h['x'][:nc], h['y'][:nc])
    cz = cz - np.median(cz[labels == 0], axis=0)
    cz = scipy.signal.decimate(cz, Q, axis=1, ftype='fir', n=256)
    cz = cadzow.cadzow_denoiser_np1(
        cz, h={'x': h['x'][:nc], 'y': h['y'][:nc]}, fs=fs / Q, bad_channels=np.isin(labels, (1, 2)))

    # crop the padding; the 2500 Hz and decimated grids both start at `first`
    i0 = round(t0 * fs - first)
    n = round(duration * fs)
    out = {'raw': raw[:, i0:i0 + n], 'lp': lp[:, i0 // Q:(i0 + n) // Q], 'cadzow': cz[:, i0 // Q:(i0 + n) // Q]}
    return out, h, labels


def transform(x, kind, h):
    """
    Apply a spatial reference to an (n_channels, n_samples) array [V].

    Parameters
    ----------
    x : numpy.ndarray
        Signal [V].
    kind : str
        'LFP' (unchanged), 'BPR' (first difference, V) or 'CSD' (A m-3).
    h : dict
        Trace header.

    Returns
    -------
    numpy.ndarray
        Transformed signal; viewephys multiplies by 1e6 for display.
    """
    x = np.asarray(x, dtype=np.float64)
    if kind == 'BPR':
        return current_source_density(x, h, n=1, scale=False)
    if kind == 'CSD':
        return current_source_density(x, h) * 1e-6  # viewephys display scale, keeps A m-3 on screen
    return x


# %%
# lfpack benchmark insertions, with the start [s] of the first snippet of 2026-10-10_snippets.csv
BENCHMARK_PIDS = {
    '1a276285-8b0e-4cc9-9f0a-a3a002978724': 2468.3,
    '1e104bf4-7a24-4624-a5b2-c2c8289c0de7': 2632.1,
    '6638cfb3-3831-4fc2-9327-194b76cf22e1': 2140.6,
    '749cb2b7-e57e-4453-a794-f6230e4d0226': 2795.9,
    'd7ec0892-0a6c-4f4f-9d8f-72083692af5c': 2304.4,
    'da8dfec1-d265-44e8-84ce-6ae9c109b8bd': 2140.6,
    'dab512bd-a02d-4c1f-8dbc-9155a163efc0': 1812.9,
    'dc7e9403-19f7-409f-9240-05ee57cb7aea': 2468.3,
    'e8f9fba4-d151-4b00-bee7-447f0f3e752c': 2501.0,
    'eebcaf65-7fa4-4118-869d-a084e84530e2': 2271.6,
    'fe380793-8035-414e-b000-09bfe5ece92a': 2894.2,
}

# %%
duration = 10
pid = '1a276285-8b0e-4cc9-9f0a-a3a002978724'
pid = 'e8f9fba4-d151-4b00-bee7-447f0f3e752c'
t0 = BENCHMARK_PIDS[pid]
br = BrainRegions()
for ev in list(EphysViewer._instances()):  # close the windows left open by a previous run
    ev.close()

stages, h, labels = load_stages(pid, t0, duration, ONE())


# %% (stage, kind) windows to open; uncomment to add more
views = [
    ('raw', 'LFP'),
    ('lp', 'LFP'),
    # ('lp', 'BPR'),
    ('cadzow', 'LFP'),
    # ('cadzow', 'BPR'),
    # ('raw', 'BPR'),
    # ('raw', 'CSD'),
    # ('lp', 'CSD'),
    # ('cadzow', 'CSD'),
]

viewers = []  # keep references so the windows are not garbage collected
for stage, kind in views:
    viewers.append(viewephys(transform(stages[stage], kind, h), fs=2500.0 if stage == 'raw' else 250.0,
                             channels=h, br=br, t0=t0, title=f'{pid[:8]} {stage} {kind}'))
