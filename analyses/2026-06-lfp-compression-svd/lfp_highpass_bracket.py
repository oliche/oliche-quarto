"""
Highpass-corner bracketing for the lfpack LFP codec: 0.5 / 1 / 2 Hz.

Goal
----
1. Verify the adaptive warmup padding (ibldsp _warmup_pad_out) keeps the decimation
   output continuous across chunk seams when the highpass corner is lowered to 0.5 Hz.
2. Compare, at each corner, the uncompressed reference against the default (ε=150, α=28)
   and aggressive (ε=450, α=96) lfpack tiers — as traces (one figure per corner) and PSDs.

A short window (DUR seconds) is streamed per PID and written to a truncated flat binary so
the *chunked* ``resample_denoise_lfp_cbin`` runs over ≥1 chunk seam without processing the
whole recording. The seam sits at output-sample multiples of CHUNK_SIZE_OUT (8192 → 32.8 s
at 250 Hz) and is drawn on every trace panel so continuity can be eyeballed.

Caveats (local test, not the production cluster path): the truncated reader carries no probe
metadata, so NP1 sample-shift dephasing is skipped and bad-channel interpolation is off
(channel_labels=None). Both apply identically across corners, so the comparison stays fair.
"""
# %%
import sys
from pathlib import Path

sys.path.insert(0, '/Users/olivier/Documents/oliche-quarto/analyses/2026-06-lfp-compression')

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import neuropixel
import numpy as np
import scipy.signal
import seaborn as sns
import spikeglx
from brainbox.io.one import SpikeSortingLoader
from one.api import ONE

from lfpack import LFPackReader, compress_to_h5

sns.set_theme(context='notebook')

# ── Parameters ───────────────────────────────────────────────────────────────
Q = 10
FS_RS = 250.0                       # decimated rate = sr.fs / Q
T0, DUR = 100.0, 64.0               # window start / length (s); 64 s → seam at 32.8 s
CHUNK_SIZE_OUT = 8192               # ibldsp resample chunk (output samples) → seam period
HIGHPASS = [0.5, 1.0, 2.0]          # low-cut corners to bracket [Hz]
TIERS = {                           # label → (epsilon, alpha); None = uncompressed reference
    'uncompressed': None,
    'default': (150.0, 28.0),
    'aggressive': (450.0, 96.0),
}
CADZOW_KWARGS = dict(rank=5, niter=1, fmax=None, nswx=64, ovx=32, gap_threshold=2.0, ppca_k=2.0)
VMAX = 0.25e-3                      # V, shared imshow colour scale (matches benchmark gallery)

CACHE_DIR = Path('/Users/olivier/scratch/lfp/highpass_bracket')
FIGURE_DIR = Path('/Users/olivier/Documents/oliche-quarto/analyses/2026-06-lfp-compression-svd/figures')
DATE = '2026-07-18'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PIDS = [
    '1a276285-8b0e-4cc9-9f0a-a3a002978724',
    '1e104bf4-7a24-4624-a5b2-c2c8289c0de7',
    '6638cfb3-3831-4fc2-9327-194b76cf22e1',
    '749cb2b7-e57e-4453-a794-f6230e4d0226',
    'd7ec0892-0a6c-4f4f-9d8f-72083692af5c',
    'da8dfec1-d265-44e8-84ce-6ae9c109b8bd',
    'dab512bd-a02d-4c1f-8dbc-9155a163efc0',  # amazing CSD
    'dc7e9403-19f7-409f-9240-05ee57cb7aea',  # static noise: positive spikes
    'e8f9fba4-d151-4b00-bee7-447f0f3e752c',
    'eebcaf65-7fa4-4118-869d-a084e84530e2',
    'fe380793-8035-414e-b000-09bfe5ece92a',
]


def stream_truncated_reader(pid, one):
    """Stream a DUR-second LFP window and expose it as a truncated flat-binary Reader.

    Returns a ``spikeglx.Reader`` over ``(ns, nc)`` float32 volts (data channels only, no
    sync), long enough to cross at least one decimation chunk seam.
    """
    bin_file = CACHE_DIR.joinpath(f'{pid}_raw.bin')
    ssl = SpikeSortingLoader(pid=pid, one=one)
    sr = ssl.raw_electrophysiology(band='lf', stream=True)
    nc = sr.nc - sr.nsync
    fs = sr.fs
    if not bin_file.exists():
        i0, i1 = int(T0 * fs), int((T0 + DUR) * fs)
        raw = np.asarray(sr[i0:i1, :nc], dtype=np.float32)  # (ns, nc) volts
        raw.tofile(bin_file)
        ns = raw.shape[0]
    else:
        ns = bin_file.stat().st_size // (nc * 4)
    reader = spikeglx.Reader(bin_file, ns=ns, nc=nc, fs=fs, dtype=np.float32, nsync=0, s2v=1.0)
    return reader, nc, fs


def build_reference(reader, hp, ref_npy):
    """Run the chunked decimation+Cadzow pipeline at highpass ``hp`` → cached (ns, nc) npy."""
    from ibldsp.voltage import resample_denoise_lfp_cbin

    if not ref_npy.exists():
        resample_denoise_lfp_cbin(
            reader, q=Q, output=ref_npy, dtype=np.float32,
            highpass_cutoff=hp, car=True, cadzow_kwargs=CADZOW_KWARGS, n_jobs=4,
        )
    return np.load(ref_npy)  # (ns, nc)


def compress_tier(ref_npy, pid, hp, epsilon, alpha, geometry):
    """Compress a reference checkpoint with lfpack and return the decoded (nc, ns) array."""
    tag = f'{pid}_hp{hp}_eps{epsilon:g}_alpha{alpha:g}'
    h5 = CACHE_DIR.joinpath(f'{tag}.h5')
    if not h5.exists():
        compress_to_h5(ref_npy, h5, recording=pid, h=geometry,
                       epsilon=epsilon, alpha=alpha, fs=FS_RS, n_jobs=4)
    reader = LFPackReader(str(h5), recording=pid)
    try:
        data, _ = reader.read(slice(0, reader.ns), slice(None))
    finally:
        reader.close()
    return data.T  # (nc, ns)


def seam_samples(ns):
    """Output-sample indices of the decimation chunk seams within a length-``ns`` window."""
    return [k * CHUNK_SIZE_OUT for k in range(1, ns // CHUNK_SIZE_OUT + 1)]


def plot_tiers_for_corner(pid, hp, tiers, seams):
    """One figure per corner: uncompressed / default / aggressive stacked, seams marked."""
    fig, axes = plt.subplots(len(tiers), 1, figsize=(14, 9), sharex=True)
    for ax, (label, data) in zip(axes, tiers.items()):
        ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-VMAX, vmax=VMAX,
                  interpolation='none', origin='lower')
        for s in seams:
            ax.axvline(s, color='k', lw=0.8, ls='--', alpha=0.6)
        ax.set_ylabel(label)
        ax.set_yticks([])
    axes[-1].set_xlabel('samples (250 Hz)  —  dashed = decimation chunk seam')
    fig.suptitle(f'{pid[:8]}  highpass = {hp} Hz', fontsize=11)
    fig.tight_layout()
    path = FIGURE_DIR.joinpath(f'{DATE}_highpass_traces_{pid[:8]}_hp{hp}.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_psds(pid, psd_by_corner, freqs):
    """One PSD figure per PID: a subplot per corner overlaying the three tiers."""
    fig, axes = plt.subplots(1, len(HIGHPASS), figsize=(16, 5), sharey=True)
    colors = {'uncompressed': 'k', 'default': '#1f77b4', 'aggressive': '#d62728'}
    for ax, hp in zip(axes, HIGHPASS):
        for label, psd in psd_by_corner[hp].items():
            ax.loglog(freqs, psd, color=colors[label], lw=1.2,
                      ls='-' if label == 'uncompressed' else '--', label=label)
        ax.axvline(hp, color='grey', lw=0.8, ls=':')
        ax.set_title(f'highpass = {hp} Hz')
        ax.set_xlabel('frequency (Hz)')
        ax.set_xlim(0.1, FS_RS / 2)
        ax.grid(True, which='both', alpha=0.3)
    axes[0].set_ylabel('PSD (V²/Hz)')
    axes[0].legend(fontsize=9)
    fig.suptitle(f'{pid[:8]}  —  PSD across highpass corners and codec tiers', fontsize=11)
    fig.tight_layout()
    path = FIGURE_DIR.joinpath(f'{DATE}_highpass_psd_{pid[:8]}.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def process_pid(pid, one):
    """Bracket the three corners for one PID: build references, compress tiers, plot."""
    print(f'{pid[:8]}: streaming {DUR:.0f} s window …', flush=True)
    reader, nc, fs = stream_truncated_reader(pid, one)
    geometry = {k: neuropixel.trace_header(version=1)[k][:nc] for k in ('x', 'y')}
    psd_by_corner = {}
    freqs = None
    for hp in HIGHPASS:
        print(f'{pid[:8]}: highpass {hp} Hz …', flush=True)
        ref_npy = CACHE_DIR.joinpath(f'{pid}_hp{hp}_ref.npy')
        ref = build_reference(reader, hp, ref_npy).T  # (nc, ns)
        tiers, psds = {}, {}
        for label, params in TIERS.items():
            data = ref if params is None else compress_tier(ref_npy, pid, hp, *params, geometry)
            tiers[label] = data
            freqs, pxx = scipy.signal.welch(data, fs=FS_RS, nperseg=2048, axis=-1)
            psds[label] = pxx.mean(axis=0)  # average across channels
        seams = seam_samples(ref.shape[1])
        fig_path = plot_tiers_for_corner(pid, hp, tiers, seams)
        print(f'{pid[:8]}: {hp} Hz → {fig_path.name}  (seams at {seams})', flush=True)
        psd_by_corner[hp] = psds
    psd_path = plot_psds(pid, psd_by_corner, freqs)
    print(f'{pid[:8]}: PSD → {psd_path.name}', flush=True)


# %%
if __name__ == '__main__':
    one = ONE()
    for pid in PIDS:
        process_pid(pid, one)
