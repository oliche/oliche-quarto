"""
Per-insertion reconstruction gallery of the three lfpack v04 tiers (shared basis m = 32, ε = 100).

For each of the 11 benchmark insertions the Cadzow-denoised checkpoint is re-encoded with the v04 codec at
α = 2.5 (fine), 7 (default) and 14 (small), then a 3 s window is shown as LFP, residual and CSD next to the
original.  The 11-probe checkpoints use a 2 Hz highpass whereas the production v04 archives use 0.5 Hz, so the
tiers are re-encoded from the same checkpoints that the behaviour decoding uses (same input, same codec) rather
than read from the archives.

Writes ``<DATE>_v04_insertion_<pid8>.png`` and ``<DATE>_v04_insertions_snr.csv`` (full-recording SNR and bytes).
"""

# %%
from pathlib import Path

import addcopyfighandler  # noqa: F401
import matplotlib.cm as mplcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from lfpack import LFPackReader, compress_to_h5

sns.set_theme(context='notebook')

SCRATCH = Path('/Users/olivier/scratch/lfp')
CACHE = SCRATCH.joinpath('v04_tiers')
OUT_DIR = Path(__file__).parent
FIGURE_DIRS = [Path.home().joinpath('Documents', 'figures'), OUT_DIR.joinpath('figures')]
DATE = '2026-10-02'
FS = 250.0
EPSILON = 100.0
TIERS = {'original': None, 'fine': 2.5, 'default': 7.0, 'small': 14.0}  # label -> α, in display order
RESID_GAIN = 5  # residual colour scale is this many times tighter than the LFP one
T0_FRAC, DUR = 0.5, 3.0  # window start (fraction of the recording) and length (s)
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


def encode(pid, alpha):
    """Encode one checkpoint with the v04 codec at ``alpha`` (cached); returns the h5 path."""
    h5 = CACHE.joinpath(f'{pid}_a{alpha:g}.h5')
    if not h5.exists():
        compress_to_h5(SCRATCH.joinpath(pid, 'lf_resampled_car_cadzow.npy'), h5, recording=pid,
                       epsilon=EPSILON, alpha=alpha, fs=FS, n_jobs=1)
    return h5


def decode(h5, pid, nsel):
    """Decoded (ns, nc) float32 for the sample slice ``nsel``."""
    reader = LFPackReader(str(h5), recording=pid)
    try:
        return reader.read(nsel, slice(None))[0]
    finally:
        reader.close()


def full_snr(h5, pid, x, step=40960):
    """Pooled SNR (dB) of the whole recording, decoded piecewise to bound memory."""
    sig = err = 0.0
    for i in range(0, x.shape[0], step):
        ref = np.asarray(x[i:i + step], dtype=np.float64)
        y = decode(h5, pid, slice(i, i + ref.shape[0]))
        sig += np.sum(ref ** 2)
        err += np.sum((ref - y) ** 2)
    return 10 * np.log10(sig / err)


def csd(x):
    """Second spatial difference between channels two rows apart (same Neuropixels column), (ns, nc)."""
    out = np.zeros_like(x)
    out[:, 2:-2] = x[:, :-4] - 2 * x[:, 2:-2] + x[:, 4:]
    return out


def process(pid):
    """Encode, score and plot one insertion; returns the per-tier summary rows."""
    x = np.load(SCRATCH.joinpath(pid, 'lf_resampled_car_cadzow.npy'), mmap_mode='r')
    ns_win = int(DUR * FS)
    a = int(x.shape[0] * T0_FRAC) // 2048 * 2048
    orig = np.asarray(x[a:a + ns_win])
    vmax = np.percentile(np.abs(orig), 99)
    vmax_csd = np.percentile(np.abs(csd(orig)), 99)
    panels, rows = {'original': orig}, []
    for label, alpha in TIERS.items():
        if alpha is None:
            continue
        h5 = encode(pid, alpha)
        panels[label] = decode(h5, pid, slice(a, a + ns_win))
        snr_file = h5.with_suffix('.snr')
        if not snr_file.exists():
            snr_file.write_text(str(full_snr(h5, pid, x)))
        rows.append(dict(pid=pid, tier=label, alpha=alpha, bytes=h5.stat().st_size, snr=float(snr_file.read_text())))

    fig, axes = plt.subplots(3, len(TIERS), figsize=(4.2 * len(TIERS), 9.5), sharex=True, sharey=True)
    ext = [0, DUR, 0, orig.shape[1]]
    for j, (label, y) in enumerate(panels.items()):
        for i, (img, vm) in enumerate([(y, vmax), (orig - y, vmax / RESID_GAIN), (csd(y), vmax_csd)]):
            ax = axes[i, j]
            ax.imshow(img.T, aspect='auto', cmap='RdBu_r', vmin=-vm, vmax=vm, origin='lower',
                      interpolation='none', extent=ext)
        title = label if label == 'original' else f"{label} (α = {TIERS[label]:g})"
        axes[0, j].set_title(title)
        if label != 'original':
            r = next(r for r in rows if r['tier'] == label)
            axes[1, j].text(0.03, 0.97, f"SNR {r['snr']:.1f} dB\n{r['bytes'] / 1e6:.1f} MB", transform=axes[1, j].transAxes,
                            va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, lw=0))
        axes[2, j].set_xlabel('time (s)')
    axes[1, 0].text(0.5, 0.5, f'original − reconstruction\n(colour scale ÷ {RESID_GAIN})', transform=axes[1, 0].transAxes,
                    ha='center', va='center', fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.8, lw=0))
    for ax, name in zip(axes[:, 0], ['LFP', 'residual', 'CSD']):
        ax.set_ylabel(f'{name}\nchannel')
    cax = fig.add_axes([0.3, 0.05, 0.4, 0.012])
    cb = fig.colorbar(mplcm.ScalarMappable(norm=mcolors.Normalize(-vmax, vmax), cmap='RdBu_r'), cax=cax,
                      orientation='horizontal')
    cb.set_label('µV (LFP row)')
    cb.set_ticks(np.linspace(-vmax, vmax, 5))
    cb.set_ticklabels([f'{t * 1e6:.0f}' for t in np.linspace(-vmax, vmax, 5)])
    fig.suptitle(f'{pid[:8]}  —  v04 tiers (m = 32, ε = {EPSILON:g}), {DUR:g} s at {a / FS / 60:.0f} min')
    fig.subplots_adjust(top=0.92, bottom=0.12, wspace=0.08, hspace=0.08)
    for d in FIGURE_DIRS:
        if d.exists():
            fig.savefig(d.joinpath(f'{DATE}_v04_insertion_{pid[:8]}.png'), dpi=110)
    plt.close(fig)
    return rows


# %%
if __name__ == '__main__':
    CACHE.mkdir(parents=True, exist_ok=True)
    res = Parallel(n_jobs=4)(delayed(process)(pid) for pid in PIDS)
    pd.DataFrame([r for rows in res for r in rows]).to_csv(OUT_DIR.joinpath(f'{DATE}_v04_insertions_snr.csv'),
                                                          index=False)
