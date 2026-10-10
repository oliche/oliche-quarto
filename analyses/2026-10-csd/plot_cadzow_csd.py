"""
Cadzow denoising seen through three spatial references, for the 11 lfpack benchmark insertions.

One figure per 3 s snippet (three per insertion): rows voltage (LFP), bipolar and CSD; columns brain regions,
original, Cadzow-denoised and difference:
  - voltage: the CAR, 250 Hz LF signal,
  - bipolar: first spatial difference between neighbouring sites of the same probe column,
  - CSD: second spatial difference, scaled by the inter-site distance and tissue conductivity.

Snippets are built with the lfpack production chunk worker, on the production chunk at mid-recording, see
`load_snippets_lfpack`. The figures use the lfpack v05 Cadzow (`ibldsp.cadzow.LFP_KWARGS_NP1`, bad channels as
missing); the v04 production run is kept to check the pipeline against the v04 archive. The June 2026 benchmark
snippets (`load_snippets_benchmark`) were made with a 2 Hz highpass and no bad-channel or saturation handling; they are
kept for reference only. One colour scale per row, ±99th percentile of |Cadzow-denoised|, RdBu_r colormap (the
original and the difference saturate in bipolar and CSD).
"""
# %%
from pathlib import Path

import addcopyfighandler  # noqa: F401
import h5py
import matplotlib.pyplot as plt
import neuropixel
import numpy as np
import pandas as pd
import seaborn as sns
from brainbox.ephys_plots import plot_brain_regions
from brainbox.io.one import SpikeSortingLoader
from iblatlas.regions import BrainRegions
from ibldsp import cadzow
from ibldsp.utils import WindowGenerator
from ibldsp.voltage import _resample_lfp_chunk, _warmup_pad_out, current_source_density
from lfpack import LFPackReader
from one.api import ONE

sns.set_theme(context='notebook')

ROOT_OUTPUT = Path('/Users/olivier/scratch/lfp')
CACHE_DIR = ROOT_OUTPUT.joinpath('csd_lfpack_v04')
ARCHIVE_FINE = Path('/Users/olivier/Documents/datadisk/lfp-processing/lfpack/v04/lf_compressed_v04_a2p5_fine_all.h5')
CACHE_DIR.mkdir(exist_ok=True)
FIGURE_DIR = Path(__file__).parent.joinpath('figures')
FIGURE_DIR.mkdir(exist_ok=True)
PIDS = pd.read_csv(
    Path(__file__).parents[1].joinpath('2026-06-lfp-compression-svd', '2026-09-26_pareto_wheel.csv')
)['pid'].drop_duplicates().sort_values().tolist()

PRCTILE = 99  # colour scale: percentile of |data| over the 3 snippets, per column
CMAP = 'RdBu_r'
FS_LF, Q = 2500.0, 10
# lfpack v04 production settings (sdsc-slurms/2026-06-lfpack/compress.py and compress_bin_to_h5 defaults)
HIGHPASS = 0.5  # Hz
MUTE_WINDOW_SAMPLES = 7
CHUNK_SIZE_OUT = 8192  # resample_denoise_lfp_cbin output chunk
CADZOW_KWARGS = {'rank': 5, 'niter': 1, 'fmax': None, 'nswx': 64, 'ovx': 32, 'gap_threshold': 2.0, 'ppca_k': 2.0}
# chunk variants: Cadzow kwargs per cached output, None = no Cadzow
VARIANTS = {'orig': None, 'cadzow': CADZOW_KWARGS, 'cadzow_ovx48': {**CADZOW_KWARGS, 'ovx': 48},
            'cadzow_v05': {**cadzow.LFP_KWARGS_NP1, 'bad_channels_as_missing': True}}  # lfpack v05 (lfpack#22)
DISPLAY_VARIANT = 'cadzow_v05'
FIG_KWARGS = {'dpi': 80, 'pil_kwargs': {'quality': 80}}  # WebP
T_SNIPPETS = [4.0, 15.0, 26.0]  # s, snippet starts within the 32.8 s production chunk
T_SNIPPET = 3.0  # s
REGIONS = BrainRegions()
H = neuropixel.trace_header(version=1)  # all 11 insertions are NP1 (3A / 3B2)


def bipolar(lfp, h):
    """
    Bipolar re-referencing: difference between consecutive sites of the same probe column.

    Parameters
    ----------
    lfp : numpy.ndarray
        LFP (n_channels, n_samples) [V].
    h : dict
        Trace header with 'col' and 'row' arrays.

    Returns
    -------
    numpy.ndarray
        Bipolar signal (n_channels, n_samples) [V]; site k holds lfp[k + 1] - lfp[k] along its column, the
        last site of each column repeats its neighbour.
    """
    out = np.zeros(lfp.shape, dtype=np.float64)
    for col in np.unique(h['col']):
        itr = np.where(h['col'] == col)[0]
        itr = itr[np.argsort(h['row'][itr])]
        out[itr[:-1]] = np.diff(lfp[itr].astype(np.float64), axis=0)
        out[itr[-1]] = out[itr[-2]]
    return out


TRANSFORMS = {  # voltage and bipolar in µV for readable colorbars
    'LFP': (lambda x: x * 1e6, 'µV'),
    'bipolar': (lambda x: bipolar(x, H) * 1e6, 'µV'),
    'CSD': (lambda x: current_source_density(np.asarray(x, dtype=np.float64), H), 'A m$^{-3}$'),
}


def load_snippets_benchmark(pid):
    """
    Load the June 2026 benchmark snippets (2 Hz highpass), as used by figure 1 of the compression page.

    Parameters
    ----------
    pid : str
        Probe insertion id.

    Returns
    -------
    list of tuple
        Three (original, denoised) pairs of (n_channels, n_samples) arrays [V].
    """
    folder = ROOT_OUTPUT.joinpath(pid)
    return [(np.load(folder.joinpath(f'cadzow_orig_{i}.npy')),
             np.load(folder.joinpath(f'cadzow_denoised_adaptive_{i}.npy'))) for i in range(3)]


def load_snippets_lfpack(pid, one, variant=DISPLAY_VARIANT):
    """
    Build snippets with exactly the lfpack v04 production preprocessing, with and without Cadzow.

    Runs the ibldsp worker used by `resample_denoise_lfp_cbin` (as called by `lfpack.compress_bin_to_h5`) on the
    production chunk that contains the middle of the recording: same chunk grid and warm-up padding, raw samples
    streamed from the LF cbin, NP1 sample shift and geometry from the SpikeGLX meta, bad-channel labels and
    saturated intervals read from the v04 archive, 0.5 Hz highpass, CAR, decimation x10, Cadzow and late
    saturation mute. One dummy sample on each side of the streamed slice keeps the worker off its
    recording-edge taper branches, as for an interior chunk in production. Cached in CACHE_DIR.

    Parameters
    ----------
    pid : str
        Probe insertion id.
    one : one.api.One
        ONE instance, used to stream the raw LF band when the insertion is not cached yet.
    variant : str
        Key of VARIANTS for the denoised output: 'cadzow' (v04 production), 'cadzow_ovx48' or 'cadzow_v05'.

    Returns
    -------
    list of tuple
        Three (original, denoised) pairs of (n_channels, n_samples) arrays [V], 250 Hz.
    int
        Index of the first output sample of the chunk in the recording (250 Hz).
    """
    files = {k: CACHE_DIR.joinpath(f'{pid}_chunk_{k}.npy') for k in ('orig', variant)}
    file_i0 = CACHE_DIR.joinpath(f'{pid}_chunk_i0.npy')
    if not all(f.exists() for f in [*files.values(), file_i0]):
        sr = SpikeSortingLoader(pid=pid, one=one).raw_electrophysiology(band='lf', stream=True)
        nc, fs = sr.nc - sr.nsync, float(sr.fs)
        with h5py.File(ARCHIVE_FINE, 'r') as f:
            labels = f[pid]['00/meta'].attrs['labels'][:].astype(np.int8)
            assert int(f[pid]['00/meta'].attrs['ns_total']) == sr.ns // Q
            sat_intervals = f[pid]['saturation'][:]
        geo = sr.geometry
        assert np.allclose(geo['y'][:nc], H['y']) and np.allclose(geo['x'][:nc], H['x'])
        # production chunk grid of resample_denoise_lfp_cbin (CHUNK_SIZE_OUT is hardcoded there)
        ns_out = sr.ns // Q
        pad = _warmup_pad_out(HIGHPASS, fs / Q, CHUNK_SIZE_OUT)
        wg = WindowGenerator(ns=ns_out, nswin=CHUNK_SIZE_OUT + 2 * pad, overlap=2 * pad)
        win_first, win_last, first_valid, last_valid = next(
            w for w in wg.firstlast_valid if w[2] <= ns_out // 2 < w[3])
        first_in, last_in = win_first * Q, min(sr.ns, win_last * Q)
        assert first_in > 0 and last_in < sr.ns  # interior chunk
        # streamed padded slice, framed by one dummy sample on each side
        raw = np.zeros((last_in - first_in + 2, sr.nc), dtype=np.float32)
        raw[1:-1] = sr[first_in:last_in, :]
        file_raw = CACHE_DIR.joinpath(f'{pid}_chunk_raw.bin')
        raw.tofile(file_raw)
        sat = np.zeros(raw.shape[0], dtype=bool)
        for start, stop in sat_intervals:
            start, stop = max(start, first_in) - first_in + 1, min(stop, last_in) - first_in + 1
            sat[start:max(start, stop)] = True
        file_sat = CACHE_DIR.joinpath(f'{pid}_chunk_saturation.npy')
        np.save(file_sat, sat)
        n_out = last_valid - first_valid
        for k, file in files.items():
            if file.exists():
                continue
            file_car = file.with_name(f'{file.stem}_car.npy')
            np.lib.format.open_memmap(file, mode='w+', dtype=np.float32, shape=(n_out, nc))
            np.lib.format.open_memmap(file_car, mode='w+', dtype=np.float32, shape=(n_out,))
            _resample_lfp_chunk((
                str(file_raw), str(file), (n_out, nc), np.dtype(np.float32).str, 0, n_out,
                1, raw.shape[0] - 1, first_valid - win_first, Q, HIGHPASS, nc, int(sr.major_version),
                np.array(geo['sample_shift'][:nc]), fs, labels, np.array(geo['x'][:nc]), np.array(geo['y'][:nc]),
                {'nc': sr.nc, 'ns': raw.shape[0], 'fs': fs, 'dtype': np.dtype(np.float32).str},
                True, str(file_car), VARIANTS[k],
                str(file_sat), MUTE_WINDOW_SAMPLES,
            ))
        np.save(file_i0, first_valid)  # same chunk for every variant
        file_raw.unlink()
    orig, den = (np.load(files[k]).T for k in ('orig', variant))
    sls = [slice(int(t * FS_LF / Q), int((t + T_SNIPPET) * FS_LF / Q)) for t in T_SNIPPETS]
    return [(orig[:, sl], den[:, sl]) for sl in sls], int(np.load(file_i0))


def load_atlas_ids(pid, one):
    """
    Load the histology atlas id of each of the 384 channels, cached in channels_atlas_ids.csv.

    Parameters
    ----------
    pid : str
        Probe insertion id.
    one : one.api.One
        ONE instance, only used when the insertion is not cached yet.

    Returns
    -------
    numpy.ndarray
        Allen atlas id per channel, in raw channel order (channel 0 at the probe tip).
    """
    file_cache = Path(__file__).parent.joinpath('channels_atlas_ids.csv')
    df = pd.read_csv(file_cache) if file_cache.exists() else pd.DataFrame(columns=['pid', 'channel', 'atlas_id'])
    if pid not in df['pid'].values:
        channels = SpikeSortingLoader(pid=pid, one=one).load_channels()
        assert np.all(channels['rawInd'] == np.arange(384))
        new = pd.DataFrame({'pid': pid, 'channel': np.arange(384), 'atlas_id': channels['atlas_id']})
        df = pd.concat([df, new]) if df.size else new
        df.to_csv(file_cache, index=False)
    return df.loc[df['pid'] == pid].sort_values('channel')['atlas_id'].values.astype(int)


def plot_snippet(pid, isnip, snippet, t, atlas_ids, labels):
    """
    Plot one snippet: rows voltage (LFP), bipolar and CSD; columns brain regions, original, Cadzow-denoised and
    difference.

    One colour scale and colorbar per row, ±PRCTILE percentile of |Cadzow-denoised|; the original and the difference
    saturate in bipolar and CSD. Dead (black) and noisy (red) channels are marked on the brain-region column.

    Parameters
    ----------
    pid : str
        Probe insertion id.
    isnip : int
        Snippet index, for the file name.
    snippet : tuple
        (original, denoised) arrays (n_channels, n_samples) [V], from `load_snippets_lfpack`.
    t : float
        Time of the snippet start in the recording [s].
    atlas_ids : numpy.ndarray
        Allen atlas id per channel, for the brain-region column.
    labels : numpy.ndarray
        Channel labels (1 dead, 2 noisy).
    """
    orig, den = snippet
    fig, axes = plt.subplots(len(TRANSFORMS), 4, figsize=(26, 14.6), layout='constrained',
                             gridspec_kw={'width_ratios': [0.22, 1, 1, 1]})
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.005, hspace=0.005)
    for irow, (kind, (fcn, unit)) in enumerate(TRANSFORMS.items()):
        ax = axes[irow, 0]
        plot_brain_regions(atlas_ids, channel_depths=np.arange(atlas_ids.size), brain_regions=REGIONS, ax=ax,
                           label='left')
        for lab, col in ((1, 'k'), (2, 'crimson')):
            ich = np.where(labels == lab)[0]
            ax.plot(np.full(ich.size, 0.93), ich, ls='none', marker='<', ms=5, color=col,
                    transform=ax.get_yaxis_transform(), clip_on=False, in_layout=False)
        ax.set_ylim(-0.5, atlas_ids.size - 0.5)
        ax.tick_params(axis='y', labelsize=7)
        ax.set_title(kind if irow else f'{kind}\nregions', fontsize=10)
        o, d = fcn(orig), fcn(den)
        vmax = np.nanpercentile(np.abs(d), PRCTILE)
        for ax, x, title in zip(axes[irow, 1:], (o, d, o - d), ('original', 'Cadzow-denoised (lfpack v05)',
                                                                'difference')):
            im = ax.imshow(x, aspect='auto', cmap=CMAP, vmin=-vmax, vmax=vmax, origin='lower', interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if irow == 0:
                ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=axes[irow, :], location='right', shrink=0.95, aspect=25, pad=0.005, label=f'{kind} [{unit}]')
    fig.suptitle(f'{pid}, t = {t:.0f}-{t + T_SNIPPET:.0f} s; triangles: dead (black) and noisy (red) channels',
                 fontsize=11)
    fig.savefig(FIGURE_DIR.joinpath(f'2026-10-10_csd_{pid[:8]}_{isnip}.webp'), **FIG_KWARGS)
    plt.close(fig)


# %%
one = ONE()
times, checks = [], []
for pid in PIDS:
    atlas_ids = load_atlas_ids(pid, one)
    with h5py.File(ARCHIVE_FINE, 'r') as f:
        labels = f[pid]['00/meta'].attrs['labels'][:]
    load_snippets_lfpack(pid, one, variant='cadzow')  # v04 production run, for the check below
    snippets, i0 = load_snippets_lfpack(pid, one)
    t0 = i0 / (FS_LF / Q)
    for isnip, snippet in enumerate(snippets):
        plot_snippet(pid, isnip, snippet, t0 + T_SNIPPETS[isnip], atlas_ids, labels)
        times.append({'pid': pid, 'snippet': isnip, 't': t0 + T_SNIPPETS[isnip]})
    # check against production: decoded v04 fine archive over the same chunk
    chunk = {k: np.load(CACHE_DIR.joinpath(f'{pid}_chunk_{k}.npy')) for k in ('orig', 'cadzow')}
    fine = LFPackReader(ARCHIVE_FINE, recording=pid)[i0:i0 + chunk['cadzow'].shape[0], :]
    checks.append({'pid': pid, 't0': t0, **{
        f'snr_{k}_vs_fine': 10 * np.log10(np.sum(fine ** 2) / np.sum((x - fine) ** 2)) for k, x in chunk.items()}})
df_checks = pd.DataFrame(checks)
df_checks.to_csv(Path(__file__).parent.joinpath('2026-10-09_check_vs_v04_fine.csv'), index=False)
print(df_checks.round(2))
pd.DataFrame(times).to_csv(Path(__file__).parent.joinpath('2026-10-10_snippets.csv'), index=False)
