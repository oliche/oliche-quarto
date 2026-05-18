"""

>>> sshfs fedora:/mnt/s0 /mnt/fedora
>>> rsync -az --relative --include='*/' --include='*.lf.*' --exclude='*' fedora:/mnt/s0/Data/2026_np1np2/./001/ fedora:/mnt/s0/Data/2026_np1np2/./004/ fedora:/mnt/s0/Data/2026_np1np2/./005/ /datadisk/Data/2026_np1np2/
"""
# %%
import addcopyfighandler
import spikeglx
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags

import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler

sns.set_theme(context="talk", palette="muted", style="whitegrid")
import ibldsp.fourier

# this directory contains 001, 002, 003, etc...
root_path = Path('/datadisk/Data/2026/np1np2')
output_fig_path = Path('/home/olivier/Documents/PYTHON/oliche-quarto/analyses/2026-05-lfp/figures')

def read_csv_datasets(root_path):
    csv_path = root_path.joinpath('datasets.csv')
    df_datasets = pd.read_csv(csv_path)

    df_datasets['full_path_ap'] = df_datasets['full_path_ap'].apply(
        lambda x: root_path.joinpath(Path(x).relative_to('/home/pranavrai/Work/int-brain-lab/projects/np1_vs_np2_steinmetz_lab/raw_data'))
    )
    df_datasets['full_path_lf'] = df_datasets['full_path_ap'].apply(
        lambda x: Path(str(x).replace('.ap.bin', '.lf.bin'))
    )

    df_datasets.set_index('label', inplace=True)
    return df_datasets


def plot_psd_log(data, fs, ax=None, plot_kwargs=None, plot_median_kwargs=None, title=None):
    """Plot log-binned PSD curves for multichannel data.

    Parameters
    ----------
    data : array-like, shape (n_channels, n_samples)
        Multichannel time-series data.
    fs : float
        Sampling frequency in Hz.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. Creates one if omitted.
    plot_kwargs : dict, optional
        Keyword arguments for per-channel curves.
    plot_median_kwargs : dict, optional
        Keyword arguments for the median curve.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The plot axis.
    """

    nc = data.shape[0]
    alpha = 10 / nc
    plot_kwargs = {'color': 'k', 'alpha': alpha} | ({} if plot_kwargs is None else plot_kwargs)
    plot_median_kwargs = {'color': 'b', 'label': 'median'} | ({} if plot_median_kwargs is None else plot_median_kwargs)
    title = 'Welchogram (Power Spectral Density)' if title is None else title

    fscale_log, psd_log = ibldsp.fourier.compute_psd_log(data, fs)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    ax.semilogx(
        fscale_log, 10 * np.log10(psd_log.T), **plot_kwargs
    )
    ax.semilogx(fscale_log, 10 * np.log10(np.median(psd_log, axis=0)), **plot_median_kwargs)

    ax.set(xlabel='Frequency (Hz)', ylabel='PSD (dB rel. V/sqrt(Hz)', title=title,
           xlim=(2, 15_000), ylim=(-150, -70))
    return ax

def get_lf_file(root_path, number, probe_type):
    """Return a single lf file for a session folder by probe type.

    probe_type : 'NP1' matches the imec1 subfolder, 'NP2' matches the imec0c subfolder.
    """
    session_path = root_path.joinpath(f"{number:03d}")
    suffix = 'imec1' if probe_type == 'NP1' else 'imec0c'
    for lf_file in session_path.rglob('*.lf.*bin'):
        if lf_file.parent.name.endswith(suffix):
            return lf_file


# %% Display probe locations
number = 4
fig, axs = plt.subplots(1, 2, figsize=(6, 10), sharey=True, sharex=True)

for i, number in enumerate([4, 5]):
    sr_np1 = spikeglx.Reader(get_lf_file(root_path, number, 'NP1'))
    sr_np2 = spikeglx.Reader(get_lf_file(root_path, number, 'NP2'))

    intercept, slope = [2601.3, 1.179] if number <= 4 else [79, 1.204]
    ax = axs[i]
    ax.scatter(sr_np1.geometry['x'], sr_np1.geometry['y'], s=10, label='NP1', marker='s')
    ax.scatter(sr_np2.geometry['x'] + sr_np1.geometry['x'].max() - sr_np2.geometry['x'].min() + 100,
               intercept + slope * sr_np2.geometry['y'], s=10, label='NP2', marker='s')
    ax.set(xlabel='x (μm)', ylabel='y (μm)', title=f'Dataset {number:03d}')
    ax.legend()

if output_fig_path:
    fig.savefig(output_fig_path / f'channel_locations', dpi=300)

# %% Display PSDs
# df_datasets = read_csv_datasets(root_path)
sample_slice = slice(30_000, int(30_000 * 30))
# shank 2 has the highest correlation
for i, number in enumerate([4, 5]):
    # sr  = spikeglx.Reader(df_datasets.loc['4_NP1_external'].full_path_ap)
    # plot_psd_log(sr[30_000: int(30_000 * 30), :].T, sr.fs, title='4_NP1_external')

    sr_np1 = spikeglx.Reader(get_lf_file(root_path, number, 'NP1'))
    sr_np2 = spikeglx.Reader(get_lf_file(root_path, number, 'NP2'))

    intercept, slope = [2601.3, 1.179] if number <= 4 else [79, 1.204]

    # ● idx_np2 has the same length as idx_np1 — entry i is the NP2 channel index closest (in transformed y)
    # to idx_np1[i]. So sr_np2[..., idx_np2] and sr_np1[..., idx_np1] are channel-aligned.
    idx_np1 = np.where(
        (sr_np1.geometry['y'] >= intercept + slope * sr_np2.geometry['y'].min()) &
        (sr_np1.geometry['y'] <= intercept + slope * sr_np2.geometry['y'].max())
    )[0]

    idx_np2 = np.argmin(np.abs(
        (intercept + slope * sr_np2.geometry['y'])[:, None] - sr_np1.geometry['y'][idx_np1][None, :]
    ), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
    plot_psd_log(sr_np1[sample_slice, idx_np1].T, sr_np1.fs, ax=axes[0], title=f'NP1 matched channels - dataset {number:03d}')
    plot_psd_log(sr_np2[sample_slice, :].T, sr_np2.fs, ax=axes[1], title=f'NP2 - dataset {number:03d}')
    axes[0].set(xlim=[0, 1250])
    if output_fig_path:
        fig.savefig(output_fig_path / f'psd_dataset_{number:03d}.png')




# %%
from viewephys.gui import viewephys
import ibldsp.cadzow
t0 = 50
number = 5
sample_slice = slice(int(sr_np1.fs * t0), int(sr_np1.fs * (t0 + 20)))
# shank 2 has the highest correlation

sr_np1 = spikeglx.Reader(get_lf_file(root_path, number, 'NP1'))
sr_np2 = spikeglx.Reader(get_lf_file(root_path, number, 'NP2'))

def preprocessing(raw, sr, k_filter=None):
    # remove DC offset
    # raw = raw - np.mean(raw, axis=1)[:, np.newaxis]
    data = {}
    data['bandpass'] = ibldsp.voltage.destripe_lfp(
        raw, fs=sr.fs, channel_labels=None, neuropixel_version=np.floor(sr.major_version), h=sr.geometry, k_filter=None)
    data['car'] = data['bandpass'] - np.mean(data['bandpass'], axis=0)
    # Compute Current Source Density (CSD) to identify current sources and sinks
    data['csd1'] = ibldsp.voltage.current_source_density(data['bandpass'], h=sr.geometry, scale=False, n=1)
    data['csd2'] = ibldsp.voltage.current_source_density(data['bandpass'], h=sr.geometry, scale=False, n=2)
    # Apply Cadzow denoising using probe geometry information
    # This performs spatiotemporal denoising based on low-rank approximation
    data['cadzow'] = ibldsp.cadzow.cadzow_np1(data['bandpass'], fs=sr.fs, fmax=200, rank=4, h=sr.geometry)
    # Compute Current Source Density (CSD) to identify current sources and sinks
    data['csd1_denoise'] = ibldsp.voltage.current_source_density(data['cadzow'], h=sr.geometry, scale=False, n=1)
    data['csd2_denoise'] = ibldsp.voltage.current_source_density(data['cadzow'], h=sr.geometry, scale=False, n=2)
    return data

# concatenate the np1 and np2 datasets in the display
stages_order = ['bandpass', 'car', 'cadzow', 'csd1', 'csd1_denoise', 'csd2', 'csd2_denoise']

eqc = {}
data_np1 = preprocessing(sr_np1[sample_slice, :-1].T, sr_np1)
data_np2 = preprocessing(sr_np2[sample_slice, :-1].T, sr_np2)


np_ch2 = 40
ch_np1 = idx_np1[np.argmin(np.abs(
    sr_np1.geometry['y'][idx_np1] - (intercept + slope * sr_np2.geometry['y'][np_ch2])
))]

stages = stages_order
t = np.arange(data_np1[stages[0]].shape[1]) / sr_np1.fs
lag_max = int(0.2 * sr_np1.fs)

fig, axes = plt.subplots(len(stages), 2, figsize=(22, 3 * len(stages)),
                         sharex='col', gridspec_kw={'width_ratios': [5, 1]})
for i, stage in enumerate(stages):
    sig1 = data_np1[stage][ch_np1, :]
    sig2 = data_np2[stage][np_ch2, :]
    axes[i, 0].plot(t, sig1, label=f'NP1 ch {ch_np1}')
    axes[i, 0].plot(t, sig2, label=f'NP2 ch {np_ch2}')
    axes[i, 0].set(ylabel=stage)
    axes[i, 0].legend(loc='upper right')
    lags = correlation_lags(len(sig1), len(sig2))
    xcorr = correlate(sig1, sig2) / np.sqrt(np.sum(sig1 ** 2) * np.sum(sig2 ** 2))
    mask = np.abs(lags) <= lag_max
    axes[i, 1].plot(lags[mask] / sr_np1.fs * 1e3, xcorr[mask])
    axes[i, 1].axvline(0, color='k', lw=0.5)
    axes[i, 1].set(ylim=(-.2, 1))
axes[0, 0].set(title=f'dataset {number:03d}', xlim=[15, 20])
axes[0, 1].set(title='Normalized X-Cor')
axes[-1, 0].set(xlabel='Time (s)')
axes[-1, 1].set(xlabel='Lag (ms)')
fig.tight_layout()
if output_fig_path:
    fig.savefig(output_fig_path / f'preprocessing_xcor_{number:03d}.png')

# %% View ephys
for stage in stages_order:
    eqc[f'np1_{stage}'] = viewephys(
        data_np1[stage][idx_np1, :], sr_np1.fs,
        channels={k: v[idx_np1] for k, v in sr_np1.geometry.items()},
        title=f'NP1 {stage} - dataset {number:03d}',
    )
    eqc[f'np2_{stage}'] = viewephys(
        data_np2[stage], sr_np2.fs,
        channels=sr_np2.geometry,
        title=f'NP2 {stage} - dataset {number:03d}',
    )
