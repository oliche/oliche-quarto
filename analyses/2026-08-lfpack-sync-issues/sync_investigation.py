# %%
# https://github.com/int-brain-lab/lfpack/issues/8
"""
Audit of ibllib.ephys.sync_probes (pyproject-migration branch) against the probes flagged
in lfpack#8 as producing NaN / high-residual nidq sync.

Downloads only the small raw sync-front datasets (_spikeglx_sync.{channels,polarities,times},
.meta, .wiring.json) for the nidq and the affected probe of each session -- never the raw
.cbin/.ch binaries -- then re-runs the 3B synchronisation locally step by step. This tells us
whether the bad fit is reproducible from the raw pulses themselves (a genuine session-level
sync defect) or is an artefact of a since-fixed extraction bug.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import addcopyfighandler  # noqa: F401


import one.alf.io as alfio
from one.alf.exceptions import ALFObjectNotFound
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
import spikeglx

sns.set_theme(context='notebook')
TODAY = '2026-08-20'
FIG_PATH = Path('figures')
FIG_PATH.mkdir(exist_ok=True)

import ibllib.ephys.sync_probes as sync_probes
from ibllib.io.extractors.ephys_fpga import get_sync_fronts, get_ibl_sync_map

one = ONE()

# 20 probes flagged in lfpack#8 (7 with NaN sync -> extraction never produced a passing fit,
# 13 more from the same sweep with high but non-NaN residuals)
pids_fail = [
    '0b8ea3ec-e75b-41a1-9442-64f5fbc11a5a',
    '0d59e3a1-86c1-44bd-b291-d1f8bc8327ba',
    '0fed7207-f747-428b-b4c0-854cabb50d9e',
    '2adc4f5d-bc7b-42a4-be76-f5df33d713d4',
    '367e94f6-df51-4120-a297-77fa88dcec31',
    '4836a465-c691-4852-a0b1-dcd2b1ce38a1',
    '53ecbf4f-e0d8-4fe6-a852-8b934a37a1c2',
    '5999eeca-10fa-4e4b-ae7c-02fab4fe41be',
    '5c63d860-1e3c-481b-a290-9f299a5421f5',
    '735fa61d-db9b-4289-990d-659793413c75',
    '7967a14e-1bf0-4666-acb4-9b08ba8f3385',
    '81f0087b-2bd1-4e48-8e86-e8206aee3d9d',
    'a5f2ec22-0ff3-4249-bd2f-6247c3990e53',
    'ad597f5f-5201-4e02-9a28-1fb1a75746cc',
    'b2746c16-7152-45a3-a7f0-477985638638',
    'b543e81e-4c8f-415e-82ec-631b177d19d2',
    'c09b3c18-c9e5-4551-9a35-7b2a069f57ff',
    'd0046384-16ea-4f69-bae9-165e8d0aeacf',
    'e6402305-5028-42aa-975b-c540c882b131',
    'f2ea7211-85f3-4394-b03e-1302a1dfe79c',
]

# same 20 probes, ordered as in the "Per-probe results" table of diagnosis_source.md (grouped by
# diagnosis, worst first) -- used only for display/figure ordering, not for the analysis itself
pids_table_order = [
    '4836a465-c691-4852-a0b1-dcd2b1ce38a1',
    'd0046384-16ea-4f69-bae9-165e8d0aeacf',
    '367e94f6-df51-4120-a297-77fa88dcec31',
    '7967a14e-1bf0-4666-acb4-9b08ba8f3385',
    'ad597f5f-5201-4e02-9a28-1fb1a75746cc',
    '0fed7207-f747-428b-b4c0-854cabb50d9e',
    '0b8ea3ec-e75b-41a1-9442-64f5fbc11a5a',
    'a5f2ec22-0ff3-4249-bd2f-6247c3990e53',
    'f2ea7211-85f3-4394-b03e-1302a1dfe79c',
    'b2746c16-7152-45a3-a7f0-477985638638',
    '81f0087b-2bd1-4e48-8e86-e8206aee3d9d',
    'e6402305-5028-42aa-975b-c540c882b131',
    '2adc4f5d-bc7b-42a4-be76-f5df33d713d4',
    '0d59e3a1-86c1-44bd-b291-d1f8bc8327ba',
    '5c63d860-1e3c-481b-a290-9f299a5421f5',
    'b543e81e-4c8f-415e-82ec-631b177d19d2',
    '53ecbf4f-e0d8-4fe6-a852-8b934a37a1c2',
    '735fa61d-db9b-4289-990d-659793413c75',
    'c09b3c18-c9e5-4551-9a35-7b2a069f57ff',
    '5999eeca-10fa-4e4b-ae7c-02fab4fe41be',
]

def _affine_residual(_s, _t):
    """
    Single global affine fit + residual -- the exact QC attach_ibl_metadata.py::compute_sync()
    runs on a probe's sync.timestamps ALF object (max residual < 1 ms to pass).

    Parameters
    ----------
    _s : np.ndarray
        Probe-side sample-index column of a sync.timestamps array.
    _t : np.ndarray
        Reference-time column of the same array.

    Returns
    -------
    dict
        slope, intercept, max_residual_ms, rmse_ms, qc_pass.
    """
    slope, intercept = np.polyfit(_s, _t, 1)
    residuals = _t - np.polyval([slope, intercept], _s)
    max_residual_ms = np.max(np.abs(residuals)) * 1e3
    median_residual_ms = np.median(np.abs(residuals)) * 1e3
    rmse_ms = np.sqrt(np.mean(residuals ** 2)) * 1e3
    return dict(slope=slope, intercept=intercept, max_residual_ms=max_residual_ms,
                median_residual_ms=median_residual_ms, rmse_ms=rmse_ms, qc_pass=bool(max_residual_ms < 1.0))


def compute_residuals_from_spike_sorting_loader(pid):
    """
    Residual of the sync currently registered on Alyx, exactly as attach_ibl_metadata.py's QC
    computes it: a single global affine fit of the probe's own sync.timestamps ALF object.

    Parameters
    ----------
    pid : str
        Probe insertion UUID.

    Returns
    -------
    dict
        slope, intercept, max_residual_ms, rmse_ms, qc_pass (see `_affine_residual`).
    """
    ssl = SpikeSortingLoader(one=one, pid=pid)
    ssl.samples2times(0)  # triggers sync download
    out = _affine_residual(ssl._sync['timestamps'][:, 0], ssl._sync['timestamps'][:, 1])
    print(f"  {pid}: max_residual={out['max_residual_ms']:.3f} ms  rmse={out['rmse_ms']:.3f} ms  "
          f"{'PASS' if out['qc_pass'] else 'FAIL'}")
    return out


def _is_needed(dataset_name):
    """A raw sync front, meta or wiring file -- never the large .cbin/.ch binaries."""
    if 'sync.' in dataset_name and dataset_name.endswith('.npy'):
        return True
    return dataset_name.endswith(('.meta', 'wiring.json'))


def download_sync_fronts(pid):
    """
    Download the small raw datasets needed to re-run synchronisation for one session: sync
    fronts, .meta and .wiring.json for the nidq (3B) and every probe -- not just the flagged
    one, since 3A sessions (no nidq) sync probes against each other.

    Parameters
    ----------
    pid : str
        Probe insertion UUID.

    Returns
    -------
    ses_path : one.alf.path.ALFPath
        Local ONE-cache session path, populated with the files sync_probes expects.
    probe : str
        Probe label, e.g. 'probe00'.
    eid : str
        Session UUID.
    """
    eid, probe = one.pid2eid(pid)
    ses_path = one.eid2path(eid)
    for dset in filter(_is_needed, one.list_datasets(eid, collection='raw_ephys_data*')):
        one.load_dataset(eid, dset, download_only=True)
    return ses_path, probe, eid


def classify_defect(sync_ref, sync_probe, version, linear_max_residual_ms):
    """
    Classify the raw-pulse pathology from independent per-channel checks on the untruncated
    front times: a sustained pulse-rate change (real edges dropped partway through the
    recording) vs an isolated duplicate/bounce edge (one transition double-triggering the
    detector), on either channel. 3A sessions use frame2ttl/camera pulses, which are naturally
    bursty (dense within a trial, sparse between), so the rate-change/burst tests are unreliable
    there -- those are flagged only by residual magnitude.

    Parameters
    ----------
    sync_ref : iblutil.util.Bunch
        Reference channel fronts (nidq for 3B, the probe with the most pulses for 3A).
    sync_probe : iblutil.util.Bunch
        Target probe's fronts.
    version : str
        '3A' or '3B'.
    linear_max_residual_ms : float
        The single global-affine residual, used only for the 3A magnitude-based fallback.

    Returns
    -------
    str
        One of 'dropped_edges (probe)', 'dropped_edges (ref)', 'duplicate_burst (probe)',
        'duplicate_burst (ref)', 'single_edge_glitch (probe)', 'single_edge_glitch (ref)',
        'irregular_3A_reference', or 'clean'.
    """
    if version == '3A':
        return 'clean' if linear_max_residual_ms < 15 else 'irregular_3A_reference'

    def rate_change(times, block=20):
        dt = np.diff(times)
        med = np.median(dt)
        if dt.size < block * 3 or med <= 0:
            return False
        block_med = np.array([np.median(dt[i:i + block]) for i in range(0, dt.size - block, block)])
        bad = np.where(np.abs(block_med - med) / med > 0.3)[0]
        return bool(bad.size > 0 and bad[-1] >= len(block_med) - 2)  # sustained to the end, not a blip

    def burst(times):
        dt = np.diff(times)
        med = np.median(dt)
        return bool(med > 0 and np.any(dt < 0.2 * med))

    def isolated_glitch(times, abs_thresh=0.005):
        # a real one-off mistimed edge, as opposed to ordinary sub-ms clock jitter -- deliberately
        # an absolute threshold: the two real cases found this way deviate by 139ms and 12ms,
        # i.e. very different *relative* sizes (28% vs 2.4% of the nominal period)
        dt = np.diff(times)
        med = np.median(dt)
        return bool(med > 0 and np.max(np.abs(dt - med)) > abs_thresh)

    if rate_change(sync_probe.times):
        return 'dropped_edges (probe)'
    if rate_change(sync_ref.times):
        return 'dropped_edges (ref)'
    if burst(sync_probe.times):
        return 'duplicate_burst (probe)'
    if burst(sync_ref.times):
        return 'duplicate_burst (ref)'
    if isolated_glitch(sync_probe.times):
        return 'single_edge_glitch (probe)'
    if isolated_glitch(sync_ref.times):
        return 'single_edge_glitch (ref)'
    return 'clean'


def reproduce_sync(ses_path, probe):
    """
    Re-run the essential steps of sync_probes.version3A()/version3B() for a single probe,
    keeping the intermediate front times and residuals that the wrappers discard.

    Parameters
    ----------
    ses_path : pathlib.Path
        Local session path populated by `download_sync_fronts`.
    probe : str
        Probe label, e.g. 'probe00'.

    Returns
    -------
    dict
        version ('3A'/'3B'), n_ref / n_probe pulse counts, the production fit type and its
        pass/fail (prod_qc), the max/rmse residual for 'linear' (global affine, mirrors the
        metadata-attach QC), 'exact' and 'smooth' fits, and the raw front times for plotting.
    """
    version = spikeglx.get_neuropixel_version_from_folder(ses_path)
    ephys_files = spikeglx.glob_ephys_files(ses_path, ext='meta', bin_exists=False)

    if version == '3B':
        ref_file = next(ef for ef in ephys_files if ef.get('nidq'))
        probe_file = next(ef for ef in ephys_files if ef.path.parts[-1] == probe)
        for ef in (ref_file, probe_file):
            ef['sync'] = alfio.load_object(ef.path, 'sync', namespace='spikeglx', short_keys=True)
            ef['sync_map'] = get_ibl_sync_map(ef, '3B')
        sync_ref = get_sync_fronts(ref_file.sync, ref_file.sync_map['imec_sync'])
        sync_probe = get_sync_fronts(probe_file.sync, probe_file.sync_map['imec_sync'])
        n = min(sync_ref.times.size, sync_probe.times.size)
        t, tref = sync_probe.times[:n], sync_ref.times[:n]
        n_ref = sync_ref.times.size
        # this is the decision the real pipeline makes (version3B): fall back to 'exact' if the
        # probe's own pulse spacing is already too noisy to trust a smoothed fit
        prod_type = 'smooth' if sync_probes._check_diff_3b(sync_probe) else 'exact'
        tol = 2.5
    else:  # 3A: probes are synced against each other via frame2ttl / camera pulses, no nidq
        for ef in ephys_files:
            ef['sync'] = alfio.load_object(ef.ap.parent, 'sync', namespace='spikeglx', short_keys=True)
            ef['sync_map'] = get_ibl_sync_map(ef, '3A')
        aux = next(a for a in ('frame2ttl', 'right_camera') if all(a in ef.sync_map for ef in ephys_files))
        fronts = [get_sync_fronts(ef.sync, ef.sync_map[aux]) for ef in ephys_files]
        n = min(f.times.size for f in fronts)
        times = np.array([f.times[:n] for f in fronts])
        iref = int(np.argmax([len(ef.sync.channels) for ef in ephys_files]))
        itarget = next(i for i, ef in enumerate(ephys_files) if ef.path.parts[-1] == probe)
        probe_file, t, tref = ephys_files[itarget], times[itarget], times[iref]
        sync_ref, sync_probe = fronts[iref], fronts[itarget]
        n_ref = fronts[iref].times.size
        prod_type = 'smooth'  # version3A always uses this default, no data-driven fallback
        tol = 2.1

    sr = sync_probes._get_sr(probe_file)
    out = dict(version=version, probe=probe, n_ref=n_ref, n_probe=t.size, prod_type=prod_type, t=t, tref=tref,
               sr=sr, tol=tol, sync_probe=sync_probe, sync_ref=sync_ref)

    # the single global affine fit, evaluated the same way attach_ibl_metadata.py's QC does --
    # NOT via sync_probe_front_times(type='linear'), whose 2-point sync_points array only
    # round-trips correctly through interp1d(..., fill_value='extrapolate'); np.interp would
    # clip outside [0, 1] and fabricate a huge bogus residual
    pol = np.polyfit(t, tref, 1)
    linear_residual = tref - np.polyval(pol, t)
    out['linear_qc'] = bool(np.max(np.abs(linear_residual)) < 1e-3)
    out['linear_max_residual_ms'] = np.max(np.abs(linear_residual)) * 1e3
    out['linear_rmse_ms'] = np.sqrt(np.mean(linear_residual ** 2)) * 1e3

    for fit_type in ('exact', 'smooth'):
        sync_points, qc = sync_probes.sync_probe_front_times(t, tref, sr, display=False, type=fit_type, tol=tol)
        residual = tref - np.interp(t, sync_points[:, 0], sync_points[:, 1])
        out[f'{fit_type}_qc'] = qc
        out[f'{fit_type}_max_residual_ms'] = np.max(np.abs(residual)) * 1e3
        out[f'{fit_type}_rmse_ms'] = np.sqrt(np.mean(residual ** 2)) * 1e3
    out['prod_qc'] = out[f'{prod_type}_qc']
    out['diagnosis'] = classify_defect(sync_ref, sync_probe, version, out['linear_max_residual_ms'])
    return out


def plot_extraction_residual(diag, ax=None):
    """
    Reproduce the exact residual plot sync_probes.sync_probe_front_times() draws during
    extraction (display=True), using the production fit type (diag['prod_type']) -- as opposed
    to the simplified global-affine scatter used elsewhere in this script. Shows the raw
    residual, the frequency-smoothed residual, and the interpolation knot points actually baked
    into the probe's sync.timestamps, over the whole session.

    Parameters
    ----------
    diag : dict
        A `reproduce_sync` result (needs 't', 'tref', 'sr', 'prod_type', 'tol'; 'pid' and
        'probe' are used for the title if present).
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new one is created if not given.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = ax or plt.axes()
    sync_probes.sync_probe_front_times(
        diag['t'], diag['tref'], diag['sr'], display=ax, type=diag['prod_type'], tol=diag['tol']
    )
    ax.set_title(f"{diag.get('pid', '')} ({diag.get('probe', '')})  type={diag['prod_type']}  qc={diag.get('prod_qc')}")
    return ax


def plot_front_diff(diag, ax=None):
    """
    Plot the absolute inter-pulse interval (diff of raw front times, in ms) for the probe and
    nidq/reference channels on a shared time axis and a single shared y-scale. Unlike the
    residual plots, this shows the pulse trains directly rather than through a fit -- rate
    changes, bursts and isolated glitches are visible as excursions from the flat nominal
    interval on whichever channel they occur, and plotting both channels on the same axis (rather
    than centring/twin axes) makes it directly visible when the two intervals genuinely diverge
    versus track each other.

    Only rising edges (polarities == 1) are used, matching `_check_diff_3b`/`classify_defect` --
    `get_sync_fronts` returns both polarities of the sync square wave, and consecutive rising +
    falling edges have a slightly different (duty-cycle-driven) interval, which would otherwise
    dominate the plot as a fast alternation between two values on every single pulse.

    No title is set -- intended as the second panel under `plot_extraction_residual`'s plot for
    the same probe, which already carries the identifying title.

    Parameters
    ----------
    diag : dict
        A `reproduce_sync` result (needs 'sync_probe', 'sync_ref').
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new one is created if not given.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = ax or plt.axes()
    t = diag['sync_probe'].times[diag['sync_probe'].polarities == 1]
    tref = diag['sync_ref'].times[diag['sync_ref'].polarities == 1]
    dt_probe = np.diff(t) * 1e3
    dt_ref = np.diff(tref) * 1e3
    ax.plot(t[:-1], dt_probe, color='C0', lw=0.6, label='probe')
    ax.plot(tref[:-1], dt_ref, color='C1', lw=0.6, label='ref')
    ax.set_ylabel('Δt (ms)')
    ax.set_xlabel('time (s)')
    ax.legend()
    return ax


def resync_and_compute_residuals(ses_path, probe, version):
    """
    Actually run sync_probes.version3A()/version3B() -- writing a fresh sync.timestamps.npy for
    `probe` from the raw fronts on disk -- then apply the exact same affine-residual check as
    `compute_residuals_from_spike_sorting_loader`, this time against what we just recomputed
    rather than whatever is currently registered on Alyx.

    Parameters
    ----------
    ses_path : pathlib.Path
        Local session path populated by `download_sync_fronts`.
    probe : str
        Probe label, e.g. 'probe00'.
    version : str
        '3A' or '3B', as returned by `spikeglx.get_neuropixel_version_from_folder`.

    Returns
    -------
    dict
        slope, intercept, max_residual_ms, rmse_ms, qc_pass (see `_affine_residual`).
    """
    if version == '3B':
        sync_probes.version3B(ses_path, display=False, probe_names=[probe])
    else:
        sync_probes.version3A(ses_path, display=False, probe_names=[probe])
    ts_file = next(ses_path.joinpath('raw_ephys_data', probe).glob('*.timestamps.npy'))
    timestamps = np.load(ts_file)
    return _affine_residual(timestamps[:, 0], timestamps[:, 1])


# %%
# Guarded so this module can be imported for its helper functions (download_sync_fronts,
# classify_defect, _affine_residual, the `one` client, etc.) without re-running the full
# 20-probe audit pipeline below as an import side effect -- unchanged when run directly as a
# script (__name__ == '__main__' either way).
if __name__ == '__main__':
    # note the order: the SpikeSortingLoader-based (production) residual is computed BEFORE
    # resync_and_compute_residuals, which overwrites the local .sync./.timestamps. cache files --
    # ONE's check_hash=True default should re-fetch them if this script is ever re-run, but there is
    # no reason to rely on that when the safe order costs nothing
    results = []
    for pid in tqdm.tqdm(pids_fail):
        try:
            ses_path, probe, eid = download_sync_fronts(pid)
            diag = reproduce_sync(ses_path, probe)
            ssl_res = compute_residuals_from_spike_sorting_loader(pid)
            resync_res = resync_and_compute_residuals(ses_path, probe, diag['version'])
        except (ALFObjectNotFound, StopIteration) as e:
            results.append(dict(pid=pid, error=f'{type(e).__name__}: {e}'))
            continue
        diag.update(
            pid=pid, eid=str(eid),
            slope_ssl=ssl_res['slope'], intercept_ssl=ssl_res['intercept'], max_residual_ssl_ms=ssl_res['max_residual_ms'],
            median_residual_ssl_ms=ssl_res['median_residual_ms'],
            slope_resync=resync_res['slope'], intercept_resync=resync_res['intercept'],
            max_residual_resync_ms=resync_res['max_residual_ms'],
            median_residual_resync_ms=resync_res['median_residual_ms'],
        )
        results.append(diag)

    df = pd.DataFrame(results)
    # ssl vs resync verified consistent (no extraction mismatch, see DIAGNOSIS_MD below) -- only
    # report one set of slope/intercept/max_residual, not both
    cols = ['pid', 'eid', 'probe', 'version', 'diagnosis', 'slope_resync', 'intercept_resync',
            'max_residual_resync_ms', 'median_residual_resync_ms', 'error']
    print(df.reindex(columns=[c for c in cols if c in df]).to_string())

    # %%
    # Export the "Per-probe results" table for the interactive OJS widget in index.qmd, in table
    # order, rounded to the same precision as the equivalent static markdown table.
    table_cols = ['pid', 'eid', 'probe', 'version', 'diagnosis', 'slope_resync', 'intercept_resync',
                  'max_residual_resync_ms', 'median_residual_resync_ms']
    df_indexed = df.set_index('pid')
    table_df = (
        df_indexed.loc[[p for p in pids_table_order if p in df_indexed.index], table_cols[1:]]
        .reset_index()
        .rename(columns={'slope_resync': 'slope', 'intercept_resync': 'intercept',
                          'max_residual_resync_ms': 'max_residual_ms', 'median_residual_resync_ms': 'median_residual_ms'})
    )
    table_df['max_residual_ms'] = table_df['max_residual_ms'].round(1)
    table_df['median_residual_ms'] = table_df['median_residual_ms'].round(1)
    table_df['intercept'] = table_df['intercept'].round(4)
    table_df['slope'] = table_df['slope'].round(8)
    table_df.to_csv('per_probe_results.csv', index=False)

    # %%
    # sessions (not probes) sharing near-identical residuals implicate the shared nidq trace,
    # not per-probe noise -- this is the symmetry the issue's root-cause comment points to
    print(df.groupby('eid')[['linear_max_residual_ms', 'prod_qc']].agg(list))

    # %%
    # visualise the raw front-time drift for the worst offenders: a late discontinuity (stitched /
    # discontinuous recording) looks like a step in the residual; smooth nonlinear drift looks like
    # a slowly varying curve; sparse-pulse drift looks noisy/jumpy throughout
    worst = df.dropna(subset=['linear_max_residual_ms']).sort_values('linear_max_residual_ms', ascending=False)
    for _, row in worst.head(6).iterrows():
        pol = np.polyfit(row.t, row.tref, 1)
        residual_ms = (row.tref - np.polyval(pol, row.t)) * 1e3
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f"{row.pid} {row.probe}  prod_type={row.prod_type}  prod_qc={row.prod_qc}")
        axs[0].plot(row.t, row.tref, '.')
        axs[0].set(xlabel='probe front time (s)', ylabel='nidq front time (s)', title='raw fronts')
        axs[1].plot(row.tref, residual_ms, '.')
        axs[1].set(xlabel='nidq front time (s)', ylabel='residual (ms)', title='global-affine residual')
        fig.tight_layout()
        fig.savefig(FIG_PATH.joinpath(f'{TODAY}_lfpack_sync_{row.pid}.png'))

    # %%
    # Extraction-time residual plot for every one of the 20 probes, in the same order as the
    # "Per-probe results" table in diagnosis_source.md (worst dropped_edges first, clean last), with
    # the inter-pulse interval (diff of raw front times) as a second panel below it. The top panel is
    # sync_probe_front_times()'s own display=True plot -- raw + smoothed residual + the interpolation
    # knots baked into sync.timestamps -- not the simplified global-affine scatter used for the
    # "worst offenders" cell above. The bottom panel shows the pulse trains directly (no fit): probe
    # left/blue axis, nidq/ref right/orange axis, each centred on its own median interval -- this is
    # what actually exposes dropped edges (a step to a longer/shorter interval), duplicate bursts (a
    # single very-short interval) and isolated glitches (one interval off, then back to normal) as raw
    # pulse-train shape.
    # NOTE: plt.sca(ax1) before plot_extraction_residual -- sync_probe_front_times()'s 'exact'-type
    # branch (unlike 'smooth') calls bare plt.plot() and ignores the `display` axes it's given, so it
    # would otherwise land on whichever axes pyplot considers current (here, ax2 from the previous
    # iteration) instead of ax1.
    df_by_pid = df.set_index('pid')
    for pid in pids_table_order:
        if pid not in df_by_pid.index or ('error' in df_by_pid and pd.notna(df_by_pid.loc[pid, 'error'])):
            continue  # download/reproduce_sync failed for this pid, see `error` column
        diag = {**df_by_pid.loc[pid].to_dict(), 'pid': pid}
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.5), height_ratios=[2, 1], sharex=True)
        plt.sca(ax1)
        plot_extraction_residual(diag, ax=ax1)
        plot_front_diff(diag, ax=ax2)
        ax1.set_xlabel('')
        ax1.tick_params(labelbottom=False)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05)  # sharex above keeps the x-scale linked; this removes the gap
        fig.savefig(FIG_PATH.joinpath(f'residual_{pid[:8]}.png'))
        plt.close(fig)

    # %%
    # 4aab0f45 is flagged 'clean' but both its probes carry a 1.2-1.3 ms median residual, well above
    # every other clean session's 0.0-0.5 ms. Both probes show the *same* ~6.3ms deficit at the *same*
    # nidq time (~1021s): 3-4 consecutive pulses arriving slightly early on the probe's own imec_sync
    # channel. Identical timing across two independent probes rules out per-probe clock noise -- it
    # points to a brief signal disturbance on the sync line shared by both probes (nidq's own copy of
    # the pulse train stays clean). Each single interval deviates by <5ms, so it falls under
    # classify_defect's isolated_glitch threshold (tuned to the 12-139ms cases) and rate_change's
    # sustained-only criterion -- a real discrete event the existing heuristics don't catch, distinct
    # from ordinary crystal-oscillator ppm mismatch.
    eid_4aab0f45 = '4aab0f45-54eb-4ba0-9049-8ad1b7598fbe'
    for _, row in df[df.eid == eid_4aab0f45].sort_values('probe').iterrows():
        dt_probe, dt_nidq = np.diff(row.t), np.diff(row.tref)
        i = np.argmax(np.abs(dt_probe - dt_nidq))
        print(f"{row.pid} ({row.probe}): burst at nidq t={row.tref[i]:.2f}s, "
              f"deficit={(dt_probe - dt_nidq)[i] * 1e3:.3f} ms, "
              f"local dt(probe)={dt_probe[max(0, i - 1):i + 3] * 1e3} ms")

