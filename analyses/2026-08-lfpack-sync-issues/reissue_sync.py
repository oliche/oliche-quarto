"""
Workstream C -- compute corrected sync knot arrays for the 11 defective probes diagnosed in
`index.qmd` (lfpack#8 sync audit: `dropped_edges`, `single_edge_glitch`, `duplicate_burst`,
`irregular_3A_reference`), as reviewable groundwork for re-issuing their registered
`sync.timestamps.npy` on Alyx.

This script does NOT touch Alyx. It only downloads the small raw sync-front datasets (reusing
`sync_investigation.download_sync_fronts`), computes a corrected 2-column `[sample, time]` array
per probe in exactly the format `ibllib.ephys.sync_probes._save_timestamps_npy` writes, saves it
locally, and prints a before/after residual comparison for human review.

Where the actual re-issue would plug in (NOT called here, requires explicit go-ahead + write
credentials)::

    from ibllib.oneibl.patcher import FTPPatcher  # or whichever Patcher is configured
    patcher = FTPPatcher(one=one)
    patcher.patch_dataset(new_timestamps_file, dset_id=..., ...)

Fix strategies (see index.qmd "Fixes" table for the full diagnosis/justification):

- `dropped_edges` / `single_edge_glitch` (probe-side onset, 7 probes): truncate the probe's own
  front times at the onset found by `classify_defect`'s rate_change/isolated_glitch tests, fit
  the trusted (pre-onset) region with the normal production `sync_probe_front_times`, then append
  one synthetic terminal anchor at (total_ap_samples, trusted_affine(total_ap_samples)) so
  `interp1d(..., fill_value='extrapolate')` continues the trusted trend past the last real knot.
- `duplicate_burst` (3 probes, nidq-side or probe-side): drop the anomalous burst of near-
  simultaneous fronts, insert one synthetic front at the midpoint of the two valid fronts
  bounding it, then re-run `sync_probe_front_times` on the corrected front-time array.
- `irregular_3A_reference` (1 probe): re-run the fit with `right_camera` instead of `frame2ttl`
  as the aux reference channel.
"""
from pathlib import Path

import numpy as np
import one.alf.io as alfio
import pandas as pd
import spikeglx
from ibllib.ephys import sync_probes
from ibllib.io.extractors.ephys_fpga import get_ibl_sync_map, get_sync_fronts
from iblutil.util import Bunch

# reuses download_sync_fronts, classify_defect, _affine_residual,
# compute_residuals_from_spike_sorting_loader and the shared `one` client -- module is import-safe
# (its 20-probe audit pipeline is guarded behind `if __name__ == '__main__':`)
import sync_investigation as si

OUT_DIR = Path(__file__).parent.joinpath('reissue_output')
OUT_DIR.mkdir(exist_ok=True)

TOL_3B = 2.5  # matches sync_probes.version3B's default tolerance (samples @ probe fs)
TOL_3A = 2.1  # matches sync_probes.version3A's default tolerance

# The 11 defective probes from index.qmd, grouped by fix strategy. `duplicate_burst` is split
# into `_nidq` (0b8ea3ec + a5f2ec22 share one bad nidq reference, fixed once) and `_probe`
# (f2ea7211, fixed directly on its own channel).
PROBE_CONFIG = [
    ('367e94f6-df51-4120-a297-77fa88dcec31', 'dropped_edges'),
    ('4836a465-c691-4852-a0b1-dcd2b1ce38a1', 'dropped_edges'),
    ('d0046384-16ea-4f69-bae9-165e8d0aeacf', 'dropped_edges'),
    ('7967a14e-1bf0-4666-acb4-9b08ba8f3385', 'dropped_edges'),
    ('ad597f5f-5201-4e02-9a28-1fb1a75746cc', 'dropped_edges'),
    ('b2746c16-7152-45a3-a7f0-477985638638', 'single_edge_glitch'),
    ('81f0087b-2bd1-4e48-8e86-e8206aee3d9d', 'single_edge_glitch'),
    ('0b8ea3ec-e75b-41a1-9442-64f5fbc11a5a', 'duplicate_burst_nidq'),
    ('a5f2ec22-0ff3-4249-bd2f-6247c3990e53', 'duplicate_burst_nidq'),
    ('f2ea7211-85f3-4394-b03e-1302a1dfe79c', 'duplicate_burst_probe'),
    ('0fed7207-f747-428b-b4c0-854cabb50d9e', 'irregular_3A_reference'),
]


def _total_ap_samples(ephys_file):
    """
    Total sample count of the AP recording, from its `.ap.meta` alone -- the same
    fileSizeBytes / nSavedChans / 2-bytes-per-int16-sample convention `spikeglx.Reader` uses to
    derive `ns` when constructed with `bin_exists=False`, so no `.cbin`/`.ch` download is needed.

    Parameters
    ----------
    ephys_file : iblutil.util.Bunch
        An entry of `spikeglx.glob_ephys_files`, as returned for the probe of interest.

    Returns
    -------
    int
        Total number of AP samples in the (un-downloaded) binary file.
    """
    meta = spikeglx.read_meta_data(ephys_file.ap.with_suffix('.meta'))
    nc = spikeglx._get_nchannels_from_meta(meta)
    return int(meta['fileSizeBytes']) // (2 * nc)


def _load_fronts_3b(ses_path, probe):
    """
    Load the nidq and probe `imec_sync` front times/polarities for a 3B session -- exactly the
    arrays `sync_probes.version3B()` feeds to `sync_probe_front_times()`.

    Parameters
    ----------
    ses_path : pathlib.Path
        Local ONE-cache session path, populated by `sync_investigation.download_sync_fronts`.
    probe : str
        Probe label, e.g. 'probe00'.

    Returns
    -------
    probe_file : iblutil.util.Bunch
        The probe's `glob_ephys_files` entry (used for `.ap.meta` and sampling rate).
    sync_nidq : iblutil.util.Bunch
        nidq `imec_sync` fronts (times, polarities), both polarities.
    sync_probe : iblutil.util.Bunch
        Probe `imec_sync` fronts (times, polarities), both polarities.
    sr : float
        Probe's own AP sampling rate.
    """
    ephys_files = spikeglx.glob_ephys_files(ses_path, ext='meta', bin_exists=False)
    nidq_file = next(ef for ef in ephys_files if ef.get('nidq'))
    probe_file = next(ef for ef in ephys_files if ef.path.parts[-1] == probe)
    for ef in (nidq_file, probe_file):
        ef['sync'] = alfio.load_object(ef.path, 'sync', namespace='spikeglx', short_keys=True)
        ef['sync_map'] = get_ibl_sync_map(ef, '3B')
    sync_nidq = get_sync_fronts(nidq_file.sync, nidq_file.sync_map['imec_sync'])
    sync_probe = get_sync_fronts(probe_file.sync, probe_file.sync_map['imec_sync'])
    sr = sync_probes._get_sr(probe_file)
    return probe_file, sync_nidq, sync_probe, sr


def find_truncation_onset(times, diagnosis, resume_tol=0.2, lookback=50):
    """
    Locate the last trustworthy front, using the same per-channel logic
    `sync_investigation.classify_defect` uses to detect (but not localise) the defect, then back
    the boundary off further to before the earliest individual-pulse timing anomaly found in the
    preceding `lookback` intervals -- a short transitional wobble (e.g. a power-down-style
    electrical transient) can precede the sustained/isolated defect by a few, possibly
    non-contiguous, pulses, which a block-median or single-argmax test alone can leave just
    inside the "trusted" region.

    Parameters
    ----------
    times : np.ndarray
        Front times (both polarities) of the defective channel.
    diagnosis : str
        'dropped_edges' (sustained rate change) or 'single_edge_glitch' (one mistimed edge).
    resume_tol : float
        Relative deviation from the trusted-region median inter-pulse interval, above which a
        preceding pulse is considered part of the same transitional anomaly and trimmed off.
    lookback : int
        Number of intervals before the initial onset to scan for such a preceding anomaly.

    Returns
    -------
    int
        Index into `times` of the last trusted front (inclusive) -- everything after this index
        is discarded.
    """
    dt = np.diff(times)
    med = np.median(dt)
    if diagnosis == 'dropped_edges':
        block = 20
        block_med = np.array([np.median(dt[i:i + block]) for i in range(0, dt.size - block, block)])
        bad_blocks = np.where(np.abs(block_med - med) / med > 0.3)[0]
        onset_dt_idx = int(bad_blocks[0] * block)
    elif diagnosis == 'single_edge_glitch':
        onset_dt_idx = int(np.argmax(np.abs(dt - med)))
    else:
        raise ValueError(diagnosis)
    trusted_med = np.median(dt[:onset_dt_idx]) if onset_dt_idx > 0 else med
    window_start = max(0, onset_dt_idx - lookback)
    local = dt[window_start:onset_dt_idx]
    if local.size:
        bad_local = np.where(np.abs(local - trusted_med) / trusted_med > resume_tol)[0]
        if bad_local.size:
            onset_dt_idx = window_start + int(bad_local[0])
    return onset_dt_idx  # dt[onset_dt_idx] = times[onset_dt_idx + 1] - times[onset_dt_idx]


def fix_truncate_and_extrapolate(sync_probe, tref, sr, onset_idx, total_samples, tol):
    """
    `dropped_edges` / `single_edge_glitch` fix: fit the pre-onset trusted fronts with the normal
    production knot generation, then append one synthetic terminal anchor from a local affine fit
    of the same trusted fronts, placed at the true end of the AP recording.

    Parameters
    ----------
    sync_probe : iblutil.util.Bunch
        Defective probe's fronts (times, polarities), both polarities, already truncated to
        `min(n_probe, n_ref)` so indices line up 1:1 with `tref`.
    tref : np.ndarray
        Reference (nidq) front times, same length/alignment as `sync_probe.times`.
    sr : float
        Probe's own AP sampling rate.
    onset_idx : int
        Index of the last trusted front, from `find_truncation_onset`.
    total_samples : int
        Total AP sample count for this probe, from `_total_ap_samples`.
    tol : float
        Tolerance (samples) passed through to `sync_probe_front_times`.

    Returns
    -------
    timestamps : np.ndarray
        Corrected `[sample, time]` 2-column array, same convention as `_save_timestamps_npy`.
    qc : bool
        Pass/fail of the trusted-region fit's own tolerance check.
    info : dict
        `fit_type`, `onset_time_s`, `n_trusted`, `n_dropped`, `t_end_s`, `tref_end_s`.
    t_trusted, tref_trusted : np.ndarray
        The genuine (non-synthetic) front times used to build the fit -- the only points with a
        real reference to check residuals against (the appended terminal anchor has none; a
        single global affine over knots that include it is dominated by its extrapolation
        leverage, not by fit quality -- see `main`'s "after" residual computation).
    """
    t_trusted = sync_probe.times[:onset_idx + 1]
    tref_trusted = tref[:onset_idx + 1]
    trusted_bunch = Bunch({'times': t_trusted, 'polarities': sync_probe.polarities[:onset_idx + 1]})

    fit_type = 'smooth' if sync_probes._check_diff_3b(trusted_bunch) else 'exact'
    sync_points, qc = sync_probes.sync_probe_front_times(
        t_trusted, tref_trusted, sr, display=False, type=fit_type, tol=tol
    )

    pol = np.polyfit(t_trusted, tref_trusted, 1)  # local affine on the trusted region only
    t_end = total_samples / sr
    tref_end = np.polyval(pol, t_end)
    sync_points = np.vstack([sync_points, [t_end, tref_end]])

    timestamps = sync_points.copy()
    timestamps[:, 0] *= np.float64(sr)
    info = {'fit_type': fit_type, 'onset_time_s': float(t_trusted[-1]), 'n_trusted': t_trusted.size,
                'n_dropped': int(sync_probe.times.size - t_trusted.size), 't_end_s': float(t_end),
                'tref_end_s': float(tref_end)}
    return timestamps, qc, info, t_trusted, tref_trusted


def fix_duplicate_burst_bunch(sync, factor=0.2, resume_tol=0.2):
    """
    Collapse a burst of near-simultaneous duplicate/bounced fronts -- `classify_defect`'s `burst`
    test (an inter-pulse interval far shorter than the nominal rate) -- down to however many
    genuine pulses the gap between the two bounding valid fronts actually implies.

    The tight `factor` threshold only reliably flags the ultra-short intervals *between* bounced
    fronts; the boundary interval connecting the last bounced front back to the next genuine one
    can land at a middling fraction of the nominal period (the bounces already ate into part of
    that true gap) without ever dropping under `factor * median`. Both boundaries are therefore
    extended one front at a time, using the looser `resume_tol`, until the gap on that side truly
    looks like nominal cadence again.

    Whether any synthetic front is needed at all depends on how many nominal periods actually
    separate the two (now correctly located) bounding fronts: if they already sit ~1 period apart,
    the burst was pure noise inside an otherwise-ordinary gap and every anomalous front in between
    is dropped outright; only a gap of ~2+ periods implies a genuine pulse was lost inside it, in
    which case the missing pulse(s) are reinstated at the expected even spacing (a single missing
    pulse lands at the midpoint, minimising squared deviation from the expected local period,
    matching index.qmd's fix description for that case).

    Parameters
    ----------
    sync : iblutil.util.Bunch
        Front times/polarities for one channel (nidq or probe), both polarities -- exactly what
        `version3B` feeds to `sync_probe_front_times`.
    factor : float
        Same 0.2 * median threshold `classify_defect`'s `burst()` test uses, to find the core of
        the anomalous run.
    resume_tol : float
        Relative deviation from nominal cadence, used to extend the run's boundaries until the
        gap on either side is no longer anomalous.

    Returns
    -------
    corrected : iblutil.util.Bunch
        `sync` with the anomalous run replaced by 0+ evenly-spaced synthetic fronts. Any synthetic
        front's polarity is set to 1 (rising) -- arbitrary but inconsequential among thousands of
        real fronts, and only used downstream to decide the production 'smooth' vs 'exact' type.
    n_dropped : int
        Number of real fronts removed (informational; excludes any synthetic fronts inserted).
    """
    dt = np.diff(sync.times)
    med = np.median(dt)
    bad = np.where(dt < factor * med)[0]
    if bad.size == 0:
        raise ValueError('no burst detected -- factor/threshold mismatch with classify_defect')
    i_before, i_after = int(bad.min()), int(bad.max()) + 1  # core anomalous run, may need widening
    while i_before > 0 and abs(dt[i_before - 1] - med) / med > resume_tol:
        i_before -= 1
    while i_after < dt.size and abs(dt[i_after - 1] - med) / med > resume_tol:
        i_after += 1

    n_periods = round((sync.times[i_after] - sync.times[i_before]) / med)
    n_missing = max(n_periods - 1, 0)
    inserted = sync.times[i_before] + med * np.arange(1, n_missing + 1)
    times = np.r_[sync.times[:i_before + 1], inserted, sync.times[i_after:]]
    polarities = np.r_[sync.polarities[:i_before + 1], np.ones(n_missing, dtype=sync.polarities.dtype),
                        sync.polarities[i_after:]]
    n_dropped = i_after - i_before - 1
    return Bunch({'times': times, 'polarities': polarities}), n_dropped


def fit_3b(probe_bunch, ref_times, sr, tol):
    """
    Re-run the production `sync_probe_front_times` on a (possibly corrected) probe/reference pair
    -- the same production function, just fed corrected input, per the `duplicate_burst` fix.

    Parameters
    ----------
    probe_bunch : iblutil.util.Bunch
        Probe fronts (times, polarities), corrected or original.
    ref_times : np.ndarray
        Reference (nidq) front times, corrected or original.
    sr : float
        Probe's own AP sampling rate.
    tol : float
        Tolerance (samples) passed through to `sync_probe_front_times`.

    Returns
    -------
    timestamps : np.ndarray
        Corrected `[sample, time]` 2-column array, same convention as `_save_timestamps_npy`.
    qc : bool
        Pass/fail of the fit's own tolerance check.
    info : dict
        `fit_type`, `n_used`.
    t, tref : np.ndarray
        The (corrected) front times the fit was built from -- for the "after" residual check in
        `main`.
    """
    n = min(probe_bunch.times.size, ref_times.size)
    t, tref = probe_bunch.times[:n], ref_times[:n]
    trimmed = Bunch({'times': t, 'polarities': probe_bunch.polarities[:n]})
    fit_type = 'smooth' if sync_probes._check_diff_3b(trimmed) else 'exact'
    sync_points, qc = sync_probes.sync_probe_front_times(t, tref, sr, display=False, type=fit_type, tol=tol)
    timestamps = sync_points.copy()
    timestamps[:, 0] *= np.float64(sr)
    return timestamps, qc, {'fit_type': fit_type, 'n_used': n}, t, tref


def fix_irregular_3a(ses_path, target_probe, tol=TOL_3A):
    """
    `irregular_3A_reference` fix: re-run the probe-vs-probe fit with `right_camera` in place of
    `frame2ttl` as the aux reference channel (see index.qmd Diagnosis for why `version3A`'s
    hardcoded frame2ttl-first precedence picked the wrong one here).

    Parameters
    ----------
    ses_path : pathlib.Path
        Local ONE-cache session path.
    target_probe : str
        Probe label to correct, e.g. 'probe01'.
    tol : float
        Tolerance (samples), matches `sync_probes.version3A`'s default.

    Returns
    -------
    timestamps : np.ndarray
        Corrected `[sample, time]` 2-column array, same convention as `_save_timestamps_npy`.
    qc : bool
        Pass/fail of the fit's own tolerance check.
    info : dict
        `fit_type` (always 'smooth', matching `version3A`), `aux`, `n_used`.
    t, tref : np.ndarray
        The `right_camera` front times the fit was built from -- for the "after" residual check
        in `main`.
    """
    ephys_files = spikeglx.glob_ephys_files(ses_path, ext='meta', bin_exists=False)
    for ef in ephys_files:
        ef['sync'] = alfio.load_object(ef.ap.parent, 'sync', namespace='spikeglx', short_keys=True)
        ef['sync_map'] = get_ibl_sync_map(ef, '3A')
    aux = 'right_camera'
    if not all(aux in ef.sync_map for ef in ephys_files):
        raise ValueError(f'{aux} not present on all probes for {ses_path}')

    fronts = [get_sync_fronts(ef.sync, ef.sync_map[aux]) for ef in ephys_files]
    n = min(f.times.size for f in fronts)
    iref = int(np.argmax([len(ef.sync.channels) for ef in ephys_files]))
    itarget = next(i for i, ef in enumerate(ephys_files) if ef.path.parts[-1] == target_probe)
    t, tref = fronts[itarget].times[:n], fronts[iref].times[:n]
    sr = sync_probes._get_sr(ephys_files[itarget])

    sync_points, qc = sync_probes.sync_probe_front_times(t, tref, sr, display=False, type='smooth', tol=tol)
    timestamps = sync_points.copy()
    timestamps[:, 0] *= np.float64(sr)
    return timestamps, qc, {'fit_type': 'smooth', 'aux': aux, 'n_used': n}, t, tref


def compute_corrected_timestamps(pid, diagnosis, nidq_fix_cache):
    """
    Dispatch to the right fix strategy for one probe and return its corrected timestamps array.

    Parameters
    ----------
    pid : str
        Probe insertion UUID.
    diagnosis : str
        One of the `PROBE_CONFIG` diagnosis labels.
    nidq_fix_cache : dict
        eid -> (corrected nidq Bunch, n_dropped), shared across probes of the same session for
        `duplicate_burst_nidq` (fixed once, reused for both `0b8ea3ec` and `a5f2ec22`).

    Returns
    -------
    timestamps : np.ndarray
    qc : bool
    info : dict
    probe : str
    t_check, tref_check : np.ndarray
        The genuine (non-synthetic) front times the fit was actually built from, in seconds --
        for the "after" residual check in `main` (see `fix_truncate_and_extrapolate`'s docstring
        for why this excludes the appended terminal anchor).
    sr : float
        Probe's own AP sampling rate, to convert `t_check` to the `timestamps` sample domain.
    """
    ses_path, probe, eid = si.download_sync_fronts(pid)
    version = spikeglx.get_neuropixel_version_from_folder(ses_path)

    if diagnosis == 'irregular_3A_reference':
        assert version == '3A', f'{pid}: expected 3A session for irregular_3A_reference'
        timestamps, qc, info, t_check, tref_check = fix_irregular_3a(ses_path, probe)
        sr = sync_probes._get_sr(next(
            ef for ef in spikeglx.glob_ephys_files(ses_path, ext='meta', bin_exists=False)
            if ef.path.parts[-1] == probe
        ))
        return timestamps, qc, info, probe, t_check, tref_check, sr

    assert version == '3B', f'{pid}: expected 3B session for {diagnosis}'
    probe_file, sync_nidq, sync_probe, sr = _load_fronts_3b(ses_path, probe)

    if diagnosis in ('dropped_edges', 'single_edge_glitch'):
        n = min(sync_probe.times.size, sync_nidq.times.size)
        trimmed_probe = Bunch({'times': sync_probe.times[:n], 'polarities': sync_probe.polarities[:n]})
        onset_idx = find_truncation_onset(trimmed_probe.times, diagnosis)
        total_samples = _total_ap_samples(probe_file)
        timestamps, qc, info, t_check, tref_check = fix_truncate_and_extrapolate(
            trimmed_probe, sync_nidq.times[:n], sr, onset_idx, total_samples, tol=TOL_3B
        )
        return timestamps, qc, info, probe, t_check, tref_check, sr

    if diagnosis == 'duplicate_burst_probe':
        probe_corrected, n_dropped = fix_duplicate_burst_bunch(sync_probe)
        timestamps, qc, info, t_check, tref_check = fit_3b(probe_corrected, sync_nidq.times, sr, tol=TOL_3B)
        info['n_dropped'] = n_dropped
        return timestamps, qc, info, probe, t_check, tref_check, sr

    if diagnosis == 'duplicate_burst_nidq':
        if eid not in nidq_fix_cache:
            nidq_fix_cache[eid] = fix_duplicate_burst_bunch(sync_nidq)
        nidq_corrected, n_dropped = nidq_fix_cache[eid]
        timestamps, qc, info, t_check, tref_check = fit_3b(sync_probe, nidq_corrected.times, sr, tol=TOL_3B)
        info['n_dropped'] = n_dropped
        return timestamps, qc, info, probe, t_check, tref_check, sr

    raise ValueError(diagnosis)


def main():
    """Compute, save and report corrected sync knot arrays for all 11 defective probes."""
    nidq_fix_cache = {}
    results = []
    for pid, diagnosis in PROBE_CONFIG:
        print(f'\n=== {pid} ({diagnosis}) ===')
        try:
            before = si.compute_residuals_from_spike_sorting_loader(pid)
        except Exception as e:  # noqa: BLE001 -- report and keep going, per-probe
            print(f'  FAILED to load currently-registered sync for {pid}: {type(e).__name__}: {e}')
            results.append({'pid': pid, 'diagnosis': diagnosis, 'error': f'before: {type(e).__name__}: {e}'})
            continue
        try:
            timestamps, qc_new, info, probe, t_check, tref_check, sr = compute_corrected_timestamps(
                pid, diagnosis, nidq_fix_cache
            )
        except Exception as e:  # noqa: BLE001 -- report and keep going, per-probe
            print(f'  FAILED to compute corrected timestamps for {pid}: {type(e).__name__}: {e}')
            results.append({'pid': pid, 'diagnosis': diagnosis, 'error': f'fix: {type(e).__name__}: {e}'})
            continue

        # "after" residual: how well the corrected knot array's own interpolant reproduces the
        # genuine (non-synthetic) front times it was built from -- the same style of check
        # `sync_probe_front_times`'s own `qc` flag makes internally. NOT a fresh independent
        # global-affine refit across all knots: that gets dominated by the leverage of the
        # appended terminal anchor (dropped_edges/single_edge_glitch), which sits far past the
        # bulk of the trusted data by construction -- see `fix_truncate_and_extrapolate`.
        fcn_vals = np.interp(t_check * sr, timestamps[:, 0], timestamps[:, 1])
        after_residual_ms = np.abs(tref_check - fcn_vals) * 1e3
        after = {'max_residual_ms': after_residual_ms.max(), 'median_residual_ms': np.median(after_residual_ms)}

        out_file = OUT_DIR.joinpath(f'{pid[:8]}_{probe}_sync.timestamps.npy')
        np.save(out_file, timestamps)

        print(f'  probe={probe}  info={info}')
        print(f'  BEFORE (registered): max={before["max_residual_ms"]:.2f} ms  median={before["median_residual_ms"]:.4f} ms')
        print(f'  AFTER  (corrected):  max={after["max_residual_ms"]:.4f} ms  median={after["median_residual_ms"]:.4f} ms  '
              f'qc={"PASS" if qc_new else "FAIL"}')
        print(f'  saved -> {out_file}')

        results.append({
            'pid': pid, 'probe': probe, 'diagnosis': diagnosis, 'qc_new': qc_new,
            'max_before_ms': before['max_residual_ms'], 'median_before_ms': before['median_residual_ms'],
            'max_after_ms': after['max_residual_ms'], 'median_after_ms': after['median_residual_ms'],
            'out_file': str(out_file), 'info': info,
        })

    df = pd.DataFrame(results)
    print('\n\n=== Summary: before vs after residuals, all 11 probes ===')
    cols = ['pid', 'probe', 'diagnosis', 'max_before_ms', 'median_before_ms',
            'max_after_ms', 'median_after_ms', 'qc_new', 'out_file', 'error']
    print(df.reindex(columns=[c for c in cols if c in df]).to_string())
    return df


if __name__ == '__main__':
    df_results = main()
