"""
lfpack shared spatial basis: projection variant and the file size vs behaviour decoding Pareto.

Codecs (see shared_basis_sweep.py for the codec re-implementation):
    'inf'  per-chunk U_scaled, float32 (current lfpack)
    '48'   re-decomposition inside the shared basis B (SVD of Bᵀ X_j), m = 48
    'p48'  projection: per-chunk codec unchanged (same SVD, rank, Vh and wavelet thresholding), only the spatial
           factor is replaced, U_scaled_j ≈ B · C_j with C_j = Bᵀ U_j float16 and a float32 per-column scale

Phases (`python pareto_sweep.py <phase>`):
    projection  p48 at ε = 100 on the size and snr tracks (α calibrated on snippets as in shared_basis_sweep.py),
                to compare against the existing m = 48 re-decomposition rows of 2026-09-25_shared-basis_arms.csv
    grid        m ∈ {32, 48} × ε ∈ {50, 70, 100} × α grid for the shared basis, plus the per-chunk-U codec at ε = 100
                on the same α grid; every point is decoded (wheel R², choice / feedback balanced accuracy)
    extra       p24, p16, p48q8, p24q8 (q8: C in int8) on the α grid at the ε of the grid with the best wheel R² at
                the mild file size

Each (probe, group of arms) runs in a fresh subprocess (`worker` phase) so the decoding memory is returned to the
OS; rows are checkpointed to CSV after each arm and finished arms are skipped on restart.  Pass-1 singular values and
U·s are cached per probe in SCRATCH/<pid>/pass1_sv_us.npz.
"""

# %%
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import shared_basis_sweep as sbs

TODAY = '2026-09-26'
OUT_DIR = Path(__file__).parent
CALIB_CSV = OUT_DIR.joinpath(f'{TODAY}_pareto_calibration.csv')
ARMS_CSV = OUT_DIR.joinpath(f'{TODAY}_pareto_arms.csv')
WHEEL_CSV = OUT_DIR.joinpath(f'{TODAY}_pareto_wheel.csv')
CATEG_CSV = OUT_DIR.joinpath(f'{TODAY}_pareto_categorical.csv')
PYTHON = '/Users/olivier/PycharmProjects/ephys-atlas/.venv/bin/python'

ALPHAS = [2.5, 3.5, 5.0, 7.0, 10.0, 14.0]
# (codec, ε) groups in priority order, so that a partial run already brackets the operating point
GRID_GROUPS = [('p48', 100.0), ('inf', 100.0), ('p48', 70.0), ('p32', 100.0), ('p48', 50.0), ('p32', 70.0),
               ('p32', 50.0)]


def pass1_cached(pid):
    """Pass-1 singular values and U·s of every chunk of one probe, from the npz cache when present."""
    f = sbs.SCRATCH.joinpath(pid, 'pass1_sv_us.npz')
    if not f.exists():
        sv, us = sbs.pass1(pid)
        np.savez(f, sv=sv, us=us)
    d = np.load(f)
    return d['sv'], d['us']


def done_keys():
    if not ARMS_CSV.exists():
        return set()
    d = pd.read_csv(ARMS_CSV, dtype={'m': str})
    return set(zip(d.pid, d.m, d.epsilon.astype(float), d.alpha.round(4), d.track))


# %% Worker: encode (and decode) a list of arms for one probe
def worker(pid, arms):
    """
    Parameters
    ----------
    pid : str
    arms : list of dict
        Keys m (str), epsilon, alpha, track, decode (bool).
    """
    from one.api import ONE

    sv, us = pass1_cached(pid)
    keys, one = done_keys(), None
    for arm in arms:
        m, eps, alpha, track = arm['m'], float(arm['epsilon']), float(arm['alpha']), arm['track']
        if (pid, m, eps, round(alpha, 4), track) in keys:
            continue
        B = sbs.arm_basis(sv, us, m, eps)
        n, e, recon, ranks = sbs.encode_probe(pid, m, eps, alpha, arm['decode'], sv, B)
        row = sbs.summarize(pid, m, eps, alpha, track, n, e, ranks)
        if arm['decode']:
            one = one or ONE()
            wheel, categ = sbs.decode_all(recon, sbs.behaviour(pid, one))
            del recon
            sbs._append(WHEEL_CSV, dict(pid=pid, m=m, epsilon=eps, alpha=alpha, track=track, **wheel))
            for target, res in categ.items():
                sbs._append(CATEG_CSV, dict(pid=pid, m=m, epsilon=eps, alpha=alpha, track=track, target=target,
                                            **res))
            row['r2'] = wheel['r2']
        sbs._append(ARMS_CSV, row)
        print(f"{track} m={m} ε={eps:.0f} α={alpha:.2f} {pid[:8]}: {row['bytes'] / 1e6:.2f} MB, "
              f"SNR {row['snr']:.2f} dB, p10 {row['snr_chunk_p10']:.2f}, R² {row.get('r2', np.nan):.4f}", flush=True)


def run_worker(pid, arms):
    """Run `worker` in a fresh interpreter; the subprocess exits and returns its memory after each group."""
    subprocess.run([PYTHON, __file__, 'worker', pid, json.dumps(arms)], check=True, cwd=OUT_DIR)


# %% Phase 1: projection variant on the size and snr tracks
def phase_projection():
    m, eps = 'p48', 100.0
    if not CALIB_CSV.exists():
        jobs = []
        for pid in sbs.PIDS:
            sv, us = pass1_cached(pid)
            jobs.append((pid, m, eps, sv, sbs.arm_basis(sv, us, m, eps)))
            print(f'pass 1 {pid[:8]}', flush=True)
        rows = Parallel(n_jobs=sbs.N_JOBS, verbose=5, max_nbytes=None)(delayed(sbs.calibrate_probe)(*j) for j in jobs)
        pd.DataFrame([r for rr in rows for r in rr]).to_csv(CALIB_CSV, index=False)
    calib = pd.concat([pd.read_csv(sbs.CALIB_CSV, dtype={'m': str}), pd.read_csv(CALIB_CSV, dtype={'m': str})])
    alphas, ref_bytes, ref_snr = sbs.solve_alphas(calib)
    alphas = alphas[alphas.m.isin(['48', m]) & (alphas.epsilon == eps)]
    print(f'reference snippets: {ref_bytes / 1e6:.2f} MB, SNR {ref_snr:.2f} dB')
    print(alphas.pivot_table(index='m', columns='track', values='alpha').round(3), flush=True)
    a = alphas.set_index(['m', 'track']).alpha
    arms = [dict(m=m, epsilon=eps, alpha=float(a[(m, 'size')]), track='size', decode=True),
            dict(m=m, epsilon=eps, alpha=float(a[(m, 'snr')]), track='snr', decode=False)]
    for pid in sbs.PIDS:
        run_worker(pid, arms)


# %% Phase 2: grid for the Pareto frontier
def phase_grid():
    for m, eps in GRID_GROUPS:
        arms = [dict(m=m, epsilon=eps, alpha=a, track='grid', decode=True) for a in ALPHAS]
        for pid in sbs.PIDS:
            run_worker(pid, arms)


# %% Phase 3: smallest spatial parts at the best ε of the grid
EXTRA_CODECS = ['p24', 'p16', 'p48q8', 'p24q8']


def best_epsilon(m='p48'):
    """ε of the grid whose mean Δ wheel R² vs mild, interpolated at the mild file size, is the largest."""
    arms = pd.read_csv(ARMS_CSV, dtype={'m': str})
    old = pd.read_csv(sbs.ARMS_CSV, dtype={'m': str})
    mild = old[(old.m == 'inf') & (old.epsilon == 100) & (old.track == 'size')].set_index('pid')
    g = arms[(arms.track == 'grid') & (arms.m == m)].assign(mild_bytes=lambda d: d.pid.map(mild.bytes),
                                                             d_r2=lambda d: d.r2 - d.pid.map(mild.r2))
    at_mild = {}
    for eps, d in g.groupby('epsilon'):
        s = d.groupby('alpha').agg(bytes=('bytes', 'sum'), mild_bytes=('mild_bytes', 'sum'), d_r2=('d_r2', 'mean'))
        s = s.assign(ratio=s.bytes / s.mild_bytes).sort_values('ratio')
        at_mild[eps] = np.interp(0.0, np.log(s.ratio.values), s.d_r2.values)
    print('Δ wheel R² at the mild file size per ε:', {k: round(v, 5) for k, v in at_mild.items()}, flush=True)
    return max(at_mild, key=at_mild.get)


def phase_extra():
    eps = best_epsilon()
    for m in EXTRA_CODECS:
        arms = [dict(m=m, epsilon=eps, alpha=a, track='grid', decode=True) for a in ALPHAS]
        for pid in sbs.PIDS:
            run_worker(pid, arms)


if __name__ == '__main__':
    phase = sys.argv[1]
    if phase == 'worker':
        worker(sys.argv[2], json.loads(sys.argv[3]))
    elif phase == 'projection':
        phase_projection()
    elif phase == 'grid':
        phase_grid()
    elif phase == 'extra':
        phase_extra()
