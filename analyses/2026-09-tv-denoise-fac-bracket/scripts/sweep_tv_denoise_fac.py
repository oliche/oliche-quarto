"""Bracket the TV-denoise `fac` per feature group and evaluate downstream Cosmos_id accuracy.

One-factor-at-a-time sweep: for each feature group in {raw_ap, raw_lf, raw_lf_csd,
waveforms}, vary its TV-denoise `fac` over FAC_GRID while holding the other three
groups at fac=1 (today's default), and measure XGBoost Cosmos_id accuracy on a
single, fixed PID-based train/test split shared across every configuration.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import sklearn.metrics
from xgboost import XGBClassifier

import iblutil.numerical
import ephysatlas.anatomy
import ephysatlas.aggregation
import ephysatlas.features
import ephysatlas.fixtures

PROJECT = "ea_active"
VINTAGE = "2026_W32"
TRAIN_LABEL = "Cosmos_id"
GROUPS = ["raw_ap", "raw_lf", "raw_lf_csd", "waveforms"]
FAC_GRID = [0, 0.1, 0.3, 1, 3, 10]
TEST_FRACTION = 0.2
RANDOM_SEED = 12345

root_path_features = Path.home().joinpath("data", "ephys-atlas", "features")
path_features = root_path_features.joinpath(PROJECT, VINTAGE, "agg_full")
path_results = Path.home().joinpath("data", "ephys-atlas", "tv_denoise_sweep")
path_results.mkdir(parents=True, exist_ok=True)
path_csv = path_results.joinpath(f"sweep_results_{VINTAGE}.csv")

# raw_ephys_features.pqt is the pre-merge, all-numeric aggregate: the correct
# input for denoise_raw_features_data (channels.pqt carries non-numeric columns
# that break its outlier/nan handling).
df_raw_ephys = pd.read_parquet(path_features.joinpath("raw_ephys_features.pqt"))
df_channels = pd.read_parquet(path_features.joinpath("channels.pqt"))

# region labels only depend on (x, y, z), which denoising never touches, so
# compute them once and reuse across every sweep point instead of remapping
# the atlas on every configuration.
brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
aids = brain_atlas.get_labels(df_channels.loc[:, ["x", "y", "z"]].values, mode="clip")
df_channels = df_channels.copy()
df_channels[TRAIN_LABEL] = brain_atlas.regions.remap(aids, "Allen", "Cosmos")

# exclude known misaligned pids, identically for every configuration
valid_mask = ~df_raw_ephys.index.get_level_values(0).isin(
    ephysatlas.fixtures.misaligned_pids
)
df_raw_ephys = df_raw_ephys.loc[valid_mask]
df_channels = df_channels.loc[df_channels.index.isin(df_raw_ephys.index)]

x_list = sorted(
    set(ephysatlas.features.voltage_features_set(GROUPS)) & set(df_raw_ephys.columns)
)

# single fixed train/test split, reused identically across every sweep point
all_pids = df_raw_ephys.index.get_level_values(0).unique().to_numpy()
rng = np.random.default_rng(RANDOM_SEED)
shuffled_pids = all_pids.copy()
rng.shuffle(shuffled_pids)
n_test = int(len(shuffled_pids) * TEST_FRACTION)
test_pids = set(shuffled_pids[:n_test])
test_idx = df_raw_ephys.index.get_level_values(0).isin(test_pids)
train_idx = ~test_idx
print(
    f"{len(all_pids)} pids ({n_test} held out for test), "
    f"{len(x_list)} features, {df_raw_ephys.shape[0]} channels"
)


def evaluate(fac):
    """Denoise df_raw_ephys with `fac`, merge region labels, train/test on the fixed split."""
    df_denoised = ephysatlas.aggregation.denoise_raw_features_data(
        df_raw_ephys, fac=fac, n_jobs=-1, verbose=0
    )
    df_denoised["outside"] = df_denoised["channel_labels"] == 3
    df_eval = df_denoised.merge(
        df_channels[[TRAIN_LABEL]], how="inner", left_index=True, right_index=True
    )
    cols = x_list + ["outside"]
    x_train = df_eval.loc[train_idx, cols].to_numpy(dtype=float)
    x_test = df_eval.loc[test_idx, cols].to_numpy(dtype=float)
    y_train = df_eval.loc[train_idx, TRAIN_LABEL].to_numpy(dtype=float)
    y_test = df_eval.loc[test_idx, TRAIN_LABEL].to_numpy(dtype=float)
    classes = np.unique(y_train)
    _, iy_train = iblutil.numerical.ismember(y_train, classes)
    classifier = XGBClassifier(n_jobs=-1)
    classifier.fit(x_train, iy_train)
    y_pred = classes[classifier.predict(x_test)]
    return sklearn.metrics.accuracy_score(y_test, y_pred)


results = []


def record(group, fac_value, accuracy, elapsed):
    results.append(
        dict(group=group, fac_value=fac_value, accuracy=accuracy, elapsed_s=elapsed)
    )
    pd.DataFrame(results).to_csv(path_csv, index=False)
    print(
        f"group={group:<12} fac={fac_value:<6} accuracy={accuracy:.4f} ({elapsed:.0f}s)"
    )


t0 = time.time()
baseline_accuracy = evaluate(fac=1)
record("baseline", 1, baseline_accuracy, time.time() - t0)

for group in GROUPS:
    for fac_value in FAC_GRID:
        if fac_value == 1:
            record(group, 1, baseline_accuracy, 0.0)
            continue
        t0 = time.time()
        accuracy = evaluate(fac={group: fac_value})
        record(group, fac_value, accuracy, time.time() - t0)

print(f"Done. Results saved to {path_csv}")
