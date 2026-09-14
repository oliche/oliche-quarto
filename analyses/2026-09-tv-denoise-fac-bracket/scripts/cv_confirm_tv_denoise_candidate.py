"""5-fold CV confirmation: candidate per-group TV-denoise fac vs the fac=1 baseline.

The single-split sweeps in sweep_tv_denoise_fac.py / sweep_waveforms_fine.py
found a candidate config (raw_ap=raw_lf=raw_lf_csd=0.1, waveforms=5) that beat
the current fac=1-everywhere default by about +1pp Cosmos_id accuracy on one
train/test split. This script checks that gain survives 5-fold PID cross-
validation, using the exact same fold assignment for both configs.
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
CONFIGS = {
    "baseline_fac1": 1,
    "candidate": {"raw_ap": 0.1, "raw_lf": 0.1, "raw_lf_csd": 0.1, "waveforms": 5},
}
N_FOLDS = 5
RANDOM_SEED = 12345

root_path_features = Path.home().joinpath("data", "ephys-atlas", "features")
path_features = root_path_features.joinpath(PROJECT, VINTAGE, "agg_full")
path_results = Path.home().joinpath("data", "ephys-atlas", "tv_denoise_sweep")
path_results.mkdir(parents=True, exist_ok=True)
path_csv = path_results.joinpath(f"cv_confirm_{VINTAGE}.csv")

df_raw_ephys = pd.read_parquet(path_features.joinpath("raw_ephys_features.pqt"))
df_channels = pd.read_parquet(path_features.joinpath("channels.pqt"))

brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
aids = brain_atlas.get_labels(df_channels.loc[:, ["x", "y", "z"]].values, mode="clip")
df_channels = df_channels.copy()
df_channels[TRAIN_LABEL] = brain_atlas.regions.remap(aids, "Allen", "Cosmos")

valid_mask = ~df_raw_ephys.index.get_level_values(0).isin(
    ephysatlas.fixtures.misaligned_pids
)
df_raw_ephys = df_raw_ephys.loc[valid_mask]
df_channels = df_channels.loc[df_channels.index.isin(df_raw_ephys.index)]

x_list = sorted(
    set(ephysatlas.features.voltage_features_set(GROUPS)) & set(df_raw_ephys.columns)
)

# fold assignment computed once, shared identically across every config
all_pids = df_raw_ephys.index.get_level_values(0).unique().to_numpy()
rng = np.random.default_rng(RANDOM_SEED)
shuffled_pids = all_pids.copy()
rng.shuffle(shuffled_pids)
ifold = np.floor(np.arange(len(shuffled_pids)) / len(shuffled_pids) * N_FOLDS)
pid_to_fold = dict(zip(shuffled_pids, ifold))
fold_of_row = df_raw_ephys.index.get_level_values(0).map(pid_to_fold).to_numpy()
print(
    f"{len(all_pids)} pids, {N_FOLDS} folds, {len(x_list)} features, "
    f"{df_raw_ephys.shape[0]} channels"
)

results = []


def record(config_name, fold, accuracy, elapsed):
    results.append(
        dict(config=config_name, fold=fold, accuracy=accuracy, elapsed_s=elapsed)
    )
    pd.DataFrame(results).to_csv(path_csv, index=False)
    print(
        f"config={config_name:<14} fold={fold} accuracy={accuracy:.4f} ({elapsed:.0f}s)"
    )


for config_name, fac in CONFIGS.items():
    t0 = time.time()
    df_denoised = ephysatlas.aggregation.denoise_raw_features_data(
        df_raw_ephys, fac=fac, n_jobs=-1, verbose=0
    )
    df_denoised["outside"] = df_denoised["channel_labels"] == 3
    df_eval = df_denoised.merge(
        df_channels[[TRAIN_LABEL]], how="inner", left_index=True, right_index=True
    )
    cols = x_list + ["outside"]
    print(f"{config_name}: denoised in {time.time() - t0:.0f}s")

    for fold in range(N_FOLDS):
        t0 = time.time()
        test_idx = fold_of_row == fold
        train_idx = ~test_idx
        x_train = df_eval.loc[train_idx, cols].to_numpy(dtype=float)
        x_test = df_eval.loc[test_idx, cols].to_numpy(dtype=float)
        y_train = df_eval.loc[train_idx, TRAIN_LABEL].to_numpy(dtype=float)
        y_test = df_eval.loc[test_idx, TRAIN_LABEL].to_numpy(dtype=float)
        classes = np.unique(y_train)
        _, iy_train = iblutil.numerical.ismember(y_train, classes)
        classifier = XGBClassifier(n_jobs=-1)
        classifier.fit(x_train, iy_train)
        y_pred = classes[classifier.predict(x_test)]
        accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
        record(config_name, fold, accuracy, time.time() - t0)

df_results = pd.DataFrame(results)
summary = df_results.groupby("config")["accuracy"].agg(["mean", "std"])
print(summary)
summary.to_json(path_results.joinpath(f"cv_confirm_summary_{VINTAGE}.json"), indent=2)
print(f"Done. Results saved to {path_csv}")
