"""
XGBoost NP1/NP2 cross-probe region classifier analysis.

Evaluates how well brain-region classifiers trained on NP1 probes (ea_active, 2026_W24)
transfer to NP2 probes (multi_area_comm, 2026_W24), and tests whether new features
(rms_lf_no_car, distance_to_tip_um, axial_um) improve transferability.

Experiments
-----------
1. Vanilla within-project 5-fold CV (NP1, NP2 separately).
2. Cross-project evaluation: NP1 model on NP2 data and vice versa.
3. Enhanced features: repeat experiments 1-2 adding rms_lf_no_car, distance_to_tip_um, axial_um.
4. Probe-aware combined model: train on both projects with neuropixel_version as a feature.
"""
# %% ---- imports ----------------------------------------------------------------
from pathlib import Path

import addcopyfighandler  # noqa: F401
import iblutil.numerical
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
from xgboost import XGBClassifier

import ephysatlas.anatomy
import ephysatlas.data
import ephysatlas.features
import ephysatlas.fixtures

sns.set_theme(style="whitegrid")

# %% ---- paths & constants ------------------------------------------------------
ROOT_FEATURES = Path.home().joinpath("data", "ephys-atlas", "features")
ROOT_PROJECTS = Path.home().joinpath("data", "ephys-atlas", "projects")
FIGURE_DIR = Path.home().joinpath("Documents", "figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

VINTAGE = "2026_W24"
PROJECT_NP1 = "ea_active"
PROJECT_NP2 = "multi_area_comm"
TRAIN_LABEL = "Cosmos_id"
RANDOM_SEED = 42
N_FOLDS = 5

# Features available in both vintages
FEATURES_BASE = ephysatlas.features.voltage_features_set(
    ["raw_lf", "raw_lf_csd", "raw_ap", "waveforms"]
) + ["outside"]

# New 2026_W24 features that could improve NP1/NP2 transferability
FEATURES_NEW = ["rms_lf_no_car", "distance_to_tip_um", "axial_um"]

FEATURES_ENHANCED = FEATURES_BASE + FEATURES_NEW


# %% ---- data download ----------------------------------------------------------
def ensure_features(project: str, vintage: str = VINTAGE) -> Path:
    """Download features if not present; return path to agg_full directory."""
    path = ROOT_FEATURES.joinpath(project, vintage, "agg_full")
    if not path.exists():
        from one.api import ONE
        one = ONE(base_url="https://alyx.internationalbrainlab.org", mode="remote")
        ephysatlas.data.download_tables(ROOT_FEATURES, label=vintage, project=project, one=one)
    return path


def ensure_probe_details(project: str) -> Path:
    """Download probe details if not present; return path."""
    path = ROOT_PROJECTS.joinpath(project, "df_probe_details.pqt")
    if not path.exists():
        ROOT_PROJECTS.joinpath(project).mkdir(parents=True, exist_ok=True)
        from one.api import ONE
        one = ONE(base_url="https://alyx.internationalbrainlab.org", mode="remote")
        ephysatlas.data.download_probe_details(ROOT_PROJECTS, project=project, one=one)
    return path


# %% ---- data loading -----------------------------------------------------------
print("Downloading / loading features ...")
brain_atlas = ephysatlas.anatomy.ClassifierAtlas()

path_np1 = ensure_features(PROJECT_NP1)
path_np2 = ensure_features(PROJECT_NP2)

df_np1 = ephysatlas.data.read_features_from_disk(path_np1, brain_atlas=brain_atlas, strict=False)
df_np2 = ephysatlas.data.read_features_from_disk(path_np2, brain_atlas=brain_atlas, strict=False)

# Remove known mis-aligned insertions
lowq = ephysatlas.fixtures.misaligned_pids
df_np1 = df_np1[~df_np1.index.get_level_values(0).isin(lowq)]
df_np2 = df_np2[~df_np2.index.get_level_values(0).isin(lowq)]

print(f"NP1 (ea_active)         : {df_np1.shape[0]:>8,} channels, "
      f"{df_np1.index.get_level_values(0).nunique():>5} probes")
print(f"NP2 (multi_area_comm)   : {df_np2.shape[0]:>8,} channels, "
      f"{df_np2.index.get_level_values(0).nunique():>5} probes")

# Check new features exist in both datasets
for feat in FEATURES_NEW:
    np1_ok = feat in df_np1.columns
    np2_ok = feat in df_np2.columns
    print(f"  {feat}: NP1={np1_ok}, NP2={np2_ok}")


# %% ---- helper: prune feature list to available columns -----------------------
def available_features(df: pd.DataFrame, feature_list: list[str]) -> list[str]:
    """Return only features present in df, with a warning for missing ones."""
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"  WARNING: missing features {missing}")
    return [f for f in feature_list if f in df.columns]


# %% ---- training helpers -------------------------------------------------------
def _detect_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    try:
        import xgboost as xgb
        # XGBoost >=2.0 exposes this helper; fall back to a fit probe otherwise
        if hasattr(xgb, "get_config"):
            test = XGBClassifier(device="cuda", verbosity=0)
            test.fit(np.zeros((4, 2)), np.arange(4))
            return "cuda"
    except Exception:
        pass
    return "cpu"


_DEVICE = _detect_device()


def xgb_classifier() -> XGBClassifier:
    """Return a default XGBClassifier using the best available device."""
    return XGBClassifier(device=_DEVICE, verbosity=0, n_jobs=-1, random_state=RANDOM_SEED)


def train_and_eval(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    x_cols: list[str],
    label_col: str = TRAIN_LABEL,
) -> dict:
    """Train an XGBClassifier on df_train and evaluate on df_test.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training feature dataframe.
    df_test : pd.DataFrame
        Test feature dataframe.
    x_cols : list of str
        Feature column names.
    label_col : str
        Target label column.

    Returns
    -------
    dict with keys: accuracy, probas, y_true, y_pred, classes, clf
    """
    classes = np.unique(df_train[label_col].values)

    x_train = df_train[x_cols].values.astype(float)
    x_test = df_test[x_cols].values.astype(float)
    y_train = df_train[label_col].values.astype(int)
    y_test = df_test[label_col].values.astype(int)

    _, iy_train = iblutil.numerical.ismember(y_train, classes)

    clf = xgb_classifier()
    clf.fit(x_train, iy_train)

    probas = clf.predict_proba(x_test)

    # Map predicted class indices back to region IDs
    # For cross-project: model classes may not cover all test labels → use NaN mask
    test_classes_idx = np.searchsorted(classes, y_test)
    in_training_set = np.isin(y_test, classes)

    # Accuracy computed only on regions the model has seen
    y_pred_idx = np.argmax(probas, axis=1)
    y_pred = classes[y_pred_idx]

    accuracy = sklearn.metrics.accuracy_score(
        y_test[in_training_set], y_pred[in_training_set]
    )
    coverage = in_training_set.mean()

    return dict(
        accuracy=accuracy,
        coverage=coverage,
        probas=probas,
        y_true=y_test,
        y_pred=y_pred,
        classes=classes,
        clf=clf,
    )


def kfold_cv(df: pd.DataFrame, x_cols: list[str], n_folds: int = N_FOLDS) -> dict:
    """Run k-fold cross-validation on a single project's data.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe indexed by (pid, channel).
    x_cols : list of str
        Feature column names.
    n_folds : int
        Number of folds.

    Returns
    -------
    dict with keys: accuracy_per_fold, accuracy_overall, df_predictions
    """
    all_pids = np.array(df.index.get_level_values(0).unique())
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(all_pids)
    fold_ids = np.floor(np.arange(len(all_pids)) / len(all_pids) * n_folds).astype(int)

    rids = np.unique(df[TRAIN_LABEL].values)
    df_pred = pd.DataFrame(index=df.index, columns=["prediction", "fold"], dtype=float)

    acc_per_fold = []
    for i in range(n_folds):
        test_pids = all_pids[fold_ids == i]
        test_mask = np.isin(df.index.get_level_values(0), test_pids)
        df_train = df.loc[~test_mask]
        df_test = df.loc[test_mask]

        res = train_and_eval(df_train, df_test, x_cols)
        acc_per_fold.append(res["accuracy"])

        df_pred.loc[test_mask, "prediction"] = res["y_pred"]
        df_pred.loc[test_mask, "fold"] = i
        print(f"  fold {i}: accuracy={res['accuracy']:.4f}  coverage={res['coverage']:.3f}")

    y_true = df[TRAIN_LABEL].values
    y_pred_all = df_pred["prediction"].values.astype(float)
    # drop channels with no prediction (shouldn't happen in CV)
    valid = ~np.isnan(y_pred_all)
    acc_overall = sklearn.metrics.accuracy_score(y_true[valid], y_pred_all[valid].astype(int))

    return dict(
        accuracy_per_fold=acc_per_fold,
        accuracy_overall=acc_overall,
        df_predictions=df_pred,
        rids=rids,
    )


# %% ---- Experiment 1 & 3: within-project CV -----------------------------------
results = {}

for feat_label, x_cols_fn in [
    ("vanilla", lambda df: available_features(df, FEATURES_BASE)),
    ("enhanced", lambda df: available_features(df, FEATURES_ENHANCED)),
]:
    for proj_label, df in [("NP1", df_np1), ("NP2", df_np2)]:
        key = f"{proj_label}_{feat_label}_cv"
        x_cols = x_cols_fn(df)
        print(f"\n=== {key} ({len(x_cols)} features) ===")
        res = kfold_cv(df, x_cols)
        print(f"  overall CV accuracy: {res['accuracy_overall']:.4f}")
        results[key] = res


# %% ---- Experiment 2 & 3: cross-project evaluation ----------------------------
# Train on all of one project, test on the other.
# Find common Cosmos regions to ensure a fair evaluation.
cosmos_np1 = set(df_np1[TRAIN_LABEL].unique())
cosmos_np2 = set(df_np2[TRAIN_LABEL].unique())
cosmos_common = cosmos_np1 & cosmos_np2
print(f"\nCosmos regions — NP1: {len(cosmos_np1)}, NP2: {len(cosmos_np2)}, common: {len(cosmos_common)}")

df_np1_common = df_np1[df_np1[TRAIN_LABEL].isin(cosmos_common)]
df_np2_common = df_np2[df_np2[TRAIN_LABEL].isin(cosmos_common)]

for feat_label, x_cols_fn in [
    ("vanilla", lambda df: available_features(df, FEATURES_BASE)),
    ("enhanced", lambda df: available_features(df, FEATURES_ENHANCED)),
]:
    for train_label, df_train, test_label, df_test in [
        ("NP1", df_np1_common, "NP2", df_np2_common),
        ("NP2", df_np2_common, "NP1", df_np1_common),
    ]:
        key = f"{train_label}_to_{test_label}_{feat_label}"
        x_train = x_cols_fn(df_train)
        x_test = [c for c in x_train if c in df_test.columns]
        if len(x_train) != len(x_test):
            print(f"  cross-project feature mismatch: {set(x_train)-set(x_test)}")
        x_cols = x_test  # use intersection
        print(f"\n=== {key} ({len(x_cols)} features) ===")
        res = train_and_eval(df_train, df_test, x_cols)
        print(f"  accuracy={res['accuracy']:.4f}  coverage={res['coverage']:.3f}")
        results[key] = res


# %% ---- Experiment 4: probe-aware combined model ------------------------------
# Train on combined NP1+NP2 with a binary probe-type feature.
print("\n=== Combined probe-aware model ===")
df_combined_list = []
for version_int, df in [(1, df_np1_common), (2, df_np2_common)]:
    df_v = df.copy()
    df_v["neuropixel_version"] = version_int
    df_combined_list.append(df_v)
df_combined = pd.concat(df_combined_list)

for feat_label, x_cols_fn in [
    ("vanilla", lambda df: available_features(df, FEATURES_BASE)),
    ("enhanced", lambda df: available_features(df, FEATURES_ENHANCED)),
]:
    x_cols_base = x_cols_fn(df_combined)
    for probe_feat in [False, True]:
        x_cols = x_cols_base + (["neuropixel_version"] if probe_feat else [])
        probe_str = "+probe" if probe_feat else ""
        key = f"combined_{feat_label}{probe_str}_cv"
        print(f"\n=== {key} ({len(x_cols)} features) ===")
        res = kfold_cv(df_combined, x_cols)
        print(f"  overall CV accuracy: {res['accuracy_overall']:.4f}")
        results[key] = res


# %% ---- Summary table ---------------------------------------------------------
rows = []
for key, res in results.items():
    acc = res.get("accuracy_overall", res.get("accuracy"))
    rows.append({"experiment": key, "accuracy": acc})

df_summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
print("\n=== Summary ===")
print(df_summary.to_string(index=False))
df_summary.to_csv(FIGURE_DIR.joinpath("2026-06-11_np1np2_summary.csv"), index=False)


# %% ---- Figure: accuracy bar chart --------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
palette = sns.color_palette("Set2", n_colors=2)

# Separate within-project CV from cross-project
mask_cv = df_summary["experiment"].str.contains("_cv$")
mask_cross = df_summary["experiment"].str.contains("_to_")

for mask, title in [(mask_cv, "Within-project CV"), (mask_cross, "Cross-project")]:
    sub = df_summary[mask]
    colors = [palette[0] if "vanilla" in e else palette[1] for e in sub["experiment"]]
    ax.bar(sub["experiment"], sub["accuracy"], color=colors)

ax.axhline(0.5, ls="--", c="grey", lw=0.8)
ax.set_ylabel("Accuracy (Cosmos regions)")
ax.set_title("NP1/NP2 cross-probe region classifier — 2026_W24")
ax.tick_params(axis="x", rotation=40)
plt.setp(ax.get_xticklabels(), ha="right")
fig.tight_layout()
fig.savefig(FIGURE_DIR.joinpath("2026-06-11_np1np2_accuracy_summary.png"), dpi=150)

# %% ---- Figure: confusion matrix for vanilla cross-project --------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, key in zip(axes, ["NP1_to_NP2_vanilla", "NP2_to_NP1_vanilla"]):
    if key not in results:
        continue
    res = results[key]
    valid = np.isin(res["y_true"], res["classes"])
    sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
        res["y_true"][valid],
        res["y_pred"][valid],
        normalize="true",
        cmap="Blues",
        ax=ax,
        im_kw=dict(vmax=0.75),
        colorbar=False,
    )
    ax.set_title(f"{key}\nacc={res['accuracy']:.3f}")
fig.suptitle("Vanilla cross-probe confusion matrices (Cosmos)")
fig.tight_layout()
fig.savefig(FIGURE_DIR.joinpath("2026-06-11_np1np2_confusion_vanilla.png"), dpi=150)

# %% ---- Feature importance: which new features help most? ---------------------
# Compare feature importances of the enhanced within-NP1 model
for proj_label, df in [("NP1", df_np1), ("NP2", df_np2)]:
    x_cols = available_features(df, FEATURES_ENHANCED)
    classes = np.unique(df[TRAIN_LABEL].values)
    _, iy = iblutil.numerical.ismember(df[TRAIN_LABEL].values.astype(int), classes)
    clf = xgb_classifier()
    clf.fit(df[x_cols].values.astype(float), iy)
    importances = pd.Series(clf.feature_importances_, index=x_cols).sort_values(ascending=False)
    print(f"\nTop 15 features ({proj_label}):")
    print(importances.head(15).to_string())

    fig, ax = plt.subplots(figsize=(10, 4))
    importances.head(20).plot.bar(ax=ax)
    ax.set_title(f"Feature importances — {proj_label} enhanced model")
    ax.set_ylabel("Importance (gain)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR.joinpath(f"2026-06-11_np1np2_feature_importance_{proj_label}.png"), dpi=150)

plt.close("all")
print("\nDone.")