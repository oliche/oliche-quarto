"""Compare region-prediction accuracy across 9 parcellations: the two existing anatomical mappings
(Cosmos, Beryl), the raw atlas partition (Allen), and the 6 new ones from `parcellation.py`
(optimal/nested x cosmos/mid/beryl-scale) — same features, same model, same train/test split.

Mirrors `packages/ibleatools/examples/training_region_predictor_gradient_boosting.py` (feature set,
XGBClassifier), but trades that script's 5-fold CV for a single pid-level train/test split shared
across all 9 targets, since the point here is a fair head-to-head comparison, not a production
model — 45 (9 targets x 5 folds) training runs would multiply the runtime for no benefit to that
comparison.

Region-level (not voxel-level) remapping: the 6 new parcellations were built over the CCF ontology
tree in PCA-whitened *voxel* space (`parcellation.py`), so `id_to_label_map` re-derives, from each
cached frontier's `(canon_row, kind)` pairs, which raw Allen atlas *ids* (the same ids
`ephysatlas.data.read_features_from_disk`'s `Allen_id` column holds per channel) fall under each
group — reusing `parcellation.build_ontology_forest`'s tree (cheap; no need to refit the PCA basis
or DP to do this, since the ontology tree itself doesn't depend on the voxel grid resolution).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.preprocessing
from xgboost import XGBClassifier

import ephysatlas.anatomy
import ephysatlas.data
import ephysatlas.features
from parcellation import build_ontology_forest, CACHE_DIR

FEATURES_PATH = Path(
    "/Users/olivier/Documents/datadisk/ephys-atlas-decoding/features/ea_active/2026_W26/agg_full"
)
FEATURE_SET = ["raw_lf", "raw_lf_csd", "raw_ap", "localisation", "waveforms"]
NEW_PARCELLATIONS = ["optimal_cosmos", "optimal_mid", "optimal_beryl",
                     "nested_cosmos", "nested_mid", "nested_beryl"]
TARGETS = ["Allen", "Cosmos", "Beryl"] + NEW_PARCELLATIONS
TEST_FRACTION = 0.2
RANDOM_SEED = 12345


def collect_subtree_positions(children, own_positions, i):
    idx = list(own_positions[i])
    for c in children[i]:
        idx.extend(collect_subtree_positions(children, own_positions, c))
    return idx


def id_to_label_map(children, own_positions, canon, br, frontier_csv):
    """Rebuild {atlas_id: group_label} for one cached parcellation, from its `(canon_row, kind)`
    frontier (see `parcellation.frontier_to_label_map`, whose `own_positions`/subtree collection
    this reuses) — `atlas_id` here is the signed real CCF id (`br.id`), matching `Allen_id`.
    """
    frontier_df = pd.read_csv(frontier_csv)
    mapping = {}
    for row in frontier_df.itertuples():
        positions = (collect_subtree_positions(children, own_positions, row.canon_row)
                     if row.kind == "subtree" else list(own_positions[row.canon_row]))
        for pos in positions:
            mapping[int(br.id[pos])] = row.acronym
    return mapping


def load_labelled_features():
    ba = ephysatlas.anatomy.ClassifierAtlas()
    df = ephysatlas.data.read_features_from_disk(FEATURES_PATH, brain_atlas=ba,
                                                  mappings=["Cosmos", "Beryl"])
    df = df.rename(columns={"Allen_id": "Allen", "Cosmos_id": "Cosmos", "Beryl_id": "Beryl"})

    children, own_positions, roots, canon = build_ontology_forest(ba)
    for tag in NEW_PARCELLATIONS:
        kind = "optimal" if tag.startswith("optimal") else "nested"
        size = tag.split("_", 1)[1]
        mapping = id_to_label_map(children, own_positions, canon, ba.regions,
                                   CACHE_DIR / f"parcellation_{kind}_{size}.csv")
        df[tag] = df["Allen"].map(mapping)

    n_total = len(df)
    df = df.dropna(subset=TARGETS)
    print(f"{n_total} channels total, {len(df)} with a valid label in all {len(TARGETS)} "
          f"parcellations ({n_total - len(df)} dropped, mostly out-of-brain/void/root voxels "
          f"the new parcellations exclude by design)")
    return df


def pid_train_test_split(df, test_fraction=TEST_FRACTION, seed=RANDOM_SEED):
    pids = df.index.get_level_values(0).unique().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n_test = int(len(pids) * test_fraction)
    test_pids = set(pids[:n_test])
    is_test = df.index.get_level_values(0).isin(test_pids)
    print(f"split: {len(pids) - len(test_pids)} train pids / {len(test_pids)} test pids "
          f"({is_test.sum()} / {len(df)} channels in test)")
    return is_test


def train_and_evaluate(x_train, x_test, y_train, y_test, label):
    encoder = sklearn.preprocessing.LabelEncoder().fit(y_train)
    # a test-set class never seen in training can't be predicted correctly by construction;
    # keep such rows in the accuracy denominator (they just count as wrong) rather than silently
    # dropping them, but they'd crash the encoder, so map them to a sentinel that's never right.
    known = set(encoder.classes_)
    iy_train = encoder.transform(y_train)

    t0 = time.time()
    classifier = XGBClassifier(n_jobs=-1, verbosity=0)
    classifier.fit(x_train, iy_train)
    fit_time = time.time() - t0

    # sentinel for a test-set class never seen in training (can't be predicted correctly by
    # construction) - must be a real, sortable value of y_test's own dtype, since sklearn's
    # accuracy/balanced-accuracy call np.unique on the combined true/pred arrays internally (a
    # bare `None` crashes there with "'<' not supported between NoneType and int").
    is_numeric = np.issubdtype(np.asarray(y_test).dtype, np.number)
    sentinel = -(10**9) if is_numeric else "__unseen__"
    y_pred = np.full(len(y_test), sentinel, dtype=y_test.dtype if is_numeric else object)
    test_known = np.array([yy in known for yy in y_test])
    if test_known.any():
        y_pred[test_known] = encoder.inverse_transform(classifier.predict(x_test[test_known]))

    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    balanced = sklearn.metrics.balanced_accuracy_score(y_test, y_pred)
    print(f"[{label}] n_classes={len(encoder.classes_)}, accuracy={accuracy:.3f}, "
          f"balanced_accuracy={balanced:.3f}, fit_time={fit_time:.0f}s")
    return {
        "target": label, "n_classes": len(encoder.classes_), "n_train": len(y_train),
        "n_test": len(y_test), "accuracy": accuracy, "balanced_accuracy": balanced,
        "fit_time_s": fit_time,
    }


def main():
    df = load_labelled_features()
    x_list = ephysatlas.features.voltage_features_set(FEATURE_SET)
    x_list = [c for c in x_list if c in df.columns]

    is_test = pid_train_test_split(df)
    x_train = df.loc[~is_test, x_list].values.astype(float)
    x_test = df.loc[is_test, x_list].values.astype(float)

    results = []
    for target in TARGETS:
        y_train = df.loc[~is_test, target].values
        y_test = df.loc[is_test, target].values
        results.append(train_and_evaluate(x_train, x_test, y_train, y_test, target))

    summary = pd.DataFrame(results).sort_values("n_classes")
    out_path = CACHE_DIR / "classifier_accuracy_summary.csv"
    summary.to_csv(out_path, index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
