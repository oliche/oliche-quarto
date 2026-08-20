"""Boundary MLP with Captum Integrated Gradients for per-class attribution.

Unlike ``boundary_classifier.py`` (which feeds a single mean-diff vector per
crossing), here the input is the **full depth profile** of each crossing:
50 evenly-spaced positions in ±500 µm around the transition × 26 features
→ 1 300-dimensional input.  A GPU-accelerated MLP is trained with 4-fold
stratified CV; Captum IntegratedGradients then assigns an attribution value to
every (depth-position, feature) pair for each class, producing spatial landmark
signatures.

Outputs
-------
figures/ig/
    ig_class_signatures.png   — per-class (depth × feature) IG heatmap
    ig_depth_profile.png      — per-class mean |IG| collapsed over features
    ig_feature_importance.png — per-class mean |IG| collapsed over depth
    ig_attributions.pkl       — raw attribution arrays for further analysis
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
import boundaries_utils as bu
from boundary_classifier import get_qualifying_transitions, load_data

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VINTAGE = '2026_W24'
FIGURES_DIR = Path(__file__).parent.joinpath('figures')
LOCAL_CACHE_DIR = Path(__file__).parent.joinpath('cache')
WINDOW_UM = 500.0
N_GRID = 25           # grid points per side → 50 total
MIN_INSERTIONS = 32
N_FOLDS = 4
RANDOM_STATE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GRID = np.linspace(-WINDOW_UM, WINDOW_UM, 2 * N_GRID)  # (50,)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_sequence_matrix(
    df: pd.DataFrame,
    brain_atlas,
    pairs: list[tuple[str, str]],
    window_um: float = WINDOW_UM,
    n_grid: int = N_GRID,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build (n_samples, 2*n_grid, n_features) input from per-channel profiles.

    For each (probe, boundary) crossing:
    1. Extract channels in ±window_um around the transition depth.
    2. Average duplicate channels at the same depth (2-shank NP1 sites).
    3. Linearly interpolate each feature at the 2*n_grid fixed grid positions.

    Parameters
    ----------
    df:
        Channel features dataframe after PCA (MultiIndex with ``pid`` level).
    brain_atlas:
        ClassifierAtlas instance.
    pairs:
        Qualifying (from_acr, to_acr) pairs.
    window_um:
        Half-window around each transition depth (µm).
    n_grid:
        Number of grid positions per side (total = 2 * n_grid).

    Returns
    -------
    X:
        Shape ``(n_samples, 2*n_grid*n_features)``.
    y:
        Integer class labels ``(n_samples,)``.
    feature_cols:
        Ordered feature column names (length n_features).
    class_names:
        ``class_names[i]`` is the string for ``y == i``.
    """
    grid = np.linspace(-window_um, window_um, 2 * n_grid)
    feature_cols = [c for c in df.columns if c not in bu.EXCLUDE_COLS]
    df_reset = df[['axial_um'] + feature_cols].reset_index(level='pid')

    # Pre-aggregate channels at the same depth within each probe.
    df_agg = (
        df_reset.groupby(['pid', 'axial_um'])[feature_cols]
        .mean()
        .reset_index()
        .sort_values(['pid', 'axial_um'])
    )

    rows: list[np.ndarray] = []
    labels: list[str] = []

    for from_acr, to_acr in pairs:
        label = f'{from_acr}_to_{to_acr}'
        crossings = bu.find_boundary_crossings(df, brain_atlas, from_acr, to_acr)
        print(f'  {label}: {len(crossings)} crossings')

        for pid, trans_depth in crossings:
            probe = df_agg[df_agg['pid'] == pid]
            rel = probe['axial_um'].values - trans_depth
            in_win = (rel >= -window_um) & (rel <= window_um)
            rel_win = rel[in_win]
            if len(rel_win) < 4:   # need enough points to interpolate
                continue

            feats_win = probe.loc[in_win, feature_cols].values  # (n_ch, n_feat)

            # Interpolate each feature column at the fixed grid.
            profile = np.column_stack([
                np.interp(grid, rel_win, feats_win[:, j])
                for j in range(len(feature_cols))
            ])  # (2*n_grid, n_features)

            rows.append(np.nan_to_num(profile.flatten()))
            labels.append(label)

    X = np.array(rows, dtype=np.float32)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return X, y, feature_cols, list(le.classes_)


def normalise(
    X: np.ndarray,
    n_features: int,
    n_positions: int,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise by per-feature mean and std pooled over all depth positions.

    Using global-per-feature statistics keeps the depth axis on the same scale
    at every position, so IG attributions at different depths are directly
    comparable.

    Parameters
    ----------
    X:
        Shape ``(n_samples, n_positions * n_features)``.
    n_features:
        Number of features.
    n_positions:
        Number of depth grid points.
    mean:
        Pre-computed means; computed from X when None.
    std:
        Pre-computed stds; computed from X when None.

    Returns
    -------
    X_norm, mean, std
    """
    X3 = X.reshape(-1, n_positions, n_features)  # (N, P, F)
    if mean is None:
        mean = X3.mean(axis=(0, 1))          # (F,)
        std = X3.std(axis=(0, 1)) + 1e-8     # (F,)
    X_norm = ((X3 - mean) / std).reshape(X.shape[0], -1)
    return X_norm.astype(np.float32), mean, std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BoundaryMLP(nn.Module):
    """Fully-connected classifier for boundary depth profiles.

    Parameters
    ----------
    n_input:
        Flattened input size (n_positions × n_features).
    n_classes:
        Number of transition classes.
    dropout:
        Dropout probability applied after each hidden layer.
    """

    def __init__(self, n_input: int, n_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_fold(
    model: BoundaryMLP,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int = 80,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> tuple[BoundaryMLP, np.ndarray]:
    """Train one CV fold and return predictions on the validation set.

    Parameters
    ----------
    model:
        Fresh BoundaryMLP instance.
    X_tr, y_tr:
        Training split (numpy).
    X_val, y_val:
        Validation split (numpy).
    n_epochs:
        Training epochs.
    batch_size:
        Mini-batch size.
    lr:
        Adam learning rate.

    Returns
    -------
    model:
        Trained model (eval mode, on DEVICE).
    val_preds:
        Integer predictions for the validation set.
    """
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    ds_tr = TensorDataset(
        torch.from_numpy(X_tr).to(DEVICE),
        torch.from_numpy(y_tr).long().to(DEVICE),
    )
    loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_val).to(DEVICE))
        val_preds = logits.argmax(dim=1).cpu().numpy()
    return model, val_preds


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_folds: int = N_FOLDS,
) -> tuple[np.ndarray, BoundaryMLP]:
    """Run stratified k-fold CV and refit on full data.

    Parameters
    ----------
    X:
        Normalised feature matrix ``(n_samples, n_input)``.
    y:
        Integer class labels.
    n_classes:
        Total number of classes.
    n_folds:
        Number of CV folds.

    Returns
    -------
    y_pred:
        CV predictions (same order as y).
    full_model:
        Model refit on the entire dataset (for IG).
    """
    n_input = X.shape[1]
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.empty_like(y)

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f'  fold {fold + 1}/{n_folds} …', flush=True)
        model = BoundaryMLP(n_input, n_classes)
        model, preds = train_fold(model, X[tr_idx], y[tr_idx], X[val_idx], y[val_idx])
        y_pred[val_idx] = preds

    # Refit on full data for IG attribution.
    print('  refitting on full data …', flush=True)
    full_model = BoundaryMLP(n_input, n_classes)
    full_model, _ = train_fold(full_model, X, y, X, y)
    return y_pred, full_model


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------

def compute_ig_attributions(
    model: BoundaryMLP,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_positions: int,
    n_features: int,
    n_steps: int = 50,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute per-class mean Integrated Gradients attribution.

    For each sample the IG is computed w.r.t. the true class label, using a
    zero baseline (= the per-feature mean, because data are normalised).
    Attributions are averaged over all samples belonging to each class.

    Parameters
    ----------
    model:
        Trained BoundaryMLP in eval mode.
    X:
        Normalised inputs ``(n_samples, n_positions * n_features)``.
    y:
        Integer class labels.
    n_classes:
        Number of transition classes.
    n_positions:
        Depth grid points (50).
    n_features:
        Feature count (26).
    n_steps:
        Riemann approximation steps for IG.
    batch_size:
        Samples processed per GPU batch (reduces VRAM usage).

    Returns
    -------
    class_attr:
        Shape ``(n_classes, n_positions, n_features)`` — mean attribution per
        class, reshaped so axes are interpretable.
    """
    from captum.attr import IntegratedGradients

    model.eval()
    ig = IntegratedGradients(model)
    baseline = torch.zeros(1, X.shape[1], device=DEVICE)

    # Accumulate sum and count per class.
    attr_sum = np.zeros((n_classes, n_positions * n_features), dtype=np.float64)
    attr_cnt = np.zeros(n_classes, dtype=np.int64)

    X_t = torch.from_numpy(X)  # keep on CPU, move batch by batch
    n = len(X)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = X_t[start:end].to(DEVICE)
        yb = y[start:end]

        for i, (xi, yi) in enumerate(zip(xb, yb)):
            inp = xi.unsqueeze(0).requires_grad_(True)
            attr = ig.attribute(inp, baseline, target=int(yi), n_steps=n_steps)
            attr_sum[int(yi)] += attr.detach().cpu().numpy().flatten()
            attr_cnt[int(yi)] += 1

        if (start // batch_size) % 5 == 0:
            print(f'  IG {end}/{n}', flush=True)

    # Avoid division by zero for empty classes.
    mask = attr_cnt > 0
    class_attr = np.zeros((n_classes, n_positions * n_features))
    class_attr[mask] = attr_sum[mask] / attr_cnt[mask, None]
    return class_attr.reshape(n_classes, n_positions, n_features)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_class_signatures(
    class_attr: np.ndarray,
    class_names: list[str],
    feature_cols: list[str],
    grid: np.ndarray,
    output_dir: Path,
) -> None:
    """Save per-class (depth × feature) IG attribution heatmap.

    Parameters
    ----------
    class_attr:
        ``(n_classes, n_positions, n_features)`` attribution array.
    class_names:
        Class label strings.
    feature_cols:
        Feature column names.
    grid:
        Depth grid (µm relative to transition).
    output_dir:
        Destination directory.
    """
    n = len(class_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 7, nrows * 5),
        squeeze=False,
    )

    vmax = np.percentile(np.abs(class_attr), 95)

    for idx, (ax, name) in enumerate(zip(axes.flat, class_names)):
        attr = class_attr[idx]  # (n_positions, n_features)
        im = ax.imshow(
            attr,
            aspect='auto',
            cmap='RdBu_r',
            vmin=-vmax,
            vmax=vmax,
            interpolation='nearest',
            origin='lower',
        )
        ax.set_title(name.replace('_to_', ' → '), fontsize=13)
        ax.set_xlabel('feature', fontsize=11)
        ax.set_ylabel('depth (µm)', fontsize=11)
        ax.set_xticks(range(len(feature_cols)))
        ax.set_xticklabels(feature_cols, rotation=90, fontsize=7)
        # Label a subset of y-ticks.
        tick_pos = np.linspace(0, len(grid) - 1, 7, dtype=int)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels([f'{grid[p]:.0f}' for p in tick_pos], fontsize=9)
        # Mark the transition.
        boundary_row = np.argmin(np.abs(grid))
        ax.axhline(boundary_row, color='k', lw=1.5, ls='--', alpha=0.6)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)

    fig.suptitle('Per-class Integrated Gradients (depth × feature)', fontsize=16)
    fig.tight_layout()
    out = output_dir.joinpath('ig_class_signatures.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'Saved {out.name}')


def plot_depth_profile(
    class_attr: np.ndarray,
    class_names: list[str],
    grid: np.ndarray,
    output_dir: Path,
) -> None:
    """Per-class mean |IG| collapsed over features, plotted vs depth.

    Parameters
    ----------
    class_attr:
        ``(n_classes, n_positions, n_features)``.
    class_names:
        Class label strings.
    grid:
        Depth grid (µm relative to transition).
    output_dir:
        Destination directory.
    """
    mean_abs = np.abs(class_attr).mean(axis=2)  # (n_classes, n_positions)

    n = len(class_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)

    for idx, (ax, name) in enumerate(zip(axes.flat, class_names)):
        ax.plot(grid, mean_abs[idx], lw=2)
        ax.axvline(0, color='k', ls='--', lw=1, alpha=0.6)
        ax.set_title(name.replace('_to_', ' → '), fontsize=11)
        ax.set_xlabel('depth (µm)', fontsize=9)
        ax.set_ylabel('mean |IG|', fontsize=9)
        ax.tick_params(labelsize=8)

    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)

    fig.suptitle('Mean |IG| vs depth (feature-averaged)', fontsize=14)
    fig.tight_layout()
    out = output_dir.joinpath('ig_depth_profile.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'Saved {out.name}')


def plot_feature_importance(
    class_attr: np.ndarray,
    class_names: list[str],
    feature_cols: list[str],
    output_dir: Path,
) -> None:
    """Per-class top features by mean |IG| collapsed over depth.

    Parameters
    ----------
    class_attr:
        ``(n_classes, n_positions, n_features)``.
    class_names:
        Class label strings.
    feature_cols:
        Feature column names.
    output_dir:
        Destination directory.
    """
    mean_abs = np.abs(class_attr).mean(axis=1)  # (n_classes, n_features)

    n = len(class_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)

    for idx, (ax, name) in enumerate(zip(axes.flat, class_names)):
        order = np.argsort(mean_abs[idx])[::-1][:10]
        ax.barh(
            [feature_cols[j] for j in reversed(order)],
            mean_abs[idx][list(reversed(order))],
            color='steelblue',
        )
        ax.set_title(name.replace('_to_', ' → '), fontsize=11)
        ax.set_xlabel('mean |IG|', fontsize=9)
        ax.tick_params(labelsize=8)

    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)

    fig.suptitle('Top-10 features by mean |IG| (depth-averaged)', fontsize=14)
    fig.tight_layout()
    out = output_dir.joinpath('ig_feature_importance.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'Saved {out.name}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full MLP + IG boundary attribution pipeline."""
    ig_dir = FIGURES_DIR.joinpath('ig')
    ig_dir.mkdir(parents=True, exist_ok=True)
    LOCAL_CACHE_DIR.mkdir(exist_ok=True)

    _seq_cache = LOCAL_CACHE_DIR.joinpath(
        f'seq_{VINTAGE}_{WINDOW_UM:.0f}um_{MIN_INSERTIONS}ins.pkl'
    )
    _ig_cache = LOCAL_CACHE_DIR.joinpath(
        f'ig_{VINTAGE}_{WINDOW_UM:.0f}um_{MIN_INSERTIONS}ins.pkl'
    )

    print(f'Device: {DEVICE}')

    # ------------------------------------------------------------------ data
    print('\n=== Loading data ===')
    df, brain_atlas = load_data()

    print('\n=== Qualifying transitions ===')
    pairs = get_qualifying_transitions(MIN_INSERTIONS)
    print(f'  {len(pairs)} transitions')

    print(f'\n=== Building sequence feature matrix (±{WINDOW_UM:.0f} µm, {2*N_GRID} grid pts) ===')
    if _seq_cache.exists():
        print(f'  Loading from cache: {_seq_cache.name}')
        with _seq_cache.open('rb') as fh:
            X_raw, y, feature_cols, class_names = pickle.load(fh)
    else:
        X_raw, y, feature_cols, class_names = build_sequence_matrix(
            df, brain_atlas, pairs, window_um=WINDOW_UM, n_grid=N_GRID
        )
        with _seq_cache.open('wb') as fh:
            pickle.dump((X_raw, y, feature_cols, class_names), fh)
        print(f'  Cached → {_seq_cache.name}')

    n_features = len(feature_cols)
    n_positions = 2 * N_GRID
    n_classes = len(class_names)
    print(f'X: {X_raw.shape},  classes: {n_classes},  features: {n_features}')

    # Per-feature normalisation (pooled over all depth positions).
    X, feat_mean, feat_std = normalise(X_raw, n_features, n_positions)
    print(f'Normalised X shape: {X.shape}')

    # ------------------------------------------------------------------ CV
    print(f'\n=== MLP {N_FOLDS}-fold stratified CV ===')
    y_pred, full_model = run_cv(X, y, n_classes)

    acc = (y_pred == y).mean()
    print(f'\nOverall CV accuracy: {acc:.3f}')
    per_class = {
        class_names[i]: float((y_pred[y == i] == i).mean())
        for i in range(n_classes)
        if (y == i).any()
    }
    for name, a in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f'  {name}: {a:.3f}')

    # ------------------------------------------------------------------ IG
    print('\n=== Integrated Gradients ===')
    if _ig_cache.exists():
        print(f'  Loading from cache: {_ig_cache.name}')
        with _ig_cache.open('rb') as fh:
            class_attr = pickle.load(fh)
    else:
        class_attr = compute_ig_attributions(
            full_model, X, y,
            n_classes=n_classes,
            n_positions=n_positions,
            n_features=n_features,
        )
        with _ig_cache.open('wb') as fh:
            pickle.dump(class_attr, fh)
        print(f'  Cached → {_ig_cache.name}')

    # ------------------------------------------------------------------ plots
    print('\n=== Saving figures ===')
    plot_class_signatures(class_attr, class_names, feature_cols, GRID, ig_dir)
    plot_depth_profile(class_attr, class_names, GRID, ig_dir)
    plot_feature_importance(class_attr, class_names, feature_cols, ig_dir)

    # Save attribution data alongside figures for downstream use.
    with ig_dir.joinpath('ig_attributions.pkl').open('wb') as fh:
        pickle.dump(
            {
                'class_attr': class_attr,
                'class_names': class_names,
                'feature_cols': feature_cols,
                'grid': GRID,
            },
            fh,
        )
    print('Saved ig_attributions.pkl')


if __name__ == '__main__':
    main()
