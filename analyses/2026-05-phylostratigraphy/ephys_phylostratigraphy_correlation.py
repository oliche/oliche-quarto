"""
Correlate ephys features (rms_ap, rms_lf) with phylostratum expression scores.

Coordinate conventions (shared by both atlases):
  x = ML, y = AP, z = DV  (meters, same in 25 µm AllenAtlas and AGEA)
  AGEA volume axes: (ML=58, DV=41, AP=67) = (0, 1, 2)

Steps:
  1 - Load ephys features
  2 - Load + z-score phylostratum volumes
  3 - Look up PS score for each channel at its (x, y, z)
  4 - Cross-plot rms_ap / rms_lf vs PS score for each phylostratum
  5 - Compute Pearson r + p-value; plot r vs phylostratum for both features
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from one.api import ONE

import ephysatlas.data
import ephysatlas.anatomy
from iblatlas.genomics import agea

# ---------------------------------------------------------------------------
FEATURES = ['rms_ap', 'rms_lf']
OUTPUT_DIR = Path(__file__).parent / 'outputs'
NPY_FILE = OUTPUT_DIR / 'summed_volumes_by_phylostratum.npy'
PROJECT = 'ea_active'

PHYLOSTRATA = {
    1:  ('Cellular org.',    3500),
    2:  ('Eukaryota',        1500),
    3:  ('Opisthokonta',     1000),
    4:  ('Metazoa',           650),
    5:  ('Bilateria',         550),
    6:  ('Chordata',          525),
    7:  ('Vertebrata',        500),
    8:  ('Gnathostomata',     450),
    9:  ('Tetrapoda',         350),
    10: ('Mammalia',          200),
    11: ('Euarchontoglires',   90),
    12: ('Rodentia',           25),
}
PHYLOSTRATA_SHOW = [3, 5, 6, 7, 8, 9, 10, 11, 12]


# ---------------------------------------------------------------------------
# 1. Load ephys features
# ---------------------------------------------------------------------------

def load_ephys_features():
    root_path = Path.home() / 'data' / 'ephys-atlas' / 'features'
    one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
    vintage = ephysatlas.data.get_latest_label(one=one, project=PROJECT)
    print(f'  vintage: {vintage}')
    path_features = root_path / PROJECT / vintage / 'agg_full'
    if not path_features.exists():
        ephysatlas.data.download_tables(root_path, label=vintage, one=one)
    brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
    df = ephysatlas.data.read_features_from_disk(path_features, brain_atlas=brain_atlas, strict=False)
    required = ['x', 'y', 'z'] + FEATURES
    mask = (df['outside'] == 0) & df[required].notna().all(axis=1)
    for feat in FEATURES:
        mask &= df[feat] > 0  # log-transform requires positive values
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Load and z-score phylostratum volumes
# ---------------------------------------------------------------------------

def load_ps_volumes():
    summed = np.load(NPY_FILE).astype(float)  # (12, 58, 41, 67)
    _, _, atlas_agea = agea.load(label='processed')
    brain_mask = atlas_agea.label > 0
    summed[:, ~brain_mask] = np.nan
    for i in range(12):
        v = summed[i][brain_mask]
        if v.std() > 0:
            summed[i][brain_mask] = (v - v.mean()) / v.std()
    return summed, atlas_agea


# ---------------------------------------------------------------------------
# 3. Look up PS score per channel
# ---------------------------------------------------------------------------

def channels_to_ps_scores(df, summed_volumes, atlas_agea):
    """Returns array (n_channels, 12) of z-scored PS expression at each channel."""
    bc = atlas_agea.bc
    n_ml, n_dv, n_ap = 58, 41, 67

    # Both atlases share x=ML, y=AP, z=DV
    i_ml = np.clip(np.round(bc.x2i(df['x'].values)).astype(int), 0, n_ml - 1)
    i_ap = np.clip(np.round(bc.y2i(df['y'].values)).astype(int), 0, n_ap - 1)
    i_dv = np.clip(np.round(bc.z2i(df['z'].values)).astype(int), 0, n_dv - 1)

    scores = np.stack([summed_volumes[ps][i_ml, i_dv, i_ap] for ps in range(12)], axis=1)
    return scores  # (n_channels, 12)


# ---------------------------------------------------------------------------
# 4. Cross-plots
# ---------------------------------------------------------------------------

def plot_cross_plots(df, ps_scores, log_features=True):
    n_ps = len(PHYLOSTRATA_SHOW)
    n_feat = len(FEATURES)
    fig, axes = plt.subplots(n_feat, n_ps, figsize=(n_ps * 2.4, n_feat * 2.6))

    for row, feat in enumerate(FEATURES):
        raw = df[feat].values
        feat_vals = np.log10(raw) if log_features else raw
        feat_label = f'log₁₀({feat})' if log_features else feat

        for col, ps in enumerate(PHYLOSTRATA_SHOW):
            ax = axes[row, col]
            scores = ps_scores[:, ps - 1]
            valid = np.isfinite(scores) & np.isfinite(feat_vals)

            hb = ax.hexbin(scores[valid], feat_vals[valid], gridsize=40, cmap='Blues', mincnt=1)

            m, b, r, p, _ = stats.linregress(scores[valid], feat_vals[valid])
            xr = np.percentile(scores[valid], [2, 98])
            ax.plot(xr, m * xr + b, 'r-', lw=1.2)
            ax.text(0.05, 0.92, f'r={r:.3f}', transform=ax.transAxes,
                    fontsize=6.5, color='red', va='top')

            ax.set_title(f'PS{ps}', fontsize=8)
            if col == 0:
                ax.set_ylabel(feat_label, fontsize=7)
            else:
                ax.set_yticklabels([])
            if row == n_feat - 1:
                name, age = PHYLOSTRATA[ps]
                ax.set_xlabel(f'{name[:10]}\n{age} Mya', fontsize=6)
            ax.tick_params(labelsize=6)

    fig.suptitle('Ephys features vs phylostratum expression score', fontsize=11)
    plt.tight_layout()
    out = OUTPUT_DIR / 'ephys_vs_phylostratum_crossplots.png'
    plt.savefig(out, dpi=150)
    print(f'  saved → {out}')
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 5. Correlation coefficient vs phylostratum
# ---------------------------------------------------------------------------

def plot_correlations(df, ps_scores, log_features=True):
    colors = {'rms_ap': 'tab:blue', 'rms_lf': 'tab:orange'}
    markers = {'rms_ap': 'o', 'rms_lf': 's'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # Left panel: r vs phylostratum
    ax_r = axes[0]
    # Right panel: -log10(p) vs phylostratum
    ax_p = axes[1]

    for feat in FEATURES:
        raw = df[feat].values
        feat_vals = np.log10(raw) if log_features else raw
        rs, pvals = [], []

        for ps in PHYLOSTRATA_SHOW:
            scores = ps_scores[:, ps - 1]
            valid = np.isfinite(scores) & np.isfinite(feat_vals)
            r, p = stats.pearsonr(scores[valid], feat_vals[valid])
            rs.append(r)
            pvals.append(p)

        rs = np.array(rs)
        pvals = np.array(pvals)
        neg_log_p = -np.log10(np.clip(pvals, 1e-300, 1))

        ax_r.plot(PHYLOSTRATA_SHOW, rs, f'{markers[feat]}-',
                  color=colors[feat], label=feat, lw=2, ms=6)
        ax_p.plot(PHYLOSTRATA_SHOW, neg_log_p, f'{markers[feat]}-',
                  color=colors[feat], label=feat, lw=2, ms=6)

    ax_r.axhline(0, color='k', lw=0.7, ls='--')
    ax_r.set_xticks(PHYLOSTRATA_SHOW)
    ax_r.set_xticklabels(
        [f'PS{ps}\n{PHYLOSTRATA[ps][0][:9]}\n{PHYLOSTRATA[ps][1]}Mya'
         for ps in PHYLOSTRATA_SHOW],
        fontsize=6.5,
    )
    ax_r.set_ylabel('Pearson r', fontsize=10)
    ax_r.set_title('Correlation coefficient vs phylostratum', fontsize=10)
    ax_r.legend(fontsize=9)
    ax_r.grid(True, alpha=0.3)

    ax_p.axhline(-np.log10(0.05), color='gray', lw=0.8, ls=':', label='p=0.05')
    ax_p.axhline(-np.log10(0.001), color='gray', lw=0.8, ls='--', label='p=0.001')
    ax_p.set_xticks(PHYLOSTRATA_SHOW)
    ax_p.set_xticklabels(
        [f'PS{ps}\n{PHYLOSTRATA[ps][0][:9]}\n{PHYLOSTRATA[ps][1]}Mya'
         for ps in PHYLOSTRATA_SHOW],
        fontsize=6.5,
    )
    ax_p.set_ylabel('−log₁₀(p)', fontsize=10)
    ax_p.set_title('Significance vs phylostratum', fontsize=10)
    ax_p.legend(fontsize=8)
    ax_p.grid(True, alpha=0.3)

    feat_label = 'log₁₀(feature)' if log_features else 'feature'
    fig.suptitle(f'Ephys features ({feat_label}) × phylostratum expression', fontsize=11)
    plt.tight_layout()
    out = OUTPUT_DIR / 'ephys_correlation_vs_phylostratum.png'
    plt.savefig(out, dpi=150)
    print(f'  saved → {out}')
    plt.show()
    return fig


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Loading ephys features...')
    df = load_ephys_features()
    print(f'  {len(df):,} channels with valid features')

    print('Loading phylostratum volumes...')
    summed_vols, atlas_agea = load_ps_volumes()

    print('Looking up PS scores per channel...')
    ps_scores = channels_to_ps_scores(df, summed_vols, atlas_agea)
    print(f'  PS scores shape: {ps_scores.shape}')
    nan_frac = np.isnan(ps_scores).mean()
    print(f'  NaN fraction: {nan_frac:.1%}')

    print('\nPlotting cross-plots...')
    plot_cross_plots(df, ps_scores)

    print('\nPlotting correlations...')
    plot_correlations(df, ps_scores)

    print('\nDone.')
