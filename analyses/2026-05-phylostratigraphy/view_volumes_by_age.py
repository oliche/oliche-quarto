"""
Visualise the 12 summed expression volumes (one per phylostratum).
Loads outputs/summed_volumes_by_phylostratum.npy produced by sum_volumes_by_age.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from iblatlas.genomics import agea

PHYLOSTRATA = {
    1:  {'name': 'Cellular organisms',  'age_mya': 3500},
    2:  {'name': 'Eukaryota',           'age_mya': 1500},
    3:  {'name': 'Opisthokonta',        'age_mya': 1000},
    4:  {'name': 'Metazoa',             'age_mya':  650},
    5:  {'name': 'Bilateria',           'age_mya':  550},
    6:  {'name': 'Chordata',            'age_mya':  525},
    7:  {'name': 'Vertebrata',          'age_mya':  500},
    8:  {'name': 'Gnathostomata',       'age_mya':  450},
    9:  {'name': 'Tetrapoda',           'age_mya':  350},
    10: {'name': 'Mammalia',            'age_mya':  200},
    11: {'name': 'Euarchontoglires',    'age_mya':   90},
    12: {'name': 'Rodentia',            'age_mya':   25},
}

OUTPUT_DIR = Path(__file__).parent / 'outputs'
NPY_FILE = OUTPUT_DIR / 'summed_volumes_by_phylostratum.npy'


def view_phylostratum_volumes(islice=None, orientation='coronal', cmap='magma'):
    """
    Plot z-scored deviation from baseline for each phylostratum in a 3×4 grid.

    # AGEA atlas axis order: (ML, DV, AP) = (58, 41, 67)
    #   axis 0 ML  (58): slice here → sagittal  (DV × AP)
    #   axis 1 DV  (41): slice here → horizontal (ML × AP)
    #   axis 2 AP  (67): slice here → coronal    (ML × DV)

    Parameters
    ----------
    islice : int or None
        Slice index along the relevant axis.  Defaults to mid-volume.
    orientation : str
        'sagittal'  → slice axis 0 (ML, size 58), default islice 29
        'coronal'   → slice axis 2 (AP, size 67), default islice 33
        'horizontal'→ slice axis 1 (DV, size 41), default islice 20
    cmap : str
        Matplotlib colormap (used for baseline; deviations use RdBu_r).
    """
    # AGEA axis order: (ML, DV, AP) = (58, 41, 67)
    # transpose=True means the raw slice needs a .T to put DV on the vertical axis
    ORIENTATIONS = {
        'sagittal':   {'axis': 0, 'size': 58, 'default': 29, 'transpose': False},  # (DV,AP) ok
        'horizontal': {'axis': 1, 'size': 41, 'default': 20, 'transpose': False},  # (ML,AP) ok
        'coronal':    {'axis': 2, 'size': 67, 'default': 33, 'transpose': True},   # (ML,DV)→(DV,ML)
    }
    info_o = ORIENTATIONS[orientation]
    axis, size, default_islice = info_o['axis'], info_o['size'], info_o['default']
    do_transpose = info_o['transpose']
    if islice is None:
        islice = default_islice

    summed_volumes = np.load(NPY_FILE)            # (12, 58, 41, 67)
    _, _, atlas_agea = agea.load(label='processed')
    brain_mask = atlas_agea.label > 0             # (58, 41, 67)

    all_vols = summed_volumes.astype(float)
    all_vols[:, ~brain_mask] = np.nan

    # z-score each phylostratum over in-brain voxels, then average → baseline
    for i in range(all_vols.shape[0]):
        v = all_vols[i][brain_mask]
        all_vols[i][brain_mask] = (v - v.mean()) / v.std()
    baseline = np.nanmean(all_vols, axis=0)       # (58, 41, 67)

    # deviation of each z-scored volume from the baseline
    diff_all = all_vols - baseline[np.newaxis]    # (12, 58, 41, 67)

    # shared symmetric colour scale across all phylostrata
    absmax = np.nanpercentile(np.abs(diff_all[:, brain_mask]), 97.5)

    def _slice(vol3d):
        img = np.take(vol3d, islice, axis=axis)
        return img.T if do_transpose else img

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for ps in range(1, 13):
        ax = axes[ps - 1]
        img = _slice(diff_all[ps - 1])

        im = ax.imshow(img, cmap='RdBu_r', vmin=-absmax, vmax=absmax, aspect='auto', origin='upper')
        plt.colorbar(im, ax=ax, shrink=0.8)

        info = PHYLOSTRATA[ps]
        ax.set_title(f'PS{ps} – {info["name"]}\n{info["age_mya"]} Mya', fontsize=8)
        ax.axis('off')

    fig.suptitle(
        f'Z-scored deviation from baseline  ({orientation}, slice {islice}/{size})',
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phylostratum_volumes.png', dpi=150)
    plt.show()
    return fig


def view_phylostratum_grid(orientation='coronal', cmap='RdBu_r'):
    """
    Grid: rows = 12 evenly-spaced slices spanning the full brain,
    columns = Allen atlas annotation + one per selected phylostratum.
    PS1, PS2 and PS4 are discarded.
    Shows z-scored deviation from the cross-phylostratum baseline.

    # AGEA atlas axis order: (ML, DV, AP) = (58, 41, 67)
    # Standard AllenAtlas volume order: (AP, DV, ML) = (67, 41, 58)
    # bc convention: i2x → AP (axis 0), i2y → DV (axis 1), i2z → ML (axis 2)
    #   Coronal  → AGEA axis 2 (AP, n=67); ap_m = ba.bc.i2x(ap_idx); slice: vol[:,:,idx].T
    #   Sagittal → AGEA axis 0 (ML, n=58); ml_m = ba.bc.i2z(ml_idx); slice: vol[idx,:,:]

    Parameters
    ----------
    orientation : str
        'coronal' (default) or 'sagittal'
    cmap : str
        Colormap for deviation maps.
    """
    from iblatlas.atlas import AllenAtlas

    PHYLOSTRATA_SHOW = [3, 5, 6, 7, 8, 9, 10, 11, 12]
    N_SLICES = 12

    summed_volumes = np.load(NPY_FILE)
    _, _, atlas_agea = agea.load(label='processed')
    ba = AllenAtlas()                               # 25 µm, downloads if needed
    brain_mask = atlas_agea.label > 0               # (58, 41, 67)

    all_vols = summed_volumes.astype(float)
    all_vols[:, ~brain_mask] = np.nan
    for i in range(all_vols.shape[0]):
        v = all_vols[i][brain_mask]
        all_vols[i][brain_mask] = (v - v.mean()) / v.std()
    baseline = np.nanmean(all_vols, axis=0)
    diff_all = all_vols - baseline[np.newaxis]      # (12, 58, 41, 67)
    absmax = np.nanpercentile(np.abs(diff_all[:, brain_mask]), 97.5)

    if orientation == 'coronal':
        slice_indices = np.round(np.linspace(3, 63, N_SLICES)).astype(int)
        def get_img(vol3d, idx):
            return vol3d[:, :, idx].T               # (ML,DV) → (DV,ML)
        def plot_ref(idx, ax):
            # bc convention: x=ML, y=AP, z=DV  (plot_cslice uses bc.y2i internally)
            # AGEA axis 2 = AP → bc.i2y gives AP in metres
            ba.plot_cslice(atlas_agea.bc.i2y(idx), volume='annotation', mapping='Allen', ax=ax)
    else:  # sagittal
        slice_indices = np.round(np.linspace(3, 54, N_SLICES)).astype(int)
        def get_img(vol3d, idx):
            return vol3d[idx, :, :]                 # (DV, AP)
        def plot_ref(idx, ax):
            # AGEA axis 0 = ML → bc.i2x gives ML in metres
            ba.plot_sslice(atlas_agea.bc.i2x(idx), volume='annotation', mapping='Allen', ax=ax)

    n_cols = 1 + len(PHYLOSTRATA_SHOW)
    fig, axes = plt.subplots(N_SLICES, n_cols, figsize=(n_cols * 2.5, N_SLICES * 2))

    for row, islice in enumerate(slice_indices):
        ax_ref = axes[row, 0]
        plot_ref(islice, ax_ref)
        ax_ref.axis('off')
        if row == 0:
            ax_ref.set_title('Allen\nAtlas', fontsize=7)

        for col, ps in enumerate(PHYLOSTRATA_SHOW):
            ax = axes[row, col + 1]
            img = get_img(diff_all[ps - 1], islice)
            ax.imshow(img, cmap=cmap, vmin=-absmax, vmax=absmax, aspect='auto', origin='upper')
            ax.axis('off')
            if row == 0:
                info = PHYLOSTRATA[ps]
                ax.set_title(f'PS{ps}\n{info["name"][:12]}\n{info["age_mya"]}Mya', fontsize=6)

    fig.suptitle(
        f'Z-scored deviation from baseline — {orientation} slices',
        fontsize=11, y=1.002,
    )
    plt.tight_layout(pad=0.3)
    plt.savefig(OUTPUT_DIR / f'phylostratum_grid_{orientation}.png', dpi=150)
    plt.show()
    return fig


def view_phylostratum_heatmap(cmap='RdBu_r', normalize='column'):
    """
    Heatmap: Cosmos brain regions (rows) × phylostrata (columns).
    normalize='column': z-score each phylostratum column across regions (default)
    normalize='volume': z-score each 3D volume then subtract cross-PS baseline
    normalize=False   : raw summed expression, averaged per region
    Rows sorted by phylostratigraphic preference (oldest-biased regions at top).
    """
    PHYLOSTRATA_SHOW = [3, 5, 6, 7, 8, 9, 10, 11, 12]

    summed_volumes = np.load(NPY_FILE)
    _, _, atlas_agea = agea.load(label='processed')
    brain_mask = atlas_agea.label > 0

    all_vols = summed_volumes.astype(float)
    all_vols[:, ~brain_mask] = np.nan

    if normalize == 'volume':
        for i in range(all_vols.shape[0]):
            v = all_vols[i][brain_mask]
            all_vols[i][brain_mask] = (v - v.mean()) / v.std()
        baseline = np.nanmean(all_vols, axis=0)
        diff_all = all_vols - baseline[np.newaxis]   # (12, 58, 41, 67)
    else:
        diff_all = all_vols                          # raw; column z-score applied after aggregation

    # label values are region indices into br; mappings['Cosmos'] maps each index to its Cosmos index
    br = atlas_agea.regions
    cosmos_label = br.mappings['Cosmos'][atlas_agea.label]   # (58, 41, 67)

    cosmos_indices = np.unique(cosmos_label[brain_mask])
    cosmos_indices = cosmos_indices[cosmos_indices > 0]      # exclude void

    records = []
    for cidx in cosmos_indices:
        vox = cosmos_label == cidx
        acr = br.acronym[cidx]
        row = {'region': acr}
        for ps in PHYLOSTRATA_SHOW:
            vals = diff_all[ps - 1][vox]
            vals = vals[~np.isnan(vals)]
            row[f'PS{ps}'] = float(vals.mean()) if len(vals) else np.nan
        records.append(row)

    ps_cols = [f'PS{ps}' for ps in PHYLOSTRATA_SHOW]
    df = pd.DataFrame(records).set_index('region')[ps_cols]

    if normalize in ('column', 'both'):
        df = (df - df.mean()) / df.std()             # z-score each phylostratum across regions
    if normalize in ('row', 'both'):
        df = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)  # z-score each region across phylostrata

    # sort rows: regions that favour older PS rise to top
    ps_nums = np.array(PHYLOSTRATA_SHOW, dtype=float)
    df['_w'] = df[ps_cols].values @ ps_nums / ps_nums.sum()
    df = df.sort_values('_w').drop(columns='_w')

    col_labels = [f'PS{ps}  {PHYLOSTRATA[ps]["name"][:10]}\n{PHYLOSTRATA[ps]["age_mya"]} Mya'
                  for ps in PHYLOSTRATA_SHOW]

    fig, ax = plt.subplots(figsize=(13, max(5, len(df) * 0.45)))
    hmap_kw = dict(xticklabels=col_labels, yticklabels=True,
                   linewidths=0.5, linecolor='#cccccc')
    if normalize in ('column', 'row', 'both'):
        hmap_kw.update(center=0, vmin=-2, vmax=2,
                       cbar_kws={'label': 'z-score'})
    elif normalize == 'volume':
        hmap_kw.update(center=0, vmin=-0.75, vmax=0.75,
                       cbar_kws={'label': 'mean z-score (deviation from baseline)'})
    else:
        hmap_kw.update(cbar_kws={'label': 'mean summed expression'})
    sns.heatmap(df, cmap=cmap, ax=ax, **hmap_kw)
    ax.set_xlabel('')
    ax.set_ylabel('Cosmos region')
    titles = {'column': 'Per-phylostratum z-score', 'row': 'Per-region z-score',
              'both': 'Z-score by phylostratum then by region',
              'volume': 'Z-scored deviation from baseline', False: 'Raw summed expression'}
    ax.set_title(titles[normalize] + ' by Cosmos region and phylostratum')
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phylostratum_heatmap_cosmos.png', dpi=150)
    plt.show()
    return fig


if __name__ == '__main__':
    view_phylostratum_heatmap(normalize='volume')
