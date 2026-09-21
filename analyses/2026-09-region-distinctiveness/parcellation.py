"""Hierarchy-consistent brain-region parcellations that maximize between-group separation in
PCA-whitened ephys-atlas feature space, at three granularities (~Cosmos, ~100 regions, ~Beryl).

Framing (2026-09-10): in whitened (orthogonal PCA) space, the total sum-of-squares over all voxels
is fixed regardless of how they're grouped (ANOVA identity: total SS = within-group SS +
between-group SS). So "maximize between-group distance" is exactly equivalent to "minimize
within-group SS" — the same objective Ward's-linkage clustering optimizes — which makes a
hierarchy-constrained partition tractable as an exact tree DP instead of an intractable
combinatorial search over pairwise group distances.

A valid grouping is a *frontier* of the CCF ontology tree (canonical, unlateralized nodes): a set
of nodes such that every raw painted region falls under exactly one selected node, i.e. every group
is a union of an ontology subtree — this is what "consistent with the hierarchy" means. Two ways to
pick a frontier of a target size K:

- `optimal_cut` (exact DP): the true minimum-within-SS frontier of exactly K nodes. Independent per
  K, so the 13/100/250-group cuts are NOT guaranteed to nest inside each other.
- `greedy_nested_cuts`: repeatedly split whichever current group's split would reduce SS the most
  (regression-tree-style growth of the ontology tree, one split at a time, starting from the 3 real
  top-level branches `grey`/`fiber tracts`/`VS` — `grooves`/`retina` carry zero real voxels and
  drop out; `root`/`void`/`void_fluid` are excluded, same as throughout this project). Slightly
  suboptimal at any fixed K, but the splitting sequence is monotonic, so the 13/100/250-group cuts
  are guaranteed strict refinements of each other.

Both need only the per-node pooled (n, mean, var) subtree statistics (`subtree_stats`), built once
from the same PCA-projected leaf-level stats `pairwise_mahalanobis.py` already computes. Kept as a
separate module rather than folded into `region_stats_table.py` since the tree-DP machinery here is
a different kind of problem (choosing a hierarchy-respecting partition) from that file's pairwise
distance utilities, even though it consumes the same PCA fit.

**Size-balanced variant (2026-09-17).** `optimal_cut`'s `size_penalty` adds a quadratic penalty on
each group's deviation from an equal K-way voxel split (`group_cost`), trading pure separation for
more equiprobable-by-voxel-count groups — motivated by the classifier comparison in `index.qmd`
showing severe class imbalance (gini up to ~0.7) drives a lot of the accuracy/balanced-accuracy gap
there. `auto_scaled_size_penalty` picks the absolute penalty weight from the unregularized
solution's own cost scale (so one dimensionless `alpha` transfers across K without hand-tuning per
K); swept `alpha` on the grey-matter K=100 cut and found gini improves from 0.533 to a floor around
0.50 by `alpha≈2.5` (larger `alpha` doesn't help further) — some imbalance is structural, not fixable
by this penalty at all: the single biggest group at K=100 (`STRd`, dorsal striatum, ~208k voxels)
has exactly one ontology child, so there's nowhere for it to split to no matter how it's weighted.
At coarse K=13 the effect is much smaller (0.430 -> 0.426) since the standard Cosmos-like divisions
leave little room to redistribute without changing K. Only `optimal_cut` got this extension, not
`greedy_nested_cuts` — the nested search was already the less-controllable of the two.
"""

import heapq
from pathlib import Path

import numpy as np
import pandas as pd

import ephysatlas.anatomy as anat
from distinctiveness import EXCLUDE_ACRONYMS, load_encoding_volume, no_coverage_mask
from region_stats_table import direct_paint_stats, fit_and_project_pca, pool_stats

VOL_PATH = Path(
    "/Users/olivier/Documents/datadisk/ephys-atlas-decoding/encoding_volumes/"
    "ea_active/2026_W26/brainwide_ephys_atlas_50um.npz"
)
CACHE_DIR = Path(__file__).parent / "cache"
VARIANCE_THRESHOLD = 0.95
TARGET_SIZES = {"cosmos": 13, "mid": 100, "beryl": 250}
GREY_MATTER_ONLY = True
SIZE_PENALTY_ALPHA = 2.5  # see auto_scaled_size_penalty; swept on K=100, past the point of
                          # diminishing returns (gini plateaus well before this) but not extreme


GREY_MATTER_ONLY_ROOTS = {"grey"}


def build_ontology_forest(ba, grey_matter_only=False):
    """Direct parent -> children graph over the CCF ontology's canonical (unlateralized, id>=0)
    nodes, plus each node's own raw-position indices (self + its `-id` hemisphere twin, if
    painted — see `pairwise_mahalanobis.py`'s note on why both twins hold real, independent
    voxels rather than one being a near-empty duplicate).

    Returns `(children, own_positions, roots, canon)`:
      - `children[i]`: canonical-row indices that are direct ontology children of row i.
      - `own_positions[i]`: raw-position indices (0..len(br.id)-1, the same indexing
        `direct_paint_stats` uses) directly painted with this node's own id or its twin's.
      - `roots`: canonical-row indices of the real top-level branches (`grey`, `fiber tracts`,
        `VS`) — `root`/`void`/`void_fluid` are dropped (`EXCLUDE_ACRONYMS`; `root` itself is
        registration-edge noise, not a valid group, and sits at ontology level 0 with no parent,
        so it structurally isn't an ancestor of grey/fiber tracts/VS anyway); `grv`/`retina` are
        real level-1 branches but carry zero voxels in this mask and are left in as roots — they
        just end up with `leaves_under == 0` downstream and drop out on their own. If
        `grey_matter_only`, `fiber tracts`/`VS`/`grv`/`retina` are dropped from `roots` too (same
        motivation as `region_distance_by_level.py`'s `GREY_MATTER_ONLY`: white matter/ventricles
        are trivially separable from grey matter and otherwise dominate the group budget at low
        K without being the interesting comparison) — `children`/`own_positions` still cover their
        subtrees structurally, they're just never reached since nothing in `roots` leads there.
      - `canon`: the canonical-row dataframe (`acronym`/`hexcolor`/`order`/... ), row-aligned with
        `children`/`own_positions`, for labelling the final groups.
    """
    br = ba.regions
    df = br.to_df()
    df["raw_position"] = np.arange(len(df))
    canon = df[df["id"] >= 0].reset_index(drop=True)

    id_to_rawpos = {int(rid): pos for pos, rid in enumerate(br.id)}
    rawpos_to_canonrow = {raw: i for i, raw in enumerate(canon["raw_position"])}

    n = len(canon)
    children = [[] for _ in range(n)]
    own_positions = [None] * n
    roots = []
    for i, row in canon.iterrows():
        pos = int(row.raw_position)
        twin = id_to_rawpos.get(-int(row.id))
        own_positions[i] = [pos] if twin is None else [pos, twin]

        if row.acronym in EXCLUDE_ACRONYMS:
            continue  # root/void/void_fluid never become a node in the tree at all

        parent_pos = id_to_rawpos.get(int(row.parent)) if not pd.isna(row.parent) else None
        parent_row = rawpos_to_canonrow.get(parent_pos) if parent_pos is not None else None
        # a parent that's missing, or itself excluded (root, for grey/fiber tracts/VS/...),
        # means this node has no *valid* ancestor left in the tree - treat it as a root.
        if parent_row is None or canon["acronym"].iat[parent_row] in EXCLUDE_ACRONYMS:
            if not grey_matter_only or row.acronym in GREY_MATTER_ONLY_ROOTS:
                roots.append(i)
        else:
            children[parent_row].append(i)

    return children, own_positions, roots, canon


def combine(parts):
    """Pool a list of (n, mean, var) tuples via the parallel-axis identity (same maths as
    `region_stats_table.pool_stats`, operating on already-computed sub-statistics instead of raw
    position indices, since a subtree's stats are built recursively from its children's)."""
    parts = [p for p in parts if p[0] > 0]
    if not parts:
        return 0, None, None
    n_total = sum(p[0] for p in parts)
    mean_total = sum(p[0] * p[1] for p in parts) / n_total
    var_total = sum(p[0] * (p[2] + (p[1] - mean_total) ** 2) for p in parts) / n_total
    return int(n_total), mean_total, var_total


def own_stats(own_positions, count_pc, mean_pc, var_pc):
    """(n, mean, var) of each canonical node's own raw-painted voxels (self + hemisphere twin)."""
    return [pool_stats(count_pc, mean_pc, var_pc, np.array(pos)) for pos in own_positions]


def subtree_stats(children, own):
    """Bottom-up pooled (n, mean, var) per node's full subtree (own voxels + all descendants'),
    by recursive parallel-axis combination. Ontology depth is shallow (<15), so plain recursion
    (memoised, one pass) is simpler than an explicit post-order stack and plenty fast for ~1300
    nodes.
    """
    n_nodes = len(own)
    memo = [None] * n_nodes

    def rec(i):
        if memo[i] is None:
            memo[i] = combine([own[i]] + [rec(c) for c in children[i]])
        return memo[i]

    return [rec(i) for i in range(n_nodes)]


def ss(stat):
    """Within-group sum-of-squares of a pooled (n, mean, var) stat, summed over PCs — this is
    `DP[node][1]`: the cost of keeping this node's whole subtree as a single, unsplit group."""
    n, mean, var = stat
    return 0.0 if n == 0 else float(n * np.sum(var))


def group_cost(n, ss_value, target_n, size_penalty):
    """`ss_value` plus an optional size-balance penalty, quadratic in the group's relative
    deviation from `target_n` (= total voxels / K, the size a perfectly-equal K-way split would
    give every group) — `((n - target_n) / target_n) ** 2`, dimensionless so `size_penalty` (its
    weight, "lambda") is comparable in spirit across different K/target_n choices.

    Applied only where a node actually *becomes* an emitted group (`optimal_cut`'s `j=1`/"self"
    leaf costs) — internal split points never appear as a group themselves, so they don't need it,
    and a child's own emitted-group costs are already folded into its `dp` array by the time a
    parent's merge sees it, so nothing here risks double-counting.

    `size_penalty=0` (the default) reproduces plain `ss_value` exactly, i.e. the original
    pure-separation objective — this is an additive extension of it, not a different codepath.
    """
    if size_penalty == 0 or target_n == 0:
        return ss_value
    return ss_value + size_penalty * ((n - target_n) / target_n) ** 2


def _knapsack_merge(parts, k):
    """Minimum-cost way to allocate a total budget across `parts` (each `(leaves_under, dp_array,
    info)`), giving every part at least 1 of the budget. Returns `{total: (cost, alloc)}` where
    `alloc` is a list of `(info, jc)` pairs (one per part) summing to `total`. Shared by every
    node's own children-merge and by the top-level merge across the forest's root branches.
    """
    running = {0: (0.0, [])}
    for lu, dp_arr, info in parts:
        new_running = {}
        for used, (cost, alloc) in running.items():
            for jc in range(1, min(lu, k - used) + 1):
                if not np.isfinite(dp_arr[jc]):
                    continue
                total, new_cost = used + jc, cost + dp_arr[jc]
                if total not in new_running or new_cost < new_running[total][0]:
                    new_running[total] = (new_cost, alloc + [(info, jc)])
        running = new_running
    return running


def optimal_cut(children, own, sub, roots, k, size_penalty=0.0):
    """Exact tree DP: the minimum-cost hierarchy-respecting frontier of exactly `k` groups.

    `dp[i][j]` = minimum cost from partitioning node `i`'s subtree into exactly `j` groups. `j=1`
    means "keep this whole subtree as one group" (`group_cost(n, ss(sub[i]), ...)`). `j>1` means
    split it up, via a knapsack-style min-cost merge (`_knapsack_merge`) over `i`'s *parts*: its
    real children (each contributing their own `leaves_under[c]`/`dp[c]`), plus, if `i` itself has
    any directly-painted voxels not belonging to any child (common — 86 of 1329 ontology nodes
    here, 1.4M voxels total, e.g. a generic "layer unspecified" residual under a named area), a
    synthetic "self" leaf part representing exactly those voxels. Without that self-part, splitting
    a node with own voxels would silently drop them from every group entirely — caught by a
    coverage check (`frontier_to_label_map`'s total raw-position count dropping as `k` grew,
    instead of staying constant) during development.

    `size_penalty` (the DP's cost is `ss + size_penalty * relative deviation from an equal K-way
    voxel split, squared` — see `group_cost`) trades the pure between-group-separation objective
    for one that also discourages very unequal group sizes; `0` (default) is the original,
    unregularized objective exactly.

    Children/self-parts with zero real voxels anywhere (`leaves_under == 0`, e.g. `grv`/`retina`)
    are skipped rather than forced to consume a budget slot. Runs once up to `k`, so a smaller
    target doesn't need a separate pass — just call again with a smaller `k` (cheap: ~1300 nodes).

    Returns the frontier as a list of `(canonical_row, kind)` pairs, `kind` one of `"subtree"`
    (this node's entire subtree is the group) or `"self"` (only this node's own residual voxels,
    not its descendants', are the group) — see `frontier_to_label_map`.
    """
    n_nodes = len(children)
    dp = [None] * n_nodes
    choice = [None] * n_nodes
    leaves_under = [0] * n_nodes
    total_n = sum(sub[r][0] for r in roots if r is not None)
    target_n = total_n / k

    def rec(i):
        if dp[i] is not None:
            return
        for c in children[i]:
            rec(c)

        parts = []
        if own[i][0] > 0:
            self_dp = np.full(k + 1, np.inf)
            self_dp[1] = group_cost(own[i][0], ss(own[i]), target_n, size_penalty)
            parts.append((1, self_dp, ("self", i)))
        for c in children[i]:
            if leaves_under[c] > 0:
                parts.append((leaves_under[c], dp[c], ("child", c)))

        arr = np.full(k + 1, np.inf)
        ch = [None] * (k + 1)
        if sub[i][0] > 0:
            arr[1] = group_cost(sub[i][0], ss(sub[i]), target_n, size_penalty)
        if parts:
            for total, (cost, alloc) in _knapsack_merge(parts, k).items():
                if total >= 2 and cost < arr[total]:
                    arr[total], ch[total] = cost, alloc

        dp[i], choice[i] = arr, ch
        feasible = np.where(np.isfinite(arr))[0]
        leaves_under[i] = int(feasible.max()) if len(feasible) else 0

    real_roots = [r for r in roots if r is not None]
    for r in real_roots:
        rec(r)

    # forest-level merge across the top-level branches (grey / fiber tracts / VS); no "j=1"
    # option here since there's no single ontology node representing the whole forest, and no
    # "self" part either (the forest itself paints nothing directly).
    parts = [(leaves_under[r], dp[r], ("child", r)) for r in real_roots if leaves_under[r] > 0]
    running = _knapsack_merge(parts, k)
    if k not in running:
        raise ValueError(f"k={k} groups is infeasible (forest supports at most "
                          f"{max(running)} groups)")
    _, alloc = running[k]

    frontier = []

    def backtrack(info, jc):
        kind, ref = info
        if kind == "self":
            frontier.append((ref, "self"))
        elif jc == 1:
            frontier.append((ref, "subtree"))
        else:
            for sub_info, sub_jc in choice[ref][jc]:
                backtrack(sub_info, sub_jc)

    for info, jc in alloc:
        backtrack(info, jc)
    return frontier


def auto_scaled_size_penalty(children, own, sub, roots, k, alpha):
    """Pick an absolute `optimal_cut(..., size_penalty=...)` weight scaled to this tree/K's own
    cost magnitude, rather than a hand-tuned constant: run the unregularized (`size_penalty=0`)
    cut once, and use `alpha` times its mean per-group SS as the absolute penalty weight.

    Needed because `group_cost`'s penalty term is O(1) per group (a squared *relative* deviation)
    while `ss` is O(total_voxels) — passing a raw `size_penalty` of, say, `1.0` directly has no
    detectable effect at all (checked: identical output from `size_penalty=0` up to `~1e5` on this
    project's grey-matter tree at K=100, where per-group SS runs ~2-4e5). Scaling by the
    unregularized solution's own mean group cost keeps one dimensionless `alpha` comparably
    "strong" whether `k` is 13 or 250, instead of needing a separately-tuned absolute number for
    each — despite `target_n` and typical per-group SS both changing by orders of magnitude
    between those two.
    """
    frontier0 = optimal_cut(children, own, sub, roots, k, size_penalty=0.0)
    total_ss = sum(ss(sub[i] if kind == "subtree" else own[i]) for i, kind in frontier0)
    return alpha * (total_ss / k)


def greedy_nested_cuts(children, own, sub, roots, target_sizes):
    """Regression-tree-style growth of the ontology tree: start from the frontier of real
    top-level branches (each as a whole-subtree group), and repeatedly replace whichever current
    group has the largest SS-reduction-if-split with its direct children — plus, if the node being
    split has any directly-painted voxels of its own not belonging to any child, a `"self"` group
    for exactly those (same reasoning as `optimal_cut`: otherwise they'd silently vanish). Every
    step only ever subdivides an existing group, so a snapshot of the frontier at any later size is
    a strict refinement of an earlier snapshot — this is what guarantees the requested sizes nest.
    A `"self"` group, once created, is terminal (nothing to split further within it).

    `target_sizes`: iterable of frontier sizes to snapshot (ascending). A node with no splittable
    children (a raw leaf, or a subtree with zero real voxels in every child) is left in the
    frontier permanently once reached.

    Returns `{size: [(canonical_row, kind), ...]}` for each reached target size (a size larger than
    the total number of real leaves is simply never reached and omitted) — see `optimal_cut` for
    the `kind` ("subtree"/"self") convention.
    """
    def reduction(i):
        kids = [c for c in children[i] if sub[c][0] > 0]
        if not kids:
            return -np.inf
        new_ss = sum(ss(sub[c]) for c in kids) + (ss(own[i]) if own[i][0] > 0 else 0.0)
        return ss(sub[i]) - new_ss

    frontier = [(r, "subtree") for r in roots if sub[r][0] > 0]
    heap = [(-reduction(i), i) for i, _ in frontier if np.isfinite(reduction(i))]
    heapq.heapify(heap)

    results = {}
    for target in sorted(target_sizes):
        while len(frontier) < target and heap:
            _, i = heapq.heappop(heap)
            frontier.remove((i, "subtree"))
            kids = [c for c in children[i] if sub[c][0] > 0]
            frontier.extend((c, "subtree") for c in kids)
            if own[i][0] > 0:
                frontier.append((i, "self"))
            for c in kids:
                r = reduction(c)
                if np.isfinite(r):
                    heapq.heappush(heap, (-r, c))
        if len(frontier) >= target:
            results[target] = list(frontier)
    return results


def frontier_to_label_map(children, own_positions, canon, frontier):
    """Map every raw painted position (`own_positions`' indexing) to the label of whichever
    frontier entry it falls under: a `"subtree"` entry collects its node's full descendant
    raw-position set (self + all descendants, both hemisphere twins — the inverse of
    `subtree_stats`' bottom-up pooling); a `"self"` entry collects only that node's own positions.
    `"self"` groups get an `-own` suffix on their acronym so they're visually distinguishable from
    a genuine child region that happens to share a name.
    """
    def collect_subtree(i):
        idx = list(own_positions[i])
        for c in children[i]:
            idx.extend(collect_subtree(c))
        return idx

    raw_to_acronym = {}
    group_meta = []
    for i, kind in frontier:
        acr, hexcolor = canon["acronym"].iat[i], canon["hexcolor"].iat[i]
        label = acr if kind == "subtree" else f"{acr}-own"
        positions = collect_subtree(i) if kind == "subtree" else list(own_positions[i])
        for pos in positions:
            raw_to_acronym[pos] = label
        group_meta.append({"acronym": label, "name": canon["name"].iat[i],
                            "hexcolor": hexcolor, "canon_row": i, "kind": kind})
    return raw_to_acronym, pd.DataFrame(group_meta)


def main():
    CACHE_DIR.mkdir(exist_ok=True)
    vol, feature_names = load_encoding_volume(VOL_PATH)
    ba = anat.ClassifierAtlas(res_um=50)
    extra_exclude = no_coverage_mask(vol)

    npz_data = np.load(VOL_PATH, allow_pickle=True)
    mean_per_feature, std_per_feature = npz_data["mean_per_feature"], npz_data["std_per_feature"]
    pca, pc_vol, pc_names, count_pc, mean_pc, var_pc = fit_and_project_pca(
        vol, ba, mean_per_feature, std_per_feature, extra_exclude, VARIANCE_THRESHOLD
    )
    print(f"PCA: k={pca['k']} components, {pca['explained_ratio'].sum():.1%} variance")

    children, own_positions, roots, canon = build_ontology_forest(ba, grey_matter_only=GREY_MATTER_ONLY)
    own = own_stats(own_positions, count_pc, mean_pc, var_pc)
    sub = subtree_stats(children, own)
    total_n = sum(sub[r][0] for r in roots)
    print(f"ontology forest: {len(children)} canonical nodes, {len(roots)} top-level branches"
          f"{' (grey matter only)' if GREY_MATTER_ONLY else ''}, {total_n} total voxels pooled")

    # raw 41-feature voxel counts, for a voxel-exact coverage check (not just raw-*position*
    # count — many ontology positions are pure organisational placeholders with zero voxels of
    # their own, e.g. `grey`/`CTX`, so "positions covered" alone isn't a meaningful invariant).
    count, _, _ = direct_paint_stats(vol, feature_names, ba, extra_exclude)

    for tag, k in TARGET_SIZES.items():
        frontier = optimal_cut(children, own, sub, roots, k)
        assert len(frontier) == k
        raw_to_acr, meta = frontier_to_label_map(children, own_positions, canon, frontier)
        covered_voxels = int(sum(count[p] for p in raw_to_acr))
        assert covered_voxels == total_n, f"voxel coverage mismatch at k={k}: {covered_voxels}"
        meta.to_csv(CACHE_DIR / f"parcellation_optimal_{tag}.csv", index=False)
        print(f"[optimal/{tag}] k={k}: {covered_voxels} voxels covered (exact), "
              f"top groups by acronym: {meta['acronym'].tolist()[:5]}...")

    for tag, k in TARGET_SIZES.items():
        lam = auto_scaled_size_penalty(children, own, sub, roots, k, SIZE_PENALTY_ALPHA)
        frontier = optimal_cut(children, own, sub, roots, k, size_penalty=lam)
        assert len(frontier) == k
        raw_to_acr, meta = frontier_to_label_map(children, own_positions, canon, frontier)
        covered_voxels = int(sum(count[p] for p in raw_to_acr))
        assert covered_voxels == total_n, f"voxel coverage mismatch at k={k}: {covered_voxels}"
        meta.to_csv(CACHE_DIR / f"parcellation_balanced_{tag}.csv", index=False)
        voxels_per_group = meta["acronym"].map(
            lambda a, r2a=raw_to_acr: sum(count[p] for p, a2 in r2a.items() if a2 == a)
        )
        print(f"[balanced/{tag}] k={k}: size_penalty={lam:.2e}, "
              f"voxel/group min={voxels_per_group.min()} max={voxels_per_group.max()}")

    nested = greedy_nested_cuts(children, own, sub, roots, TARGET_SIZES.values())
    prev_raw_to_acr = None
    for k in sorted(TARGET_SIZES.values()):
        frontier = nested[k]
        raw_to_acr, meta = frontier_to_label_map(children, own_positions, canon, frontier)
        covered_voxels = int(sum(count[p] for p in raw_to_acr))
        assert covered_voxels == total_n, f"voxel coverage mismatch at k={k}: {covered_voxels}"
        tag = {v: k_ for k_, v in TARGET_SIZES.items()}[k]
        meta.to_csv(CACHE_DIR / f"parcellation_nested_{tag}.csv", index=False)
        if prev_raw_to_acr is not None:
            # refinement check: every raw position's *coarser* group must be a deterministic
            # function of its finer group (no finer group straddles two coarser groups).
            coarse_of_fine = {}
            ok = True
            for pos, coarse_acr in prev_raw_to_acr.items():
                fine_acr = raw_to_acr.get(pos)
                if fine_acr is None:
                    continue
                if coarse_of_fine.setdefault(fine_acr, coarse_acr) != coarse_acr:
                    ok = False
                    break
            print(f"[nested/{tag}] k={k}: {len(frontier)} groups, "
                  f"refines previous level: {ok}")
        else:
            print(f"[nested/{tag}] k={k}: {len(frontier)} groups")
        prev_raw_to_acr = raw_to_acr


if __name__ == "__main__":
    main()
