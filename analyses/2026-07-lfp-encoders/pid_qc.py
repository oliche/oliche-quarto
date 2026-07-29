"""PID-level data-quality classification for the 2026-07-07 resweep.

``check_resweep.py`` found that the 46 PIDs whose ``default``-tier fit collapses
(median held-out R² < -0.5 in >=1 band) are not a single population -- see
``DATA_ISSUES.md`` for the full derivation:

- **Group A** (``GROUP_A_BAD_RECORDING``): collapses even in the *uncompressed*
  (Cadzow) reference -- a pre-existing recording problem, unrelated to
  compression.
- **Group B** (``GROUP_B_NC24_CORRUPTION``): every PID with the archive's rare
  ``nc=24`` channel-binning count (4 of 699 total) reconstructs to near-zero
  across most/all CV folds in both compressed tiers -- a compression-pipeline
  data-corruption bug, not a quality tradeoff.
- The remaining 38 (``DEFAULT_COMPRESSION_ARTIFACT``) are healthy uncompressed
  and only collapse once compression is applied -- the genuine compression-
  quality population.
- ``AGGRESSIVE_ONLY_COLLAPSE``: PIDs that pass both uncompressed and default
  but collapse specifically under the aggressive tier.

``KEEP_PIDS`` (653) is every PID except Group A / B / the 38 -- i.e. PIDs that
pass both the uncompressed and default tiers. Science figures/analyses should
restrict to ``KEEP_PIDS`` (and, for anything specifically about the aggressive
tier, additionally consider excluding ``AGGRESSIVE_ONLY_COLLAPSE``). Diagnostic
functions that *characterise* the collapse itself (``region_aggregate
.collapse_rate``/``pid_band_r2``) should stay unfiltered.
"""

from __future__ import annotations

import json
from pathlib import Path

_DATA = json.loads(Path(__file__).resolve().parent.joinpath("pid_qc_groups.json").read_text())

GROUP_A_BAD_RECORDING: frozenset[str] = frozenset(_DATA["group_a_bad_recording"])
GROUP_B_NC24_CORRUPTION: frozenset[str] = frozenset(_DATA["group_b_nc24_corruption"])
DEFAULT_COMPRESSION_ARTIFACT: frozenset[str] = frozenset(_DATA["remaining_default_compression_artifact"])
AGGRESSIVE_ONLY_COLLAPSE: frozenset[str] = frozenset(_DATA["aggressive_only_collapse"])
KEEP_PIDS: frozenset[str] = frozenset(_DATA["keep_pids"])

EXCLUDED_PIDS: frozenset[str] = GROUP_A_BAD_RECORDING | GROUP_B_NC24_CORRUPTION | DEFAULT_COMPRESSION_ARTIFACT
