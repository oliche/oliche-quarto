"""
  from results_io import load_scores  # from the co-located module
  d = load_scores("~/ceph/lfp-encoders/results_bwm/default")
  a = load_scores("~/ceph/lfp-encoders/results_bwm/aggressive")
  u = load_scores("~/ceph/lfp-encoders/results_bwm/uncompressed")
  # per-PID/channel R² (and per-group drop-R²); uncompressed − compressed = the
  # behaviour signal lost to SVD+WP compression, split by band and by region.

  The headline comparison is u vs d vs a: how much recoverable behaviour R² the standard and aggressive tiers cost you relative to the pre-compression Cadzow reference — broken down by band (delta/theta/beta/gamma) and brain region via the channel metadata in the scores. Want me to sketch that
  comparison/plot on the laptop side (it imports these same modules via sys.path), or is your quarto analysis already wired for it?


rsync -av --progress -e ssh popeye:~/ceph/lfp-encoders/results_bwm_2026-07-05.tar.gz ./


"""

from pathlib import Path
Path('/Users/olivier/Document1s/datadisk/lfp-processing/lfp-encoders')