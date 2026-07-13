"""Evaluation metrics for design-space generalization and co-design fidelity.

Ranking metrics (Spearman, pairwise accuracy, top-k regret) quantify how well
world-model fitness predictions order held-out designs against true-simulation
ground truth. Front metrics (hypervolume ratio, IGD+) quantify how well a
surrogate-driven Pareto front matches the true-simulation reference front;
they use pymoo's indicator implementations (pymoo is already a dependency of
the co-design stack).
"""

from __future__ import annotations

import numpy as np
import torch


def spearman(pred: torch.Tensor, true: torch.Tensor) -> float:
  """Spearman rank correlation between two 1-D score vectors."""
  if pred.numel() != true.numel() or pred.numel() < 2:
    raise ValueError("Need two equal-length vectors with >= 2 entries.")
  ranks_p = _ranks(pred.flatten().float())
  ranks_t = _ranks(true.flatten().float())
  rp = ranks_p - ranks_p.mean()
  rt = ranks_t - ranks_t.mean()
  denom = rp.norm() * rt.norm()
  if float(denom) == 0.0:
    return 0.0
  return float((rp @ rt) / denom)


def pairwise_ranking_accuracy(pred: torch.Tensor, true: torch.Tensor) -> float:
  """Fraction of design pairs ordered consistently with ground truth."""
  p, t = pred.flatten().float(), true.flatten().float()
  dp = p.unsqueeze(0) - p.unsqueeze(1)
  dt = t.unsqueeze(0) - t.unsqueeze(1)
  mask = torch.triu(torch.ones_like(dp, dtype=torch.bool), diagonal=1)
  mask &= dt != 0
  if int(mask.sum()) == 0:
    return 1.0
  return float(((dp * dt > 0) & mask).sum() / mask.sum())


def topk_regret(pred: torch.Tensor, true: torch.Tensor, k: int = 10) -> float:
  """True-fitness gap between the overall best design and the best design
  among the top-k *predicted* designs (0 = the surrogate's shortlist contains
  the true optimum)."""
  p, t = pred.flatten().float(), true.flatten().float()
  shortlist = torch.topk(p, min(k, p.numel())).indices
  return float(t.max() - t[shortlist].max())


def hypervolume(front: np.ndarray, ref_point: np.ndarray) -> float:
  """Hypervolume of a minimization front w.r.t. ``ref_point`` (pymoo)."""
  from pymoo.indicators.hv import HV

  value = HV(ref_point=np.asarray(ref_point, dtype=float))(np.asarray(front))
  assert value is not None
  return float(value)


def igd_plus(front: np.ndarray, reference_front: np.ndarray) -> float:
  """IGD+ of a minimization front against a reference front (pymoo)."""
  from pymoo.indicators.igd_plus import IGDPlus

  value = IGDPlus(np.asarray(reference_front, dtype=float))(np.asarray(front))
  assert value is not None
  return float(value)


def hypervolume_ratio(
  front: np.ndarray, reference_front: np.ndarray, ref_point: np.ndarray | None = None
) -> float:
  """HV(front) / HV(reference_front), sharing one reference point.

  The default reference point is the joint nadir of both fronts plus a 5%
  margin, so the ratio is scale-free and comparable across runs.
  """
  front = np.asarray(front, dtype=float)
  reference_front = np.asarray(reference_front, dtype=float)
  if ref_point is None:
    both = np.concatenate([front, reference_front], axis=0)
    span = both.max(axis=0) - both.min(axis=0)
    ref_point = both.max(axis=0) + 0.05 * np.where(span > 0, span, 1.0)
  hv_ref = hypervolume(reference_front, ref_point)
  if hv_ref == 0.0:
    raise ValueError("Reference front has zero hypervolume at the ref point.")
  return hypervolume(front, ref_point) / hv_ref


def _ranks(x: torch.Tensor) -> torch.Tensor:
  """Average ranks (ties get the mean of their positions)."""
  order = torch.argsort(x)
  ranks = torch.empty_like(x)
  ranks[order] = torch.arange(x.numel(), dtype=x.dtype)
  # Average ties.
  unique, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True)
  if unique.numel() != x.numel():
    sums = torch.zeros_like(unique).scatter_add_(0, inverse, ranks)
    ranks = (sums / counts)[inverse]
  return ranks
