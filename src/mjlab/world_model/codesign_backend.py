"""World-model surrogate backend for the NSGA-II co-design loop.

Drop-in replacement for the true-simulation backends in
``scripts/codesign_ga.py`` (same duck-typed surface as ``CodesignBackend``:
``evaluate(tau_LJ, n_seeds) -> np.ndarray`` plus ``last_rms_LJ`` for the
thermal constraint), scoring whole populations in imagination.

Surrogate-assisted protocol: every generation the world model scores the full
population; the top fraction by predicted performance, plus any design whose
ensemble uncertainty exceeds a running percentile of previously seen
uncertainties, is re-scored in true simulation by an optional ``verifier``
backend. Verified scores overwrite predictions and feed an affine calibration
(pred -> true) that debiases subsequent predictions. The final reported front
must always be re-evaluated in true sim (honest-reporting rule) — the GA
driver handles that.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from mjlab.world_model.config import WorldModelBackendCfg
from mjlab.world_model.imagination import DesignEvaluator, DesignScores


class _Verifier(Protocol):
  """True-simulation backend surface used for verification."""

  last_rms_LJ: np.ndarray | None

  def evaluate(self, tau_LJ: torch.Tensor, n_seeds: int) -> np.ndarray: ...


class _Scorer(Protocol):
  """Imagination-evaluator surface (satisfied by DesignEvaluator; fakes in
  tests provide the same method)."""

  def evaluate_joint_tau(
    self,
    tau_LJ: torch.Tensor,
    joint_order: list[str] | tuple[str, ...],
    n_seeds: int,
  ) -> DesignScores: ...


class WorldModelBackend:
  """Imagination-based population scoring with optional true-sim verification."""

  def __init__(
    self,
    evaluator: _Scorer,
    joint_order: list[str],
    rollout_steps: int,
    metric: Any = None,
    device: str = "cpu",
    verifier: _Verifier | None = None,
    cfg: WorldModelBackendCfg | None = None,
    num_envs: int = 4096,
  ) -> None:
    self.evaluator = evaluator
    self.joint_order = list(joint_order)
    self.rollout_steps = rollout_steps
    self.metric = metric
    self.device = device
    self.num_envs = num_envs
    self.verifier = verifier
    self.cfg = cfg or WorldModelBackendCfg()
    self.last_rms_LJ: np.ndarray | None = None
    # Running state for calibration and uncertainty thresholding.
    self._calibration: list[tuple[float, float]] = []  # (pred_raw, true)
    self._uncertainty_history: list[float] = []
    self.total_verified = 0
    self.total_evaluated = 0

  @classmethod
  def from_checkpoint(
    cls,
    checkpoint: str | Path,
    task: str,
    policy_path: str | None,
    joint_order: list[str],
    rollout_steps: int,
    device: str,
    metric: Any = None,
    verifier: _Verifier | None = None,
    cfg: WorldModelBackendCfg | None = None,
    num_envs: int = 4096,
    seed: int = 0,
  ) -> "WorldModelBackend":
    from mjlab.world_model.collect import load_policy

    cfg = cfg or WorldModelBackendCfg()
    imag_cfg = cfg.imagination
    imag_cfg.rollout_steps = rollout_steps
    evaluator = DesignEvaluator(
      checkpoint, torch.nn.Identity(), device, imag_cfg, seed=seed
    )
    action_dim = evaluator.model.dims.action_dim
    evaluator.policy = load_policy(task, policy_path, device, action_dim).to(device)
    return cls(
      evaluator,
      joint_order,
      rollout_steps,
      metric=metric,
      device=device,
      verifier=verifier,
      cfg=cfg,
      num_envs=num_envs,
    )

  # -- CodesignBackend surface -----------------------------------------------

  def evaluate(self, tau_LJ: torch.Tensor, n_seeds: int) -> np.ndarray:
    scores: DesignScores = self.evaluator.evaluate_joint_tau(
      tau_LJ, self.joint_order, n_seeds
    )
    pred_raw = scores.performance.numpy().astype(np.float64)
    pred = self._calibrate(pred_raw)
    rms = scores.rms_torque.numpy().astype(np.float64)
    unc = scores.uncertainty.numpy()
    self._uncertainty_history.extend(float(u) for u in unc)
    self.total_evaluated += len(pred)

    if self.verifier is not None and self.cfg.verify_topk > 0.0:
      verify_idx = self._select_for_verification(pred, unc)
      if verify_idx.size > 0:
        true_scores = self.verifier.evaluate(tau_LJ[verify_idx], n_seeds)
        pred[verify_idx] = true_scores
        true_rms = self.verifier.last_rms_LJ
        if true_rms is not None:
          rms[verify_idx] = true_rms
        for i, true in zip(verify_idx, true_scores, strict=True):
          self._calibration.append((float(pred_raw[i]), float(true)))
        self.total_verified += verify_idx.size

    self.last_rms_LJ = rms
    return pred

  # -- Internals ----------------------------------------------------------------

  def _select_for_verification(
    self, pred: np.ndarray, uncertainty: np.ndarray
  ) -> np.ndarray:
    n = len(pred)
    k = max(1, math.ceil(self.cfg.verify_topk * n))
    top = set(np.argsort(pred)[::-1][:k].tolist())
    if len(self._uncertainty_history) >= 20:
      threshold = float(
        np.percentile(self._uncertainty_history, self.cfg.verify_uncertainty_pct)
      )
      top |= set(np.nonzero(uncertainty > threshold)[0].tolist())
    return np.array(sorted(top), dtype=np.int64)

  def _calibrate(self, pred: np.ndarray) -> np.ndarray:
    """Affine debiasing (true ~ a*pred + b) fit on verified pairs."""
    if (
      not self.cfg.calibrate or len(self._calibration) < self.cfg.min_calibration_pairs
    ):
      return pred.copy()
    p = np.array([c[0] for c in self._calibration])
    t = np.array([c[1] for c in self._calibration])
    var = float(np.var(p))
    if var < 1e-12:
      return pred.copy()
    a = float(np.cov(p, t, bias=True)[0, 1]) / var
    b = float(t.mean() - a * p.mean())
    return a * pred + b

  @property
  def verified_fraction(self) -> float:
    if self.total_evaluated == 0:
      return 0.0
    return self.total_verified / self.total_evaluated
