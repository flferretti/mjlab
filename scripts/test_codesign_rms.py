# SPDX-FileCopyrightText: Generative Bionics S.R.L.
# SPDX-License-Identifier: LicenseRef-GenerativeBionics-AllRightsReserved

"""Self-check for the RMS thermal constraint in codesign_ga.

Runs without Isaac/mjlab: a fake backend replays a scripted torque trace so we can
assert (a) evaluate() computes the correct per-design RMS, and (b) the RMS
overload formula (worst-joint RMS - ratio*peak) has the right sign.

Run: uv run --no-sync python scripts/test_codesign_rms.py
"""

from __future__ import annotations

import numpy as np
import torch
from codesign_ga import CodesignBackend, RewardMetric


class FakeBackend(CodesignBackend):
  """Replays a fixed per-joint torque each step; reward = mean applied torque."""

  def __init__(self, num_envs: int, n_joints: int, torque_LJ: torch.Tensor) -> None:
    super().__init__(rollout_steps=10, metric=RewardMetric())
    self.device = "cpu"
    self.num_envs = num_envs
    self.joint_order = [f"j{i}" for i in range(n_joints)]
    self._torque_LJ = torque_LJ  # (num_envs, n_joints)

  def _reset(self):
    return {}

  def _step(self, actions):
    return {}, torch.zeros(self.num_envs)

  def _actor_obs(self, obs):
    return torch.zeros(self.num_envs, 1)

  def _policy(self, actor_obs):
    return torch.zeros(self.num_envs, len(self.joint_order))

  def _root_height(self):
    return torch.ones(self.num_envs)

  def _write_tau(self, tau_full):
    pass

  def action_dim(self):
    return len(self.joint_order)

  def _applied_torque_LJ(self):
    return self._torque_LJ


def test_constant_torque_rms():
  # Two designs, 3 joints. Constant torque -> RMS == |torque|.
  torque = torch.tensor([[10.0, -20.0, 30.0], [5.0, 5.0, 5.0]])
  be = FakeBackend(num_envs=2, n_joints=3, torque_LJ=torque)
  perf = be.evaluate(torque, n_seeds=1)  # L=2, 1 seed each
  assert perf.shape == (2,)
  rms = be.last_rms_LJ
  assert rms is not None and rms.shape == (2, 3)
  assert np.allclose(rms, np.abs(torque.numpy()), atol=1e-4), rms


def test_sinusoidal_torque_rms():
  # A sinusoid sampled over a full period has RMS = amplitude / sqrt(2).
  amp = 40.0
  steps = 200
  t = torch.linspace(0, 2 * np.pi, steps)

  class Sinusoid(FakeBackend):
    def __init__(self):
      super().__init__(1, 1, torch.zeros(1, 1))
      self.rollout_steps = steps
      self._k = 0

    def _applied_torque_LJ(self):
      val = amp * torch.sin(t[self._k % steps])
      self._k += 1
      return torch.tensor([[val]])

  be = Sinusoid()
  be.evaluate(torch.tensor([[100.0]]), n_seeds=1)
  rms = float(be.last_rms_LJ[0, 0])
  assert abs(rms - amp / np.sqrt(2)) < 1.0, rms


def test_overload_formula():
  # RMS overload = worst-joint (RMS - ratio*peak). Sign check.
  ratio = 0.25
  peak = np.array([100.0, 80.0])  # motor peaks
  rms_ok = np.array([20.0, 15.0])  # 20 < 25, 15 < 20 -> feasible (<=0)
  rms_bad = np.array([20.0, 30.0])  # 30 > 20 -> violated (+10)
  assert np.max(rms_ok - ratio * peak) <= 0.0
  assert abs(np.max(rms_bad - ratio * peak) - 10.0) < 1e-9


if __name__ == "__main__":
  test_constant_torque_rms()
  test_sinusoidal_torque_rms()
  test_overload_formula()
  print("OK: RMS constraint self-checks passed")
