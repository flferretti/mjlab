"""Tests for the actuator design-parameter space."""

from dataclasses import asdict

import pytest
import torch
from conftest import get_test_device

from mjlab.world_model.design_space import (
  QDD_DESIGN_SPACE_V1,
  DesignParamCfg,
  DesignSpaceCfg,
  current_design,
  design_space_from_dict,
)

SPACE = DesignSpaceCfg(
  name="toy",
  tau_obs_range=(5.0, 90.0),
  params=(
    DesignParamCfg("a", "tau_max", ("l_a", "r_a"), (10.0, 100.0)),
    DesignParamCfg("b", "tau_max", ("l_b", "r_b"), (5.0, 50.0)),
  ),
)


def test_normalize_round_trip():
  theta = SPACE.sample(16, seed=0, region="train")
  unit = SPACE.normalize(theta)
  assert (unit >= 0).all() and (unit <= 1).all()
  assert torch.allclose(SPACE.denormalize(unit), theta, atol=1e-5)


def test_train_samples_avoid_holdout_and_shell():
  holdout = SPACE.holdout_designs(50, seed=0)
  theta = SPACE.sample(200, seed=1, region="train", holdout=holdout)
  unit = SPACE.normalize(theta)
  m = SPACE.extrapolation_margin
  assert (unit >= m).all() and (unit <= 1 - m).all()
  dist = (
    (SPACE.normalize(theta).unsqueeze(1) - SPACE.normalize(holdout).unsqueeze(0))
    .abs()
    .amax(dim=-1)
  )
  assert (dist.amin(dim=1) > 0.05).all()


def test_extrap_samples_live_in_shell():
  theta = SPACE.sample(200, seed=2, region="extrap")
  unit = SPACE.normalize(theta)
  m = SPACE.extrapolation_margin
  in_shell = ((unit < m) | (unit > 1 - m)).any(dim=-1)
  assert in_shell.all()


def test_holdout_is_deterministic():
  a = SPACE.holdout_designs(20, seed=0)
  b = SPACE.holdout_designs(20, seed=0)
  assert torch.equal(a, b)


def test_joint_tau_round_trip():
  theta = SPACE.sample(8, seed=3, region="train")
  tau = SPACE.joint_tau_from_theta(theta)
  assert tau.shape == (8, 4)
  # Joints within a group share the group's value.
  assert torch.equal(tau[:, 0], tau[:, 1])
  joint_order = ["r_b", "l_a", "l_b", "r_a"]  # scrambled GA order
  tau_ga = tau[:, [3, 0, 2, 1]]  # reorder columns to joint_order
  restored = SPACE.theta_from_joint_tau(tau_ga, joint_order)
  assert torch.allclose(restored, theta, atol=1e-5)


def test_from_dict_round_trip():
  restored = design_space_from_dict(asdict(QDD_DESIGN_SPACE_V1))
  assert restored == QDD_DESIGN_SPACE_V1


def test_duplicate_joints_rejected():
  with pytest.raises(ValueError, match="at most one"):
    DesignSpaceCfg(
      name="bad",
      tau_obs_range=(0.0, 1.0),
      params=(
        DesignParamCfg("a", "tau_max", ("j",), (0.0, 1.0)),
        DesignParamCfg("b", "tau_max", ("j",), (0.0, 1.0)),
      ),
    )


@pytest.mark.slow
def test_apply_writes_per_env_force_limits():
  """Designs land in the per-env actuator force limits and the theta buffer."""
  # The task registry transitively requires the private motion-prior package.
  pytest.importorskip("gb_motion_prior_lupin")
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  device = get_test_device()
  cfg = load_env_cfg("Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond", play=True)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    space = QDD_DESIGN_SPACE_V1
    env_ids = torch.arange(2, device=device)
    theta = torch.stack(
      [space.bounds_tensor(device)[0], space.bounds_tensor(device)[1]]
    )  # env0 = all lower bounds, env1 = all upper bounds.
    space.apply(env, env_ids, theta)

    assert torch.allclose(current_design(env, space), theta)
    robot = env.scene["robot"]
    checked = 0
    for actuator in robot.actuators:
      force_limit = getattr(actuator, "force_limit", None)
      if force_limit is None:
        continue
      for i, jname in enumerate(actuator.target_names):
        if jname not in space.joint_names:
          continue
        j = list(space.joint_names).index(jname)
        expected_lo = space.joint_tau_from_theta(theta[:1])[0, j]
        expected_hi = space.joint_tau_from_theta(theta[1:])[0, j]
        assert float(force_limit[0, i]) == pytest.approx(float(expected_lo))
        assert float(force_limit[1, i]) == pytest.approx(float(expected_hi))
        checked += 1
    assert checked == len(space.joint_names)
  finally:
    env.close()
