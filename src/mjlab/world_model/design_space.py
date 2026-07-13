"""Actuator design-parameter spaces for design-conditioned world models.

A :class:`DesignSpaceCfg` declares the vector of design parameters theta the
world model is conditioned on, together with their bounds and the joints each
parameter drives. It is the single write path for designs: the data
collector's per-reset randomization, the evaluator's normalization, and
GA-time injection all go through ``DesignSpaceCfg.apply``, which delegates to
:func:`mjlab.tasks.velocity.mdp.motor_randomization.set_motor_tau_max` —
generalizing that function's "single source of truth" doctrine.

v1 supports ``kind="tau_max"`` (per-group peak torque, applied through the
actuator ``force_limit`` buffer — no model-field expansion or recompile
needed), which matches the genome of ``scripts/codesign_ga.py`` exactly.
Contact-relevant physical parameters (reflected rotor inertia via
``dof_armature``, motor-mass feedback via body inertials) are the planned
Phase-4 extension and go through the ``mjlab.envs.mdp.dr`` model-field
expansion machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_THETA_BUFFER_ATTR = "_wm_design_theta"
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

DesignParamKind = Literal["tau_max"]
DesignRegion = Literal["train", "interp", "extrap"]


@dataclass(frozen=True)
class DesignParamCfg:
  """One scalar design parameter shared by a group of joints."""

  name: str
  kind: DesignParamKind
  joints: tuple[str, ...]
  """Joints driven by this parameter (e.g. a left/right symmetric pair)."""
  bounds: tuple[float, float]

  def __post_init__(self) -> None:
    if self.bounds[0] >= self.bounds[1]:
      raise ValueError(f"{self.name}: bounds {self.bounds} must be (lo, hi).")
    if not self.joints:
      raise ValueError(f"{self.name}: at least one joint is required.")


@dataclass(frozen=True)
class DesignSpaceCfg:
  """A named, bounded design-parameter space theta in R^D."""

  name: str
  params: tuple[DesignParamCfg, ...]
  tau_obs_range: tuple[float, float]
  """Normalization range of the policy's ``motor_tau_max`` observation. Must
  match the range the frozen design-conditioned policy was trained with."""
  extrapolation_margin: float = 0.1
  """Outer fraction of each bound range reserved as the extrapolation shell
  (never sampled during training-data collection)."""

  def __post_init__(self) -> None:
    names = [p.name for p in self.params]
    if len(set(names)) != len(names):
      raise ValueError(f"Duplicate design parameter names: {names}")
    joints = [j for p in self.params for j in p.joints]
    if len(set(joints)) != len(joints):
      raise ValueError("A joint may belong to at most one design parameter.")

  @property
  def dim(self) -> int:
    return len(self.params)

  @property
  def joint_names(self) -> tuple[str, ...]:
    """All driven joints, in parameter-declaration order."""
    return tuple(j for p in self.params for j in p.joints)

  def bounds_tensor(self, device: torch.device | str = "cpu") -> torch.Tensor:
    """(2, D) tensor of (lo, hi) bounds."""
    lo = [p.bounds[0] for p in self.params]
    hi = [p.bounds[1] for p in self.params]
    return torch.tensor([lo, hi], dtype=torch.float32, device=device)

  def normalize(self, theta: torch.Tensor) -> torch.Tensor:
    """Map theta to [0, 1]^D."""
    b = self.bounds_tensor(theta.device)
    return (theta - b[0]) / (b[1] - b[0])

  def denormalize(self, unit: torch.Tensor) -> torch.Tensor:
    b = self.bounds_tensor(unit.device)
    return b[0] + unit * (b[1] - b[0])

  # -- Sampling --------------------------------------------------------------

  def sample(
    self,
    n: int,
    seed: int = 0,
    region: DesignRegion = "train",
    holdout: torch.Tensor | None = None,
    holdout_radius: float = 0.05,
    device: torch.device | str = "cpu",
  ) -> torch.Tensor:
    """Sample ``n`` designs from a region of the space.

    Regions (in normalized coordinates, with margin m = extrapolation_margin):
      - ``train``: uniform over the interior box [m, 1-m]^D, rejecting samples
        within an L-inf ball of ``holdout_radius`` around any holdout design.
      - ``interp``: low-discrepancy (Sobol) points in the interior box — used
        to carve the interpolation holdout set.
      - ``extrap``: uniform over the shell (at least one coordinate in the
        outer margin).
    """
    m = self.extrapolation_margin
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if region == "interp":
      sobol = torch.quasirandom.SobolEngine(self.dim, scramble=True, seed=seed)
      unit = m + (1.0 - 2.0 * m) * sobol.draw(n).to(torch.float32)
      return self.denormalize(unit.to(device))

    unit = torch.empty(0, self.dim)
    max_rounds = 200
    for _ in range(max_rounds):
      if unit.shape[0] >= n:
        break
      cand = torch.rand((2 * n, self.dim), generator=gen)
      if region == "train":
        cand = m + (1.0 - 2.0 * m) * cand
        if holdout is not None and holdout.numel() > 0:
          h = self.normalize(holdout.cpu().float())
          dist = (cand.unsqueeze(1) - h.unsqueeze(0)).abs().amax(dim=-1)
          cand = cand[dist.amin(dim=1) > holdout_radius]
      elif region == "extrap":
        in_shell = ((cand < m) | (cand > 1.0 - m)).any(dim=-1)
        cand = cand[in_shell]
      else:
        raise ValueError(f"Unknown region '{region}'.")
      unit = torch.cat([unit, cand], dim=0)
    if unit.shape[0] < n:
      raise RuntimeError(
        f"Could not draw {n} '{region}' samples after {max_rounds} rounds; "
        f"check extrapolation_margin/holdout_radius."
      )
    return self.denormalize(unit[:n].to(device))

  def holdout_designs(
    self, n: int, seed: int = 0, device: torch.device | str = "cpu"
  ) -> torch.Tensor:
    """Deterministic interpolation holdout set (Sobol over the interior)."""
    return self.sample(n, seed=seed, region="interp", device=device)

  # -- Conversions to/from the GA's per-joint torque vectors ------------------

  def joint_tau_from_theta(self, theta: torch.Tensor) -> torch.Tensor:
    """(N, D) group values -> (N, n_joints) per-joint torques (tau params)."""
    cols = []
    for i, p in enumerate(self.params):
      if p.kind != "tau_max":
        raise NotImplementedError(f"kind={p.kind!r} lands with theta_v2.")
      cols.append(theta[:, i : i + 1].expand(-1, len(p.joints)))
    return torch.cat(cols, dim=-1)

  def theta_from_joint_tau(
    self, tau: torch.Tensor, joint_order: list[str] | tuple[str, ...]
  ) -> torch.Tensor:
    """(N, len(joint_order)) per-joint torques -> (N, D) group values.

    Joints within a group are expected to share one value (the GA genome is
    per-group); the first joint of each group is read.
    """
    idx = []
    for p in self.params:
      if p.kind != "tau_max":
        raise NotImplementedError(f"kind={p.kind!r} lands with theta_v2.")
      idx.append(list(joint_order).index(p.joints[0]))
    return tau[:, idx]

  # -- Application to a live environment --------------------------------------

  def apply(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    theta: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    """Write designs into the simulation and the policy observation.

    ``theta`` is ``(len(env_ids), D)``. tau_max parameters go through
    :func:`set_motor_tau_max` so physics clamps and the ``motor_tau_max``
    observation stay consistent. The per-env theta is also cached on the env
    (``_wm_design_theta``) for the data collector.
    """
    # Local import: importing anything under mjlab.tasks triggers full task
    # registration (and viewer dependencies), which must not happen at
    # mjlab.world_model import time — wm-train/wm-eval run on machines
    # without the simulation/viewer stack (e.g. ROCm training boxes).
    from mjlab.tasks.velocity.mdp.motor_randomization import set_motor_tau_max

    if theta.shape != (env_ids.shape[0], self.dim):
      raise ValueError(
        f"theta shape {tuple(theta.shape)} != ({env_ids.shape[0]}, {self.dim})"
      )
    buffer = _theta_buffer(env, self.dim)
    buffer[env_ids] = theta.to(buffer.dtype)
    tau_joints = self.joint_tau_from_theta(theta)
    set_motor_tau_max(
      env,
      env_ids,
      tau_joints,
      list(self.joint_names),
      self.tau_obs_range,
      asset_cfg,
    )


def _theta_buffer(env: ManagerBasedRlEnv, dim: int) -> torch.Tensor:
  buffer = getattr(env, _THETA_BUFFER_ATTR, None)
  if buffer is None or buffer.shape != (env.num_envs, dim):
    buffer = torch.zeros((env.num_envs, dim), device=env.device)
    setattr(env, _THETA_BUFFER_ATTR, buffer)
  return buffer


def current_design(env: ManagerBasedRlEnv, space: DesignSpaceCfg) -> torch.Tensor:
  """The (num_envs, D) designs most recently applied to the env."""
  return _theta_buffer(env, space.dim)


@dataclass
class _SamplerState:
  """Per-env cache for the design-randomization event."""

  space: DesignSpaceCfg
  holdout: torch.Tensor
  draws: int = 0


def randomize_design(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  design_space: DesignSpaceCfg,
  region: DesignRegion = "train",
  n_holdout: int = 200,
  holdout_seed: int = 0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset-mode event: sample a fresh design per resetting env and apply it.

  In the ``train`` region, samples reject the deterministic interpolation
  holdout set (seeded by ``holdout_seed``) and the extrapolation shell, so
  held-out designs are never seen during training-data collection. In the
  ``interp``/``extrap`` regions, designs are drawn from the respective
  held-out pools instead (used to collect evaluation ground truth).
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = int(env_ids.shape[0])
  if n == 0:
    return

  cache_attr = "_wm_design_sampler_state"
  state: _SamplerState | None = getattr(env, cache_attr, None)
  if state is None or state.space is not design_space:
    holdout = design_space.holdout_designs(n_holdout, seed=holdout_seed)
    state = _SamplerState(space=design_space, holdout=holdout)
    setattr(env, cache_attr, state)

  seed = holdout_seed * 1_000_003 + int(env.cfg.seed or 0) * 7919 + state.draws
  state.draws += 1
  if region == "train":
    theta = design_space.sample(n, seed=seed, region="train", holdout=state.holdout)
  elif region == "interp":
    idx = torch.randint(
      state.holdout.shape[0], (n,), generator=torch.Generator().manual_seed(seed)
    )
    theta = state.holdout[idx]
  else:
    theta = design_space.sample(n, seed=seed, region="extrap")
  design_space.apply(env, env_ids, theta.to(env.device), asset_cfg)


def design_space_from_dict(data: dict) -> DesignSpaceCfg:
  """Rebuild a :class:`DesignSpaceCfg` from ``dataclasses.asdict`` output
  (e.g. the ``design_space`` entry of a dataset's ``meta.json``)."""
  params = tuple(
    DesignParamCfg(
      name=p["name"],
      kind=p["kind"],
      joints=tuple(p["joints"]),
      bounds=(float(p["bounds"][0]), float(p["bounds"][1])),
    )
    for p in data["params"]
  )
  return DesignSpaceCfg(
    name=data["name"],
    params=params,
    tau_obs_range=(
      float(data["tau_obs_range"][0]),
      float(data["tau_obs_range"][1]),
    ),
    extrapolation_margin=float(data.get("extrapolation_margin", 0.1)),
  )


# ---------------------------------------------------------------------------
# Presets.
# ---------------------------------------------------------------------------

# GBionics QDD biped: mirrors QDD_GROUPS in scripts/codesign_ga.py exactly, so
# world-model Pareto fronts are directly comparable to MjlabBackend fronts.
QDD_DESIGN_SPACE_V1 = DesignSpaceCfg(
  name="qdd_v1",
  tau_obs_range=(5.0, 90.0),
  params=(
    DesignParamCfg("hip_pitch", "tau_max", ("l_hip_pitch", "r_hip_pitch"), (30, 150)),
    DesignParamCfg("hip_roll", "tau_max", ("l_hip_roll", "r_hip_roll"), (30, 150)),
    DesignParamCfg("hip_yaw", "tau_max", ("l_hip_yaw", "r_hip_yaw"), (10, 100)),
    DesignParamCfg("knee", "tau_max", ("l_knee", "r_knee"), (30, 180)),
    DesignParamCfg(
      "ankle_pitch", "tau_max", ("l_ankle_pitch", "r_ankle_pitch"), (15, 120)
    ),
    DesignParamCfg("ankle_roll", "tau_max", ("l_ankle_roll", "r_ankle_roll"), (5, 80)),
  ),
)

# Unitree Go1 quadruped (public robot, reproducibility track): one parameter
# per joint type, shared across all four legs. Nominal actuators are 23.7 Nm
# (hip/thigh) and 35.55 Nm (calf).
_GO1_LEGS = ("FL", "FR", "RL", "RR")
GO1_DESIGN_SPACE_V1 = DesignSpaceCfg(
  name="go1_v1",
  tau_obs_range=(5.0, 70.0),
  params=(
    DesignParamCfg(
      "hip", "tau_max", tuple(f"{leg}_hip_joint" for leg in _GO1_LEGS), (5, 50)
    ),
    DesignParamCfg(
      "thigh", "tau_max", tuple(f"{leg}_thigh_joint" for leg in _GO1_LEGS), (5, 50)
    ),
    DesignParamCfg(
      "calf", "tau_max", tuple(f"{leg}_calf_joint" for leg in _GO1_LEGS), (8, 70)
    ),
  ),
)

DESIGN_SPACES: dict[str, DesignSpaceCfg] = {
  QDD_DESIGN_SPACE_V1.name: QDD_DESIGN_SPACE_V1,
  GO1_DESIGN_SPACE_V1.name: GO1_DESIGN_SPACE_V1,
}
