"""Go1 motor-conditioned velocity environment configuration.

Mirrors the QDD motor-conditioned config: at every reset, each environment
samples a new per-joint-type maximum torque (shared across all four legs) from
a fixed range. The sampled limits are applied to the actuators and exposed to
the policy through an observation term, yielding a policy that can run with
arbitrary motor configurations — the behavior policy for world-model data
collection and design evaluation on the public Go1 robot.

Run with:
  uv run train Mjlab-Velocity-Flat-Unitree-Go1-MotorCond
"""

import dataclasses

from mjlab.actuator import BuiltinPositionActuatorCfg, IdealPdActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg
from mjlab.tasks.velocity.mdp.motor_randomization import (
  motor_tau_max,
  randomize_motor_tau_max,
)

_LEGS = ("FL", "FR", "RL", "RR")

GO1_ACTUATED_JOINTS: list[str] = [
  f"{leg}_{joint}_joint" for leg in _LEGS for joint in ("hip", "thigh", "calf")
]

# All four legs share one sampled tau_max per joint type. The symmetry map
# collapses each joint onto the front-left leg's joint of the same type.
GO1_SYMMETRY_PAIRS: list[tuple[str, str]] = [
  (f"FL_{joint}_joint", f"{leg}_{joint}_joint")
  for joint in ("hip", "thigh", "calf")
  for leg in ("FR", "RL", "RR")
]

# Absolute torque range (Nm) sampled uniformly per joint-type group. Nominal
# Go1 actuators are 23.7 Nm (hip/thigh) and 35.55 Nm (calf).
MOTOR_TAU_RANGE: tuple[float, float] = (5.0, 70.0)


def unitree_go1_motor_cond_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Go1 flat config conditioned on randomized per-joint max torque."""
  cfg = unitree_go1_flat_env_cfg(play=play)

  # The stock Go1 uses MuJoCo built-in <position> actuators, whose effort
  # limit is baked into the model and cannot vary per environment. Swap them
  # for ideal PD actuators (same gains/armature) whose per-env ``force_limit``
  # buffer is what ``set_motor_tau_max`` writes, so sampled designs actually
  # clamp the physics. The XML effort limit is set to the top of the sampling
  # range so the torch-side clamp is always the binding constraint.
  robot = cfg.scene.entities["robot"]
  assert robot.articulation is not None
  pd_actuators = []
  for actuator in robot.articulation.actuators:
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    pd_actuators.append(
      IdealPdActuatorCfg(
        target_names_expr=actuator.target_names_expr,
        stiffness=actuator.stiffness,
        damping=actuator.damping,
        effort_limit=MOTOR_TAU_RANGE[1],
        armature=actuator.armature,
        frictionloss=actuator.frictionloss,
      )
    )
  cfg.scene.entities["robot"] = dataclasses.replace(
    robot,
    articulation=dataclasses.replace(robot.articulation, actuators=tuple(pd_actuators)),
  )

  joint_names = list(GO1_ACTUATED_JOINTS)

  cfg.events["randomize_motor_tau_max"] = EventTermCfg(
    mode="reset",
    func=randomize_motor_tau_max,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "joint_names": joint_names,
      "symmetry_pairs": GO1_SYMMETRY_PAIRS,
      "tau_range": MOTOR_TAU_RANGE,
    },
  )

  for group in ("actor", "critic"):
    cfg.observations[group].terms["motor_tau_max"] = ObservationTermCfg(
      func=motor_tau_max,
      params={
        "joint_names": joint_names,
        "tau_range": MOTOR_TAU_RANGE,
      },
    )

  return cfg
