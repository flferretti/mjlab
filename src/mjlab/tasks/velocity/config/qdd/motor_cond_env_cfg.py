"""QDD motor-conditioned velocity environment configuration.

Extends the flat QDD config so that, at every reset, each environment samples a
new per-joint maximum torque (shared across left/right pairs) from a fixed
range. The sampled limits are applied to the actuators and exposed to the policy
through an observation term, yielding a policy that can run with arbitrary motor
configurations. A downstream genetic algorithm can then use this frozen policy
to search for the best torque combination without retraining.

Run with:
  uv run train Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond
"""

from mjlab.asset_zoo.robots.gbionics_qdd.qdd_constants import QDD_ACTUATED_JOINTS
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.qdd.env_cfgs import gbionics_qdd_flat_env_cfg
from mjlab.tasks.velocity.mdp.motor_randomization import (
  motor_tau_max,
  randomize_motor_tau_max,
)

# Left-right symmetry pairs for the QDD: each pair shares one sampled tau_max.
QDD_SYMMETRY_PAIRS: list[tuple[str, str]] = [
  ("l_hip_pitch", "r_hip_pitch"),
  ("l_hip_roll", "r_hip_roll"),
  ("l_hip_yaw", "r_hip_yaw"),
  ("l_knee", "r_knee"),
  ("l_ankle_pitch", "r_ankle_pitch"),
  ("l_ankle_roll", "r_ankle_roll"),
]

# Absolute torque range (Nm) sampled uniformly per symmetric joint group.
MOTOR_TAU_RANGE: tuple[float, float] = (5.0, 90.0)


def gbionics_qdd_motor_cond_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """QDD flat config conditioned on randomized per-joint max torque."""
  cfg = gbionics_qdd_flat_env_cfg(play=play)

  joint_names = list(QDD_ACTUATED_JOINTS)

  # -- Event: resample per-joint tau_max each reset and apply to actuators ----
  cfg.events["randomize_motor_tau_max"] = EventTermCfg(
    mode="reset",
    func=randomize_motor_tau_max,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "joint_names": joint_names,
      "symmetry_pairs": QDD_SYMMETRY_PAIRS,
      "tau_range": MOTOR_TAU_RANGE,
    },
  )

  # -- Observation: expose normalized tau_max to actor and critic -------------
  for group in ("actor", "critic"):
    cfg.observations[group].terms["motor_tau_max"] = ObservationTermCfg(
      func=motor_tau_max,
      params={
        "joint_names": joint_names,
        "tau_range": MOTOR_TAU_RANGE,
      },
    )

  return cfg
