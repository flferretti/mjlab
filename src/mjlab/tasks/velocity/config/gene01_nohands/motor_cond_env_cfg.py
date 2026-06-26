"""Gene01 motor-conditioned velocity environment configuration.

Extends the flat Gene01 nohands config so that, at every reset, each
environment samples a new per-joint maximum torque (shared across left/right
pairs) from a fixed range. The sampled limits are applied to the actuators and
exposed to the policy through an observation term, yielding a policy that can
run with arbitrary motor configurations.
"""

from mjlab.asset_zoo.robots import GENE01_NOHANDS_ACTUATED_JOINTS
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.gene01_nohands.env_cfgs import (
  gbionics_gene01_nohands_flat_env_cfg,
)
from mjlab.tasks.velocity.mdp.motor_randomization import (
  motor_tau_max,
  randomize_motor_tau_max,
)

# Left-right symmetry pairs. Torso joints are intentionally unpaired.
GENE01_NOHANDS_SYMMETRY_PAIRS: list[tuple[str, str]] = [
  ("l_hip_pitch", "r_hip_pitch"),
  ("l_hip_roll", "r_hip_roll"),
  ("l_hip_yaw", "r_hip_yaw"),
  ("l_shoulder_pitch", "r_shoulder_pitch"),
  ("l_knee", "r_knee"),
  ("l_shoulder_roll", "r_shoulder_roll"),
  ("l_ankle_motor_1", "r_ankle_motor_1"),
  ("l_ankle_motor_2", "r_ankle_motor_2"),
  ("l_shoulder_yaw", "r_shoulder_yaw"),
  ("l_elbow", "r_elbow"),
]

# Absolute torque range (Nm) sampled uniformly per symmetric joint group.
MOTOR_TAU_RANGE: tuple[float, float] = (20.0, 200.0)


def gbionics_gene01_nohands_motor_cond_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Gene01 nohands flat config conditioned on randomized per-joint max torque."""
  cfg = gbionics_gene01_nohands_flat_env_cfg(play=play)
  joint_names = list(GENE01_NOHANDS_ACTUATED_JOINTS)

  # Resample per-joint tau_max each reset and apply to actuators.
  cfg.events["randomize_motor_tau_max"] = EventTermCfg(
    mode="reset",
    func=randomize_motor_tau_max,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "joint_names": joint_names,
      "symmetry_pairs": GENE01_NOHANDS_SYMMETRY_PAIRS,
      "tau_range": MOTOR_TAU_RANGE,
    },
  )

  # Expose normalized tau_max to actor and critic.
  for group in ("actor", "critic"):
    cfg.observations[group].terms["motor_tau_max"] = ObservationTermCfg(
      func=motor_tau_max,
      params={
        "joint_names": joint_names,
        "tau_range": MOTOR_TAU_RANGE,
      },
    )

  return cfg
