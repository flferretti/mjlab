"""QDD codesign velocity environment configuration.

Extends the flat QDD config with:
1. CodesignPdActuator (soft saturation + alternating optimization)
2. Codesign torque reward penalty (RMS + peak + saturation + rate)
3. Codesign scheduler hook in the training loop

Run with:
  uv run python -m mjlab.rl.train --task Mjlab-Velocity-Flat-Gbionics-QDD-Codesign
"""

from mjlab.actuator import CodesignPdActuatorCfg
from mjlab.asset_zoo.robots.gbionics_qdd.qdd_constants import (
  QDD_ACTUATED_JOINTS,
  EntityArticulationInfoCfg,
  get_qdd_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.qdd.env_cfgs import gbionics_qdd_flat_env_cfg
from mjlab.tasks.velocity.mdp.codesign_rewards import codesign_torque_composite

# Left-right symmetry pairs for the QDD.
QDD_SYMMETRY_PAIRS: list[tuple[str, str]] = [
  ("l_hip_pitch", "r_hip_pitch"),
  ("l_hip_roll", "r_hip_roll"),
  ("l_hip_yaw", "r_hip_yaw"),
  ("l_knee", "r_knee"),
  ("l_ankle_pitch", "r_ankle_pitch"),
  ("l_ankle_roll", "r_ankle_roll"),
]


def gbionics_qdd_codesign_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """QDD flat config with motor-type codesign actuator.

  Replaces the 6 separate IdealPdActuatorCfg entries with a single
  CodesignPdActuatorCfg that uses soft saturation and learns the
  optimal motor-type assignment.
  """
  cfg = gbionics_qdd_flat_env_cfg(play=play)

  # -- Replace actuators with codesign actuator --------------------------
  # Use a single actuator for all joints with codesign-learned limits.
  # Average stiffness/damping across all motor types as starting point.
  codesign_actuator = CodesignPdActuatorCfg(
    target_names_expr=(r".*",),
    stiffness=100.0,
    damping=5.0,
    effort_limit=80.0,  # Upper bound (max_tau).
    armature=0.03,
    viscous_damping=0.001 * 9.0**2,
    # --- Codesign params ---
    n_types=3,
    init_tau_max=[80.0, 50.0, 17.0],  # Start from current AJD12/10/8.
    joint_names=list(QDD_ACTUATED_JOINTS),
    symmetry_pairs=QDD_SYMMETRY_PAIRS,
    temperature_init=1.0,
    temperature_min=0.1,
    temperature_decay=0.9995,
    lambda_types=0.01,
    lambda_balance=0.05,
    lambda_tau=5e-4,
    lambda_saturation=0.1,
    lambda_rms=0.05,
    lambda_peak=0.1,
    min_tau=5.0,
    max_tau=120.0,
    codesign_lr=3e-3,
    codesign_interval=10,
  )

  # Override the robot config with codesign actuator.
  robot_cfg = get_qdd_robot_cfg()
  robot_cfg.articulation = EntityArticulationInfoCfg(
    actuators=(codesign_actuator,),
    soft_joint_pos_limit_factor=0.9,
  )
  cfg.scene.entities = {"robot": robot_cfg}

  # -- Add codesign torque reward ----------------------------------------
  # Replace the old per-motor-type rated torque penalties with the
  # composite codesign penalty that doesn't assume fixed motor assignment.
  cfg.rewards.pop("ajd8_rated_torque", None)
  cfg.rewards.pop("ajd10_rated_torque", None)
  cfg.rewards.pop("ajd12_rated_torque", None)

  cfg.rewards["codesign_torque"] = RewardTermCfg(
    func=codesign_torque_composite,
    weight=-0.1,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=tuple(QDD_ACTUATED_JOINTS)),
      "w_rms": 1.0,
      "w_peak": 2.0,
      "w_saturation": 3.0,
      "w_rate": 0.5,
      "tau_nominal": 80.0,
      "saturation_threshold": 0.8,
    },
  )

  return cfg
