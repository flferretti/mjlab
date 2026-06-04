"""QDD codesign velocity environment configuration.

Extends the flat QDD config with a codesign hook that learns the optimal
motor-type assignment (τ_max per joint) while keeping the original 6
actuator configs (with their per-joint PD gains) intact.

The codesign module runs as a standalone component alongside PPO:
1. Original actuators handle PD control with their tuned gains
2. Codesign module reads torques and learns motor-type assignment
3. Effort limits on actuators are updated based on learned assignment

Run with:
  uv run train Mjlab-Velocity-Flat-Gbionics-QDD-Codesign
"""

from mjlab.asset_zoo.robots.gbionics_qdd.qdd_constants import QDD_ACTUATED_JOINTS
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

# Codesign hyperparameters stored alongside the env config so the
# training loop can access them without importing a separate module.
CODESIGN_CFG: dict = {
  "n_types": 3,
  "init_tau_max": [80.0, 50.0, 17.0],
  "joint_names": list(QDD_ACTUATED_JOINTS),
  "symmetry_pairs": QDD_SYMMETRY_PAIRS,
  "temperature_init": 1.0,
  "temperature_min": 0.1,
  "temperature_decay": 0.9995,
  "lambda_types": 0.01,
  "lambda_balance": 0.05,
  "lambda_tau": 1e-4,
  "lambda_saturation": 0.05,
  "lambda_rms": 0.01,
  "lambda_peak": 0.02,
  "min_tau": 5.0,
  "max_tau": 120.0,
  "codesign_lr": 3e-3,
  "codesign_interval": 10,
  "warmup_iters": 2000,
}


def gbionics_qdd_codesign_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """QDD flat config with motor-type codesign.

  Keeps the original 6 actuator configs (per-joint PD gains, armature,
  damping) and adds codesign torque penalties. The codesign module
  (GumbelSoftmaxActuator) runs as a standalone hook in the training
  loop that modifies effort limits on the existing actuators.
  """
  cfg = gbionics_qdd_flat_env_cfg(play=play)

  # -- Replace per-motor rated torque with composite codesign penalty --
  # The original rated_torque penalties assume fixed motor assignment.
  # The composite penalty works with the codesign-learned assignment.
  cfg.rewards.pop("ajd8_rated_torque", None)
  cfg.rewards.pop("ajd10_rated_torque", None)
  cfg.rewards.pop("ajd12_rated_torque", None)

  cfg.rewards["codesign_torque"] = RewardTermCfg(
    func=codesign_torque_composite,
    weight=-0.05,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=tuple(QDD_ACTUATED_JOINTS)),
      "w_rms": 1.0,
      "w_peak": 1.0,
      "w_saturation": 2.0,
      "w_rate": 0.3,
      "tau_nominal": 80.0,
      "saturation_threshold": 0.85,
    },
  )

  # Store codesign config on the env config so the runner can access it.
  cfg.codesign = CODESIGN_CFG

  return cfg
