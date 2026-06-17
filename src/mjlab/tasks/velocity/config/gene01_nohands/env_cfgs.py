"""Gbionics Gene01 Nohands velocity environment configurations.

Reward weights and MDP settings are matched to gb-rl-locomotion's Gene01 Nohands config.
AMP is currently disabled due to motion loader compatibility issues.
"""

from mjlab.asset_zoo.robots import (
  GENE01_NOHANDS_ACTION_SCALE,
  GENE01_NOHANDS_ACTUATED_JOINTS,
  X6_60_JOINTS,
  X8_120_JOINTS,
  X10_200_JOINTS,
  get_gene01_nohands_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.observations import base_ang_vel, base_lin_vel
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.viewer import ViewerConfig

# AMP_JOINT_NAMES and AMP_DATASET_CFG are disabled; use standard PPO training instead.


def gbionics_gene01_nohands_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Gbionics Gene01 Nohands flat terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  # -- Simulation --------------------------------------------------------
  # Match gb-rl-locomotion simulation settings.
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.mujoco.impratio = 1
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500

  # Match gb-rl-locomotion: 16 s episodes at 50 Hz (dt=0.005, decimation=4).
  cfg.episode_length_s = 16.0

  # -- Scene / Robot -----------------------------------------------------
  cfg.scene.entities = {"robot": get_gene01_nohands_robot_cfg()}
  cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
  cfg.viewer.entity_name = "robot"

  # Set raycast sensor frame to Gene01 root link (torso).
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      from mjlab.sensor import RayCastSensorCfg

      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "torso_1"

  # Foot height scan using sole sites (for foot clearance during swing).
  site_names = ("l_sole", "r_sole")
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )

  # Contact sensor for fall detection.
  contact_cfg = ContactSensorCfg(
    name="contact_forces",
    primary=ContactMatch(
      mode="body", pattern=("torso_1", "l_lower_leg", "r_lower_leg"), entity="robot"
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (contact_cfg,)

  # -- Actions -----------------------------------------------------------
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = tuple(GENE01_NOHANDS_ACTUATED_JOINTS)
  joint_pos_action.scale = GENE01_NOHANDS_ACTION_SCALE
  joint_pos_action.use_default_offset = False

  # -- Observations (matched to gb-rl-locomotion flat) -------------------
  # Replace IMU-based observations with custom functions that compute base vel from kinematic chain.
  # The base environment expects IMU sensors, but Gene01 doesn't have them, so we use computed observations.

  # Modify actor observations to use only actuated joints and add history stacking.
  actor_joint_pos = cfg.observations["actor"].terms["joint_pos"]
  assert isinstance(actor_joint_pos, ObservationTermCfg)
  actor_joint_pos.params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=tuple(GENE01_NOHANDS_ACTUATED_JOINTS)
  )
  actor_joint_pos.delay_max_lag = 4

  actor_joint_vel = cfg.observations["actor"].terms["joint_vel"]
  assert isinstance(actor_joint_vel, ObservationTermCfg)
  actor_joint_vel.params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=tuple(GENE01_NOHANDS_ACTUATED_JOINTS)
  )
  actor_joint_vel.delay_max_lag = 4

  actor_actions = cfg.observations["actor"].terms["actions"]
  assert isinstance(actor_actions, ObservationTermCfg)
  actor_actions.delay_max_lag = 4

  # Replace IMU observations with custom base velocity functions
  cfg.observations["actor"].terms["base_lin_vel"] = ObservationTermCfg(
    func=base_lin_vel,
  )
  cfg.observations["actor"].terms["base_lin_vel"].delay_max_lag = 4

  cfg.observations["actor"].terms["base_ang_vel"] = ObservationTermCfg(
    func=base_ang_vel,
  )
  cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 4

  # Modify critic observations to include all joints (not just actuated).
  # History is handled via delay_max_lag = 4.
  crit_joint_vel = cfg.observations["critic"].terms["joint_vel"]
  assert isinstance(crit_joint_vel, ObservationTermCfg)
  crit_joint_vel.delay_max_lag = 4

  crit_joint_pos = cfg.observations["critic"].terms["joint_pos"]
  assert isinstance(crit_joint_pos, ObservationTermCfg)
  crit_joint_pos.delay_max_lag = 4

  crit_actions = cfg.observations["critic"].terms["actions"]
  assert isinstance(crit_actions, ObservationTermCfg)
  crit_actions.delay_max_lag = 4

  # Replace critic IMU observations with custom base velocity functions
  cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
    func=base_lin_vel,
  )
  cfg.observations["critic"].terms["base_lin_vel"].delay_max_lag = 4

  cfg.observations["critic"].terms["base_ang_vel"] = ObservationTermCfg(
    func=base_ang_vel,
  )
  cfg.observations["critic"].terms["base_ang_vel"].delay_max_lag = 4

  # Remove observations that depend on sensors we haven't configured.
  # These are from the base velocity env config but Gene01 doesn't have all sensors.
  for group in ("actor", "critic"):
    for term_to_remove in ("foot_air_time", "foot_contact", "foot_contact_forces"):
      if term_to_remove in cfg.observations[group].terms:
        del cfg.observations[group].terms[term_to_remove]

  # AMP observations disabled due to motion loader issues.
  # cfg.observations["amp"] = ObservationGroupCfg(
  #   terms={
  #     "joint_pos": ObservationTermCfg(
  #       func=amp_joint_pos_fn,
  #       params={"asset_cfg": SceneEntityCfg("robot", joint_names=AMP_JOINT_NAMES)},
  #     ),
  #     "joint_vel": ObservationTermCfg(
  #       func=amp_joint_vel_fn,
  #       params={"asset_cfg": SceneEntityCfg("robot", joint_names=AMP_JOINT_NAMES)},
  #     ),
  #   },
  # )

  # -- Rewards (matched to gb-rl-locomotion) --------------------------------
  # Configure pose reward with standard deviations per joint.
  cfg.rewards["pose"].params["std_standing"] = {r".*": 0.5}
  cfg.rewards["pose"].params["std_walking"] = {r".*": 0.7}
  cfg.rewards["pose"].params["std_running"] = {r".*": 1.0}

  # Remove foot-related rewards that require missing feet sensors.
  for reward_name in (
    "foot_clearance",
    "foot_swing_height",
    "air_time",
    "foot_slip",
    "soft_landing",
  ):
    try:
      del cfg.rewards[reward_name]
    except KeyError:
      pass

  # -- Events (domain randomization & resets) ----------------------------
  # Startup randomization.
  cfg.events["randomize_robot_mass"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "operation": "scale",
      "ranges": (0.95, 1.05),
    },
  )

  cfg.events["randomize_actuator_gains"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "kp_range": (0.7, 1.3),
      "kd_range": (0.7, 1.3),
    },
  )

  cfg.events["randomize_joint_friction"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.dof_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "ranges": (0.9, 1.1),
    },
  )

  cfg.events["joint_default_pos_noise"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.joint_default_pos,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "add",
      "ranges": (-0.02, 0.02),
    },
  )

  # Reset robot state from AMP motion data.
  # TEMPORARILY DISABLED: AMP motion loader has compatibility issues with motion data format.
  # TODO: Debug AMPLoader vs motion data format mismatch.
  # cfg.events["reset_robot_joints"] = EventTermCfg(
  #   func=reset_state_from_amp,
  #   mode="reset",
  #   params={
  #     "dataset_cfg": AMP_DATASET_CFG,
  #     "position_range": (0.0, 0.0),
  #     "velocity_range": (0.0, 0.0),
  #     "base_lin_vel_range": (0.0, 0.0),
  #     "base_ang_vel_range": (0.0, 0.0),
  #   },
  # )

  # Periodic push (every 5-10 seconds) - modify base config event.
  cfg.events["push_robot"].interval_range_s = (5.0, 10.0)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.5, 0.5),
    "y": (-0.01, 0.01),
  }

  # -- Terminations (matched to gb-rl-locomotion) -------------------------
  cfg.terminations["time_out"] = TerminationTermCfg(func=envs_mdp.time_out)

  cfg.terminations["base_height"] = TerminationTermCfg(
    func=envs_mdp.root_height_below_minimum,
    params={"minimum_height": 0.3, "asset_cfg": SceneEntityCfg("robot")},
  )

  # -- Rewards (matched to gb-rl-locomotion flat) -------------------------
  # Velocity tracking.
  cfg.rewards["track_lin_vel_xy_exp"] = RewardTermCfg(
    func=mdp.track_lin_vel_xy_exp,
    weight=10.0,
    params={"command_name": "twist", "std": 0.5},
  )

  cfg.rewards["track_ang_vel_z_exp"] = RewardTermCfg(
    func=mdp.track_ang_vel_z_exp,
    weight=5.0,
    params={"command_name": "twist", "std": 0.5},
  )

  # Action smoothness.
  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=envs_mdp.action_rate_l2,
    weight=-5e-2,
  )

  # Joint limits & smoothness penalties.
  cfg.rewards["joint_pos_limits"] = RewardTermCfg(
    func=envs_mdp.joint_pos_limits,
    weight=-1e-3,
  )

  cfg.rewards["joint_vel_l1"] = RewardTermCfg(
    func=envs_mdp.joint_vel_l1,
    weight=-1e-2,
  )

  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=envs_mdp.joint_torques_l2,
    weight=-1e-9,
  )

  # Power consumption (matching gb-rl-locomotion electrical_power_cost).
  # This uses motor current model: I = tau / ktau, then P = I^2 * R * gearbox_eff.
  # Build parameter lists following GENE01_NOHANDS_ACTUATED_JOINTS order.
  _ktau_list: list[float] = []
  _gear_list: list[float] = []
  _res_list: list[float] = []
  _x6_60_set = set(X6_60_JOINTS)
  _x8_120_set = set(X8_120_JOINTS)
  _x10_200_set = set(X10_200_JOINTS)

  for jname in GENE01_NOHANDS_ACTUATED_JOINTS:
    if jname in _x6_60_set:
      _ktau_list.append(0.11)  # X6-60 Nm/A
      _gear_list.append(19.612)
      _res_list.append(0.41)
    elif jname in _x8_120_set:
      _ktau_list.append(0.12)  # X8-120 Nm/A
      _gear_list.append(19.612)
      _res_list.append(0.18)
    else:  # X10-200
      _ktau_list.append(0.26)  # X10-200 Nm/A
      _gear_list.append(20.0)
      _res_list.append(0.27)

  cfg.rewards["power_consumption"] = RewardTermCfg(
    func=mdp.electrical_power_cost_detailed,
    weight=-1.0,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=tuple(GENE01_NOHANDS_ACTUATED_JOINTS)
      ),
      "ktau": _ktau_list,
      "gear_ratio": _gear_list,
      "resistance": _res_list,
      "power_limit": 1000.0,  # W, PSU/battery limit
      "power_ref": 500.0,  # W, "reasonable" power usage
      "gearbox_efficiency": 0.95,
      "aux_power": 100.0,  # PC + sensors etc.
      "peak_importance": 1.0,
      "energy_importance": 0.2,
    },
  )

  # Foot slide penalty disabled - Gene01 does not have feet_ground_contact sensor.
  # cfg.rewards["foot_slip"] = RewardTermCfg(
  #   func=mdp.feet_slip,
  #   weight=-1.5,
  #   params={
  #     "sensor_name": "contact_forces",
  #     "command_name": "twist",
  #     "command_threshold": 0.01,
  #   },
  # )

  # -- Velocity commands -----------------------------------------------
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.resampling_time_range = (10.0, 10.0)
  twist_cmd.rel_standing_envs = 0.2
  twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  return cfg
