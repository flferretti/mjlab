"""Gbionics QDD velocity environment configurations.

Reward weights and MDP settings are matched to gb-rl-locomotion's QDD config
for a direct comparison.  The AMP discriminator provides gait-style shaping;
there are no explicit gait-shaping rewards.
"""

import math

from gb_motion_prior_lupin import resolve_dataset_dir

from mjlab.asset_zoo.robots import (
  QDD_ACTION_OFFSET,
  QDD_ACTION_SCALE,
  get_qdd_robot_cfg,
)
from mjlab.asset_zoo.robots.gbionics_qdd.qdd_constants import QDD_ACTUATED_JOINTS
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.observations import base_ang_vel, base_lin_vel
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
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
from mjlab.tasks.velocity.mdp.amp_events import reset_state_from_amp
from mjlab.tasks.velocity.mdp.amp_observations import (
  joint_pos as amp_joint_pos_fn,
)
from mjlab.tasks.velocity.mdp.amp_observations import (
  joint_vel as amp_joint_vel_fn,
)
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# AMP dataset configuration (shared across flat/rough).
AMP_JOINT_NAMES: list[str] = list(QDD_ACTUATED_JOINTS)
AMP_DATASET_CFG: dict = {
  "amp_data_path": resolve_dataset_dir("lowerbodyqdd-mixamo"),
  "datasets": {
    "Happy Left Turn Slow": 0.5,
    "Happy Right Turn Slow": 0.5,
    "Happy Left Turn Fast": 0.5,
    "Happy Right Turn Fast": 0.5,
    "Standing": 0.2,
    "Start Walking": 1.0,
    "Stop Walking": 1.0,
    "Walking Left Turn": 1.0,
    "Walking Right Turn": 1.0,
    "Walking Backwards": 1.0,
  },
  "slow_down_factor": 1.0,
  "amp_joint_names": AMP_JOINT_NAMES,
}


def gbionics_qdd_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Gbionics QDD rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  # -- Simulation --------------------------------------------------------
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.mujoco.impratio = 1
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500

  # Match gb-rl-locomotion: 16 s episodes at 50 Hz (dt=0.005, decimation=4).
  cfg.episode_length_s = 16.0

  # -- Scene / Robot -----------------------------------------------------
  cfg.scene.entities = {"robot": get_qdd_robot_cfg()}

  # Set raycast sensor frame to QDD root link (torso body).
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      from mjlab.sensor import RayCastSensorCfg

      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "root_link"

  site_names = ("l_sole", "r_sole")
  foot_body_names = ("l_foot_roll", "r_foot_roll")

  # Wire foot height scan to per-foot sole sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      from mjlab.sensor import RingPatternCfg

      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="body", pattern=foot_body_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  trunk_ground_cfg = ContactSensorCfg(
    name="trunk_ground_touch",
    primary=ContactMatch(
      mode="body",
      entity="robot",
      pattern=("root_link",),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    trunk_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # -- Actions (match gb-rl rescale_to_limits with scale=0.5) -------------
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = QDD_ACTION_SCALE
  joint_pos_action.offset = QDD_ACTION_OFFSET
  joint_pos_action.use_default_offset = False

  # -- Viewer ------------------------------------------------------------
  cfg.viewer.body_name = "root_link"
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -10.0

  # -- Commands (match gb-rl-locomotion) ---------------------------------
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.resampling_time_range = (10.0, 10.0)
  twist_cmd.rel_standing_envs = 0.2
  twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  # -- Observations: AMP group (ground-truth, no noise) -------------------
  cfg.observations["amp"] = ObservationGroupCfg(
    terms={
      "joint_pos": ObservationTermCfg(func=amp_joint_pos_fn),
      "joint_vel": ObservationTermCfg(func=amp_joint_vel_fn),
      "base_lin_vel": ObservationTermCfg(func=base_lin_vel),
      "base_ang_vel": ObservationTermCfg(func=base_ang_vel),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  # -- Observations: actor/critic alignment with gb-rl-locomotion ---------
  # gb-rl BlindPolicyCfg uses history_length=5 on proprioceptive terms and
  # scale factors (ang_vel 0.25, joint_vel 0.05).  The base factory does
  # not set these, so we patch them here.
  for group in ("actor", "critic"):
    obs = cfg.observations[group]
    obs.terms["base_ang_vel"].history_length = 5
    obs.terms["base_ang_vel"].scale = 0.25
    obs.terms["projected_gravity"].history_length = 5
    obs.terms["joint_pos"].history_length = 5
    obs.terms["joint_vel"].history_length = 5
    obs.terms["joint_vel"].scale = 0.05

  # -- Events ------------------------------------------------------------
  del cfg.events["foot_friction"]
  cfg.events["foot_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=foot_body_names),
      "operation": "abs",
      "axes": [0],
      "ranges": (0.3, 1.5),
      "shared_random": True,
    },
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("root_link",)

  # Match gb-rl-locomotion push force (±0.01 m/s — very weak).
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.01, 0.01),
    "y": (-0.01, 0.01),
  }
  cfg.events["push_robot"].interval_range_s = (5.0, 10.0)

  # -- Domain randomization matching gb-rl-locomotion --------------------
  # Mass randomization (0.95–1.05 scale).
  cfg.events["randomize_robot_mass"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "operation": "scale",
      "ranges": (0.95, 1.05),
    },
  )
  # Actuator PD gains randomization (stiffness/damping 0.7–1.3 scale).
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
  # Joint friction randomization (0.9–1.1 scale, matching gb-rl armature DR).
  cfg.events["randomize_joint_friction"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.dof_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "ranges": (0.9, 1.1),
    },
  )
  # Joint default position calibration noise (±0.02 rad).
  cfg.events["joint_default_pos_noise"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.joint_default_pos,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "add",
      "ranges": (-0.02, 0.02),
    },
  )

  # Reset robot state from AMP motion data (zero perturbation, matching gb-rl).
  cfg.events["reset_robot_joints"] = EventTermCfg(
    func=reset_state_from_amp,
    mode="reset",
    params={
      "dataset_cfg": AMP_DATASET_CFG,
      "position_range": (0.0, 0.0),
      "velocity_range": (0.0, 0.0),
      "base_lin_vel_range": (0.0, 0.0),
      "base_ang_vel_range": (0.0, 0.0),
    },
  )

  # -- Rewards (matched to gb-rl-locomotion) -----------------------------
  # Velocity tracking — use XY-only / Z-only variants matching IsaacLab.
  # The base factory's track_linear_velocity penalises vertical bounce and
  # track_angular_velocity penalises roll/pitch — both natural during walking
  # and absent in gb-rl.  Replace them with exact equivalents.
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=mdp.track_lin_vel_xy_exp,
    weight=10.0,
    params={"command_name": "twist", "std": 0.5},
  )
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=mdp.track_ang_vel_z_exp,
    weight=5.0,
    params={"command_name": "twist", "std": 0.5},
  )

  # Disable rewards not present in gb-rl-locomotion.
  cfg.rewards["upright"].weight = 0.0
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("root_link",)
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
  cfg.rewards["pose"].weight = 0.0
  cfg.rewards["pose"].params["std_standing"] = {r".*": 1.0}
  cfg.rewards["pose"].params["std_walking"] = {r".*": 1.0}
  cfg.rewards["pose"].params["std_running"] = {r".*": 1.0}

  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("root_link",)
  cfg.rewards["angular_momentum"].weight = 0.0

  # Disable gait-shaping rewards — AMP discriminator handles gait style.
  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names
  cfg.rewards["air_time"].weight = 0.0
  cfg.rewards["foot_clearance"].weight = 0.0
  cfg.rewards["foot_swing_height"].weight = 0.0
  cfg.rewards["soft_landing"].weight = 0.0
  cfg.rewards["foot_slip"].weight = 0.0

  # Action rate — gb-rl uses -5e-2.
  cfg.rewards["action_rate_l2"].weight = -5e-2

  # Joint pos limits penalty (matches gb-rl joint_effort_limit_penalty).
  cfg.rewards["dof_pos_limits"].weight = -1e-3

  # Joint velocity penalty — gb-rl uses L1.
  cfg.rewards["joint_vel_l1"] = RewardTermCfg(
    func=envs_mdp.joint_vel_l1,
    weight=-1e-2,
  )

  # Joint torque penalty — gb-rl uses -1e-9.
  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=envs_mdp.joint_torques_l2,
    weight=-1e-9,
  )

  # -- Rated torque penalties per motor type (gb-rl QDDRewardsCfg) --------
  cfg.rewards["ajd8_rated_torque"] = RewardTermCfg(
    func=mdp.applied_torque_over_value,
    weight=-1e-3,
    params={
      "value": 6.0,
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=("l_ankle_roll", "r_ankle_roll")
      ),
    },
  )
  cfg.rewards["ajd10_rated_torque"] = RewardTermCfg(
    func=mdp.applied_torque_over_value,
    weight=-1e-3,
    params={
      "value": 20.0,
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=("l_hip_yaw", "r_hip_yaw", "l_ankle_pitch", "r_ankle_pitch"),
      ),
    },
  )
  cfg.rewards["ajd12_rated_torque"] = RewardTermCfg(
    func=mdp.applied_torque_over_value,
    weight=-1e-3,
    params={
      "value": 40.0,
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          "l_hip_pitch",
          "r_hip_pitch",
          "l_hip_roll",
          "r_hip_roll",
          "l_knee",
          "r_knee",
        ),
      ),
    },
  )

  # -- Electrical power cost (detailed motor model) ----------------------
  # Motor constants per joint, following QDD_ACTUATED_JOINTS order.
  _ktau: list[float] = []
  _gear: list[float] = []
  _res: list[float] = []
  _ajd8_joints = {"l_ankle_roll", "r_ankle_roll"}
  _ajd10_joints = {"l_hip_yaw", "r_hip_yaw", "l_ankle_pitch", "r_ankle_pitch"}
  for jname in QDD_ACTUATED_JOINTS:
    if jname in _ajd8_joints:
      _ktau.append(0.156364)
      _gear.append(7.75)
      _res.append(0.29)
    elif jname in _ajd10_joints:
      _ktau.append(0.241679)
      _gear.append(9.0)
      _res.append(0.25)
    else:  # AJD12
      _ktau.append(0.233894)
      _gear.append(9.0)
      _res.append(0.8)
  cfg.rewards["power_consumption"] = RewardTermCfg(
    func=mdp.electrical_power_cost_detailed,
    weight=-1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=tuple(QDD_ACTUATED_JOINTS)),
      "ktau": _ktau,
      "gear_ratio": _gear,
      "resistance": _res,
      "power_limit": 1000.0,
      "power_ref": 500.0,
      "gearbox_efficiency": 0.95,
      "aux_power": 100.0,
      "peak_importance": 1.0,
      "energy_importance": 0.2,
    },
  )

  # -- Anti-backward-flight penalties ------------------------------------
  cfg.rewards["no_fly_backward"] = RewardTermCfg(
    func=mdp.no_fly_backward_penalty,
    weight=-2.0,
    params={
      "command_name": "twist",
      "sensor_name": feet_ground_cfg.name,
      "threshold": 1.0,
      "cmd_threshold": -0.1,
    },
  )
  cfg.rewards["lin_vel_z_backward"] = RewardTermCfg(
    func=mdp.lin_vel_z_backward_l2,
    weight=-2.0,
    params={
      "command_name": "twist",
      "cmd_threshold": -0.1,
    },
  )

  # -- Feet orientation and slide ----------------------------------------
  cfg.rewards["feet_flat_ori"] = RewardTermCfg(
    func=mdp.feet_orientation_contact,
    weight=-1.5,
    params={
      "sensor_name": feet_ground_cfg.name,
      "asset_cfg": SceneEntityCfg("robot", body_names=("l_foot_roll", "r_foot_roll")),
    },
  )
  cfg.rewards["feet_slide"] = RewardTermCfg(
    func=mdp.feet_slip,
    weight=-1.5,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "twist",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )

  # -- Double support during standing (prevent single-leg balance exploit) --
  cfg.rewards["standing_double_support"] = RewardTermCfg(
    func=mdp.standing_double_support,
    weight=2.0,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )

  # -- Terminations ------------------------------------------------------
  # Height-based fall detection: terminate when root drops below 0.3 m.
  # The trunk contact sensor doesn't fire reliably in MuJoCo (root_link
  # geometry rarely touches terrain), so we use height + orientation.
  cfg.terminations.pop("fell_over", None)
  cfg.terminations["base_height"] = TerminationTermCfg(
    func=envs_mdp.root_height_below_minimum,
    params={"minimum_height": 0.3},
  )
  cfg.terminations["bad_orientation"] = TerminationTermCfg(
    func=envs_mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  # -- Curriculum --------------------------------------------------------
  # Disable velocity command curriculum to match gb-rl (no stages).
  cfg.curriculum.pop("command_vel", None)

  # -- Play mode ---------------------------------------------------------
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def gbionics_qdd_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Gbionics QDD flat terrain velocity configuration."""
  cfg = gbionics_qdd_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove sensors not needed on flat (keep trunk_ground_touch for diagnostics).
  remove_sensors = {"terrain_scan"}
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # upright debug_vis looks up terrain_scan — remove the reference on flat.
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  # Fall detection: inherits base_height + bad_orientation from rough.
  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum.
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  return cfg
