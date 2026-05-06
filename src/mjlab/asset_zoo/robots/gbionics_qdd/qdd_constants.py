"""Gbionics QDD lower-body humanoid constants.

Actuator parameters, joint groups, and robot configuration matching
the gb-rl-locomotion IsaacLab definition (``assets/qdd.py``).
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

QDD_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "gbionics_qdd" / "xmls" / "qdd_bat.xml"
)
assert QDD_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(QDD_XML))


##
# Joint groups.
##

QDD_ACTUATED_JOINTS: list[str] = [
  "l_hip_pitch",
  "r_hip_pitch",
  "l_hip_roll",
  "r_hip_roll",
  "l_hip_yaw",
  "r_hip_yaw",
  "l_knee",
  "r_knee",
  "l_ankle_pitch",
  "r_ankle_pitch",
  "l_ankle_roll",
  "r_ankle_roll",
]

##
# Actuator configs.
#
# Three motor types (AJD8, AJD10, AJD12) with per-joint stiffness and
# damping from gb-rl-locomotion qdd.py.
##

# AJD12 — hip pitch, hip roll, knee.
# ktau=0.233894 Nm/A, resistance=0.8 Ω, gear_ratio=9.0
QDD_AJD12_HIP_PITCH_CFG = IdealPdActuatorCfg(
  target_names_expr=(".*_hip_pitch",),
  stiffness=100.0,
  damping=8.331,
  effort_limit=80.0,
  armature=0.04,
  viscous_damping=0.001 * 9.0**2,
)
QDD_AJD12_HIP_ROLL_CFG = IdealPdActuatorCfg(
  target_names_expr=(".*_hip_roll",),
  stiffness=100.0,
  damping=9.998,
  effort_limit=80.0,
  armature=0.04,
  viscous_damping=0.001 * 9.0**2,
)
QDD_AJD12_KNEE_CFG = IdealPdActuatorCfg(
  target_names_expr=(".*_knee",),
  stiffness=100.0,
  damping=6.663,
  effort_limit=80.0,
  armature=0.04,
  viscous_damping=0.001 * 9.0**2,
)

# AJD10 — hip yaw, ankle pitch.
# ktau=0.241679 Nm/A, resistance=0.25 Ω, gear_ratio=9.0
QDD_AJD10_HIP_YAW_CFG = IdealPdActuatorCfg(
  target_names_expr=(".*_hip_yaw",),
  stiffness=100.0,
  damping=2.332,
  effort_limit=40.0,
  armature=0.02,
  viscous_damping=0.0008 * 9.0**2,
)
QDD_AJD10_ANKLE_PITCH_CFG = IdealPdActuatorCfg(
  target_names_expr=(".*_ankle_pitch",),
  stiffness=40.0,
  damping=1.329,
  effort_limit=50.0,
  armature=0.02,
  viscous_damping=0.0008 * 9.0**2,
)

# AJD8 — ankle roll.
# ktau=0.156364 Nm/A, resistance=0.29 Ω, gear_ratio=7.75
QDD_AJD8_ANKLE_ROLL_CFG = IdealPdActuatorCfg(
  target_names_expr=(".*_ankle_roll",),
  stiffness=40.0,
  damping=0.67036062,
  effort_limit=17.0,
  armature=0.0042,
  viscous_damping=0.0003 * 7.75**2,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.8),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^(l|r)_foot_roll$"

# Keep only foot box geoms as contact surfaces; disable mesh geom contacts.
FOOT_COLLISION = CollisionCfg(
  geom_names_expr=(r".*",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

##
# Final config.
##

QDD_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    QDD_AJD12_HIP_PITCH_CFG,
    QDD_AJD12_HIP_ROLL_CFG,
    QDD_AJD10_HIP_YAW_CFG,
    QDD_AJD12_KNEE_CFG,
    QDD_AJD10_ANKLE_PITCH_CFG,
    QDD_AJD8_ANKLE_ROLL_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_qdd_robot_cfg() -> EntityCfg:
  """Get a fresh QDD robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FOOT_COLLISION,),
    spec_fn=get_spec,
    articulation=QDD_ARTICULATION,
  )


# Action scale and offset matching gb-rl-locomotion's rescale_to_limits.
#
# gb-rl uses EMAJointPositionToLimitsActionCfg(scale=0.5, rescale_to_limits=True)
# which maps actions in [-1, 1] to the middle 50 % of each joint's range:
#   offset = (lower + upper) / 2      (center of joint range)
#   scale  = 0.5 * (upper - lower) / 2  (quarter of full span)
#
# The offset dict is used with use_default_offset=False so the action's
# zero-point sits at the joint-range midpoint (e.g. bent knee), not at
# the MJCF default qpos (straight legs).


def _compute_action_scale_offset() -> tuple[dict[str, float], dict[str, float]]:
  m = mujoco.MjModel.from_xml_path(str(QDD_XML))
  scale: dict[str, float] = {}
  offset: dict[str, float] = {}
  for i in range(m.njnt):
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    if m.jnt_type[i] != 3:  # hinge joints only
      continue
    lo, hi = float(m.jnt_range[i, 0]), float(m.jnt_range[i, 1])
    span = hi - lo
    scale[jname] = 0.5 * span / 2.0
    offset[jname] = (lo + hi) / 2.0
  return scale, offset


QDD_ACTION_SCALE, QDD_ACTION_OFFSET = _compute_action_scale_offset()


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_qdd_robot_cfg())

  viewer.launch(robot.spec.compile())
