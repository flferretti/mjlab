"""Gbionics Gene01 Nohands humanoid constants.

Actuator parameters, joint groups, and robot configuration matching
the gb-rl-locomotion IsaacLab definition (``assets/gene01_nohands.py``).
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

GENE01_NOHANDS_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "gbionics_gene01_nohands"
  / "xmls"
  / "gene01_nohands.xml"
)


def get_spec() -> mujoco.MjSpec:
  """Load and return MuJoCo spec for Gene01 Nohands robot."""
  if not GENE01_NOHANDS_XML.exists():
    raise FileNotFoundError(
      f"Gene01 Nohands model not found at {GENE01_NOHANDS_XML}. "
      "Ensure the robot XML file is present in the asset_zoo."
    )
  spec = mujoco.MjSpec.from_file(str(GENE01_NOHANDS_XML))
  # Resolve meshdir to an absolute path so that mesh loading works after
  # MjSpec.attach() (which loses the original modelfiledir context) and
  # regardless of CWD or broken symlinks.
  meshdir = Path(spec.modelfiledir) / spec.compiler.meshdir
  spec.compiler.meshdir = str(meshdir.resolve())
  _name_collision_geoms(spec)
  return spec


def _name_collision_geoms(spec: mujoco.MjSpec) -> None:
  """Assign names to all geoms for collision configuration.

  This ensures collision configuration can be done by name in the
  CollisionCfg rather than relying on positional indices.
  """
  for body in spec.bodies:
    if body.name == "world":
      continue
    for k, geom in enumerate(body.geoms):
      # Name sole/foot collision geoms for foot height sensors.
      if body.name in ("l_sole", "r_sole"):
        geom.name = f"{body.name}_collision"
      else:
        geom.name = f"{body.name}_visual_{k}"


##
# Joint groups.
##

# All joints in the GENE01_NOHANDS model.
GENE01_NOHANDS_JOINTS: list[str] = [
  "l_hip_pitch",
  "r_hip_pitch",
  "torso_yaw",
  "l_hip_roll",
  "r_hip_roll",
  "torso_roll",
  "l_hip_yaw",
  "r_hip_yaw",
  "l_shoulder_pitch",
  "r_shoulder_pitch",
  "l_knee",
  "r_knee",
  "l_shoulder_roll",
  "r_shoulder_roll",
  "l_ankle_motor_1",
  "l_ankle_motor_2",
  "l_ankle_pitch",
  "r_ankle_motor_1",
  "r_ankle_motor_2",
  "r_ankle_pitch",
  "l_shoulder_yaw",
  "r_shoulder_yaw",
  "l_rod_1_u_0",
  "l_rod_2_u_0",
  "l_ankle_roll",
  "r_rod_1_u_0",
  "r_rod_2_u_0",
  "r_ankle_roll",
  "l_elbow",
  "r_elbow",
  "l_rod_1_u_1",
  "l_rod_2_u_1",
  "r_rod_1_u_1",
  "r_rod_2_u_1",
]

# Actuated joints (controlled by PD actuators).
GENE01_NOHANDS_ACTUATED_JOINTS: list[str] = [
  "l_hip_pitch",
  "r_hip_pitch",
  "torso_yaw",
  "l_hip_roll",
  "r_hip_roll",
  "torso_roll",
  "l_hip_yaw",
  "r_hip_yaw",
  "l_shoulder_pitch",
  "r_shoulder_pitch",
  "l_knee",
  "r_knee",
  "l_shoulder_roll",
  "r_shoulder_roll",
  "l_ankle_motor_1",
  "l_ankle_motor_2",
  "r_ankle_motor_1",
  "r_ankle_motor_2",
  "l_shoulder_yaw",
  "r_shoulder_yaw",
  "l_elbow",
  "r_elbow",
]

# Passive joints (no PD control).
GENE01_NOHANDS_NOT_ACTUATED_JOINTS: list[str] = [
  "l_ankle_pitch",
  "r_ankle_pitch",
  "l_ankle_roll",
  "r_ankle_roll",
]

# Universal joints (passive, with minimal friction).
GENE01_NOHANDS_UNIVERSAL_JOINTS: list[str] = [
  "l_rod_1_u_0",
  "l_rod_1_u_1",
  "l_rod_2_u_0",
  "l_rod_2_u_1",
  "r_rod_1_u_0",
  "r_rod_1_u_1",
  "r_rod_2_u_0",
  "r_rod_2_u_1",
]

# Joint groups by actuator type (matching gb-rl-locomotion).
X6_60_JOINTS: list[str] = [
  "l_shoulder_roll",
  "r_shoulder_roll",
  "l_shoulder_yaw",
  "r_shoulder_yaw",
  "l_elbow",
  "r_elbow",
  "l_ankle_motor_1",
  "l_ankle_motor_2",
  "r_ankle_motor_1",
  "r_ankle_motor_2",
]

X8_120_JOINTS: list[str] = [
  "torso_roll",
  "torso_yaw",
  "l_shoulder_pitch",
  "r_shoulder_pitch",
  "l_hip_yaw",
  "r_hip_yaw",
]

X10_200_JOINTS: list[str] = [
  "l_hip_pitch",
  "r_hip_pitch",
  "l_hip_roll",
  "r_hip_roll",
  "l_knee",
  "r_knee",
]

##
# Actuator configurations.
#
# Three motor types (X6-60, X8-120, X10-200) with per-motor-type PD gains
# from gb-rl-locomotion (gene01_nohands.py).
#
# Motor specs (from gb_rl_locomotion/actuators/myactuator.py):
#   X6-60:   ktau=0.11 Nm/A, resistance=0.41 Ω, gear_ratio=19.612, effort=60 Nm
#   X8-120:  ktau=0.12 Nm/A, resistance=0.18 Ω, gear_ratio=19.612, effort=120 Nm
#   X10-200: ktau=0.26 Nm/A, resistance=0.27 Ω, gear_ratio=20.0, effort=200 Nm
##

# X6-60 — arms and ankle actuators.
GENE01_NOHANDS_X6_60_CFG = IdealPdActuatorCfg(
  target_names_expr=tuple(X6_60_JOINTS),
  stiffness=50.0,
  damping=6.7953,
  effort_limit=60.0,
)

# X8-120 — torso and hip yaw.
GENE01_NOHANDS_X8_120_CFG = IdealPdActuatorCfg(
  target_names_expr=tuple(X8_120_JOINTS),
  stiffness=100.0,
  damping=9.6085,
  effort_limit=120.0,
)

# X10-200 — leg actuators (hips and knees).
GENE01_NOHANDS_X10_200_CFG = IdealPdActuatorCfg(
  target_names_expr=tuple(X10_200_JOINTS),
  stiffness=200.0,
  damping=13.5906,
  effort_limit=200.0,
)

# Passive joints (ankle pitch/roll) — no PD control (zero stiffness/damping).
GENE01_NOHANDS_NOT_ACTUATED_CFG = IdealPdActuatorCfg(
  target_names_expr=tuple(GENE01_NOHANDS_NOT_ACTUATED_JOINTS),
  stiffness=0.0,
  damping=0.0,
)

# Universal joints (rod joints) — passive.
GENE01_NOHANDS_UNIVERSAL_CFG = IdealPdActuatorCfg(
  target_names_expr=tuple(GENE01_NOHANDS_UNIVERSAL_JOINTS),
  stiffness=0.0,
  damping=0.0,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.15),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Enable collision on foot soles only; disable everywhere else.
FOOT_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(l|r)_sole_collision$",),
  contype=0,
  conaffinity=1,
  condim={r"^(l|r)_sole_collision$": 3},
  priority={r"^(l|r)_sole_collision$": 1},
  friction={r"^(l|r)_sole_collision$": (0.6,)},
  solimp={r"^(l|r)_sole_collision$": (0.9, 0.95, 0.023)},
  disable_other_geoms=True,
)

##
# Final config.
##

GENE01_NOHANDS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GENE01_NOHANDS_X6_60_CFG,
    GENE01_NOHANDS_X8_120_CFG,
    GENE01_NOHANDS_X10_200_CFG,
    GENE01_NOHANDS_NOT_ACTUATED_CFG,
    GENE01_NOHANDS_UNIVERSAL_CFG,
  ),
  soft_joint_pos_limit_factor=0.95,
)


def get_gene01_nohands_robot_cfg() -> EntityCfg:
  """Get a fresh Gene01 Nohands robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FOOT_COLLISION,),
    spec_fn=get_spec,
    articulation=GENE01_NOHANDS_ARTICULATION,
  )


# Action scale matching gb-rl-locomotion's rescale_to_limits.
#
# gb-rl uses EMAJointPositionToLimitsActionCfg(scale=0.5, rescale_to_limits=True)
# with soft_joint_pos_limit_factor=0.95, which maps actions in [-1, 1] to:
#   action_scale = 0.25 * effort_limit / stiffness
#
# This matches the gb-rl formula: 0.25 * e[n] / s[n] for each joint n.


def _compute_action_scales() -> dict[str, float]:
  """Compute action scales for each actuated joint.

  The scale is 0.25 * effort_limit / stiffness for each motor type.
  """
  scales: dict[str, float] = {}

  # X6-60
  for joint in X6_60_JOINTS:
    scales[joint] = 0.25 * 60.0 / 50.0

  # X8-120
  for joint in X8_120_JOINTS:
    scales[joint] = 0.25 * 120.0 / 100.0

  # X10-200
  for joint in X10_200_JOINTS:
    scales[joint] = 0.25 * 200.0 / 200.0

  return scales


GENE01_NOHANDS_ACTION_SCALE = _compute_action_scales()


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_gene01_nohands_robot_cfg())
  viewer.launch(robot.spec.compile())
