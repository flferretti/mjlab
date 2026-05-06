"""Sim2sim policy runner using plain MuJoCo (C) + ONNX Runtime.

Replays a trained MJLab policy in a standalone MuJoCo viewer, without
Warp or any training infrastructure.  This is the MJLab equivalent of
gb-rl-locomotion's ``deployment/sim2sim/run_policy.py``.

Usage:
    uv run python scripts/sim2sim.py \
        --onnx logs/rsl_rl/qdd_velocity/.../policy_32000.onnx

Controls (keyboard):
    ↑/↓   : increase/decrease forward velocity
    ←/→   : increase/decrease yaw velocity
    Space  : reset command to zero
    R      : reset robot to initial pose
"""

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

# ─── QDD robot constants (must match training config) ───────────────────────

# gb-lowerbody-models world.xml: includes model with actuators + ground plane.
WORLD_XML_PATH = Path(
  "/home/fferretti/git/gb-lowerbody-models/build/setup_mjcf_staging"
  "/lowerbodyqddbat/world.xml"
)

COMMANDED_JOINTS = [
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

# Observation joint order must match MJLab's Entity internal order (MJCF order:
# all left joints first, then all right joints).  This differs from the action
# order (COMMANDED_JOINTS) which is interleaved L/R.
OBS_JOINT_ORDER = [
  "l_hip_pitch",
  "l_hip_roll",
  "l_hip_yaw",
  "l_knee",
  "l_ankle_pitch",
  "l_ankle_roll",
  "r_hip_pitch",
  "r_hip_roll",
  "r_hip_yaw",
  "r_knee",
  "r_ankle_pitch",
  "r_ankle_roll",
]

# PD gains per joint (same order as COMMANDED_JOINTS).
STIFFNESS = {
  "l_hip_pitch": 100.0, "r_hip_pitch": 100.0,
  "l_hip_roll": 100.0, "r_hip_roll": 100.0,
  "l_hip_yaw": 100.0, "r_hip_yaw": 100.0,
  "l_knee": 100.0, "r_knee": 100.0,
  "l_ankle_pitch": 40.0, "r_ankle_pitch": 40.0,
  "l_ankle_roll": 40.0, "r_ankle_roll": 40.0,
}

DAMPING = {
  "l_hip_pitch": 8.331, "r_hip_pitch": 8.331,
  "l_hip_roll": 9.998, "r_hip_roll": 9.998,
  "l_hip_yaw": 2.332, "r_hip_yaw": 2.332,
  "l_knee": 6.663, "r_knee": 6.663,
  "l_ankle_pitch": 1.329, "r_ankle_pitch": 1.329,
  "l_ankle_roll": 0.670, "r_ankle_roll": 0.670,
}

EFFORT_LIMIT = {
  # Use gb-rl-locomotion's elevated DEPLOYMENT limits (higher than training)
  # for better PD tracking headroom during sim2sim.
  "l_hip_pitch": 110.0, "r_hip_pitch": 110.0,
  "l_hip_roll": 110.0, "r_hip_roll": 110.0,
  "l_hip_yaw": 60.0, "r_hip_yaw": 60.0,
  "l_knee": 110.0, "r_knee": 110.0,
  "l_ankle_pitch": 60.0, "r_ankle_pitch": 60.0,
  "l_ankle_roll": 17.0, "r_ankle_roll": 17.0,
}

# Observation scaling (same as training).
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_JOINT_VEL = 0.05

# Joint dynamics overrides (MJLab overwrites MJCF defaults during scene build).
VISCOUS_DAMPING = {
  "l_hip_pitch": 0.001 * 9.0**2, "r_hip_pitch": 0.001 * 9.0**2,
  "l_hip_roll": 0.001 * 9.0**2, "r_hip_roll": 0.001 * 9.0**2,
  "l_hip_yaw": 0.0008 * 9.0**2, "r_hip_yaw": 0.0008 * 9.0**2,
  "l_knee": 0.001 * 9.0**2, "r_knee": 0.001 * 9.0**2,
  "l_ankle_pitch": 0.0008 * 9.0**2, "r_ankle_pitch": 0.0008 * 9.0**2,
  "l_ankle_roll": 0.0003 * 7.75**2, "r_ankle_roll": 0.0003 * 7.75**2,
}

ARMATURE = {
  "l_hip_pitch": 0.04, "r_hip_pitch": 0.04,
  "l_hip_roll": 0.04, "r_hip_roll": 0.04,
  "l_hip_yaw": 0.02, "r_hip_yaw": 0.02,
  "l_knee": 0.04, "r_knee": 0.04,
  "l_ankle_pitch": 0.02, "r_ankle_pitch": 0.02,
  "l_ankle_roll": 0.0042, "r_ankle_roll": 0.0042,
}

# Action EMA (alpha=1.0 means no smoothing, scale=0.5 for rescale_to_limits).
EMA_ALPHA = 1.0
ACTION_SCALE_FACTOR = 0.5
SOFT_JOINT_POS_LIMIT_FACTOR = 0.9

# Simulation.
SIM_DT = 0.005
DECIMATION = 4
SPAWN_HEIGHT = 0.78
HISTORY_LENGTH = 5


@dataclass
class ObservationHistory:
  """Ring buffer for proprioceptive observation history.

  Mimics MJLab's CircularBuffer backfill: on the first push after creation
  or reset, all history slots are filled with the first real observation
  (not zeros).
  """

  ang_vel: deque  # each entry: (3,)
  gravity: deque  # each entry: (3,)
  joint_pos: deque  # each entry: (12,)
  joint_vel: deque  # each entry: (12,)
  _initialized: bool = False

  @classmethod
  def create(cls, num_joints: int = 12) -> "ObservationHistory":
    return cls(
      ang_vel=deque([np.zeros(3, dtype=np.float32)] * HISTORY_LENGTH,
                    maxlen=HISTORY_LENGTH),
      gravity=deque([np.zeros(3, dtype=np.float32)] * HISTORY_LENGTH,
                    maxlen=HISTORY_LENGTH),
      joint_pos=deque([np.zeros(num_joints, dtype=np.float32)] * HISTORY_LENGTH,
                      maxlen=HISTORY_LENGTH),
      joint_vel=deque([np.zeros(num_joints, dtype=np.float32)] * HISTORY_LENGTH,
                      maxlen=HISTORY_LENGTH),
      _initialized=False,
    )

  def push(
    self,
    ang_vel: np.ndarray,
    gravity: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
  ) -> None:
    a = ang_vel.astype(np.float32)
    g = gravity.astype(np.float32)
    p = joint_pos.astype(np.float32)
    v = joint_vel.astype(np.float32)
    if not self._initialized:
      # Backfill all slots (matches MJLab CircularBuffer behavior on reset).
      for i in range(HISTORY_LENGTH):
        self.ang_vel[i] = a.copy()
        self.gravity[i] = g.copy()
        self.joint_pos[i] = p.copy()
        self.joint_vel[i] = v.copy()
      self._initialized = True
    else:
      self.ang_vel.append(a)
      self.gravity.append(g)
      self.joint_pos.append(p)
      self.joint_vel.append(v)

  def get_flat(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened history arrays (oldest first)."""
    return (
      np.concatenate(list(self.ang_vel)),
      np.concatenate(list(self.gravity)),
      np.concatenate(list(self.joint_pos)),
      np.concatenate(list(self.joint_vel)),
    )

  def get_flat(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened history arrays (oldest first)."""
    return (
      np.concatenate(list(self.ang_vel)),
      np.concatenate(list(self.gravity)),
      np.concatenate(list(self.joint_pos)),
      np.concatenate(list(self.joint_vel)),
    )


class Sim2SimRunner:
  def __init__(self, onnx_path: str, mjcf_path: str | None = None):
    # Load world.xml from gb-lowerbody-models (has actuators + floor).
    xml = mjcf_path or str(WORLD_XML_PATH)
    self.model = mujoco.MjModel.from_xml_path(xml)
    self.model.opt.timestep = SIM_DT
    self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    self.model.opt.impratio = 1

    # Convert motor actuators to position actuators and patch joint dynamics
    # to exactly match MJLab's training configuration.
    for i, jname in enumerate(COMMANDED_JOINTS):
      jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
      aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
      dof = self.model.jnt_dofadr[jid]
      # Joint dynamics.
      self.model.dof_damping[dof] = VISCOUS_DAMPING[jname]
      self.model.dof_armature[dof] = ARMATURE[jname]
      self.model.jnt_stiffness[jid] = 0.0
      # Convert motor → position actuator: force = kp*(ctrl-pos) - kd*vel.
      self.model.actuator_gaintype[aid] = mujoco.mjtGain.mjGAIN_FIXED
      self.model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_AFFINE
      self.model.actuator_gainprm[aid, 0] = STIFFNESS[jname]
      self.model.actuator_biasprm[aid, 1] = -STIFFNESS[jname]
      self.model.actuator_biasprm[aid, 2] = -DAMPING[jname]
      self.model.actuator_ctrllimited[aid] = 0
      self.model.actuator_forcelimited[aid] = 1
      self.model.actuator_forcerange[aid] = [
        -EFFORT_LIMIT[jname], EFFORT_LIMIT[jname]
      ]

    self.data = mujoco.MjData(self.model)

    # Resolve joint indices for OBSERVATIONS (MJCF order: all-L then all-R).
    self.obs_qpos_idx = []
    self.obs_qvel_idx = []
    for jname in OBS_JOINT_ORDER:
      jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
      if jid < 0:
        raise ValueError(f"Joint '{jname}' not found in MJCF")
      self.obs_qpos_idx.append(self.model.jnt_qposadr[jid])
      self.obs_qvel_idx.append(self.model.jnt_dofadr[jid])
    self.obs_qpos_idx = np.array(self.obs_qpos_idx)
    self.obs_qvel_idx = np.array(self.obs_qvel_idx)

    # Actuator indices for ACTIONS (interleaved L/R order).
    self.actuator_idx = np.array([
      mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
      for j in COMMANDED_JOINTS
    ])

    # Action scale/offset (rescale_to_limits with scale=0.5).
    self.action_offset = np.zeros(len(COMMANDED_JOINTS), dtype=np.float32)
    self.action_scale = np.zeros(len(COMMANDED_JOINTS), dtype=np.float32)
    for i, jname in enumerate(COMMANDED_JOINTS):
      jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
      lo = self.model.jnt_range[jid, 0]
      hi = self.model.jnt_range[jid, 1]
      self.action_offset[i] = (lo + hi) / 2.0
      self.action_scale[i] = (
        ACTION_SCALE_FACTOR * (hi - lo) * SOFT_JOINT_POS_LIMIT_FACTOR / 2.0
      )

    # ONNX session.
    self.session = ort.InferenceSession(onnx_path)
    self.input_name = self.session.get_inputs()[0].name
    expected_dim = self.session.get_inputs()[0].shape[1]
    print(f"ONNX model loaded: input dim = {expected_dim}")

    # State.
    self.history = ObservationHistory.create(len(COMMANDED_JOINTS))
    self.last_action = np.zeros(len(COMMANDED_JOINTS), dtype=np.float32)
    self.command = np.zeros(3, dtype=np.float32)  # [vx, vy, yaw_rate]

  def reset(self):
    """Reset robot to standing pose."""
    mujoco.mj_resetData(self.model, self.data)
    # Set upright orientation [w,x,y,z] and spawn height.
    self.data.qpos[0:3] = [0.0, 0.0, SPAWN_HEIGHT]
    self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # upright (override MJCF default)
    mujoco.mj_forward(self.model, self.data)
    self.history = ObservationHistory.create(len(COMMANDED_JOINTS))
    self.last_action = np.zeros(len(COMMANDED_JOINTS), dtype=np.float32)

  def _get_obs(self) -> np.ndarray:
    """Build the flat observation vector matching training (168 dims)."""
    # MuJoCo free-joint: qvel[3:6] is body-frame angular velocity,
    # but qvel[0:3] is world-frame linear velocity.
    ang_vel_body = self.data.qvel[3:6].astype(np.float32)

    base_quat = self.data.qpos[3:7]  # [w, x, y, z]
    rot = Rotation.from_quat(base_quat, scalar_first=True)
    lin_vel_body = rot.apply(self.data.qvel[0:3], inverse=True).astype(np.float32)

    # Projected gravity (world gravity rotated into body frame).
    gravity_body = rot.apply([0, 0, -1], inverse=True).astype(np.float32)

    # Joint positions and velocities (in MJCF/observation order).
    joint_pos = self.data.qpos[self.obs_qpos_idx].astype(np.float32)
    joint_vel = self.data.qvel[self.obs_qvel_idx].astype(np.float32)

    # Push to history.
    self.history.push(ang_vel_body, gravity_body, joint_pos, joint_vel)

    # Build observation: [lin_vel(3), ang_vel_hist(15), grav_hist(15),
    #                     jpos_hist(60), jvel_hist(60), last_action(12), cmd(3)]
    ang_hist, grav_hist, jpos_hist, jvel_hist = self.history.get_flat()

    obs = np.concatenate([
      lin_vel_body,                      # 3
      ang_hist * OBS_SCALE_ANG_VEL,     # 15
      grav_hist,                         # 15
      jpos_hist,                         # 60
      jvel_hist * OBS_SCALE_JOINT_VEL,  # 60
      self.last_action,                  # 12
      self.command,                      # 3
    ]).astype(np.float32)

    return obs

  def _infer(self, obs: np.ndarray) -> np.ndarray:
    """Run ONNX inference."""
    result = self.session.run(None, {self.input_name: obs[np.newaxis, :]})
    return result[0][0]  # (12,)

  def _apply_action(self, raw_action: np.ndarray):
    """Apply action: set position targets on actuators."""
    # EMA filter (alpha=1.0 means pass-through).
    smoothed = EMA_ALPHA * raw_action + (1.0 - EMA_ALPHA) * self.last_action
    self.last_action = smoothed.copy()

    # Convert to joint target positions.
    target_pos = self.action_offset + self.action_scale * smoothed

    # Position actuators: ctrl = target position. Step multiple substeps.
    for _ in range(DECIMATION):
      self.data.ctrl[self.actuator_idx] = target_pos
      mujoco.mj_step(self.model, self.data)

  def _key_callback(self, keycode: int):
    step = 0.1
    if keycode == 265:  # UP
      self.command[0] = min(self.command[0] + step, 1.0)
    elif keycode == 264:  # DOWN
      self.command[0] = max(self.command[0] - step, -1.0)
    elif keycode == 263:  # LEFT
      self.command[2] = min(self.command[2] + step, 1.0)
    elif keycode == 262:  # RIGHT
      self.command[2] = max(self.command[2] - step, -1.0)
    elif keycode == 32:  # SPACE
      self.command[:] = 0.0
    elif keycode == 82:  # R
      self.reset()

  def run(self):
    self.reset()
    with mujoco.viewer.launch_passive(
      self.model, self.data, key_callback=self._key_callback
    ) as viewer:
      print("Controls: ↑↓ forward, ←→ yaw, Space=stop, R=reset")
      print(f"Command: [vx={self.command[0]:.1f}, vy={self.command[1]:.1f},"
            f" yaw={self.command[2]:.1f}]")
      while viewer.is_running():
        t0 = time.time()

        obs = self._get_obs()
        action = self._infer(obs)
        self._apply_action(action)

        # Camera follows robot.
        viewer.cam.lookat[:] = self.data.qpos[:3]
        viewer.sync()

        # Realtime pacing.
        elapsed = time.time() - t0
        target = SIM_DT * DECIMATION
        if elapsed < target:
          time.sleep(target - elapsed)


def main():
  parser = argparse.ArgumentParser(description="Sim2sim: MuJoCo C + ONNX")
  parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX policy")
  parser.add_argument("--mjcf", type=str, default=None, help="Override MJCF path")
  args = parser.parse_args()

  if not Path(args.onnx).exists():
    raise FileNotFoundError(f"ONNX file not found: {args.onnx}")

  runner = Sim2SimRunner(args.onnx, args.mjcf)
  runner.run()


if __name__ == "__main__":
  main()
