"""Sim2sim policy runner using plain MuJoCo (C) + ONNX Runtime.

Replays a trained MJLab policy in a standalone MuJoCo viewer, without
Warp or any training infrastructure.  This is the MJLab equivalent of
gb-rl-locomotion's ``deployment/sim2sim/run_policy.py``.

Usage (interactive viewer):
    uv run python scripts/sim2sim.py \
        --onnx logs/rsl_rl/qdd_velocity/.../policy_32000.onnx

Usage (minimal headless check with fixed command):
    uv run python scripts/sim2sim.py \
        --onnx ./policy_62000.onnx \
        --headless --steps 2000 --cmd-x 0.8

Controls (keyboard):
    ↑/↓   : increase/decrease forward velocity
    ←/→   : increase/decrease yaw velocity
    Space  : reset command to zero
    R      : reset robot to initial pose
"""

import argparse
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

# ─── QDD robot constants (must match training config) ───────────────────────

# Preferred world from gb-lowerbody-models (includes terrain), with local fallback.
DEFAULT_MJCF_CANDIDATES = [
  Path(
    "/home/fferretti/git/gb-lowerbody-models/build/setup_mjcf_staging"
    "/lowerbodyqddbat/world.xml"
  ),
  Path(
    "/home/fferretti/git/gb-lowerbody-models/share/lowerbody/robots/lowerbodyqdd/world.xml"
  ),
  Path(
    "/home/fferretti/git/mjlab/src/mjlab/asset_zoo/robots/gbionics_qdd/xmls/qdd.xml"
  ),
]

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
  "l_hip_pitch": 100.0,
  "r_hip_pitch": 100.0,
  "l_hip_roll": 100.0,
  "r_hip_roll": 100.0,
  "l_hip_yaw": 100.0,
  "r_hip_yaw": 100.0,
  "l_knee": 100.0,
  "r_knee": 100.0,
  "l_ankle_pitch": 40.0,
  "r_ankle_pitch": 40.0,
  "l_ankle_roll": 40.0,
  "r_ankle_roll": 40.0,
}

DAMPING = {
  "l_hip_pitch": 8.331,
  "r_hip_pitch": 8.331,
  "l_hip_roll": 9.998,
  "r_hip_roll": 9.998,
  "l_hip_yaw": 2.332,
  "r_hip_yaw": 2.332,
  "l_knee": 6.663,
  "r_knee": 6.663,
  "l_ankle_pitch": 1.329,
  "r_ankle_pitch": 1.329,
  "l_ankle_roll": 0.670,
  "r_ankle_roll": 0.670,
}

EFFORT_LIMIT = {
  # Use gb-rl-locomotion's elevated DEPLOYMENT limits (higher than training)
  # for better PD tracking headroom during sim2sim.
  "l_hip_pitch": 110.0,
  "r_hip_pitch": 110.0,
  "l_hip_roll": 110.0,
  "r_hip_roll": 110.0,
  "l_hip_yaw": 60.0,
  "r_hip_yaw": 60.0,
  "l_knee": 110.0,
  "r_knee": 110.0,
  "l_ankle_pitch": 60.0,
  "r_ankle_pitch": 60.0,
  "l_ankle_roll": 17.0,
  "r_ankle_roll": 17.0,
}

# Observation scaling (same as training).
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_JOINT_VEL = 0.05

# Joint dynamics overrides (MJLab overwrites MJCF defaults during scene build).
VISCOUS_DAMPING = {
  "l_hip_pitch": 0.001 * 9.0**2,
  "r_hip_pitch": 0.001 * 9.0**2,
  "l_hip_roll": 0.001 * 9.0**2,
  "r_hip_roll": 0.001 * 9.0**2,
  "l_hip_yaw": 0.0008 * 9.0**2,
  "r_hip_yaw": 0.0008 * 9.0**2,
  "l_knee": 0.001 * 9.0**2,
  "r_knee": 0.001 * 9.0**2,
  "l_ankle_pitch": 0.0008 * 9.0**2,
  "r_ankle_pitch": 0.0008 * 9.0**2,
  "l_ankle_roll": 0.0003 * 7.75**2,
  "r_ankle_roll": 0.0003 * 7.75**2,
}

ARMATURE = {
  "l_hip_pitch": 0.04,
  "r_hip_pitch": 0.04,
  "l_hip_roll": 0.04,
  "r_hip_roll": 0.04,
  "l_hip_yaw": 0.02,
  "r_hip_yaw": 0.02,
  "l_knee": 0.04,
  "r_knee": 0.04,
  "l_ankle_pitch": 0.02,
  "r_ankle_pitch": 0.02,
  "l_ankle_roll": 0.0042,
  "r_ankle_roll": 0.0042,
}

# Action EMA (alpha=1.0 means no smoothing, scale=0.5 for rescale_to_limits).
EMA_ALPHA = 1.0
ACTION_SCALE_FACTOR = 0.5
SOFT_JOINT_POS_LIMIT_FACTOR = 0.9

# Simulation.
SIM_DT = 0.005
DECIMATION = 4
# gb-rl-locomotion lowerbodyqdd amp config uses spawn_height = 1.0.
SPAWN_HEIGHT = 1.0
HISTORY_LENGTH = 5
COMMAND_FILTER_ALPHA = 0.01
GBRL_ANG_VEL_HISTORY = 5
GBRL_GRAVITY_HISTORY = 5
GBRL_JOINT_POS_HISTORY = 5
GBRL_JOINT_VEL_HISTORY = 5
DEFAULT_TAU_OBS_RANGE = (5.0, 90.0)
DEFAULT_VIDEO_FOLLOW_DISTANCE = 4.5
DEFAULT_VIDEO_FOLLOW_HEIGHT = 0.15
DEFAULT_VIDEO_FOLLOW_SMOOTHING = 0.15
DEFAULT_VIDEO_FOLLOW_AZIMUTH = 130.0
DEFAULT_VIDEO_FOLLOW_ELEVATION = -18.0


def _ensure_runtime_mjcf(xml_path: str) -> tuple[str, bool]:
  """Return an MJCF path with named actuators available for sim2sim."""
  model = mujoco.MjModel.from_xml_path(xml_path)
  if model.nu > 0:
    return xml_path, False

  src = Path(xml_path)
  tree = ET.parse(src)
  root = tree.getroot()

  compiler = root.find("compiler")
  if compiler is not None:
    meshdir = compiler.get("meshdir")
    if meshdir and not Path(meshdir).is_absolute():
      compiler.set("meshdir", str((src.parent / meshdir).resolve()))

  worldbody = root.find("worldbody")
  if worldbody is None:
    raise ValueError(f"Invalid MJCF (missing worldbody): {xml_path}")

  has_plane = any(geom.get("type") == "plane" for geom in worldbody.findall("geom"))
  if not has_plane:
    worldbody.insert(
      0,
      ET.Element(
        "geom",
        {
          "name": "sim2sim_ground",
          "type": "plane",
          "size": "0 0 0.1",
          "pos": "0 0 0",
          "rgba": "0.85 0.85 0.85 1",
          "friction": "1.0 0.005 0.0001",
        },
      ),
    )

  actuator = root.find("actuator")
  if actuator is None:
    actuator = ET.SubElement(root, "actuator")

  if len(list(actuator)) == 0:
    for jname in COMMANDED_JOINTS:
      ET.SubElement(
        actuator,
        "motor",
        {
          "name": jname,
          "joint": jname,
          "forcelimited": "true",
          "forcerange": f"{-EFFORT_LIMIT[jname]} {EFFORT_LIMIT[jname]}",
          "ctrllimited": "false",
        },
      )

  with tempfile.NamedTemporaryFile(
    mode="wb", suffix=".xml", prefix="sim2sim_runtime_", delete=False
  ) as f:
    tree.write(f, encoding="utf-8", xml_declaration=True)
    return f.name, True


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
      ang_vel=deque(
        [np.zeros(3, dtype=np.float32)] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH
      ),
      gravity=deque(
        [np.zeros(3, dtype=np.float32)] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH
      ),
      joint_pos=deque(
        [np.zeros(num_joints, dtype=np.float32)] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH
      ),
      joint_vel=deque(
        [np.zeros(num_joints, dtype=np.float32)] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH
      ),
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


@dataclass
class GbrlBlindObservationHistory:
  """History buffers matching gb-rl-locomotion Network input packing."""

  ang_vel: deque  # each entry: (3,)
  gravity: deque  # each entry: (3,)
  joint_pos: deque  # each entry: (12,)
  joint_vel: deque  # each entry: (12,)
  old_policy: deque  # each entry: (12,)
  command: deque  # each entry: (3,)
  ang_hist: int
  grav_hist: int
  joint_pos_hist: int
  joint_vel_hist: int
  old_policy_hist: int
  command_hist: int
  _initialized: bool = False

  @classmethod
  def create(
    cls,
    num_joints: int,
    old_policy_hist: int,
    command_hist: int,
  ) -> "GbrlBlindObservationHistory":
    return cls(
      ang_vel=deque(
        [np.zeros(3, dtype=np.float32)] * GBRL_ANG_VEL_HISTORY,
        maxlen=GBRL_ANG_VEL_HISTORY,
      ),
      gravity=deque(
        [np.zeros(3, dtype=np.float32)] * GBRL_GRAVITY_HISTORY,
        maxlen=GBRL_GRAVITY_HISTORY,
      ),
      joint_pos=deque(
        [np.zeros(num_joints, dtype=np.float32)] * GBRL_JOINT_POS_HISTORY,
        maxlen=GBRL_JOINT_POS_HISTORY,
      ),
      joint_vel=deque(
        [np.zeros(num_joints, dtype=np.float32)] * GBRL_JOINT_VEL_HISTORY,
        maxlen=GBRL_JOINT_VEL_HISTORY,
      ),
      old_policy=deque(
        [np.zeros(num_joints, dtype=np.float32)] * old_policy_hist,
        maxlen=old_policy_hist,
      ),
      command=deque(
        [np.zeros(3, dtype=np.float32)] * command_hist,
        maxlen=command_hist,
      ),
      ang_hist=GBRL_ANG_VEL_HISTORY,
      grav_hist=GBRL_GRAVITY_HISTORY,
      joint_pos_hist=GBRL_JOINT_POS_HISTORY,
      joint_vel_hist=GBRL_JOINT_VEL_HISTORY,
      old_policy_hist=old_policy_hist,
      command_hist=command_hist,
      _initialized=False,
    )

  def push(
    self,
    ang_vel: np.ndarray,
    gravity: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    old_policy: np.ndarray,
    command: np.ndarray,
  ) -> None:
    a = ang_vel.astype(np.float32)
    g = gravity.astype(np.float32)
    p = joint_pos.astype(np.float32)
    v = joint_vel.astype(np.float32)
    o = old_policy.astype(np.float32)
    c = command.astype(np.float32)
    if not self._initialized:
      for i in range(self.ang_hist):
        self.ang_vel[i] = a.copy()
      for i in range(self.grav_hist):
        self.gravity[i] = g.copy()
      for i in range(self.joint_pos_hist):
        self.joint_pos[i] = p.copy()
      for i in range(self.joint_vel_hist):
        self.joint_vel[i] = v.copy()
      for i in range(self.old_policy_hist):
        self.old_policy[i] = o.copy()
      for i in range(self.command_hist):
        self.command[i] = c.copy()
      self._initialized = True
    else:
      self.ang_vel.append(a)
      self.gravity.append(g)
      self.joint_pos.append(p)
      self.joint_vel.append(v)
      self.old_policy.append(o)
      self.command.append(c)

  def get_flat(
    self,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
      np.concatenate(list(self.ang_vel)),
      np.concatenate(list(self.gravity)),
      np.concatenate(list(self.joint_pos)),
      np.concatenate(list(self.joint_vel)),
      np.concatenate(list(self.old_policy)),
      np.concatenate(list(self.command)),
    )


class Sim2SimRunner:
  def __init__(
    self,
    onnx_path: str,
    mjcf_path: str | None = None,
    obs_layout: str = "auto",
    tau_obs_range: tuple[float, float] = DEFAULT_TAU_OBS_RANGE,
    tau_obs_nm: np.ndarray | None = None,
  ):
    # Resolve model path (explicit --mjcf, otherwise best available candidate).
    if mjcf_path is not None:
      xml = mjcf_path
    else:
      xml = None
      for candidate in DEFAULT_MJCF_CANDIDATES:
        if candidate.exists():
          xml = str(candidate)
          break
      if xml is None:
        raise FileNotFoundError(
          "No default MJCF found. Pass --mjcf explicitly. Checked:\n"
          + "\n".join(str(p) for p in DEFAULT_MJCF_CANDIDATES)
        )
    runtime_xml, generated_xml = _ensure_runtime_mjcf(xml)
    self._generated_xml_path = runtime_xml if generated_xml else None
    print(f"Using MJCF: {runtime_xml}")
    self.model = mujoco.MjModel.from_xml_path(runtime_xml)
    self.model.opt.timestep = SIM_DT

    # Patch joint dynamics and actuator force limits.
    self.cmd_qpos_idx = []
    self.cmd_qvel_idx = []
    for jname in COMMANDED_JOINTS:
      jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
      aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
      if jid < 0:
        raise ValueError(f"Joint '{jname}' not found in MJCF")
      if aid < 0:
        raise ValueError(
          f"Actuator '{jname}' not found. Provide an MJCF with named actuators "
          "or use the default runtime-generated model."
        )
      # Keep actuator model as-is (gb-rl style controller computes torques).
      self.model.actuator_ctrllimited[aid] = 0
      self.model.actuator_forcelimited[aid] = 1
      self.model.actuator_forcerange[aid] = [-EFFORT_LIMIT[jname], EFFORT_LIMIT[jname]]
      self.cmd_qpos_idx.append(self.model.jnt_qposadr[jid])
      self.cmd_qvel_idx.append(self.model.jnt_dofadr[jid])
    self.cmd_qpos_idx = np.array(self.cmd_qpos_idx)
    self.cmd_qvel_idx = np.array(self.cmd_qvel_idx)

    self.data = mujoco.MjData(self.model)
    self.imu_quat_slice = self._sensor_slice("imu_frame_framequat")
    self.imu_ang_vel_slice = self._sensor_slice("imu_ang_vel")
    if self.imu_ang_vel_slice is None:
      self.imu_ang_vel_slice = self._sensor_slice("imu_frame_gyro")

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
    self.actuator_idx = np.array(
      [
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
        for j in COMMANDED_JOINTS
      ]
    )
    self.kp = np.array([STIFFNESS[j] for j in COMMANDED_JOINTS], dtype=np.float32)
    self.kd = np.array([DAMPING[j] for j in COMMANDED_JOINTS], dtype=np.float32)
    self.effort_limit = np.array(
      [EFFORT_LIMIT[j] for j in COMMANDED_JOINTS], dtype=np.float32
    )
    tau_lo, tau_hi = float(tau_obs_range[0]), float(tau_obs_range[1])
    if tau_hi <= tau_lo:
      raise ValueError(
        f"Invalid tau observation range: ({tau_lo}, {tau_hi}). Expected max > min."
      )
    self.tau_obs_range = (tau_lo, tau_hi)
    if tau_obs_nm is None:
      tau_nm = self.effort_limit.copy()
    else:
      tau_nm = np.asarray(tau_obs_nm, dtype=np.float32)
      if tau_nm.shape != (len(COMMANDED_JOINTS),):
        raise ValueError(
          f"tau_obs_nm must have {len(COMMANDED_JOINTS)} values in commanded "
          f"joint order, got shape {tau_nm.shape}."
        )
    self.motor_tau_obs = np.clip((tau_nm - tau_lo) / (tau_hi - tau_lo), 0.0, 1.0)
    self.include_motor_tau_obs = False

    # Action scale/offset (rescale_to_limits with scale=0.5).
    self.action_offset = np.zeros(len(COMMANDED_JOINTS), dtype=np.float32)
    self.action_scale = np.zeros(len(COMMANDED_JOINTS), dtype=np.float32)
    self.soft_lower = np.zeros(len(COMMANDED_JOINTS), dtype=np.float32)
    self.soft_upper = np.zeros(len(COMMANDED_JOINTS), dtype=np.float32)
    for i, jname in enumerate(COMMANDED_JOINTS):
      jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
      lo = self.model.jnt_range[jid, 0]
      hi = self.model.jnt_range[jid, 1]
      center = (lo + hi) / 2.0
      half_range = (hi - lo) / 2.0
      self.action_offset[i] = center
      self.action_scale[i] = ACTION_SCALE_FACTOR * half_range
      self.soft_lower[i] = center - SOFT_JOINT_POS_LIMIT_FACTOR * half_range
      self.soft_upper[i] = center + SOFT_JOINT_POS_LIMIT_FACTOR * half_range

    # ONNX session.
    self.session = ort.InferenceSession(onnx_path)
    self.input_name = self.session.get_inputs()[0].name
    self.expected_obs_dim = int(self.session.get_inputs()[0].shape[1])
    print(f"ONNX model loaded: input dim = {self.expected_obs_dim}")

    # Observation layout selection.
    self.obs_layout = "pad"
    self.old_policy_hist = 1
    self.command_hist = 1
    self._resolve_obs_layout(obs_layout)

    # State.
    self.history = ObservationHistory.create(len(COMMANDED_JOINTS))
    self.gbrl_history = GbrlBlindObservationHistory.create(
      len(COMMANDED_JOINTS), self.old_policy_hist, self.command_hist
    )
    self.last_action = np.zeros(len(COMMANDED_JOINTS), dtype=np.float32)
    self.last_policy_output = np.zeros(len(COMMANDED_JOINTS), dtype=np.float32)
    self.command = np.zeros(3, dtype=np.float32)  # [vx, vy, yaw_rate]
    self.filtered_command = np.zeros(3, dtype=np.float32)  # gb-rl style LPF command

  def __del__(self):
    if self._generated_xml_path is not None:
      Path(self._generated_xml_path).unlink(missing_ok=True)

  def _sensor_slice(self, sensor_name: str) -> slice | None:
    sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sid < 0:
      return None
    adr = int(self.model.sensor_adr[sid])
    dim = int(self.model.sensor_dim[sid])
    return slice(adr, adr + dim)

  def _fit_gbrl_blind_layout(self, expected_dim: int) -> tuple[int, int, bool] | None:
    base = (
      3 * GBRL_ANG_VEL_HISTORY
      + 3 * GBRL_GRAVITY_HISTORY
      + len(COMMANDED_JOINTS) * GBRL_JOINT_POS_HISTORY
      + len(COMMANDED_JOINTS) * GBRL_JOINT_VEL_HISTORY
    )
    best: tuple[int, int, bool] | None = None
    best_score: tuple[int, int, int] | None = None
    for with_tau in (True, False):
      extra = len(COMMANDED_JOINTS) if with_tau else 0
      for cmd_hist in range(1, 6):
        rem = expected_dim - base - extra - 3 * cmd_hist
        if rem <= 0 or rem % len(COMMANDED_JOINTS) != 0:
          continue
        old_hist = rem // len(COMMANDED_JOINTS)
        if old_hist < 1:
          continue
        score = (
          abs(old_hist - 1) + abs(cmd_hist - 1),
          0 if with_tau else 1,
          old_hist,
        )
        if best_score is None or score < best_score:
          best_score = score
          best = (old_hist, cmd_hist, with_tau)
    return best

  def _resolve_obs_layout(self, requested_layout: str) -> None:
    """Resolve observation packing mode from ONNX size + user preference."""
    valid_layouts = {"auto", "mjlab-168", "gbrl-blind", "pad"}
    if requested_layout not in valid_layouts:
      raise ValueError(
        f"Unknown obs layout '{requested_layout}'. Expected one of: "
        f"{', '.join(sorted(valid_layouts))}"
      )

    if requested_layout == "mjlab-168":
      self.obs_layout = "mjlab-168"
      print("Observation layout: mjlab-168")
      return

    if requested_layout == "gbrl-blind":
      self.obs_layout = "gbrl-blind"
      fit = self._fit_gbrl_blind_layout(self.expected_obs_dim)
      if fit is None:
        raise ValueError(
          "gbrl-blind layout requires obs dim compatible with "
          "ang/gravity/joint histories + old_policy history + command history "
          "(optionally + motor_tau_max vector). "
          f"Got ONNX dim={self.expected_obs_dim}."
        )
      self.old_policy_hist, self.command_hist, self.include_motor_tau_obs = fit
      print(
        "Observation layout: gbrl-blind "
        f"(old_policy_history={self.old_policy_hist}, "
        f"command_history={self.command_hist}, "
        f"motor_tau_obs={self.include_motor_tau_obs})"
      )
      if self.include_motor_tau_obs:
        print(
          "motor_tau_max normalization range: "
          f"[{self.tau_obs_range[0]:.1f}, {self.tau_obs_range[1]:.1f}] Nm"
        )
      return

    if requested_layout == "pad":
      self.obs_layout = "pad"
      print("Observation layout: pad (zero-pad/truncate)")
      return

    # auto
    if self.expected_obs_dim == 168:
      self.obs_layout = "mjlab-168"
      print("Observation layout: auto → mjlab-168")
      return

    fit = self._fit_gbrl_blind_layout(self.expected_obs_dim)
    if fit is not None:
      self.obs_layout = "gbrl-blind"
      self.old_policy_hist, self.command_hist, self.include_motor_tau_obs = fit
      print(
        "Observation layout: auto → gbrl-blind "
        f"(old_policy_history={self.old_policy_hist}, "
        f"command_history={self.command_hist}, "
        f"motor_tau_obs={self.include_motor_tau_obs})"
      )
      if self.include_motor_tau_obs:
        print(
          "motor_tau_max normalization range: "
          f"[{self.tau_obs_range[0]:.1f}, {self.tau_obs_range[1]:.1f}] Nm"
        )
      return

    self.obs_layout = "pad"
    print(
      "Observation layout: auto → pad (zero-pad/truncate fallback). "
      "Built obs dims are 168 (mjlab-168) or gb-rl blind variants "
      "(with/without motor_tau_max); "
      f"ONNX expects {self.expected_obs_dim}."
    )

  def reset(self):
    """Reset robot to standing pose."""
    mujoco.mj_resetData(self.model, self.data)
    # Set upright orientation [w,x,y,z] and spawn height.
    self.data.qpos[0:3] = [0.0, 0.0, SPAWN_HEIGHT]
    self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # upright (override MJCF default)
    mujoco.mj_forward(self.model, self.data)
    self.history = ObservationHistory.create(len(COMMANDED_JOINTS))
    self.gbrl_history = GbrlBlindObservationHistory.create(
      len(COMMANDED_JOINTS), self.old_policy_hist, self.command_hist
    )
    self.last_action = np.zeros(len(COMMANDED_JOINTS), dtype=np.float32)
    self.last_policy_output = np.zeros(len(COMMANDED_JOINTS), dtype=np.float32)
    self.filtered_command = np.zeros(3, dtype=np.float32)

  def _update_filtered_command(self):
    """Match gb-rl-locomotion command smoothing: v_des = 0.99*v_des + 0.01*cmd."""
    self.filtered_command = (
      (1.0 - COMMAND_FILTER_ALPHA) * self.filtered_command
      + COMMAND_FILTER_ALPHA * self.command
    ).astype(np.float32)

  def _get_obs(self) -> np.ndarray:
    """Build the flat observation vector according to selected layout."""
    base_quat = self.data.qpos[3:7]  # [w, x, y, z]
    rot = Rotation.from_quat(base_quat, scalar_first=True)
    lin_vel_body = rot.apply(self.data.qvel[0:3], inverse=True).astype(np.float32)

    # Defaults from base free-joint kinematics.
    ang_vel_body = self.data.qvel[3:6].astype(np.float32)
    gravity_body = rot.apply([0, 0, -1], inverse=True).astype(np.float32)

    if self.obs_layout == "gbrl-blind":
      # gb-rl uses IMU sensor signals (gyro + projected gravity from IMU quat).
      if self.imu_quat_slice is not None and self.imu_ang_vel_slice is not None:
        imu_quat = self.data.sensordata[self.imu_quat_slice].astype(np.float32)
        imu_ang = self.data.sensordata[self.imu_ang_vel_slice].astype(np.float32)
        if imu_quat.shape[0] == 4 and imu_ang.shape[0] == 3:
          imu_rot = Rotation.from_quat(imu_quat, scalar_first=True)
          ang_vel_body = imu_ang
          gravity_body = imu_rot.apply([0, 0, -1], inverse=True).astype(np.float32)

      # gb-rl sim2sim feeds joints in commanded actuator order (interleaved L/R).
      joint_pos = self.data.qpos[self.cmd_qpos_idx].astype(np.float32)
      joint_vel = self.data.qvel[self.cmd_qvel_idx].astype(np.float32)
      self.gbrl_history.push(
        ang_vel_body,
        gravity_body,
        joint_pos,
        joint_vel,
        self.last_policy_output,
        self.filtered_command,
      )
      ang_hist, grav_hist, jpos_hist, jvel_hist, old_pol_hist, cmd_hist = (
        self.gbrl_history.get_flat()
      )
      # gb-rl Network input order:
      # [imu_ang_vel_hist, projected_gravity_hist, joint_pos_hist, joint_vel_hist,
      #  old_policy_hist, command_hist]
      blocks = [
        ang_hist * OBS_SCALE_ANG_VEL,
        grav_hist,
        jpos_hist,
        jvel_hist * OBS_SCALE_JOINT_VEL,
        old_pol_hist,
        cmd_hist,
      ]
      if self.include_motor_tau_obs:
        blocks.append(self.motor_tau_obs)
      obs = np.concatenate(blocks).astype(np.float32)
    else:
      # mjlab actor observation uses entity joint order (all left, then right).
      joint_pos = self.data.qpos[self.obs_qpos_idx].astype(np.float32)
      joint_vel = self.data.qvel[self.obs_qvel_idx].astype(np.float32)
      self.history.push(ang_vel_body, gravity_body, joint_pos, joint_vel)
      # mjlab actor flat observation:
      # [lin_vel(3), ang_vel_hist(15), grav_hist(15), jpos_hist(60),
      #  jvel_hist(60), last_action(12), cmd(3)].
      ang_hist, grav_hist, jpos_hist, jvel_hist = self.history.get_flat()
      obs = np.concatenate(
        [
          lin_vel_body,
          ang_hist * OBS_SCALE_ANG_VEL,
          grav_hist,
          jpos_hist,
          jvel_hist * OBS_SCALE_JOINT_VEL,
          self.last_action,
          self.filtered_command,
        ]
      ).astype(np.float32)

    return obs

  def _infer(self, obs: np.ndarray) -> np.ndarray:
    """Run ONNX inference."""
    if self.obs_layout == "pad" and obs.shape[0] < self.expected_obs_dim:
      obs = np.pad(
        obs,
        (0, self.expected_obs_dim - obs.shape[0]),
        mode="constant",
      )
    elif self.obs_layout == "pad" and obs.shape[0] > self.expected_obs_dim:
      obs = obs[: self.expected_obs_dim]
    elif self.obs_layout != "pad" and obs.shape[0] != self.expected_obs_dim:
      raise ValueError(
        f"Observation dim mismatch for layout '{self.obs_layout}': built "
        f"{obs.shape[0]}, expected {self.expected_obs_dim}. Use --obs-layout pad "
        "or select the correct layout."
      )
    result = self.session.run(None, {self.input_name: obs[np.newaxis, :]})
    return result[0][0]  # (12,)

  def _apply_action(self, raw_action: np.ndarray):
    """Apply action using gb-rl-style EMA + PD torque control."""
    # EMA filter (alpha=1.0 means pass-through).
    smoothed = (
      EMA_ALPHA * raw_action.astype(np.float32) + (1.0 - EMA_ALPHA) * self.last_action
    )
    self.last_action = smoothed.copy()

    # Convert to joint target positions.
    target_pos = self.action_offset + self.action_scale * smoothed
    target_pos = np.clip(target_pos, self.soft_lower, self.soft_upper)

    # Compute torques with PD and apply to motor actuators (gb-rl sim2sim style).
    for _ in range(DECIMATION):
      joint_pos = self.data.qpos[self.cmd_qpos_idx].astype(np.float32)
      joint_vel = self.data.qvel[self.cmd_qvel_idx].astype(np.float32)
      torques = self.kp * (target_pos - joint_pos) - self.kd * joint_vel
      torques = np.clip(torques, -self.effort_limit, self.effort_limit)
      self.data.ctrl[self.actuator_idx] = torques
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

  def _step_once(self):
    """One policy-control step (decimated internally)."""
    self._update_filtered_command()
    obs = self._get_obs()
    action = self._infer(obs)
    # gb-rl Network stores raw policy output as old_policy history for next step.
    self.last_policy_output = action.astype(np.float32).copy()
    self._apply_action(action)

  def run_headless(
    self,
    steps: int,
    cmd_x: float,
    cmd_y: float,
    cmd_yaw: float,
    video_path: str | None = None,
    video_fps: int = 50,
    video_width: int = 1280,
    video_height: int = 720,
    video_camera: str | None = None,
    video_follow_subject: bool = True,
    video_follow_distance: float = DEFAULT_VIDEO_FOLLOW_DISTANCE,
    video_follow_height: float = DEFAULT_VIDEO_FOLLOW_HEIGHT,
    video_follow_smoothing: float = DEFAULT_VIDEO_FOLLOW_SMOOTHING,
    video_follow_azimuth: float = DEFAULT_VIDEO_FOLLOW_AZIMUTH,
    video_follow_elevation: float = DEFAULT_VIDEO_FOLLOW_ELEVATION,
  ):
    """Minimal MuJoCo-C sim2sim check without viewer.

    Useful to quickly test whether an ONNX policy transfers at all.
    """
    self.reset()
    self.command[:] = [cmd_x, cmd_y, cmd_yaw]

    lin_vel_x_body = []
    base_heights = []
    action_norms = []

    renderer = None
    camera = -1
    follow_camera = None
    follow_lookat = None
    writer_ctx = nullcontext()
    if video_path is not None:
      try:
        import mediapy as media
      except ImportError as exc:
        raise RuntimeError(
          "Video recording requires mediapy. Run with: "
          "uv run --with mediapy python scripts/sim2sim.py ..."
        ) from exc

      max_width = int(self.model.vis.global_.offwidth)
      max_height = int(self.model.vis.global_.offheight)
      if video_width > max_width or video_height > max_height:
        print(
          f"Requested video size {video_width}x{video_height} exceeds model offscreen "
          f"framebuffer {max_width}x{max_height}; using {min(video_width, max_width)}x"
          f"{min(video_height, max_height)}."
        )
        video_width = min(video_width, max_width)
        video_height = min(video_height, max_height)

      renderer = mujoco.Renderer(self.model, height=video_height, width=video_width)
      if video_camera is None:
        camera = -1
      elif video_camera.lower() == "free":
        camera = -1
      else:
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, video_camera)
        if cam_id < 0:
          raise ValueError(
            f"Camera '{video_camera}' not found in MJCF. "
            "Use --video-camera free or an existing camera name."
          )
        camera = video_camera

      if camera == -1 and video_follow_subject:
        if not 0.0 <= video_follow_smoothing <= 1.0:
          raise ValueError(
            f"video_follow_smoothing must be in [0, 1], got {video_follow_smoothing}."
          )
        follow_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(follow_camera)
        follow_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        follow_camera.distance = float(video_follow_distance)
        follow_camera.azimuth = float(video_follow_azimuth)
        follow_camera.elevation = float(video_follow_elevation)
        follow_lookat = self.data.qpos[:3].astype(np.float32).copy()
        follow_lookat[2] += float(video_follow_height)
        follow_camera.lookat[:] = follow_lookat

      writer_ctx = media.VideoWriter(
        video_path,
        shape=(video_height, video_width),
        fps=float(video_fps),
      )
      camera_label = (
        "follow-free"
        if follow_camera is not None
        else ("free" if camera == -1 else str(camera))
      )
      print(
        f"Recording video: {video_path} ({video_width}x{video_height} @ {video_fps} fps,"
        f" camera={camera_label})"
      )

    try:
      with writer_ctx as writer:
        for _ in range(steps):
          self._step_once()

          base_quat = self.data.qpos[3:7]  # [w, x, y, z]
          rot = Rotation.from_quat(base_quat, scalar_first=True)
          v_body = rot.apply(self.data.qvel[0:3], inverse=True).astype(np.float32)

          lin_vel_x_body.append(float(v_body[0]))
          base_heights.append(float(self.data.qpos[2]))
          action_norms.append(float(np.linalg.norm(self.last_action)))

          if renderer is not None:
            if follow_camera is not None:
              target = self.data.qpos[:3].astype(np.float32).copy()
              target[2] += float(video_follow_height)
              follow_lookat = (
                1.0 - video_follow_smoothing
              ) * follow_lookat + video_follow_smoothing * target
              follow_camera.lookat[:] = follow_lookat
              renderer.update_scene(self.data, camera=follow_camera)
            else:
              renderer.update_scene(self.data, camera=camera)
            writer.add_image(renderer.render())
    finally:
      if renderer is not None:
        renderer.close()

    lin_vel_x_body = np.array(lin_vel_x_body)
    base_heights = np.array(base_heights)
    action_norms = np.array(action_norms)

    print("\n=== Headless sim2sim summary ===")
    print(f"steps: {steps}")
    print(f"command set [vx, vy, yaw]: [{cmd_x:.3f}, {cmd_y:.3f}, {cmd_yaw:.3f}]")
    print(
      f"filtered command final [vx, vy, yaw]: "
      f"[{self.filtered_command[0]:.3f}, {self.filtered_command[1]:.3f}, {self.filtered_command[2]:.3f}]"
    )
    print(
      f"body vx (m/s): mean={lin_vel_x_body.mean():.3f}, "
      f"p10={np.percentile(lin_vel_x_body, 10):.3f}, "
      f"p90={np.percentile(lin_vel_x_body, 90):.3f}"
    )
    print(
      f"base height (m): min={base_heights.min():.3f}, mean={base_heights.mean():.3f}"
    )
    print(
      f"action L2 norm: mean={action_norms.mean():.3f}, max={action_norms.max():.3f}"
    )
    if video_path is not None:
      print(f"video: {video_path}")

  def run(self):
    self.reset()
    with mujoco.viewer.launch_passive(
      self.model, self.data, key_callback=self._key_callback
    ) as viewer:
      print("Controls: ↑↓ forward, ←→ yaw, Space=stop, R=reset")
      print(
        f"Command: [vx={self.command[0]:.1f}, vy={self.command[1]:.1f},"
        f" yaw={self.command[2]:.1f}]"
      )
      while viewer.is_running():
        t0 = time.time()
        self._step_once()

        # Camera follows robot.
        viewer.cam.lookat[:] = self.data.qpos[:3]
        viewer.sync()

        # Realtime pacing.
        elapsed = time.time() - t0
        target = SIM_DT * DECIMATION
        if elapsed < target:
          time.sleep(target - elapsed)


def main():
  def _parse_csv_floats(text: str, expected_len: int | None = None) -> np.ndarray:
    values = np.array([float(x.strip()) for x in text.split(",")], dtype=np.float32)
    if expected_len is not None and values.shape[0] != expected_len:
      raise ValueError(
        f"Expected {expected_len} comma-separated values, got {values.shape[0]}."
      )
    return values

  parser = argparse.ArgumentParser(description="Sim2sim: MuJoCo C + ONNX")
  parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX policy")
  parser.add_argument("--mjcf", type=str, default=None, help="Override MJCF path")
  parser.add_argument(
    "--headless",
    action="store_true",
    help="Run without viewer and print transfer diagnostics.",
  )
  parser.add_argument(
    "--steps",
    type=int,
    default=2000,
    help="Headless rollout steps.",
  )
  parser.add_argument(
    "--video-path",
    type=str,
    default=None,
    help="Optional output MP4 path for headless rollout recording.",
  )
  parser.add_argument(
    "--video-fps",
    type=int,
    default=50,
    help="Headless video frame rate.",
  )
  parser.add_argument(
    "--video-width",
    type=int,
    default=1280,
    help="Headless video width in pixels.",
  )
  parser.add_argument(
    "--video-height",
    type=int,
    default=720,
    help="Headless video height in pixels.",
  )
  parser.add_argument(
    "--video-camera",
    type=str,
    default=None,
    help="Camera name for headless video (or 'free'). Defaults to free camera.",
  )
  parser.add_argument(
    "--no-video-follow-subject",
    action="store_true",
    help="Disable free-camera subject following during headless video recording.",
  )
  parser.add_argument(
    "--video-follow-distance",
    type=float,
    default=DEFAULT_VIDEO_FOLLOW_DISTANCE,
    help="Distance of the follow camera from the robot when using free camera.",
  )
  parser.add_argument(
    "--video-follow-height",
    type=float,
    default=DEFAULT_VIDEO_FOLLOW_HEIGHT,
    help="Vertical look-at offset (meters) for follow camera framing.",
  )
  parser.add_argument(
    "--video-follow-smoothing",
    type=float,
    default=DEFAULT_VIDEO_FOLLOW_SMOOTHING,
    help="EMA smoothing in [0,1] for follow camera look-at updates.",
  )
  parser.add_argument(
    "--video-follow-azimuth",
    type=float,
    default=DEFAULT_VIDEO_FOLLOW_AZIMUTH,
    help="Free-camera azimuth angle (degrees) for follow camera framing.",
  )
  parser.add_argument(
    "--video-follow-elevation",
    type=float,
    default=DEFAULT_VIDEO_FOLLOW_ELEVATION,
    help="Free-camera elevation angle (degrees) for follow camera framing.",
  )
  parser.add_argument(
    "--cmd-x", type=float, default=0.8, help="Commanded forward velocity."
  )
  parser.add_argument(
    "--cmd-y", type=float, default=0.0, help="Commanded lateral velocity."
  )
  parser.add_argument("--cmd-yaw", type=float, default=0.0, help="Commanded yaw rate.")
  parser.add_argument(
    "--obs-layout",
    choices=["auto", "mjlab-168", "gbrl-blind", "pad"],
    default="auto",
    help=(
      "Observation packing mode. auto picks mjlab-168 for 168-dim ONNX, or "
      "gb-rl blind layout when dimensions match its history structure."
    ),
  )
  parser.add_argument(
    "--tau-obs-range",
    type=str,
    default=f"{DEFAULT_TAU_OBS_RANGE[0]},{DEFAULT_TAU_OBS_RANGE[1]}",
    help=(
      "Normalization range for motor_tau_max observation as 'min,max'. "
      "Used only when gbrl-blind layout includes motor_tau_max."
    ),
  )
  parser.add_argument(
    "--tau-obs-nm",
    type=str,
    default=None,
    help=(
      "Optional 12 comma-separated tau_max values (Nm) in commanded joint order. "
      "If omitted, defaults to effort limits from config."
    ),
  )
  args = parser.parse_args()

  if not Path(args.onnx).exists():
    raise FileNotFoundError(f"ONNX file not found: {args.onnx}")

  tau_range_vals = _parse_csv_floats(args.tau_obs_range, expected_len=2)
  tau_obs_nm = (
    None
    if args.tau_obs_nm is None
    else _parse_csv_floats(args.tau_obs_nm, expected_len=len(COMMANDED_JOINTS))
  )

  runner = Sim2SimRunner(
    args.onnx,
    args.mjcf,
    obs_layout=args.obs_layout,
    tau_obs_range=(float(tau_range_vals[0]), float(tau_range_vals[1])),
    tau_obs_nm=tau_obs_nm,
  )
  if args.headless:
    runner.run_headless(
      args.steps,
      args.cmd_x,
      args.cmd_y,
      args.cmd_yaw,
      video_path=args.video_path,
      video_fps=args.video_fps,
      video_width=args.video_width,
      video_height=args.video_height,
      video_camera=args.video_camera,
      video_follow_subject=not args.no_video_follow_subject,
      video_follow_distance=args.video_follow_distance,
      video_follow_height=args.video_follow_height,
      video_follow_smoothing=args.video_follow_smoothing,
      video_follow_azimuth=args.video_follow_azimuth,
      video_follow_elevation=args.video_follow_elevation,
    )
  else:
    runner.run()


if __name__ == "__main__":
  main()
