#!/usr/bin/env python3
"""Validate a humanoid co-design by running a long rollout with video rendering.

This script loads a design from the Pareto front (codesign_pareto.npz) and
verifies the robot can walk by running a long rollout and optionally rendering
a video.

Example:
  uv run python scripts/validate_codesign.py \
    --policy ./model_30000.pt \
    --design-idx 2 \
    --rollout-steps 1000 \
    --output-video design_validation.mp4
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import mediapy as media
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.amp_runner import MjlabAmpOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp.motor_randomization import set_motor_tau_max

# QDD robot constants (same as in codesign_ga.py)
QDD_JOINT_ORDER = [
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
TAU_OBS_RANGE = (5.0, 90.0)
SELECTION_SCREEN_STEPS = 200
SELECTION_MIN_MEAN_VX = 0.25
SELECTION_MIN_MIN_HEIGHT = 0.45


class FlatActorPolicy(torch.nn.Module):
  """Wrap an rsl_rl actor so it accepts a flat observation tensor."""

  def __init__(self, actor: torch.nn.Module) -> None:
    super().__init__()
    self.obs_normalizer = actor.obs_normalizer
    self.mlp = actor.mlp

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    latent = self.obs_normalizer(obs)
    return self.mlp(latent)


class OnnxPolicy(torch.nn.Module):
  """Wrap an ONNX policy for inference (handles obs dimension mismatch)."""

  def __init__(self, onnx_path: str, device: str = "cpu") -> None:
    super().__init__()
    import onnxruntime as rt

    self._sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    self._input_name = self._sess.get_inputs()[0].name
    self._output_name = self._sess.get_outputs()[0].name
    self._onnx_obs_dim = int(self._sess.get_inputs()[0].shape[1])

  def _adapt_obs(self, obs: torch.Tensor) -> np.ndarray:
    """Convert mjlab flat obs to ONNX input contract."""
    mjlab_obs_dim = int(obs.shape[-1])
    if mjlab_obs_dim == self._onnx_obs_dim:
      return obs.cpu().numpy().astype(np.float32)

    # Mjlab actor observations include base linear velocity as the leading 3 dims.
    # gb-rl / IsaacLab blind ONNX contracts do not include this block.
    if mjlab_obs_dim == self._onnx_obs_dim + 3:
      return obs[:, 3:].cpu().numpy().astype(np.float32)

    if mjlab_obs_dim < self._onnx_obs_dim:
      batch_size = obs.shape[0]
      padded = torch.zeros(
        batch_size, self._onnx_obs_dim, device=obs.device, dtype=obs.dtype
      )
      padded[:, :mjlab_obs_dim] = obs
      return padded.cpu().numpy().astype(np.float32)
    else:
      return obs[:, : self._onnx_obs_dim].cpu().numpy().astype(np.float32)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    batch_size = obs.shape[0]
    obs_adapted = self._adapt_obs(obs)
    actions_list = []
    for i in range(batch_size):
      obs_single = obs_adapted[i : i + 1]
      action = self._sess.run([self._output_name], {self._input_name: obs_single})[0]
      actions_list.append(action[0])
    actions_np = np.stack(actions_list, axis=0)
    return torch.from_numpy(actions_np).to(obs.device).to(obs.dtype)


def _load_mjlab_policy_module(
  task: str, policy_path: str, device: str
) -> torch.nn.Module:
  """Load a policy: ONNX, TorchScript, or rsl_rl checkpoint."""
  # Try ONNX first
  if policy_path.endswith(".onnx"):
    return OnnxPolicy(policy_path, device=device).eval()

  # Try TorchScript
  try:
    return torch.jit.load(policy_path, map_location=device).eval()
  except (RuntimeError, ValueError) as exc:
    msg = str(exc)
    if "constants.pkl" not in msg and "PytorchStreamReader" not in msg:
      raise

  # Fall back to rsl_rl checkpoint
  import mjlab.tasks  # noqa: F401

  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = 1
  rl_cfg = load_rl_cfg(task)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapper = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  try:
    runner = MjlabAmpOnPolicyRunner(wrapper, asdict(rl_cfg), device=device)
    runner.load(policy_path, load_cfg={"actor": True}, strict=True)
    actor = runner.actor_critic.actor.to(device).eval()
    return FlatActorPolicy(actor).eval()
  finally:
    wrapper.close()


def load_pareto_set(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
  """Load objectives (F), designs (X), and joint group names from Pareto npz."""
  data = np.load(path, allow_pickle=True)
  return data["F"], data["X"], data["groups"]


def select_design(
  F: np.ndarray,
  criteria: str = "reward",
  efficiency_min_reward_fraction: float = 0.97,
) -> int:
  """Select a design from the results set.

  For single-objective (n_obj=1): F[:, 0] is the scalarized fitness.
  Lower fitness = better (performance - cost).
  
  For multi-objective (n_obj=2): F[:, 0] = -performance, F[:, 1] = cost.
  Note: now using single-objective, so only criteria='reward' is applicable.
  """
  if F.shape[1] == 1:
    # Single-objective: just pick the best (lowest) fitness
    idx = np.argmin(F[:, 0])
  elif F.shape[1] == 2:
    # Multi-objective (legacy support)
    if criteria == "reward":
      idx = np.argmin(F[:, 0])  # Highest reward = most negative F[0]
    elif criteria == "efficiency":
      rewards = -F[:, 0]
      costs = F[:, 1]
      ratio = rewards / (costs + 1e-6)
      reward_floor = efficiency_min_reward_fraction * float(np.max(rewards))
      feasible = rewards >= reward_floor
      if np.any(feasible):
        feasible_indices = np.where(feasible)[0]
        idx = feasible_indices[np.argmax(ratio[feasible_indices])]
      else:
        idx = np.argmax(ratio)
    else:
      raise ValueError(f"Unknown criteria: {criteria}")
  else:
    raise ValueError(f"Unexpected F.shape: {F.shape}")
  return int(idx)


def design_walkability_score(metrics: dict[str, float]) -> float:
  """Higher values mean more likely to be a stable walking rollout."""
  return metrics["mean_vx"] + metrics["mean_height"] + metrics["min_height"]


def walkability_passes(
  metrics: dict[str, float],
  min_mean_vx: float = SELECTION_MIN_MEAN_VX,
  min_min_height: float = SELECTION_MIN_MIN_HEIGHT,
) -> bool:
  return metrics["mean_vx"] >= min_mean_vx and metrics["min_height"] >= min_min_height


def _select_by_criteria(
  F: np.ndarray,
  candidate_indices: np.ndarray,
  criteria: str,
  efficiency_min_reward_fraction: float,
) -> int:
  if len(candidate_indices) == 0:
    raise ValueError("No candidate designs available for selection")
  if criteria == "reward":
    local = candidate_indices[np.argmin(F[candidate_indices, 0])]
  elif criteria == "efficiency":
    rewards = -F[candidate_indices, 0]
    costs = F[candidate_indices, 1]
    ratio = rewards / (costs + 1e-6)
    reward_floor = efficiency_min_reward_fraction * float(np.max(rewards))
    feasible = rewards >= reward_floor
    if np.any(feasible):
      feasible_indices = candidate_indices[np.where(feasible)[0]]
      local = feasible_indices[np.argmax(ratio[feasible])]
    else:
      local = candidate_indices[np.argmax(ratio)]
  else:
    raise ValueError(f"Unknown criteria: {criteria}")
  return int(local)


def select_walkable_design(
  F: np.ndarray,
  X: np.ndarray,
  policy_module: torch.nn.Module,
  rollout_steps: int,
  criteria: str = "reward",
  efficiency_min_reward_fraction: float = 0.97,
  task: str = "Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond",
  forward_only: bool = True,
  selection_rollout_steps: int = SELECTION_SCREEN_STEPS,
  selection_min_mean_vx: float = SELECTION_MIN_MEAN_VX,
  selection_min_min_height: float = SELECTION_MIN_MIN_HEIGHT,
) -> tuple[int, dict[str, float]]:
  """Pick a design, preferring rollouts that actually walk."""
  initial_idx = _select_by_criteria(
    F,
    np.arange(len(X)),
    criteria,
    efficiency_min_reward_fraction,
  )
  initial_metrics = evaluate_design(
    design=dict(X[initial_idx]),
    policy_module=policy_module,
    rollout_steps=min(selection_rollout_steps, rollout_steps),
    render=False,
    output_video=None,
    task=task,
    forward_only=forward_only,
    verbose=False,
  )
  if walkability_passes(
    initial_metrics,
    min_mean_vx=selection_min_mean_vx,
    min_min_height=selection_min_min_height,
  ):
    return initial_idx, initial_metrics

  metrics_by_idx: dict[int, dict[str, float]] = {initial_idx: initial_metrics}
  for idx in range(len(X)):
    if idx == initial_idx:
      continue
    metrics_by_idx[idx] = evaluate_design(
      design=dict(X[idx]),
      policy_module=policy_module,
      rollout_steps=min(selection_rollout_steps, rollout_steps),
      render=False,
      output_video=None,
      task=task,
      forward_only=forward_only,
      verbose=False,
    )

  walkable_indices = np.array(
    [
      idx
      for idx, metrics in metrics_by_idx.items()
      if walkability_passes(
        metrics,
        min_mean_vx=selection_min_mean_vx,
        min_min_height=selection_min_min_height,
      )
    ],
    dtype=int,
  )
  if len(walkable_indices) > 0:
    selected_idx = _select_by_criteria(
      F,
      walkable_indices,
      criteria,
      efficiency_min_reward_fraction,
    )
    return selected_idx, metrics_by_idx[selected_idx]

  best_idx = max(
    metrics_by_idx, key=lambda idx: design_walkability_score(metrics_by_idx[idx])
  )
  return int(best_idx), metrics_by_idx[best_idx]


def design_to_tau_vector(design: dict) -> torch.Tensor:
  """Convert design dict to per-joint tau vector."""
  # Design has 'tau_joint_group' keys; map them to per-joint torques
  tau_dict = {}
  for key, value in design.items():
    if key.startswith("tau_"):
      group_name = key[4:]  # Remove "tau_" prefix
      tau_dict[group_name] = float(value)

  # Map to per-joint torques in QDD_JOINT_ORDER
  tau_vec = []
  for joint_name in QDD_JOINT_ORDER:
    # Extract group name from joint (e.g., "l_hip_pitch" -> "hip_pitch")
    parts = joint_name.split("_", 1)  # Split on first underscore
    group_name = parts[1] if len(parts) > 1 else joint_name
    tau_val = tau_dict.get(group_name, 80.0)  # Default to 80 if not found
    tau_vec.append(tau_val)

  return torch.tensor(tau_vec, dtype=torch.float32)


def evaluate_design(
  design: dict,
  policy_module: torch.nn.Module,
  rollout_steps: int = 1000,
  render: bool = True,
  output_video: str = None,
  task: str = "Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond",
  forward_only: bool = True,
  verbose: bool = True,
) -> dict:
  """Run a long rollout with a single design and optionally render video.

  Args:
      forward_only: If True, command only forward velocity (no backward).
  """
  device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
  device_torch = torch.device(device_str)
  if verbose:
    print(f"[Device] Using {device_str}")

  # Load environment
  if verbose:
    print(f"[Env] Loading {task}...")
  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = 1

  # Force forward-only velocity command during validation
  if forward_only:
    if hasattr(env_cfg, "commands") and "twist" in env_cfg.commands:
      twist_cmd = env_cfg.commands["twist"]
      twist_cmd.ranges.lin_vel_x = (0.8, 0.8)  # Match sim2sim / GA deployment
      twist_cmd.ranges.ang_vel_z = (0.0, 0.0)  # No rotation (zero angular velocity)
      if verbose:
        print(
          "[Config] Forcing forward-only velocity "
          "(lin_vel_x: 0.8 m/s fixed, ang_vel_z: 0.0)"
        )

  # Disable training-time torque randomization
  if getattr(env_cfg, "events", None) is not None:
    for name in (
      "encoder_bias",
      "base_com",
      "foot_friction_slide",
      "randomize_robot_mass",
      "randomize_actuator_gains",
      "randomize_joint_friction",
      "joint_default_pos_noise",
      "randomize_terrain",
    ):
      env_cfg.events.pop(name, None)
    env_cfg.events.pop("randomize_motor_tau_max", None)

  # Improve video quality: higher resolution and better camera angle
  if render:
    env_cfg.viewer.height = 720  # Increase from 240
    env_cfg.viewer.width = 960  # Increase from 320
    # Better viewing angle: slightly elevated and 90 degrees azimuth
    env_cfg.viewer.elevation = -30.0  # Slightly higher viewpoint
    env_cfg.viewer.distance = 4.0  # Closer to robot

  # Pass render_mode to constructor for video capture
  render_mode = "rgb_array" if render else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device_str, render_mode=render_mode)

  try:
    # Reset env and apply design
    if verbose:
      print("\n[Design] Applying from design dict")
    obs_dict, info = env.reset()
    obs = obs_dict["actor"] if isinstance(obs_dict, dict) else obs_dict

    # Decode design to tau vector and apply
    tau_vec = design_to_tau_vector(design)
    tau_vec_full = tau_vec.unsqueeze(0).to(device_str)  # shape: (1, 12)
    if verbose:
      print(f"  Tau values (Nm): {tau_vec.tolist()}")

    env_ids = torch.arange(1, dtype=torch.int32, device=device_torch)

    # Set torques before reset so first obs reflects the design
    set_motor_tau_max(
      env.unwrapped,
      env_ids,
      tau_vec_full,
      QDD_JOINT_ORDER,
      tau_range=TAU_OBS_RANGE,
    )

    # Reset to start fresh
    obs_dict, info = env.reset()
    obs = obs_dict["actor"] if isinstance(obs_dict, dict) else obs_dict

    # Re-apply torques after reset (reset events may touch limits)
    set_motor_tau_max(
      env.unwrapped,
      env_ids,
      tau_vec_full,
      QDD_JOINT_ORDER,
      tau_range=TAU_OBS_RANGE,
    )

    trajectory = {"actions": [], "rewards": []}
    if render:
      trajectory["frames"] = []
      # Capture initial frame before any steps
      try:
        frame = env.render()
        if frame is not None:
          trajectory["frames"].append(frame)
      except Exception as e:
        if verbose:
          print(f"Warning: Initial render failed: {e}")

    total_reward = 0.0
    total_vx = 0.0
    total_height = 0.0
    min_height = float("inf")
    if verbose:
      print(f"\n[Rollout] Running {rollout_steps} steps...")
    for step in range(rollout_steps):
      with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device_torch)
        if obs_tensor.dim() == 1:
          obs_tensor = obs_tensor.unsqueeze(0)
        action = policy_module(obs_tensor)
        if action.dim() > 1:
          action = action.squeeze(0)

      # Step env
      obs_dict, reward, dones, truncs, info = env.step(action.unsqueeze(0))
      obs = obs_dict["actor"] if isinstance(obs_dict, dict) else obs_dict

      step_reward = (
        float(reward[0]) if isinstance(reward, torch.Tensor) else float(reward[0])
      )
      total_reward += step_reward
      total_vx += float(obs[0, 0])
      height = float(env.unwrapped.scene["robot"].data.root_link_pos_w[0, 2])
      total_height += height
      min_height = min(min_height, height)

      trajectory["actions"].append(
        action.cpu().numpy() if isinstance(action, torch.Tensor) else action
      )
      trajectory["rewards"].append(step_reward)

      # Re-assert torques after step (auto-reset on termination keeps the design)
      set_motor_tau_max(
        env.unwrapped,
        env_ids,
        tau_vec_full,
        QDD_JOINT_ORDER,
        tau_range=TAU_OBS_RANGE,
      )

      # Render frame every step for smooth 50 fps video (render AFTER step to show result)
      if render:
        try:
          frame = env.render()
          if frame is not None:
            trajectory["frames"].append(frame)
        except Exception as e:
          if verbose:
            print(f"Warning: Rendering failed at step {step}: {e}")
          render = False

      if verbose and (step % 200 == 0 or step == rollout_steps - 1):
        print(
          f"  Step {step:4d}/{rollout_steps}, cumulative_reward={total_reward:7.3f}"
        )

    trajectory["actions"] = np.array(trajectory["actions"])
    trajectory["rewards"] = np.array(trajectory["rewards"])

    result = {
      "total_reward": total_reward,
      "mean_reward": total_reward / rollout_steps,
      "mean_vx": total_vx / rollout_steps,
      "mean_height": total_height / rollout_steps,
      "min_height": min_height,
      "walk_score": (
        total_vx / rollout_steps + total_height / rollout_steps + min_height
      ),
      "steps_completed": rollout_steps,
      "design": design,
    }

    # Save video if requested
    if output_video and trajectory.get("frames"):
      num_frames = len(trajectory["frames"])
      # FPS = 50 since we capture every physics step (0.02s per step = 50 Hz)
      fps = 50
      if verbose:
        print(
          f"\n[Video] Rendering {num_frames} frames to {output_video} @ {fps} fps..."
        )
      try:
        media.write_video(
          output_video,
          trajectory["frames"],
          fps=fps,
        )
        result["video_path"] = output_video
        if verbose:
          print(f"✓ Saved video to {output_video}")
      except Exception as e:
        if verbose:
          print(f"Error: Video save failed: {e}")
        import traceback

        traceback.print_exc()
    elif output_video and verbose:
      print("\n[Video] No frames captured; video will not be saved.")

    return result
  finally:
    env.close()


def run_validation(
  design: dict,
  policy_module: torch.nn.Module,
  rollout_steps: int = 1000,
  render: bool = True,
  output_video: str = None,
  task: str = "Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond",
  forward_only: bool = True,
) -> dict:
  return evaluate_design(
    design=design,
    policy_module=policy_module,
    rollout_steps=rollout_steps,
    render=render,
    output_video=output_video,
    task=task,
    forward_only=forward_only,
    verbose=True,
  )


def main():
  parser = argparse.ArgumentParser(
    description="Validate a co-design from the Pareto front by running a long rollout.",
  )
  parser.add_argument(
    "--pareto",
    default="codesign_pareto.npz",
    help="Path to Pareto npz file from codesign_ga.py",
  )
  parser.add_argument(
    "--policy",
    required=True,
    help="Path to policy checkpoint (.pt or exported TorchScript)",
  )
  parser.add_argument(
    "--design-idx",
    type=int,
    default=None,
    help="Index of design in Pareto set (0=first). If None, selects by --criteria.",
  )
  parser.add_argument(
    "--criteria",
    choices=["reward", "efficiency"],
    default="efficiency",
    help="Selection criteria if --design-idx not specified (default: efficiency).",
  )
  parser.add_argument(
    "--efficiency-min-reward-fraction",
    type=float,
    default=0.97,
    help="For --criteria efficiency, require reward >= this fraction of max reward.",
  )
  parser.add_argument(
    "--selection-rollout-steps",
    type=int,
    default=SELECTION_SCREEN_STEPS,
    help="Short rollout length used to screen candidate designs for walkability.",
  )
  parser.add_argument(
    "--selection-min-mean-vx",
    type=float,
    default=SELECTION_MIN_MEAN_VX,
    help="Minimum mean forward velocity required to treat a design as walkable.",
  )
  parser.add_argument(
    "--selection-min-min-height",
    type=float,
    default=SELECTION_MIN_MIN_HEIGHT,
    help="Minimum rollout min height required to treat a design as walkable.",
  )
  parser.add_argument(
    "--rollout-steps",
    type=int,
    default=1000,
    help="Number of simulation steps (at 50 Hz = 20 per sec).",
  )
  parser.add_argument(
    "--output-video",
    default=None,
    help="Path to save rendered video (.mp4).",
  )
  parser.add_argument(
    "--task",
    default="Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond",
    help="Task name.",
  )
  parser.add_argument(
    "--forward-only",
    action="store_true",
    default=True,
    help="Force forward-only velocity during validation (default: True).",
  )
  parser.add_argument(
    "--allow-backward",
    action="store_true",
    help="Allow backward velocity (overrides --forward-only).",
  )

  args = parser.parse_args()
  forward_only = not args.allow_backward  # Default True unless --allow-backward

  # Load Pareto set
  if not Path(args.pareto).exists():
    print(f"Error: Pareto file not found: {args.pareto}")
    return 1

  F, X, groups = load_pareto_set(args.pareto)
  print(f"\n[Pareto] Loaded {len(X)} designs")
  print("\nObjectives (performance, cost):")
  for i, (f, _x) in enumerate(zip(F, X, strict=True)):
    performance = -f[0]  # Convert back to positive
    print(f"  {i}: performance={performance:8.3f}, cost={f[1]:7.3f}")

  device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
  print(f"\n[Policy] Loading from {args.policy}...")
  policy_module = _load_mjlab_policy_module(args.task, args.policy, device_str)

  # Select design
  if args.design_idx is None:
    idx, screening_metrics = select_walkable_design(
      F,
      X,
      policy_module,
      args.rollout_steps,
      args.criteria,
      efficiency_min_reward_fraction=args.efficiency_min_reward_fraction,
      task=args.task,
      forward_only=forward_only,
      selection_rollout_steps=args.selection_rollout_steps,
      selection_min_mean_vx=args.selection_min_mean_vx,
      selection_min_min_height=args.selection_min_min_height,
    )
    print(f"\n[Selection] Choosing best by --criteria={args.criteria}: design {idx}")
    if args.criteria == "efficiency":
      print(
        "            with performance floor "
        f"{args.efficiency_min_reward_fraction:.2f} × max performance"
      )
    print(
      "            walk screening: "
      f"mean_vx={screening_metrics['mean_vx']:.3f}, "
      f"mean_height={screening_metrics['mean_height']:.3f}, "
      f"min_height={screening_metrics['min_height']:.3f}, "
      f"walk_score={screening_metrics['walk_score']:.3f}"
    )
  else:
    idx = args.design_idx
    if idx >= len(X):
      print(f"Error: design-idx {idx} out of range [0, {len(X) - 1}]")
      return 1

  design = dict(X[idx])
  performance_negated, cost = F[idx]
  performance = (
    -performance_negated
  )  # F[0] stored as -performance; convert back to positive
  print(f"\nSelected design {idx}:")
  print(f"  Performance: {performance:.3f}")
  print(f"  Cost: {cost:.3f}")

  # Run validation
  result = run_validation(
    design=design,
    policy_module=policy_module,
    rollout_steps=args.rollout_steps,
    render=args.output_video is not None,
    output_video=args.output_video,
    task=args.task,
    forward_only=forward_only,
  )

  # Print results
  print(f"\n{'=' * 60}")
  print("[RESULTS]")
  print(f"{'=' * 60}")
  print(f"  Total reward: {result['total_reward']:.3f}")
  print(f"  Mean reward per step: {result['mean_reward']:.6f}")
  print(f"  Mean forward velocity: {result['mean_vx']:.6f}")
  print(f"  Mean base height: {result['mean_height']:.6f}")
  print(f"  Min base height: {result['min_height']:.6f}")
  print(f"  Steps completed: {result['steps_completed']}")
  if "video_path" in result:
    print(f"  Video saved to: {result['video_path']}")

  print("\n✓ Validation complete!")
  return 0


if __name__ == "__main__":
  exit(main())
