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

import numpy as np
import torch
import mediapy as media

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.amp_runner import MjlabAmpOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp.motor_randomization import set_motor_tau_max

# QDD robot constants (same as in codesign_ga.py)
QDD_JOINT_ORDER = [
    "l_hip_pitch", "r_hip_pitch", "l_hip_roll", "r_hip_roll",
    "l_hip_yaw", "r_hip_yaw", "l_knee", "r_knee",
    "l_ankle_pitch", "r_ankle_pitch", "l_ankle_roll", "r_ankle_roll",
]
TAU_OBS_RANGE = (5.0, 90.0)


class FlatActorPolicy(torch.nn.Module):
    """Wrap an rsl_rl actor so it accepts a flat observation tensor."""

    def __init__(self, actor: torch.nn.Module) -> None:
        super().__init__()
        self.obs_normalizer = actor.obs_normalizer
        self.mlp = actor.mlp

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.obs_normalizer(obs)
        return self.mlp(latent)


def _load_mjlab_policy_module(task: str, policy_path: str, device: str) -> torch.nn.Module:
    """Load either a TorchScript policy or a standard rsl_rl checkpoint."""
    try:
        return torch.jit.load(policy_path, map_location=device).eval()
    except (RuntimeError, ValueError) as exc:
        msg = str(exc)
        if "constants.pkl" not in msg and "PytorchStreamReader" not in msg:
            raise

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


def select_design(F: np.ndarray, criteria: str = "reward") -> int:
    """Select a design from the Pareto set."""
    if criteria == "reward":
        idx = np.argmax(F[:, 0])
    elif criteria == "efficiency":
        ratio = F[:, 0] / (F[:, 1] + 1e-6)
        idx = np.argmax(ratio)
    else:
        raise ValueError(f"Unknown criteria: {criteria}")
    return int(idx)


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


def run_validation(
    design: dict,
    policy_path: str,
    rollout_steps: int = 1000,
    render: bool = True,
    output_video: str = None,
    task: str = "Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond",
) -> dict:
    """Run a long rollout with a single design and optionally render video."""
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_torch = torch.device(device_str)
    print(f"[Device] Using {device_str}")

    # Load environment
    print(f"[Env] Loading {task}...")
    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = 1
    # Disable training-time torque randomization
    if getattr(env_cfg, "events", None) is not None:
        env_cfg.events.pop("randomize_motor_tau_max", None)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device_str)

    # Load policy
    print(f"[Policy] Loading from {policy_path}...")
    policy_module = _load_mjlab_policy_module(task, policy_path, device_str)

    # Reset env and apply design
    print(f"\n[Design] Applying from design dict")
    obs_dict, info = env.reset()
    obs = obs_dict["actor"] if isinstance(obs_dict, dict) else obs_dict

    # Decode design to tau vector and apply
    tau_vec = design_to_tau_vector(design)
    print(f"  Tau values (Nm): {tau_vec.tolist()}")
    
    env_ids = torch.arange(1, dtype=torch.int32, device=device_torch)
    set_motor_tau_max(
        env.unwrapped,
        env_ids,
        tau_vec.unsqueeze(0).to(device_str),
        QDD_JOINT_ORDER,
        tau_range=TAU_OBS_RANGE,
    )

    trajectory = {"actions": [], "rewards": []}
    if render:
        trajectory["frames"] = []

    total_reward = 0.0
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

        step_reward = float(reward[0]) if isinstance(reward, torch.Tensor) else float(reward[0])
        total_reward += step_reward

        trajectory["actions"].append(action.cpu().numpy() if isinstance(action, torch.Tensor) else action)
        trajectory["rewards"].append(step_reward)

        # Render frame every 2 steps for 25 fps video
        if render and (step % 2 == 0):
            try:
                frame = env.render()
                if frame is not None:
                    trajectory["frames"].append(frame)
            except Exception as e:
                print(f"Warning: Rendering failed at step {step}: {e}")
                render = False

        if step % 200 == 0 or step == rollout_steps - 1:
            print(f"  Step {step:4d}/{rollout_steps}, cumulative_reward={total_reward:7.3f}")

    trajectory["actions"] = np.array(trajectory["actions"])
    trajectory["rewards"] = np.array(trajectory["rewards"])

    result = {
        "total_reward": total_reward,
        "mean_reward": total_reward / rollout_steps,
        "steps_completed": rollout_steps,
        "design": design,
    }

    # Save video if requested
    if output_video and trajectory.get("frames"):
        num_frames = len(trajectory["frames"])
        print(f"\n[Video] Rendering {num_frames} frames to {output_video}...")
        try:
            media.write_video(
                output_video,
                trajectory["frames"],
                fps=25,
            )
            result["video_path"] = output_video
            print(f"✓ Saved video to {output_video}")
        except Exception as e:
            print(f"Error: Video save failed: {e}")
            import traceback
            traceback.print_exc()
    elif output_video:
        print(f"\n[Video] No frames captured; video will not be saved.")

    env.close()
    return result


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
        default="reward",
        help="Selection criteria if --design-idx not specified.",
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

    args = parser.parse_args()

    # Load Pareto set
    if not Path(args.pareto).exists():
        print(f"Error: Pareto file not found: {args.pareto}")
        return 1

    F, X, groups = load_pareto_set(args.pareto)
    print(f"\n[Pareto] Loaded {len(X)} designs")
    print(f"\nObjectives (reward, cost):")
    for i, (f, x) in enumerate(zip(F, X)):
        print(f"  {i}: reward={f[0]:8.3f}, cost={f[1]:7.3f}")

    # Select design
    if args.design_idx is None:
        idx = select_design(F, args.criteria)
        print(f"\n[Selection] Choosing best by --criteria={args.criteria}: design {idx}")
    else:
        idx = args.design_idx
        if idx >= len(X):
            print(f"Error: design-idx {idx} out of range [0, {len(X)-1}]")
            return 1

    design = dict(X[idx])
    reward, cost = F[idx]
    print(f"\nSelected design {idx}:")
    print(f"  Reward: {reward:.3f}")
    print(f"  Cost: {cost:.3f}")

    # Run validation
    result = run_validation(
        design=design,
        policy_path=args.policy,
        rollout_steps=args.rollout_steps,
        render=args.output_video is not None,
        output_video=args.output_video,
        task=args.task,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"[RESULTS]")
    print(f"{'='*60}")
    print(f"  Total reward: {result['total_reward']:.3f}")
    print(f"  Mean reward per step: {result['mean_reward']:.6f}")
    print(f"  Steps completed: {result['steps_completed']}")
    if "video_path" in result:
        print(f"  Video saved to: {result['video_path']}")

    print(f"\n✓ Validation complete!")
    return 0


if __name__ == "__main__":
    exit(main())
