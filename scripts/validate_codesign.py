#!/usr/bin/env python3
"""Validate a humanoid co-design by running a long rollout with video rendering.

This script loads a design from the Pareto front (codesign_pareto.npz) and
verifies the robot can walk by running a long rollout and optionally rendering
a video.

Example:
  uv run python scripts/validate_codesign.py \
    --policy ./model_30000.pt \
    --design-idx 6 \
    --rollout-steps 1000 \
    --output-video design_6_validation.mp4
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
    """Load objectives (F), designs (X), and joint group names from Pareto npz.

    Returns:
        (F, X, groups) where:
          - F: shape (n_designs, 2) with columns [reward, cost]
          - X: shape (n_designs,) array of design dicts
          - groups: list of joint group names
    """
    data = np.load(path, allow_pickle=True)
    return data["F"], data["X"], data["groups"]


def select_design(F: np.ndarray, X: np.ndarray, criteria: str = "reward") -> int:
    """Select a design from the Pareto set.

    Args:
        F: shape (n_designs, 2) with [reward, cost]
        X: shape (n_designs,) of design dicts
        criteria: "reward" (max reward) or "efficiency" (best reward/cost ratio)

    Returns:
        Index into X of the selected design
    """
    if criteria == "reward":
        idx = np.argmax(F[:, 0])
    elif criteria == "efficiency":
        ratio = F[:, 0] / (F[:, 1] + 1e-6)
        idx = np.argmax(ratio)
    else:
        raise ValueError(f"Unknown criteria: {criteria}")
    return int(idx)


def run_validation(
    design: dict,
    policy_path: str,
    rollout_steps: int = 1000,
    render: bool = True,
    output_video: str = None,
    task: str = "Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond",
) -> dict:
    """Run a long rollout with a single design and optionally render video.

    Args:
        design: Dict mapping joint group name to tau_max value
        policy_path: Path to .pt checkpoint or TorchScript export
        rollout_steps: Number of simulation steps to run
        render: Whether to collect rendering data
        output_video: Optional path to save video (.mp4)
        task: Task environment string

    Returns:
        Dict with keys:
          - total_reward: Sum of reward over the rollout
          - mean_reward: Average reward per step
          - video_path: Path to saved video (if output_video provided)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # Load environment
    print(f"[Env] Loading {task}...")
    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = 1
    # Disable training-time torque randomization so the GA controls tau_max
    if getattr(env_cfg, "events", None) is not None:
        env_cfg.events.pop("randomize_motor_tau_max", None)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    # Load policy
    print(f"[Policy] Loading from {policy_path}...")
    policy_module = _load_mjlab_policy_module(task, policy_path, device)

    # Reset env
    print(f"\n[Design] Applying tau_max: {design}")
    obs_dict, info = env.reset()
    obs = obs_dict["actor"] if isinstance(obs_dict, dict) else obs_dict

    # Set the motor tau_max from design
    env.set_motor_tau_max(design)

    trajectory = {"actions": [], "rewards": []}
    if render:
        trajectory["frames"] = []

    total_reward = 0.0

    print(f"[Rollout] Running {rollout_steps} steps...")
    for step in range(rollout_steps):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            action = policy_module(obs_tensor)
            if action.dim() > 1:
                action = action.squeeze(0)

        # Step env
        obs_dict, reward, dones, truncs, info = env.step(action)
        obs = obs_dict["actor"] if isinstance(obs_dict, dict) else obs_dict

        step_reward = float(reward[0]) if isinstance(reward, torch.Tensor) else float(reward[0])
        total_reward += step_reward

        # Record trajectory
        trajectory["actions"].append(action.cpu().numpy() if isinstance(action, torch.Tensor) else action)
        trajectory["rewards"].append(step_reward)

        # Render frame
        if render and (step % 2 == 0):  # Capture every 2 steps for 25 fps video
            try:
                frame = env.render()
                if frame is not None:
                    trajectory["frames"].append(frame)
            except Exception as e:
                print(f"Warning: Rendering failed at step {step}: {e}")
                render = False

        if step % 200 == 0 or step == rollout_steps - 1:
            print(f"  Step {step}/{rollout_steps}, cumulative_reward={total_reward:.3f}")

    # Convert lists to arrays
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
        print(f"\n[Video] Rendering {len(trajectory['frames'])} frames to {output_video}...")
        try:
            media.write_video(
                output_video,
                trajectory["frames"],
                fps=25,  # 50 Hz physics, capture every 2 steps
            )
            result["video_path"] = output_video
            print(f"✓ Saved video to {output_video}")
        except Exception as e:
            print(f"Warning: Video save failed: {e}")

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
    print(f"\n[Pareto] Loaded {len(X)} designs with {len(groups)} joint groups: {list(groups)}")
    print(f"\nObjectives (reward, cost):")
    for i, (f, x) in enumerate(zip(F, X)):
        print(f"  {i}: reward={f[0]:.3f}, cost={f[1]:.3f}, design={dict(x)}")

    # Select design
    if args.design_idx is None:
        idx = select_design(F, X, args.criteria)
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
    print(f"  Config: {design}")

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
    print(f"  Design: {result['design']}")
    print(f"  Total reward: {result['total_reward']:.3f}")
    print(f"  Mean reward per step: {result['mean_reward']:.6f}")
    print(f"  Steps completed: {result['steps_completed']}")
    if "video_path" in result:
        print(f"  Video saved to: {result['video_path']}")

    print(f"\n✓ Validation complete!")
    return 0


if __name__ == "__main__":
    exit(main())
