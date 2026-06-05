"""Verify that a trained codesign policy respects the learned τ_max limits.

Loads a checkpoint with codesign state, rolls out the policy for N steps,
and reports per-joint torque statistics vs. the assigned motor limits.

Usage:
  uv run python scripts/verify_codesign_torques.py \
    --task Mjlab-Velocity-Flat-Gbionics-QDD-Codesign \
    --checkpoint logs/rsl_rl/qdd_velocity/<run>/model_XXXXX.pt \
    --num-steps 1000
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401 — register tasks


def main():
  parser = argparse.ArgumentParser(description="Verify codesign torque compliance")
  parser.add_argument("--task", required=True, help="Task ID")
  parser.add_argument("--checkpoint", required=True, help="Checkpoint path (.pt)")
  parser.add_argument("--num-steps", type=int, default=1000, help="Rollout steps")
  parser.add_argument("--device", default="cpu", help="Device (cpu or cuda:X)")
  args = parser.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  # Load configs.
  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = 1

  # Create env.
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # Create runner and load checkpoint.
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    args.checkpoint,
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)

  # Get codesign state.
  codesign = runner._codesign_module
  if codesign is None:
    print("[ERROR] No codesign module found. Is this a codesign checkpoint?")
    return

  print("\n" + "=" * 70)
  print("CODESIGN MOTOR ASSIGNMENT")
  print("=" * 70)
  print(codesign.summary())

  # Get per-joint tau_max from the learned assignment.
  with torch.no_grad():
    tau_eff = codesign.tau_eff(use_gumbel=False)  # (n_joints,)

  joint_names = codesign.symmetry.joint_names
  tau_limits = {name: tau_eff[i].item() for i, name in enumerate(joint_names)}

  # Apply learned effort limits to the actual actuators (enables clipping).
  robot = env.unwrapped.scene["robot"]
  with torch.no_grad():
    for actuator in robot.actuators:
      if actuator.force_limit is None:
        continue
      for i, jname in enumerate(actuator.target_names):
        if jname in joint_names:
          j_idx = joint_names.index(jname)
          actuator.force_limit[:, i] = tau_eff[j_idx]

  print("\nPer-joint τ_max (learned) — applied to actuators:")
  for name, limit in tau_limits.items():
    print(f"  {name:20s}: {limit:.1f} Nm")

  # Rollout and collect torques.
  print(f"\n{'=' * 70}")
  print(f"ROLLING OUT {args.num_steps} STEPS...")
  print("=" * 70)

  obs = env.get_observations().to(args.device)
  all_torques = []

  for _step in range(args.num_steps):
    with torch.no_grad():
      actions = policy(obs)
    obs, _, _, _ = env.step(actions)

    robot = env.unwrapped.scene["robot"]
    tau = robot.data.actuator_force[0].detach().cpu()  # (n_joints,)
    all_torques.append(tau)

  all_torques = torch.stack(all_torques, dim=0)  # (num_steps, n_joints)

  # Compute statistics and check compliance.
  print(f"\n{'=' * 70}")
  print("TORQUE COMPLIANCE REPORT")
  print("=" * 70)
  print(
    f"{'Joint':<20} {'τ_max':>7} {'Peak':>7} {'RMS':>7} "
    f"{'%Peak':>7} {'%RMS':>7} {'Violations':>10}"
  )
  print("-" * 70)

  total_violations = 0
  total_steps = all_torques.shape[0]

  for j, name in enumerate(joint_names):
    limit = tau_limits[name]
    tau_j = all_torques[:, j]
    peak = tau_j.abs().max().item()
    rms = tau_j.pow(2).mean().sqrt().item()
    pct_peak = 100 * peak / limit if limit > 0 else float("inf")
    pct_rms = 100 * rms / limit if limit > 0 else float("inf")
    violations = int((tau_j.abs() > limit).sum().item())
    total_violations += violations

    status = "✓" if violations == 0 else "✗"
    print(
      f"{status} {name:<18} {limit:>6.1f} {peak:>6.1f} {rms:>6.1f} "
      f"{pct_peak:>6.1f}% {pct_rms:>6.1f}% {violations:>10}"
    )

  print("-" * 70)
  violation_rate = 100 * total_violations / (total_steps * len(joint_names))
  print(
    f"\nTotal violations: {total_violations} / "
    f"{total_steps * len(joint_names)} samples ({violation_rate:.2f}%)"
  )

  if total_violations == 0:
    print("\n✓ ALL JOINTS RESPECT THEIR ASSIGNED τ_max LIMITS")
  else:
    print(
      f"\n✗ {total_violations} LIMIT VIOLATIONS DETECTED"
      "\n  (Some torque clipping may not be active during play)"
    )

  env.close()


if __name__ == "__main__":
  main()
