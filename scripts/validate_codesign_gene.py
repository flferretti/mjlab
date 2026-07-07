#!/usr/bin/env python3
"""Validate and visualize a Gene01 actuator co-design result.

Companion to ``codesign_ga.py --robot gene``. Given a Pareto ``.npz`` produced by
the GA (single best design) or by ``--w-torque-sweep`` (a reward-vs-cost front),
this script:

  1. Selects one design (by raw reward or by reward/cost efficiency).
  2. Plots the Pareto front (performance vs cumulative torque), highlighting the
     selected design and annotating each point with its motor sizing.
  3. Optionally validates the selected design in the Isaac Lab gene env by writing
     its per-joint max-torque vector, rolling out the frozen policy, and reporting
     walkability (mean forward velocity, min root height, survival) and mean
     reward. With ``--video`` it also records an mp4.

The plotting step needs no simulator, so ``--plot-only`` runs anywhere. The
validation step reuses the exact env/obs contract from ``codesign_ga`` so the
result matches what the GA optimized.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Reuse the design space + policy wrapper from the GA script (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from codesign_ga import (  # noqa: E402
  GENE_GROUPS,
  GENE_JOINT_ORDER,
  GENE_TASK_DEFAULT,
  GENE_TAU_OBS_RANGE,
  CodesignConfig,
  decode_individual,
)


def load_pareto_set(path: str):
  data = np.load(path, allow_pickle=True)
  return data["F"], data["X"], list(data["groups"])


def select_design(
  F: np.ndarray, criteria: str, efficiency_min_reward_fraction: float = 0.97
) -> int:
  """Pick one design index from the Pareto set.

  Single-objective npz (F has one column) stores scalarized fitness (lower is
  better), so only 'reward' applies. Multi-objective/sweep npz stores
  ``F[:, 0] = -performance`` and ``F[:, 1] = cost`` (cumulative torque).
  """
  if F.shape[1] == 1:
    return int(np.argmin(F[:, 0]))
  if criteria == "reward":
    return int(np.argmin(F[:, 0]))
  # efficiency: among high-reward designs, maximize reward / cost.
  rewards = -F[:, 0]
  costs = F[:, 1]
  ratio = rewards / (costs + 1e-6)
  reward_floor = efficiency_min_reward_fraction * float(np.max(rewards))
  feasible = rewards >= reward_floor
  masked = np.where(feasible, ratio, -np.inf)
  return int(np.argmax(masked))


def plot_pareto(
  F: np.ndarray,
  X: np.ndarray,
  cfg: CodesignConfig,
  selected_idx: int,
  out_path: Path,
) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  if F.shape[1] == 1:
    print("[plot] Single-objective npz (one design); skipping front scatter.")
    return

  rewards = -F[:, 0]
  costs = F[:, 1]

  # Identify the non-dominated (Pareto-efficient) set: no other design has both
  # >= reward and <= cost. Higher reward + lower cost dominates.
  n = len(costs)
  dominated = np.zeros(n, dtype=bool)
  for i in range(n):
    for j in range(n):
      if j == i:
        continue
      if (
        rewards[j] >= rewards[i]
        and costs[j] <= costs[i]
        and (rewards[j] > rewards[i] or costs[j] < costs[i])
      ):
        dominated[i] = True
        break
  eff = np.where(~dominated)[0]
  eff = eff[np.argsort(costs[eff])]

  fig, ax = plt.subplots(figsize=(7, 5))
  ax.plot(
    costs[eff],
    rewards[eff],
    "-",
    color="tab:green",
    lw=2,
    zorder=1,
    label="Pareto frontier",
  )
  dom = np.where(dominated)[0]
  if len(dom):
    ax.scatter(costs[dom], rewards[dom], c="0.6", s=45, zorder=2, label="dominated")
  ax.scatter(
    costs[~dominated],
    rewards[~dominated],
    c="tab:blue",
    s=70,
    zorder=3,
    label="Pareto designs",
  )
  ax.scatter(
    costs[selected_idx],
    rewards[selected_idx],
    c="tab:red",
    s=180,
    marker="*",
    zorder=4,
    label="selected",
  )
  for i in range(len(costs)):
    genome = X[i] if isinstance(X[i], dict) else dict(X[i])
    _tau, _n, cum_tau = decode_individual(genome, cfg)
    ax.annotate(
      f"{cum_tau:.0f}Nm",
      (costs[i], rewards[i]),
      textcoords="offset points",
      xytext=(6, 4),
      fontsize=8,
      color="0.3",
    )
  ax.set_xlabel("Cumulative max torque (Nm)  —  hardware cost")
  ax.set_ylabel("Performance (mean task reward)")
  ax.set_title("Gene01 actuator co-design: reward vs cost Pareto front")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  fig.savefig(out_path, dpi=150)
  print(f"[plot] Saved Pareto front to {out_path}")


def design_tau_vector(genome: dict, cfg: CodesignConfig) -> np.ndarray:
  """Per-joint max torque (Nm) in GENE_JOINT_ORDER for one design genome."""
  tau_vec, _n, _cum = decode_individual(genome, cfg)
  return tau_vec


def validate_in_isaac(
  genome: dict,
  cfg: CodesignConfig,
  policy_path: str,
  task: str,
  num_envs: int,
  steps: int,
  device: str,
  metrics_json: Path | None = None,
) -> dict:
  """Roll out the selected design in the Isaac Lab gene env; return metrics.

  Reuses the GA's ``IsaacLabBackend`` verbatim so the env/obs/tau contract is
  identical to what was optimized. Crucially the design's max-torque vector is
  written BEFORE the first reset: the ``motor_tau_max`` observation term reads
  that per-env buffer while reset computes the initial observation, so writing
  after reset crashes on an uninitialized buffer.
  """
  from isaaclab.app import AppLauncher  # type: ignore

  app_launcher = AppLauncher(headless=True)
  simulation_app = app_launcher.app
  try:
    import torch

    import gb_rl_locomotion.networks  # type: ignore # noqa: F401
    import gb_rl_locomotion.tasks  # type: ignore # noqa: F401

    from codesign_ga import IsaacLabBackend, RewardMetric  # local

    backend = IsaacLabBackend(
      task=task,
      policy_path=policy_path,
      num_envs=num_envs,
      joint_order=cfg.joint_order,
      tau_obs_range=cfg.tau_obs_range,
      rollout_steps=steps,
      metric=RewardMetric(),
      device=device,
    )
    dev = backend.device

    tau_vec = torch.as_tensor(
      design_tau_vector(genome, cfg), dtype=torch.float32, device=dev
    )
    tau_full = tau_vec.unsqueeze(0).expand(num_envs, -1).contiguous()

    def _log(msg: str) -> None:
      print(f"[validate] {msg}", flush=True)

    # Proven order (see IsaacLabBackend._rollout): write tau BEFORE reset so the
    # first observation reflects the design, then re-assert after reset events.
    _log(f"writing tau (pre-reset), shape={tuple(tau_full.shape)}")
    backend._write_tau(tau_full)
    _log("first reset")
    obs = backend._reset()
    _log("writing tau (post-reset)")
    backend._write_tau(tau_full)
    _log("entering rollout loop")

    robot = backend.env.unwrapped.scene["robot"]
    alive = torch.ones(num_envs, dtype=torch.bool, device=dev)
    vx_sum = torch.zeros(num_envs, device=dev)
    rew_sum = torch.zeros(num_envs, device=dev)
    min_h = torch.full((num_envs,), float("inf"), device=dev)
    n_alive_steps = torch.zeros(num_envs, device=dev)

    with torch.no_grad():
      for step_i in range(steps):
        actions = backend._policy(backend._actor_obs(obs))
        obs, reward, terminated, truncated, _info = backend.env.step(actions)
        backend._write_tau(tau_full)  # keep design after any auto-reset
        if step_i == 0:
          _log("first policy+step OK")
        elif (step_i + 1) % 100 == 0:
          _log(f"step {step_i + 1}/{steps}, alive={int(alive.sum().item())}")
        vx = robot.data.root_lin_vel_b[:, 0]
        h = robot.data.root_link_pos_w[:, 2]
        vx_sum += torch.where(alive, vx, torch.zeros_like(vx))
        rew_sum += torch.where(alive, reward, torch.zeros_like(reward))
        n_alive_steps += alive.float()
        min_h = torch.minimum(min_h, torch.where(alive, h, min_h))
        alive = alive & ~(terminated | truncated)

    denom = torch.clamp(n_alive_steps, min=1.0)
    metrics = {
      "mean_vx": float((vx_sum / denom).mean().item()),
      "mean_reward": float((rew_sum / denom).mean().item()),
      "min_height": float(min_h.min().item()),
      "survival_frac": float((n_alive_steps / steps).mean().item()),
      "num_envs": num_envs,
      "steps": steps,
    }
    # Emit results BEFORE closing the env: Isaac's native shutdown can segfault
    # and truncate stdout, which would otherwise lose the just-computed metrics.
    _log(f"metrics: {metrics}")
    if metrics_json is not None:
      metrics_json.parent.mkdir(parents=True, exist_ok=True)
      metrics_json.write_text(json.dumps(metrics, indent=2))
      _log(f"wrote metrics to {metrics_json}")
    backend.env.close()
    return metrics
  finally:
    # Isaac's simulation_app.close() can hang in a shutdown spin-loop
    # (_app_control_on_stop_handle_fn -> render -> cuda.set_device). Metrics are
    # already written to JSON above, so hard-exit to avoid blocking a batch of
    # sequential validations.
    # ponytail: os._exit skips Isaac cleanup; metrics_json already persisted
    if metrics_json is not None and metrics_json.exists():
      sys.stdout.flush()
      sys.stderr.flush()
      os._exit(0)
    simulation_app.close()


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--pareto", required=True, help="npz from codesign_ga.py (gene).")
  p.add_argument(
    "--policy",
    default=None,
    help="ONNX/TorchScript policy (needed unless --plot-only).",
  )
  p.add_argument("--task", default=GENE_TASK_DEFAULT)
  p.add_argument(
    "--criteria",
    choices=["reward", "efficiency"],
    default="efficiency",
    help="Design selection from a multi-objective/sweep front.",
  )
  p.add_argument("--efficiency-min-reward-fraction", type=float, default=0.97)
  p.add_argument("--out-dir", default="codesign_results")
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--steps", type=int, default=400)
  p.add_argument("--device", default="cuda:0")
  p.add_argument(
    "--plot-only", action="store_true", help="Only make the Pareto plot; no sim."
  )
  args = p.parse_args()

  if not Path(args.pareto).exists():
    print(f"Error: Pareto file not found: {args.pareto}")
    return 1
  if not args.plot_only and args.policy is None:
    print("Error: --policy is required unless --plot-only is set.")
    return 1

  cfg = CodesignConfig(
    groups=GENE_GROUPS,
    joint_order=GENE_JOINT_ORDER,
    tau_obs_range=GENE_TAU_OBS_RANGE,
  )

  F, X, _groups = load_pareto_set(args.pareto)
  F = np.atleast_2d(F)
  X = np.atleast_1d(X)
  selected_idx = select_design(F, args.criteria, args.efficiency_min_reward_fraction)
  genome = (
    X[selected_idx] if isinstance(X[selected_idx], dict) else dict(X[selected_idx])
  )
  tau_vec, n_choices, cum_tau = decode_individual(genome, cfg)
  design = {
    g.name: round(float(tau_vec[cfg.joint_order.index(g.joints[0])]), 1)
    for g in cfg.groups
  }
  print(f"\n=== Selected design (criteria={args.criteria}) ===")
  print(f"  index={selected_idx}  cum_tau={cum_tau:.1f}Nm  motor_types={n_choices}")
  print(f"  design={design}")

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  plot_pareto(F, X, cfg, selected_idx, out_dir / "gene_codesign_pareto.png")

  if args.plot_only:
    return 0

  metrics = validate_in_isaac(
    genome,
    cfg,
    policy_path=args.policy,
    task=args.task,
    num_envs=args.num_envs,
    steps=args.steps,
    device=args.device,
    metrics_json=out_dir / "validation_metrics.json",
  )
  print("\n=== Validation rollout ===")
  for k, v in metrics.items():
    print(f"  {k:16s}: {v}")
  walk_ok = metrics["mean_vx"] >= 0.25 and metrics["min_height"] >= 0.45
  print(f"\n  walkable: {'YES' if walk_ok else 'NO'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
