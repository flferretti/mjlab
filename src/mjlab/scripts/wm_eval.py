"""Evaluate a trained world model on held-out designs.

Scores interpolation/extrapolation holdout designs in imagination and, when
requested, ground-truths them with true-simulation rollouts (same protocol as
the GA's MjlabBackend: frozen policy, fixed forward command, mean step reward
over the rollout). Reports ranking metrics (Spearman, pairwise accuracy,
top-10 regret) and saves raw predictions for downstream analysis.
"""

import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import tyro

import mjlab
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.utils.torch import configure_torch_backends
from mjlab.world_model import CollectCfg, DesignEvaluator, ImaginationCfg
from mjlab.world_model.collect import build_collect_env, load_policy
from mjlab.world_model.design_space import DesignSpaceCfg
from mjlab.world_model.metrics import (
  pairwise_ranking_accuracy,
  spearman,
  topk_regret,
)


@dataclass(frozen=True)
class WmEvalConfig:
  checkpoint: str
  """World-model checkpoint (wm-train output)."""
  policy: str | None = None
  """Frozen policy for imagination + true-sim rollouts."""
  split: Literal["interp", "extrap"] = "interp"
  n_designs: int = 200
  n_seeds: int = 4
  imagination: ImaginationCfg = field(default_factory=ImaginationCfg)
  true_sim: bool = False
  """Ground-truth the designs with live simulation rollouts (needs the sim)."""
  true_sim_envs: int = 512
  ground_truth: str | None = None
  """npz with arrays theta (N, D) and performance (N,) as an alternative to
  --true-sim (e.g. produced by an earlier run)."""
  out: str = "wm_eval"
  """Output prefix: writes <out>.json and <out>.npz."""
  device: str | None = None


def main() -> None:
  maybe_print_top_level_help("wm-eval")
  cfg = tyro.cli(WmEvalConfig, config=mjlab.TYRO_FLAGS)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  evaluator = DesignEvaluator(
    cfg.checkpoint, torch.nn.Identity(), device, cfg.imagination
  )
  task = evaluator.meta["task"]
  evaluator.policy = load_policy(
    task, cfg.policy, device, evaluator.model.dims.action_dim
  )
  space = evaluator.design_space
  if cfg.split == "interp":
    theta = space.holdout_designs(cfg.n_designs, seed=0)
  else:
    theta = space.sample(cfg.n_designs, seed=1, region="extrap")

  scores = evaluator.evaluate_designs(theta, cfg.n_seeds)
  results: dict = {
    "checkpoint": cfg.checkpoint,
    "task": task,
    "split": cfg.split,
    "n_designs": cfg.n_designs,
    "n_seeds": cfg.n_seeds,
    "design_space": space.name,
  }

  true_perf: torch.Tensor | None = None
  if cfg.ground_truth is not None:
    gt = np.load(cfg.ground_truth)
    if not np.allclose(gt["theta"], theta.numpy(), atol=1e-5):
      raise SystemExit(
        "--ground-truth designs do not match this split; regenerate with the "
        "same split/seed."
      )
    true_perf = torch.as_tensor(gt["performance"], dtype=torch.float32)
  elif cfg.true_sim:
    true_perf = _true_sim_scores(
      task,
      space,
      theta,
      cfg.policy,
      cfg.n_seeds,
      cfg.imagination.rollout_steps,
      cfg.true_sim_envs,
      device,
    )

  if true_perf is not None:
    results["spearman"] = spearman(scores.performance, true_perf)
    results["pairwise_accuracy"] = pairwise_ranking_accuracy(
      scores.performance, true_perf
    )
    results["top10_regret"] = topk_regret(scores.performance, true_perf, k=10)

  np.savez(
    f"{cfg.out}.npz",
    theta=theta.numpy(),
    performance=scores.performance.numpy(),
    rms_torque=scores.rms_torque.numpy(),
    uncertainty=scores.uncertainty.numpy(),
    true_performance=(true_perf.numpy() if true_perf is not None else np.array([])),
  )
  with open(f"{cfg.out}.json", "w") as f:
    json.dump(results, f, indent=2)
  printable = {k: v for k, v in results.items() if isinstance(v, (int, float))}
  print(f"[wm-eval] {cfg.split}: {printable}")
  print(f"[wm-eval] Wrote {cfg.out}.json / {cfg.out}.npz")


@torch.no_grad()
def _true_sim_scores(
  task: str,
  space: DesignSpaceCfg,
  theta: torch.Tensor,
  policy_path: str | None,
  n_seeds: int,
  rollout_steps: int,
  num_envs: int,
  device: str,
) -> torch.Tensor:
  """Mean step reward per design from live rollouts (MjlabBackend protocol)."""
  collect_cfg = CollectCfg(num_envs=num_envs, steps=rollout_steps, seed=0)
  # Designs are injected explicitly below instead of via the reset event.
  env = build_collect_env(task, collect_cfg, space, device, with_design_event=False)
  try:
    policy = load_policy(task, policy_path, device, env.action_manager.total_action_dim)
    env_ids = torch.arange(num_envs, device=device)
    per_design: list[float] = []
    designs_per_batch = num_envs // n_seeds
    for start in range(0, theta.shape[0], designs_per_batch):
      block = theta[start : start + designs_per_batch].to(device)
      rep = block.repeat_interleave(n_seeds, dim=0)
      if rep.shape[0] < num_envs:
        rep = torch.cat([rep, rep[-1:].expand(num_envs - rep.shape[0], -1)], dim=0)
      space.apply(env, env_ids, rep)
      obs, _ = env.reset()
      space.apply(env, env_ids, rep)  # Reset events may touch limits.
      total = torch.zeros(num_envs, device=device)
      for _ in range(rollout_steps):
        actions = policy(obs["actor"])
        obs, reward, _, _, _ = env.step(actions)
        total += reward
        space.apply(env, env_ids, rep)  # Keep designs through auto-resets.
      mean_r = (total / rollout_steps)[: block.shape[0] * n_seeds]
      per_design.extend(mean_r.view(block.shape[0], n_seeds).mean(dim=1).cpu().tolist())
    return torch.tensor(per_design, dtype=torch.float32)
  finally:
    env.close()


if __name__ == "__main__":
  main()
