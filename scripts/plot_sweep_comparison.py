# SPDX-FileCopyrightText: Generative Bionics S.R.L.
# SPDX-License-Identifier: LicenseRef-GenerativeBionics-AllRightsReserved

"""Cross-run comparison plots for a codesign_ga.py sweep.

Overlays every run's Pareto front on a single, unit-consistent axis
(reward vs cumulative torque in Nm, recomputed from the designs so legacy and
powerlaw cost models are comparable), plus the frozen-policy baseline front,
and marks each run's efficiency pick. Reuses the npz files written by
``codesign_ga.py --multiobjective``.

Usage:
  uv run --no-sync python scripts/plot_sweep_comparison.py \
      --sweep-dir codesign_results/sweep62000 \
      --baseline codesign_results/gene_multiobj62000_large_pareto.npz \
      --output-dir codesign_results/sweep62000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# torso joints are single-actuator; every other group is a symmetric l/r pair.
SINGLE_JOINT_GROUPS = {"torso_yaw", "torso_roll"}
EFF_REWARD_FRACTION = 0.97  # matches codesign_ga efficiency-pick threshold


def cum_tau_nm(design: dict, groups: list[str]) -> float:
  """Total actuator torque budget in Nm (2x paired groups + 1x torso)."""
  total = 0.0
  for g in groups:
    n = 1 if g in SINGLE_JOINT_GROUPS else 2
    total += n * float(design[f"tau_{g}"])
  return total


def load_run(path: Path) -> dict:
  d = np.load(path, allow_pickle=True)
  F, X = d["F"], d["X"]
  groups = [str(g) for g in d["groups"]]
  reward = -F[:, 0]
  tau = np.array([cum_tau_nm(dict(x), groups) for x in X])
  order = np.argsort(tau)
  return {"reward": reward[order], "tau": tau[order]}


def efficiency_pick(reward: np.ndarray, tau: np.ndarray) -> tuple[float, float]:
  """Best reward/torque design among near-optimal-reward designs."""
  keep = reward >= EFF_REWARD_FRACTION * reward.max()
  eff = reward[keep] / tau[keep]
  i = np.argmax(eff)
  return reward[keep][i], tau[keep][i]


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--sweep-dir", required=True)
  p.add_argument("--baseline", default=None, help="Optional baseline front npz")
  p.add_argument("--output-dir", default=None)
  args = p.parse_args()

  sweep_dir = Path(args.sweep_dir)
  npzs = sorted(sweep_dir.glob("*.npz"))
  if not npzs:
    print(f"No npz files in {sweep_dir}")
    return 1
  out_dir = Path(args.output_dir or args.sweep_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  runs = {}
  for path in npzs:
    tag = path.stem.replace("gene62000rms_", "").replace("gene62000_", "")
    runs[tag] = load_run(path)

  fig, (ax_f, ax_e) = plt.subplots(1, 2, figsize=(15, 6))
  cmap = plt.get_cmap("tab10")

  # ---- Left: overlaid Pareto fronts (reward vs cum torque Nm) ----
  if args.baseline and Path(args.baseline).exists():
    b = load_run(Path(args.baseline))
    ax_f.plot(
      b["tau"],
      b["reward"],
      "k--o",
      lw=2,
      ms=4,
      label="baseline (pop96/gen40)",
      zorder=1,
    )
  for i, (tag, r) in enumerate(runs.items()):
    c = cmap(i % 10)
    ax_f.plot(r["tau"], r["reward"], "-o", color=c, ms=5, label=tag, zorder=2)
    er, et = efficiency_pick(r["reward"], r["tau"])
    ax_f.scatter([et], [er], color=c, s=180, marker="*", edgecolor="k", zorder=3)
  ax_f.set_xlabel("Cumulative actuator torque [Nm]  (lower = lighter/cheaper)")
  ax_f.set_ylabel("Task reward  (higher = better)")
  ax_f.set_title("Pareto fronts across sweep\n(★ = efficiency pick per run)")
  ax_f.grid(True, alpha=0.3)
  ax_f.legend(fontsize=8, loc="lower right")

  # ---- Right: efficiency-pick summary bars ----
  tags = list(runs.keys())
  picks = [efficiency_pick(runs[t]["reward"], runs[t]["tau"]) for t in tags]
  rewards = [pr for pr, _ in picks]
  taus = [pt for _, pt in picks]
  x = np.arange(len(tags))
  ax_t = ax_e.twinx()
  ax_e.bar(x - 0.2, rewards, 0.4, color="steelblue", label="reward")
  ax_t.bar(x + 0.2, taus, 0.4, color="indianred", label="cum torque [Nm]")
  ax_e.set_xticks(x)
  ax_e.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
  ax_e.set_ylabel("Reward", color="steelblue")
  ax_t.set_ylabel("Cumulative torque [Nm]", color="indianred")
  ax_e.set_title("Efficiency pick per run: reward vs torque budget")
  for xi, (pr, pt) in zip(x, picks, strict=True):
    ax_e.annotate(f"{pr:.3f}", (xi - 0.2, pr), ha="center", va="bottom", fontsize=7)
    ax_t.annotate(f"{pt:.0f}", (xi + 0.2, pt), ha="center", va="bottom", fontsize=7)

  fig.tight_layout()
  out = out_dir / "sweep_comparison.png"
  fig.savefig(out, dpi=150, bbox_inches="tight")
  print(f"Saved {out}")

  # Text summary.
  print("\nrun                 eff_reward   eff_cum_tau_Nm")
  for t, (pr, pt) in zip(tags, picks, strict=True):
    print(f"{t:<20}{pr:>10.3f}   {pt:>12.1f}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
