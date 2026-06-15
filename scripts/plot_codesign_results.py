#!/usr/bin/env python3
"""Visualize GA co-design results: Pareto front, design parameters, and performance.

This script creates publication-quality plots showing:
  1. Pareto front (reward vs hardware cost)
  2. Design parameter heatmaps (joint torques)
  3. Design characteristics (custom actuator count, cumulative torque)
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# Motor type definitions (Nm capacity ranges)
MOTOR_TYPES = {
  1: {"capacity": 80, "color": "#3498db", "label": "Type 1 (80 Nm)"},    # Blue
  2: {"capacity": 110, "color": "#e74c3c", "label": "Type 2 (110 Nm)"},  # Red
  3: {"capacity": 180, "color": "#f39c12", "label": "Type 3 (180 Nm)"},  # Orange
}

# QDD robot constants
QDD_GROUPS = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
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


def load_pareto_set(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
  """Load objectives (F), designs (X), and joint group names from Pareto npz."""
  data = np.load(path, allow_pickle=True)
  return data["F"], data["X"], data["groups"]


def assign_motor_types(taus: np.ndarray) -> tuple[np.ndarray, int]:
  """Assign motor types based on best fit (minimizing unused capacity).
  
  For each torque demand, selects the smallest motor type that can handle it.
  
  Returns:
    motor_types: Array of motor type (1, 2, or 3) for each joint
    n_types: Total number of distinct motor types used
  """
  # Motor type capacities
  capacities = {1: 80, 2: 110, 3: 180}
  
  motor_types = np.zeros(len(taus), dtype=int)
  for i, tau in enumerate(taus):
    # Find smallest type that fits
    for mtype in [1, 2, 3]:
      if tau <= capacities[mtype]:
        motor_types[i] = mtype
        break
    else:
      # Fallback to largest type
      motor_types[i] = 3
  
  # Count distinct types actually used
  n_distinct = len(np.unique(motor_types))
  
  return motor_types, n_distinct


def decode_design(design_dict: dict) -> dict:
  """Extract design parameters: torque limits per joint group."""
  params = {
    "tau": {},
  }
  for key, value in design_dict.items():
    if key.startswith("tau_"):
      group = key[4:]
      params["tau"][group] = float(value)
  return params


def create_pareto_plot(
  F: np.ndarray, X: np.ndarray, output_path: str = "pareto_front.png"
):
  """Create Pareto front plot: reward vs hardware cost."""
  fig, ax = plt.subplots(figsize=(10, 6))

  rewards = F[:, 0]
  costs = F[:, 1]

  # Sort by cost for cleaner line
  order = np.argsort(costs)

  # Plot Pareto front
  ax.plot(
    costs[order], rewards[order], "b-", linewidth=2.5, alpha=0.7, label="Pareto Front"
  )

  # Plot individual designs as points
  scatter = ax.scatter(
    costs,
    rewards,
    c=rewards,
    cmap="viridis",
    s=150,
    edgecolors="black",
    linewidth=1.5,
    zorder=5,
    alpha=0.8,
  )

  # Annotate each point with its index
  for i, (cost, reward) in enumerate(zip(costs, rewards)):
    ax.annotate(
      f"{i}",
      xy=(cost, reward),
      xytext=(5, 5),
      textcoords="offset points",
      fontsize=10,
      fontweight="bold",
    )

  # Highlight best by reward (highest/least negative)
  best_idx = np.argmax(rewards)
  ax.scatter(
    [costs[best_idx]],
    [rewards[best_idx]],
    s=400,
    marker="*",
    color="gold",
    edgecolors="red",
    linewidth=2.5,
    zorder=10,
    label=f"Best Design (#{best_idx})",
  )

  ax.set_xlabel(
    "Hardware Cost (torque capacity penalty)", fontsize=12, fontweight="bold"
  )
  ax.set_ylabel("Reward (negative = good)", fontsize=12, fontweight="bold")
  ax.set_title(
    "Co-Design Pareto Front: Performance vs Hardware Trade-off",
    fontsize=14,
    fontweight="bold",
  )
  ax.grid(True, alpha=0.3)
  ax.legend(fontsize=11, loc="best")

  plt.tight_layout()
  plt.savefig(output_path, dpi=300, bbox_inches="tight")
  print(f"✓ Saved Pareto front to {output_path}")
  plt.close()


def create_design_comparison_plot(
  F: np.ndarray,
  X: np.ndarray,
  groups: list[str],
  output_path: str = "design_comparison.png",
):
  """Create heatmap showing joint torques and motor types for each design."""
  fig, axes = plt.subplots(1, len(X), figsize=(5.5 * len(X), 6), sharey=True)
  if len(X) == 1:
    axes = [axes]

  for design_idx, (ax, design_dict, (reward, cost)) in enumerate(zip(axes, X, F)):
    design = dict(design_dict)
    params = decode_design(design)

    # Extract tau values for each group
    tau_values = np.array([params["tau"].get(g, 80.0) for g in groups])
    
    # Assign motor types
    motor_types, n_types = assign_motor_types(tau_values)

    # Create bar plot with colors by motor type
    bars = ax.bar(
      range(len(groups)),
      tau_values,
      color=[MOTOR_TYPES[mt]["color"] for mt in motor_types],
      edgecolor="black",
      linewidth=1.5,
      alpha=0.8,
    )

    # Add value labels and motor type on bars
    for bar, tau, mtype in zip(bars, tau_values, motor_types):
      height = bar.get_height()
      ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 3,
        f"T{mtype}\n{tau:.0f}Nm",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
      )

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 200)
    ax.set_title(
      f"Design {design_idx} ({n_types} motor types)\nReward={reward:.3f}, Cost={cost:.3f}",
      fontsize=11,
      fontweight="bold",
    )
    ax.set_xlabel("Joint Group", fontsize=10)
    if design_idx == 0:
      ax.set_ylabel("Max Torque (Nm)", fontsize=11, fontweight="bold")
    
    ax.grid(True, alpha=0.2, axis="y")

  # Add legend for motor types
  from matplotlib.patches import Patch
  legend_elements = [
    Patch(facecolor=MOTOR_TYPES[i]["color"], edgecolor="black", label=MOTOR_TYPES[i]["label"])
    for i in [1, 2, 3]
  ]
  fig.legend(handles=legend_elements, loc="upper center", ncol=3, fontsize=10, 
             bbox_to_anchor=(0.5, 0.98))

  plt.suptitle(
    "Design Parameters with Motor Type Assignments (T1/T2/T3)",
    fontsize=14,
    fontweight="bold",
    y=0.995,
  )
  plt.tight_layout(rect=[0, 0, 1, 0.96])
  plt.savefig(output_path, dpi=300, bbox_inches="tight")
  print(f"✓ Saved design comparison to {output_path}")
  plt.close()


def create_summary_plot(
  F: np.ndarray,
  X: np.ndarray,
  groups: list[str],
  output_path: str = "codesign_summary.png",
):
  """Create a comprehensive summary plot with motor type assignments."""
  fig = plt.figure(figsize=(18, 11))
  gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

  rewards = F[:, 0]
  costs = F[:, 1]

  # 1. Main Pareto front (large, top-left)
  ax_pareto = fig.add_subplot(gs[0:2, 0:2])
  order = np.argsort(costs)
  ax_pareto.plot(costs[order], rewards[order], "b-", linewidth=2.5, alpha=0.7)
  scatter = ax_pareto.scatter(
    costs,
    rewards,
    c=rewards,
    cmap="viridis",
    s=200,
    edgecolors="black",
    linewidth=1.5,
    zorder=5,
    alpha=0.8,
  )
  for i, (cost, reward) in enumerate(zip(costs, rewards)):
    ax_pareto.annotate(
      f"{i}",
      xy=(cost, reward),
      xytext=(5, 5),
      textcoords="offset points",
      fontsize=10,
      fontweight="bold",
    )
  best_idx = np.argmax(rewards)
  ax_pareto.scatter(
    [costs[best_idx]],
    [rewards[best_idx]],
    s=500,
    marker="*",
    color="gold",
    edgecolors="red",
    linewidth=2.5,
    zorder=10,
  )
  ax_pareto.set_xlabel("Hardware Cost", fontsize=11, fontweight="bold")
  ax_pareto.set_ylabel("Reward", fontsize=11, fontweight="bold")
  ax_pareto.set_title("Pareto Front", fontsize=12, fontweight="bold")
  ax_pareto.grid(True, alpha=0.3)

  # 2. Cost distribution (top-right)
  ax_cost = fig.add_subplot(gs[0, 2])
  ax_cost.hist(costs, bins=8, color="#ff6b6b", edgecolor="black", alpha=0.7)
  ax_cost.axvline(
    costs[best_idx],
    color="gold",
    linewidth=2.5,
    linestyle="--",
    label=f"Best: {costs[best_idx]:.2f}",
  )
  ax_cost.set_xlabel("Cost", fontsize=10)
  ax_cost.set_ylabel("Count", fontsize=10)
  ax_cost.set_title("Cost Distribution", fontsize=11, fontweight="bold")
  ax_cost.legend(fontsize=9)
  ax_cost.grid(True, alpha=0.3, axis="y")

  # 3. Reward distribution (middle-right)
  ax_reward = fig.add_subplot(gs[1, 2])
  ax_reward.hist(rewards, bins=8, color="#4ecdc4", edgecolor="black", alpha=0.7)
  ax_reward.axvline(
    rewards[best_idx],
    color="gold",
    linewidth=2.5,
    linestyle="--",
    label=f"Best: {rewards[best_idx]:.3f}",
  )
  ax_reward.set_xlabel("Reward", fontsize=10)
  ax_reward.set_ylabel("Count", fontsize=10)
  ax_reward.set_title("Reward Distribution", fontsize=11, fontweight="bold")
  ax_reward.legend(fontsize=9)
  ax_reward.grid(True, alpha=0.3, axis="y")

  # 4. Design parameters heatmap with motor types (bottom, spanning all columns)
  ax_heat = fig.add_subplot(gs[2, :])

  tau_matrix = np.zeros((len(X), len(groups)))
  mtype_matrix = np.zeros((len(X), len(groups)), dtype=int)
  
  for i, design_dict in enumerate(X):
    design = dict(design_dict)
    params = decode_design(design)
    taus = np.array([params["tau"].get(g, 80.0) for g in groups])
    tau_matrix[i] = taus
    mtypes, _ = assign_motor_types(taus)
    mtype_matrix[i] = mtypes

  # Create custom colormap for motor types
  motor_type_colors = np.zeros((len(X), len(groups), 3))
  for i in range(len(X)):
    for j in range(len(groups)):
      mtype = int(mtype_matrix[i, j])
      rgb = to_rgb(MOTOR_TYPES[mtype]["color"])
      motor_type_colors[i, j] = rgb

  # Use imshow with custom RGB colors
  ax_heat.imshow(motor_type_colors, aspect="auto")
  ax_heat.set_xticks(np.arange(len(groups)))
  ax_heat.set_yticks(np.arange(len(X)))
  ax_heat.set_xticklabels(groups, rotation=45, ha="right")
  ax_heat.set_yticklabels([f"Design {i}" for i in range(len(X))])
  ax_heat.set_ylabel("Design", fontsize=11, fontweight="bold")

  # Add text annotations with torque value and motor type
  for i in range(len(X)):
    for j in range(len(groups)):
      tau = tau_matrix[i, j]
      mtype = int(mtype_matrix[i, j])
      text = ax_heat.text(
        j,
        i,
        f"T{mtype}\n{tau:.0f}",
        ha="center",
        va="center",
        color="white",
        fontsize=8,
        fontweight="bold",
      )

  ax_heat.set_title(
    "Joint Torque Limits & Motor Type Assignments",
    fontsize=12,
    fontweight="bold",
  )

  # Add motor type legend
  from matplotlib.patches import Patch
  legend_elements = [
    Patch(facecolor=MOTOR_TYPES[i]["color"], edgecolor="black", label=MOTOR_TYPES[i]["label"])
    for i in [1, 2, 3]
  ]
  ax_heat.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.35),
    ncol=3,
    fontsize=10,
    frameon=True,
  )

  # Overall title
  plt.suptitle(
    f"Co-Design GA Results: {len(X)} Designs Evaluated (Motor Types Highlighted)",
    fontsize=15,
    fontweight="bold",
    y=0.995,
  )

  plt.savefig(output_path, dpi=300, bbox_inches="tight")
  print(f"✓ Saved summary plot to {output_path}")
  plt.close()


def print_summary(F: np.ndarray, X: np.ndarray, groups: list[str]):
  """Print a text summary of the results with motor type assignments."""
  print("\n" + "=" * 80)
  print("CO-DESIGN OPTIMIZATION RESULTS SUMMARY")
  print("=" * 80)

  print(f"\n📊 Designs evaluated: {len(X)}")
  print(f"📈 Objectives: Reward (maximize) vs Cost (minimize)")

  rewards = F[:, 0]
  costs = F[:, 1]

  # Note: rewards are stored as negative (for minimization)
  print(f"\n📉 Reward Range: [{rewards.min():.3f}, {rewards.max():.3f}]")
  print(f"💰 Cost Range: [{costs.min():.2f}, {costs.max():.2f}]")

  best_idx = np.argmax(rewards)
  best_reward = rewards[best_idx]
  best_cost = costs[best_idx]

  print(f"\n🏆 BEST DESIGN: #{best_idx}")
  print(f"   └─ Reward: {-best_reward:.4f} (highest)")
  print(f"   └─ Cost: {best_cost:.2f}")

  design_dict = dict(X[best_idx])
  params = decode_design(design_dict)

  taus = np.array([params["tau"].get(group, 80.0) for group in groups])
  mtypes, n_types = assign_motor_types(taus)

  print(f"   └─ Motor Types: {n_types} distinct type(s)")
  print(f"\n🤖 Best Design Torque Configuration with Motor Types:")
  for group, tau, mtype in zip(groups, taus, mtypes):
    motor_label = MOTOR_TYPES[mtype]["label"]
    utilization = (tau / MOTOR_TYPES[mtype]["capacity"]) * 100
    print(f"   • {group:14s}: Type {mtype} ({tau:6.1f} Nm @ {utilization:5.1f}% util.)")

  # Most efficient design
  efficiency = -rewards / (costs + 1e-6)  # Positive reward / cost
  efficient_idx = np.argmax(efficiency)

  if efficient_idx != best_idx:
    print(f"\n⚡ MOST EFFICIENT DESIGN: #{efficient_idx}")
    print(f"   └─ Reward/Cost Ratio: {efficiency[efficient_idx]:.4f}")
    print(f"   └─ Reward: {rewards[efficient_idx]:.4f}")
    print(f"   └─ Cost: {costs[efficient_idx]:.2f}")
    
    design_dict = dict(X[efficient_idx])
    params = decode_design(design_dict)
    taus = np.array([params["tau"].get(group, 80.0) for group in groups])
    mtypes, n_types = assign_motor_types(taus)
    print(f"   └─ Motor Types: {n_types} distinct type(s)")

  print("\n" + "=" * 80)
  print("MOTOR TYPE REFERENCE:")
  print("=" * 80)
  for mtype in [1, 2, 3]:
    info = MOTOR_TYPES[mtype]
    print(f"  {info['label']}: Capacity {info['capacity']} Nm")
  print("=" * 80)


def main():
  parser = argparse.ArgumentParser(
    description="Visualize GA co-design results with Pareto front and design parameters.",
  )
  parser.add_argument(
    "--pareto",
    default="codesign_pareto.npz",
    help="Path to Pareto npz file from codesign_ga.py",
  )
  parser.add_argument(
    "--output-dir",
    default=".",
    help="Directory to save plots",
  )
  parser.add_argument(
    "--summary-only",
    action="store_true",
    help="Only print text summary, don't create plots",
  )

  args = parser.parse_args()

  # Load data
  if not Path(args.pareto).exists():
    print(f"Error: Pareto file not found: {args.pareto}")
    return 1

  F, X, groups = load_pareto_set(args.pareto)

  # Print summary
  print_summary(F, X, groups)

  if args.summary_only:
    return 0

  # Create plots
  output_dir = Path(args.output_dir)
  output_dir.mkdir(exist_ok=True)

  create_pareto_plot(F, X, output_dir / "pareto_front.png")
  create_design_comparison_plot(F, X, groups, output_dir / "design_comparison.png")
  create_summary_plot(F, X, groups, output_dir / "codesign_summary.png")

  print(f"\n✓ All plots saved to {output_dir}")
  return 0


if __name__ == "__main__":
  exit(main())
