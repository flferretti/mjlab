#!/usr/bin/env python3
"""Visualize GA co-design results: Pareto front, design parameters, and performance.

This script creates publication-quality plots showing:
  1. Pareto front (reward vs hardware cost)
  2. Design parameter heatmaps (joint torques)
  3. Design characteristics (custom actuator count, cumulative torque)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist

CLASS_CMAP = plt.get_cmap("tab20")
TORQUE_CLUSTER_THRESHOLD_NM = 12.0


def class_color(class_id: int):
  """Return a stable color for a torque class id."""
  return CLASS_CMAP((class_id - 1) % CLASS_CMAP.N)


def class_label(class_id: int) -> str:
  """Return a human-readable label for a torque class id."""
  return f"Torque class {class_id}"


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


def assign_torque_classes(taus: np.ndarray) -> tuple[np.ndarray, int]:
  """Assign low/mid/high torque classes using hierarchical clustering.

  The GA optimizes continuous torque limits, so the plot should visualize
  relative torque clusters rather than inventing discrete motor names.

  Returns:
    classes: Array of class ids starting at 1, sorted by increasing torque
    n_classes: Number of distinct classes present
  """
  if len(taus) <= 1:
    return np.ones(len(taus), dtype=int), len(taus)

  taus_reshaped = taus.reshape(-1, 1)
  distances = pdist(taus_reshaped, metric="euclidean")
  linkage_matrix = linkage(distances, method="complete")
  cluster_labels = fcluster(
    linkage_matrix, t=TORQUE_CLUSTER_THRESHOLD_NM, criterion="distance"
  )

  unique_clusters = np.unique(cluster_labels)
  cluster_medians = {c: np.median(taus[cluster_labels == c]) for c in unique_clusters}
  sorted_clusters = sorted(unique_clusters, key=lambda c: cluster_medians[c])

  cluster_to_class = {c: (i + 1) for i, c in enumerate(sorted_clusters)}
  classes = np.array([cluster_to_class[c] for c in cluster_labels], dtype=int)

  return classes, len(unique_clusters)


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
  """Create Pareto front plot: reward vs hardware cost (multi-objective only)."""
  if F.shape[1] != 2:
    print(f"⚠️  Skipping Pareto front plot: single-objective problem (F.shape={F.shape})")
    return

  fig, ax = plt.subplots(figsize=(10, 6))

  rewards = -F[:, 0]
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

  # Highlight best by reward
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
  ax.set_ylabel("Reward (higher = better)", fontsize=12, fontweight="bold")
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
  """Create bar plots showing joint torques and torque classes for each design (multi-objective only)."""
  if F.shape[1] != 2:
    print(f"⚠️  Skipping design comparison plot: single-objective problem (F.shape={F.shape})")
    return

  fig, axes = plt.subplots(1, len(X), figsize=(5.5 * len(X), 6), sharey=True)
  if len(X) == 1:
    axes = [axes]

  design_class_ids = []

  for design_idx, (ax, design_dict, (reward_negated, cost)) in enumerate(
    zip(axes, X, F)
  ):
    design = dict(design_dict)
    params = decode_design(design)
    reward = -reward_negated

    # Extract tau values for each group
    tau_values = np.array([params["tau"].get(g, 80.0) for g in groups])

    # Assign torque classes
    torque_classes, n_classes = assign_torque_classes(tau_values)
    design_class_ids.append(torque_classes)

    # Create bar plot with colors by torque class
    bars = ax.bar(
      range(len(groups)),
      tau_values,
      color=[class_color(tc) for tc in torque_classes],
      edgecolor="black",
      linewidth=1.5,
      alpha=0.8,
    )

    # Add value labels on bars
    for bar, tau in zip(bars, tau_values):
      height = bar.get_height()
      ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 3,
        f"{tau:.0f} Nm",
        ha="center",
        va="bottom",
        fontsize=8,
      )

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 200)
    ax.set_title(
      f"Design {design_idx} ({n_classes} torque classes)\nReward={reward:.3f}, Cost={cost:.3f}",
      fontsize=11,
      fontweight="bold",
    )
    ax.set_xlabel("Joint Group", fontsize=10)
    if design_idx == 0:
      ax.set_ylabel("Max Torque (Nm)", fontsize=11, fontweight="bold")

    ax.grid(True, alpha=0.2, axis="y")

  # Add legend for torque classes
  from matplotlib.patches import Patch

  max_class = int(max(np.max(classes) for classes in design_class_ids))
  legend_elements = [
    Patch(facecolor=class_color(i), edgecolor="black", label=class_label(i))
    for i in range(1, max_class + 1)
  ]
  fig.legend(
    handles=legend_elements,
    loc="upper center",
    ncol=3,
    fontsize=10,
    bbox_to_anchor=(0.5, 0.98),
  )

  plt.suptitle(
    "Design Parameters with Torque-Class Colors",
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
  """Create a comprehensive summary plot with torque-class assignments (multi-objective only)."""
  if F.shape[1] != 2:
    print(f"⚠️  Skipping summary plot: single-objective problem (F.shape={F.shape})")
    return

  fig = plt.figure(figsize=(18, 11))
  gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

  rewards = -F[:, 0]
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

  # 4. Design parameters heatmap with torque classes (bottom, spanning all columns)
  ax_heat = fig.add_subplot(gs[2, :])

  tau_matrix = np.zeros((len(X), len(groups)))
  class_matrix = np.zeros((len(X), len(groups)), dtype=int)

  for i, design_dict in enumerate(X):
    design = dict(design_dict)
    params = decode_design(design)
    taus = np.array([params["tau"].get(g, 80.0) for g in groups])
    tau_matrix[i] = taus
    classes, _ = assign_torque_classes(taus)
    class_matrix[i] = classes

  # Create custom colormap for torque classes
  class_colors = np.zeros((len(X), len(groups), 3))
  for i in range(len(X)):
    for j in range(len(groups)):
      class_id = int(class_matrix[i, j])
      rgb = class_color(class_id)[:3]
      class_colors[i, j] = rgb

  # Use imshow with custom RGB colors
  ax_heat.imshow(class_colors, aspect="auto")
  ax_heat.set_xticks(np.arange(len(groups)))
  ax_heat.set_yticks(np.arange(len(X)))
  ax_heat.set_xticklabels(groups, rotation=45, ha="right")
  ax_heat.set_yticklabels([f"Design {i}" for i in range(len(X))])
  ax_heat.set_ylabel("Design", fontsize=11, fontweight="bold")

  # Add text annotations with torque value only
  for i in range(len(X)):
    for j in range(len(groups)):
      tau = tau_matrix[i, j]
      text = ax_heat.text(
        j,
        i,
        f"{tau:.0f}",
        ha="center",
        va="center",
        color="white",
        fontsize=8,
      )

  ax_heat.set_title(
    "Joint Torque Limits & Torque-Class Colors",
    fontsize=12,
    fontweight="bold",
  )

  # Add torque-class legend
  from matplotlib.patches import Patch

  max_class = int(class_matrix.max()) if class_matrix.size else 0
  legend_elements = [
    Patch(facecolor=class_color(i), edgecolor="black", label=class_label(i))
    for i in range(1, max_class + 1)
  ]
  ax_heat.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.35),
    ncol=min(4, max(1, max_class)),
    fontsize=10,
    frameon=True,
  )

  # Overall title
  plt.suptitle(
    f"Co-Design GA Results: {len(X)} Designs Evaluated (Torque Classes Highlighted)",
    fontsize=15,
    fontweight="bold",
    y=0.995,
  )

  plt.savefig(output_path, dpi=300, bbox_inches="tight")
  print(f"✓ Saved summary plot to {output_path}")
  plt.close()


def create_clustering_plot(
  F: np.ndarray,
  X: np.ndarray,
  groups: list[str],
  output_path: str = "motor_clustering.png",
):
  """Create a dendrogram + sorted bar plot for the best design's torque clusters."""
  best_idx = np.argmin(F[:, 0])
  design = dict(X[best_idx])
  params = decode_design(design)
  taus = np.array([params["tau"].get(g, 80.0) for g in groups], dtype=float)

  if len(taus) <= 1:
    return

  classes, n_classes = assign_torque_classes(taus)
  sorted_idx = np.argsort(taus)
  taus_sorted = taus[sorted_idx]
  groups_sorted = [groups[i] for i in sorted_idx]
  classes_sorted = classes[sorted_idx]

  linkage_matrix = linkage(
    pdist(taus.reshape(-1, 1), metric="euclidean"), method="complete"
  )

  fig, (ax_dendro, ax_bars) = plt.subplots(
    2,
    1,
    figsize=(12, 9),
    gridspec_kw={"height_ratios": [2.0, 1.2]},
  )

  dendrogram(
    linkage_matrix,
    labels=[f"{g}\n{t:.0f} Nm" for g, t in zip(groups, taus)],
    ax=ax_dendro,
    color_threshold=TORQUE_CLUSTER_THRESHOLD_NM,
    above_threshold_color="gray",
  )
  ax_dendro.axhline(
    y=TORQUE_CLUSTER_THRESHOLD_NM,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Cut threshold = {TORQUE_CLUSTER_THRESHOLD_NM:.0f} Nm",
  )
  ax_dendro.set_ylabel("Cluster distance (Nm)")
  ax_dendro.set_title(
    f"Best Design #{best_idx}: Hierarchical Clustering of Torque Limits"
  )
  ax_dendro.legend(loc="upper right")

  bars = ax_bars.bar(
    range(len(taus_sorted)),
    taus_sorted,
    color=[class_color(c)[:3] for c in classes_sorted],
    edgecolor="black",
    linewidth=1.2,
  )
  for bar, tau, group, class_id in zip(
    bars, taus_sorted, groups_sorted, classes_sorted
  ):
    ax_bars.text(
      bar.get_x() + bar.get_width() / 2.0,
      tau + 1.0,
      f"{group}\n{tau:.0f}",
      ha="center",
      va="bottom",
      fontsize=8,
    )
  ax_bars.set_xticks([])
  ax_bars.set_ylabel("Tau max (Nm)")
  ax_bars.set_title(f"Sorted torques colored by {n_classes} cluster(s)")
  ax_bars.grid(True, axis="y", alpha=0.25)

  from matplotlib.patches import Patch

  legend_elements = [
    Patch(facecolor=class_color(i), edgecolor="black", label=class_label(i))
    for i in range(1, n_classes + 1)
  ]
  ax_bars.legend(handles=legend_elements, loc="upper right")

  plt.tight_layout()
  plt.savefig(output_path, dpi=300, bbox_inches="tight")
  print(f"✓ Saved clustering plot to {output_path}")
  plt.close()


def print_summary(F: np.ndarray, X: np.ndarray, groups: list[str]):
  """Print a text summary of the results with torque-class assignments."""
  print("\n" + "=" * 80)
  print("CO-DESIGN OPTIMIZATION RESULTS SUMMARY")
  print("=" * 80)

  print(f"\n📊 Designs evaluated: {len(X)}")
  
  if F.shape[1] != 2:
    # Single-objective
    fitnesses = F[:, 0]
    print(f"📈 Objective: Scalarized (Performance - Cost)")
    print(f"\n📊 Fitness Range: [{fitnesses.min():.3f}, {fitnesses.max():.3f}]")
    
    best_idx = np.argmin(fitnesses)
    best_fitness = fitnesses[best_idx]
    
    print(f"\n🏆 BEST DESIGN: #{best_idx}")
    print(f"   └─ Fitness: {best_fitness:.4f} (minimized)")
    
    design_dict = dict(X[best_idx])
    params = decode_design(design_dict)
    
    taus = np.array([params["tau"].get(group, 80.0) for group in groups])
    classes, n_classes = assign_torque_classes(taus)
    
    print(f"   └─ Torque classes: {n_classes} distinct class(es)")
    print("\n🤖 Best Design Torque Configuration:")
    for group, tau, class_id in zip(groups, taus, classes):
      print(f"   • {group:14s}: class {class_id} ({tau:6.1f} Nm)")
    
    print("\n   Cluster spans:")
    for class_id in sorted(np.unique(classes)):
      class_taus = taus[classes == class_id]
      print(
        f"   • class {int(class_id)}: {class_taus.min():.1f}–{class_taus.max():.1f} Nm "
        f"({len(class_taus)} joints)"
      )
    return

  # Multi-objective (legacy support)
  print("📈 Objectives: Reward (maximize) vs Cost (minimize)")

  rewards = -F[:, 0]
  costs = F[:, 1]

  print(f"\n📉 Reward Range: [{rewards.min():.3f}, {rewards.max():.3f}]")
  print(f"💰 Cost Range: [{costs.min():.2f}, {costs.max():.2f}]")

  best_idx = np.argmax(rewards)
  best_reward = rewards[best_idx]
  best_cost = costs[best_idx]

  print(f"\n🏆 BEST DESIGN: #{best_idx}")
  print(f"   └─ Reward: {best_reward:.4f} (highest)")
  print(f"   └─ Cost: {best_cost:.2f}")

  design_dict = dict(X[best_idx])
  params = decode_design(design_dict)

  taus = np.array([params["tau"].get(group, 80.0) for group in groups])
  classes, n_classes = assign_torque_classes(taus)

  print(f"   └─ Torque classes: {n_classes} distinct class(es)")
  print("\n🤖 Best Design Torque Configuration:")
  for group, tau, class_id in zip(groups, taus, classes):
    print(f"   • {group:14s}: class {class_id} ({tau:6.1f} Nm)")

  print("\n   Cluster spans:")
  for class_id in sorted(np.unique(classes)):
    class_taus = taus[classes == class_id]
    print(
      f"   • class {int(class_id)}: {class_taus.min():.1f}–{class_taus.max():.1f} Nm "
      f"({len(class_taus)} joints)"
    )

  # Most efficient design
  efficiency = rewards / (costs + 1e-6)  # Reward / cost
  efficient_idx = np.argmax(efficiency)

  if efficient_idx != best_idx:
    print(f"\n⚡ MOST EFFICIENT DESIGN: #{efficient_idx}")
    print(f"   └─ Reward/Cost Ratio: {efficiency[efficient_idx]:.4f}")
    print(f"   └─ Reward: {rewards[efficient_idx]:.4f}")
    print(f"   └─ Cost: {costs[efficient_idx]:.2f}")

    design_dict = dict(X[efficient_idx])
    params = decode_design(design_dict)
    taus = np.array([params["tau"].get(group, 80.0) for group in groups])
    classes, n_classes = assign_torque_classes(taus)
    print(f"   └─ Torque classes: {n_classes} distinct class(es)")

  print("\n" + "=" * 80)
  print("TORQUE CLASS REFERENCE:")
  print("=" * 80)
  for class_id in sorted(np.unique(classes)):
    print(f"  {class_label(int(class_id))}")
  print("=" * 80)


def create_fitness_plot(
  F: np.ndarray,
  X: np.ndarray,
  groups: list[str],
  output_path: str = "fitness_landscape.png",
):
  """Create fitness plot for single-objective optimization."""
  if F.shape[1] != 1:
    return

  fitnesses = F[:, 0]
  
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
  
  # Plot 1: Fitness distribution
  best_idx = np.argmin(fitnesses)
  ax1.bar(
    range(len(fitnesses)),
    fitnesses,
    color=['red' if i == best_idx else 'steelblue' for i in range(len(fitnesses))],
    edgecolor='black',
    linewidth=1.5,
    alpha=0.8,
  )
  ax1.set_xlabel("Design Index", fontsize=12, fontweight="bold")
  ax1.set_ylabel("Fitness (lower = better)", fontsize=12, fontweight="bold")
  ax1.set_title("Fitness Distribution Across Designs", fontsize=13, fontweight="bold")
  ax1.grid(True, alpha=0.3, axis='y')
  
  # Annotate best
  ax1.scatter([best_idx], [fitnesses[best_idx]], s=300, marker='*', color='gold',
              edgecolors='red', linewidth=2, zorder=10, label=f"Best (#{best_idx})")
  ax1.legend(fontsize=11)
  
  # Plot 2: Best design torques
  design_dict = dict(X[best_idx])
  params = decode_design(design_dict)
  taus = np.array([params["tau"].get(group, 80.0) for group in groups])
  classes, n_classes = assign_torque_classes(taus)
  
  colors = [class_color(tc) for tc in classes]
  ax2.bar(
    range(len(groups)),
    taus,
    color=colors,
    edgecolor='black',
    linewidth=1.5,
    alpha=0.8,
  )
  ax2.set_xlabel("Joint Group", fontsize=12, fontweight="bold")
  ax2.set_ylabel("Torque Limit (Nm)", fontsize=12, fontweight="bold")
  ax2.set_title(f"Best Design Torques ({n_classes} classes)", fontsize=13, fontweight="bold")
  ax2.set_xticks(range(len(groups)))
  ax2.set_xticklabels(groups, rotation=45, ha='right', fontsize=10)
  ax2.grid(True, alpha=0.3, axis='y')
  
  plt.tight_layout()
  plt.savefig(output_path, dpi=300, bbox_inches="tight")
  print(f"✓ Saved fitness plot to {output_path}")
  plt.close()


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

  # Single-objective: fitness plot
  if F.shape[1] == 1:
    create_fitness_plot(F, X, groups, output_dir / "fitness_landscape.png")
  else:
    # Multi-objective: Pareto plots
    create_pareto_plot(F, X, output_dir / "pareto_front.png")
    create_design_comparison_plot(F, X, groups, output_dir / "design_comparison.png")
    create_summary_plot(F, X, groups, output_dir / "codesign_summary.png")
    create_clustering_plot(F, X, groups, output_dir / "motor_clustering.png")

  print(f"\n✓ All plots saved to {output_dir}")
  return 0


if __name__ == "__main__":
  exit(main())
