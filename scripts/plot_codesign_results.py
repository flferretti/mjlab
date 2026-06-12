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
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# QDD robot constants
QDD_GROUPS = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
QDD_JOINT_ORDER = [
    "l_hip_pitch", "r_hip_pitch", "l_hip_roll", "r_hip_roll",
    "l_hip_yaw", "r_hip_yaw", "l_knee", "r_knee",
    "l_ankle_pitch", "r_ankle_pitch", "l_ankle_roll", "r_ankle_roll",
]


def load_pareto_set(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load objectives (F), designs (X), and joint group names from Pareto npz."""
    data = np.load(path, allow_pickle=True)
    return data["F"], data["X"], data["groups"]


def decode_design(design_dict: dict) -> dict:
    """Extract design parameters: custom actuator flags and torque limits."""
    params = {
        "use_custom": {},
        "tau": {},
    }
    for key, value in design_dict.items():
        if key.startswith("z_"):
            group = key[2:]
            params["use_custom"][group] = bool(value)
        elif key.startswith("tau_"):
            group = key[4:]
            params["tau"][group] = float(value)
    return params


def create_pareto_plot(F: np.ndarray, X: np.ndarray, output_path: str = "pareto_front.png"):
    """Create Pareto front plot: reward vs hardware cost."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rewards = F[:, 0]
    costs = F[:, 1]
    
    # Sort by cost for cleaner line
    order = np.argsort(costs)
    
    # Plot Pareto front
    ax.plot(costs[order], rewards[order], 'b-', linewidth=2.5, alpha=0.7, label='Pareto Front')
    
    # Plot individual designs as points
    scatter = ax.scatter(costs, rewards, c=rewards, cmap='viridis', s=150, 
                        edgecolors='black', linewidth=1.5, zorder=5, alpha=0.8)
    
    # Annotate each point with its index
    for i, (cost, reward) in enumerate(zip(costs, rewards)):
        ax.annotate(f'{i}', xy=(cost, reward), xytext=(5, 5), 
                   textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Highlight best by reward (highest/least negative)
    best_idx = np.argmax(rewards)
    ax.scatter([costs[best_idx]], [rewards[best_idx]], s=400, marker='*', 
              color='gold', edgecolors='red', linewidth=2.5, zorder=10, 
              label=f'Best Design (#{best_idx})')
    
    ax.set_xlabel('Hardware Cost (# custom + Σ τ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reward (negative = good)', fontsize=12, fontweight='bold')
    ax.set_title('Co-Design Pareto Front: Performance vs Hardware Trade-off', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Pareto front to {output_path}")
    plt.close()


def create_design_comparison_plot(F: np.ndarray, X: np.ndarray, groups: list[str],
                                  output_path: str = "design_comparison.png"):
    """Create heatmap showing joint torques for each design."""
    fig, axes = plt.subplots(1, len(X), figsize=(4 * len(X), 5), sharey=True)
    if len(X) == 1:
        axes = [axes]
    
    for design_idx, (ax, design_dict, (reward, cost)) in enumerate(
        zip(axes, X, F)
    ):
        design = dict(design_dict)
        params = decode_design(design)
        
        # Extract tau values for each group
        tau_values = [params["tau"].get(g, 80.0) for g in groups]
        
        # Create bar plot
        colors = []
        for g in groups:
            if params["use_custom"].get(g, False):
                colors.append('#ff6b6b')  # Red for custom
            else:
                colors.append('#4ecdc4')  # Teal for standard
        
        bars = ax.bar(groups, tau_values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, tau_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylim(0, 120)
        ax.set_title(f'Design {design_idx}\nR={reward:.3f}, C={cost:.3f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Joint Group', fontsize=10)
        if design_idx == 0:
            ax.set_ylabel('Max Torque (Nm)', fontsize=11, fontweight='bold')
        
        # Rotate x labels
        ax.set_xticklabels(groups, rotation=45, ha='right')
    
    # Add legend
    custom_patch = mpatches.Patch(color='#ff6b6b', label='Custom Actuator')
    standard_patch = mpatches.Patch(color='#4ecdc4', label='Standard Actuator')
    fig.legend(handles=[custom_patch, standard_patch], loc='upper right', 
              bbox_to_anchor=(0.98, 0.95), fontsize=10)
    
    plt.suptitle('Design Parameters Across Pareto Set', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved design comparison to {output_path}")
    plt.close()


def create_summary_plot(F: np.ndarray, X: np.ndarray, groups: list[str],
                       output_path: str = "codesign_summary.png"):
    """Create a comprehensive summary plot with multiple subplots."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    rewards = F[:, 0]
    costs = F[:, 1]
    
    # 1. Main Pareto front (large, top-left)
    ax_pareto = fig.add_subplot(gs[0:2, 0:2])
    order = np.argsort(costs)
    ax_pareto.plot(costs[order], rewards[order], 'b-', linewidth=2.5, alpha=0.7)
    scatter = ax_pareto.scatter(costs, rewards, c=rewards, cmap='viridis', s=200, 
                               edgecolors='black', linewidth=1.5, zorder=5, alpha=0.8)
    for i, (cost, reward) in enumerate(zip(costs, rewards)):
        ax_pareto.annotate(f'{i}', xy=(cost, reward), xytext=(5, 5), 
                          textcoords='offset points', fontsize=10, fontweight='bold')
    best_idx = np.argmax(rewards)
    ax_pareto.scatter([costs[best_idx]], [rewards[best_idx]], s=500, marker='*', 
                     color='gold', edgecolors='red', linewidth=2.5, zorder=10)
    ax_pareto.set_xlabel('Hardware Cost', fontsize=11, fontweight='bold')
    ax_pareto.set_ylabel('Reward', fontsize=11, fontweight='bold')
    ax_pareto.set_title('Pareto Front', fontsize=12, fontweight='bold')
    ax_pareto.grid(True, alpha=0.3)
    
    # 2. Cost distribution (top-right)
    ax_cost = fig.add_subplot(gs[0, 2])
    ax_cost.hist(costs, bins=8, color='#ff6b6b', edgecolor='black', alpha=0.7)
    ax_cost.axvline(costs[best_idx], color='gold', linewidth=2.5, linestyle='--', 
                   label=f'Best: {costs[best_idx]:.2f}')
    ax_cost.set_xlabel('Cost', fontsize=10)
    ax_cost.set_ylabel('Count', fontsize=10)
    ax_cost.set_title('Cost Distribution', fontsize=11, fontweight='bold')
    ax_cost.legend(fontsize=9)
    ax_cost.grid(True, alpha=0.3, axis='y')
    
    # 3. Reward distribution (middle-right)
    ax_reward = fig.add_subplot(gs[1, 2])
    ax_reward.hist(rewards, bins=8, color='#4ecdc4', edgecolor='black', alpha=0.7)
    ax_reward.axvline(rewards[best_idx], color='gold', linewidth=2.5, linestyle='--',
                     label=f'Best: {rewards[best_idx]:.3f}')
    ax_reward.set_xlabel('Reward', fontsize=10)
    ax_reward.set_ylabel('Count', fontsize=10)
    ax_reward.set_title('Reward Distribution', fontsize=11, fontweight='bold')
    ax_reward.legend(fontsize=9)
    ax_reward.grid(True, alpha=0.3, axis='y')
    
    # 4. Design parameters heatmap (bottom, spanning all columns)
    ax_heat = fig.add_subplot(gs[2, :])
    
    tau_matrix = np.zeros((len(X), len(groups)))
    for i, design_dict in enumerate(X):
        design = dict(design_dict)
        params = decode_design(design)
        tau_matrix[i] = [params["tau"].get(g, 80.0) for g in groups]
    
    im = ax_heat.imshow(tau_matrix, cmap='YlOrRd', aspect='auto')
    ax_heat.set_xticks(np.arange(len(groups)))
    ax_heat.set_yticks(np.arange(len(X)))
    ax_heat.set_xticklabels(groups, rotation=45, ha='right')
    ax_heat.set_yticklabels([f'Design {i}' for i in range(len(X))])
    ax_heat.set_ylabel('Design', fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(len(X)):
        for j in range(len(groups)):
            text = ax_heat.text(j, i, f'{tau_matrix[i, j]:.0f}',
                              ha="center", va="center", color="black", fontsize=9)
    
    ax_heat.set_title('Joint Torque Limits (Nm) per Design', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax_heat, orientation='vertical', pad=0.01)
    cbar.set_label('Torque (Nm)', fontsize=10)
    
    # Overall title
    plt.suptitle(f'Co-Design GA Results: {len(X)} Designs Evaluated', 
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved summary plot to {output_path}")
    plt.close()


def print_summary(F: np.ndarray, X: np.ndarray, groups: list[str]):
    """Print a text summary of the results."""
    print("\n" + "="*70)
    print("CO-DESIGN OPTIMIZATION RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n📊 Designs evaluated: {len(X)}")
    print(f"📈 Objectives: Reward (minimize) vs Cost (minimize)")
    
    rewards = F[:, 0]
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
    
    print(f"\n🤖 Best Design Actuator Configuration:")
    for group in groups:
        tau = params["tau"].get(group, 80.0)
        use_custom = params["use_custom"].get(group, False)
        actuator_type = "CUSTOM" if use_custom else "standard"
        print(f"   • {group:14s}: τ_max = {tau:6.1f} Nm  [{actuator_type}]")
    
    # Most efficient design
    efficiency = rewards / (costs + 1e-6)
    efficient_idx = np.argmax(efficiency)
    
    if efficient_idx != best_idx:
        print(f"\n⚡ MOST EFFICIENT DESIGN: #{efficient_idx}")
        print(f"   └─ Reward/Cost Ratio: {efficiency[efficient_idx]:.4f}")
        print(f"   └─ Reward: {rewards[efficient_idx]:.4f}")
        print(f"   └─ Cost: {costs[efficient_idx]:.2f}")
    
    print("\n" + "="*70)


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
