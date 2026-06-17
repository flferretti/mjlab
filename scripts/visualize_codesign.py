#!/usr/bin/env python3
"""Complete co-design results visualization with plots and best-design video.

This script:
  1. Creates publication-quality plots of the GA search and Pareto front
  2. Generates a video of the best design walking
  3. Prints summary statistics
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
from validate_codesign import design_to_tau_vector, select_design


def load_pareto_set(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
  """Load objectives (F), designs (X), and joint group names from Pareto npz."""
  data = np.load(path, allow_pickle=True)
  return data["F"], data["X"], data["groups"]


def main():
  parser = argparse.ArgumentParser(
    description="Complete co-design results visualization: plots + video.",
  )
  parser.add_argument(
    "--pareto",
    default="codesign_pareto.npz",
    help="Path to Pareto npz file from codesign_ga.py",
  )
  parser.add_argument(
    "--policy",
    required=True,
    help="Path to policy checkpoint (.pt or TorchScript)",
  )
  parser.add_argument(
    "--output-dir",
    default="codesign_results",
    help="Directory to save all results",
  )
  parser.add_argument(
    "--video-steps",
    type=int,
    default=500,
    help="Number of steps for best design video (500 = 10 sec @ 50Hz)",
  )
  parser.add_argument(
    "--plots-only",
    action="store_true",
    help="Only create plots, skip video",
  )
  parser.add_argument(
    "--criteria",
    choices=["reward", "efficiency"],
    default="efficiency",
    help="How to select the final design from Pareto set (default: efficiency).",
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
    default=200,
    help="Short rollout length used to screen candidate designs for walkability.",
  )
  parser.add_argument(
    "--selection-min-mean-vx",
    type=float,
    default=0.25,
    help="Minimum mean forward velocity required to treat a design as walkable.",
  )
  parser.add_argument(
    "--selection-min-min-height",
    type=float,
    default=0.45,
    help="Minimum rollout min height required to treat a design as walkable.",
  )

  args = parser.parse_args()

  # Load Pareto set
  if not Path(args.pareto).exists():
    print(f"❌ Error: Pareto file not found: {args.pareto}")
    return 1

  F, X, groups = load_pareto_set(args.pareto)
  selected_idx = select_design(
    F,
    args.criteria,
    efficiency_min_reward_fraction=args.efficiency_min_reward_fraction,
  )
  selected_design = dict(X[selected_idx])
  
  # For single-objective: F[i, 0] is fitness (performance - cost)
  # For multi-objective: F[i, 0] is -performance, F[i, 1] is cost (legacy)
  if F.shape[1] == 1:
    selected_fitness = F[selected_idx, 0]
    selected_performance = None  # Not decomposable
    selected_cost = None
  else:
    selected_performance = -F[selected_idx, 0]
    selected_cost = F[selected_idx, 1]

  # Create output directory
  output_dir = Path(args.output_dir)
  output_dir.mkdir(exist_ok=True)
  print(f"\n📁 Results directory: {output_dir}")

  # Step 1: Create plots
  print("\n" + "=" * 70)
  print("STEP 1: Creating visualization plots...")
  print("=" * 70)

  plot_cmd = [
    "uv",
    "run",
    "python",
    "scripts/plot_codesign_results.py",
    "--pareto",
    str(args.pareto),
    "--output-dir",
    str(output_dir),
  ]

  result = subprocess.run(plot_cmd, cwd=Path.cwd())
  if result.returncode != 0:
    print("❌ Failed to create plots")
    return 1

  # Step 2: Generate best design video
  if not args.plots_only:
    print("\n" + "=" * 70)
    print("STEP 2: Generating video of best design walking...")
    print("=" * 70)

    video_path = output_dir / f"best_design_{selected_idx}_sim2sim.mp4"
    tau_csv = ",".join(
      f"{value:.6f}" for value in design_to_tau_vector(selected_design).tolist()
    )
    sim2sim_cmd = [
      "uv",
      "run",
      "python",
      "scripts/sim2sim.py",
      "--onnx",
      str(args.policy),
      "--headless",
      "--steps",
      str(args.video_steps),
      "--cmd-x",
      "0.8",
      "--cmd-y",
      "0.0",
      "--cmd-yaw",
      "0.0",
      "--tau-obs-nm",
      tau_csv,
      "--video-path",
      str(video_path),
    ]

    result = subprocess.run(
      sim2sim_cmd,
      cwd=Path.cwd(),
      text=True,
      capture_output=True,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
      if result.stderr:
        print(result.stderr, end="")
      print("⚠️  Video generation failed (but plots are complete)")
    elif video_path.exists():
      video_size_mb = video_path.stat().st_size / (1024**2)
      print(f"\n✅ Video saved: {video_path} ({video_size_mb:.1f} MB)")

  # Step 3: Summary
  print("\n" + "=" * 70)
  print("✅ COMPLETE: Results Summary")
  print("=" * 70)

  print("\n📊 Plots generated:")
  print(f"   • {output_dir}/pareto_front.png")
  print(f"   • {output_dir}/design_comparison.png")
  print(f"   • {output_dir}/codesign_summary.png")

  if not args.plots_only:
    video_file = output_dir / f"best_design_{selected_idx}_sim2sim.mp4"
    if video_file.exists():
      print("\n🎥 Video:")
      print(f"   • {video_file}")
      print(f"     (Shows best design walking for {args.video_steps / 50:.0f} seconds)")

  print("\n📈 Key Results:")
  if F.shape[1] == 1:
    # Single-objective
    fitnesses = F[:, 0]
    print(f"   • Designs evaluated: {len(X)}")
    print(f"   • Fitness range: [{fitnesses.min():.3f}, {fitnesses.max():.3f}]")
    print(f"   • Best design: #{selected_idx} (fitness={selected_fitness:.4f})")
  else:
    # Multi-objective (legacy)
    rewards = -F[:, 0]
    print(f"   • Designs evaluated: {len(X)}")
    print(f"   • Performance range: [{rewards.min():.3f}, {rewards.max():.3f}]")
    print(f"   • Cost range: [{F[:, 1].min():.2f}, {F[:, 1].max():.2f}]")
    print(f"   • Selection criterion: {args.criteria}")
    if args.criteria == "efficiency":
      print(
        "   • Efficiency performance floor: "
        f"{args.efficiency_min_reward_fraction:.2f} × max performance"
      )
    print(
      f"   • Best design: #{selected_idx} "
      f"(performance={selected_performance:.4f}, cost={selected_cost:.2f})"
    )

  print("\n✅ All done! Check the plots and video to visualize your results.\n")
  return 0


if __name__ == "__main__":
  exit(main())
