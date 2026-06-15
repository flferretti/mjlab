#!/usr/bin/env python3
"""Complete co-design results visualization with plots and best-design video.

This script:
  1. Creates publication-quality plots of the GA search and Pareto front
  2. Generates a video of the best design walking
  3. Prints summary statistics
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


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

    args = parser.parse_args()

    # Load Pareto set
    if not Path(args.pareto).exists():
        print(f"❌ Error: Pareto file not found: {args.pareto}")
        return 1

    F, X, groups = load_pareto_set(args.pareto)
    # F[:, 0] = -performance (stored negative in GA for minimization)
    # Higher reward = more negative F[0], so use argmin
    best_idx = np.argmin(F[:, 0])

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Results directory: {output_dir}")

    # Step 1: Create plots
    print("\n" + "="*70)
    print("STEP 1: Creating visualization plots...")
    print("="*70)
    
    plot_cmd = [
        "uv", "run", "python", "scripts/plot_codesign_results.py",
        "--pareto", str(args.pareto),
        "--output-dir", str(output_dir),
    ]
    
    result = subprocess.run(plot_cmd, cwd=Path.cwd())
    if result.returncode != 0:
        print("❌ Failed to create plots")
        return 1

    # Step 2: Generate best design video
    if not args.plots_only:
        print("\n" + "="*70)
        print("STEP 2: Generating video of best design walking...")
        print("="*70)
        
        best_reward = -F[best_idx, 0]  # F[0] = -performance; convert back to positive
        best_cost = F[best_idx, 1]
        print(f"\n🏆 Best Design: #{best_idx}")
        print(f"   Reward: {best_reward:.4f}")
        print(f"   Cost: {best_cost:.2f}")
        
        video_path = output_dir / f"best_design_#{best_idx}_walk.mp4"
        
        video_cmd = [
            "uv", "run", "python", "scripts/validate_codesign.py",
            "--policy", str(args.policy),
            "--design-idx", str(best_idx),
            "--rollout-steps", str(args.video_steps),
            "--output-video", str(video_path),
        ]
        
        result = subprocess.run(video_cmd, cwd=Path.cwd())
        if result.returncode != 0:
            print("⚠️  Video generation failed (but plots are complete)")
        else:
            if video_path.exists():
                video_size_mb = video_path.stat().st_size / (1024**2)
                print(f"\n✅ Video saved: {video_path} ({video_size_mb:.1f} MB)")

    # Step 3: Summary
    print("\n" + "="*70)
    print("✅ COMPLETE: Results Summary")
    print("="*70)
    
    print(f"\n📊 Plots generated:")
    print(f"   • {output_dir}/pareto_front.png")
    print(f"   • {output_dir}/design_comparison.png")
    print(f"   • {output_dir}/codesign_summary.png")
    
    if not args.plots_only:
        video_file = output_dir / f"best_design_#{best_idx}_walk.mp4"
        if video_file.exists():
            print(f"\n🎥 Video:")
            print(f"   • {video_file}")
            print(f"     (Shows best design walking for {args.video_steps/50:.0f} seconds)")
    
    print(f"\n📈 Key Results:")
    print(f"   • Designs evaluated: {len(X)}")
    print(f"   • Reward range: [{F[:, 0].min():.3f}, {F[:, 0].max():.3f}]")
    print(f"   • Cost range: [{F[:, 1].min():.2f}, {F[:, 1].max():.2f}]")
    print(f"   • Best design: #{best_idx} (reward={best_reward:.4f})")
    
    print("\n✅ All done! Check the plots and video to visualize your results.\n")
    return 0


if __name__ == "__main__":
    exit(main())
