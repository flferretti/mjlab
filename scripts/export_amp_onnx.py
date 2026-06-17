"""Export AMP policy checkpoint to ONNX for sim2sim deployment.

Usage:
    uv run python scripts/export_amp_onnx.py \
        --checkpoint logs/rsl_rl/qdd_velocity/2026-05-05_11-53-45/model_32000.pt \
        --task Mjlab-Velocity-Flat-Gbionics-QDD

The exported ONNX takes a single flat observation tensor and produces actions.
Compatible with gb-rl-locomotion's sim2sim runner.
"""

import argparse
import copy
import os
from pathlib import Path

import torch
import torch.nn as nn


class FlatActorExporter(nn.Module):
  """Wraps an rsl_rl 5.x MLPModel actor to accept a flat tensor input."""

  def __init__(self, actor):
    super().__init__()
    # Extract the raw MLP and normalizer from the actor.
    self.obs_normalizer = copy.deepcopy(actor.obs_normalizer)
    self.mlp = copy.deepcopy(actor.mlp)
    self.obs_groups = actor.obs_groups
    self.obs_dim = actor.obs_dim

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    # The actor's get_latent normally selects from TensorDict and concatenates.
    # Since we pass a pre-concatenated flat tensor, just normalize and forward.
    latent = self.obs_normalizer(obs)
    return self.mlp(latent)


def main():
  parser = argparse.ArgumentParser(description="Export AMP policy to ONNX")
  parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to .pt checkpoint"
  )
  parser.add_argument(
    "--task",
    type=str,
    default="Mjlab-Velocity-Flat-Gbionics-QDD",
    help="Task registry name",
  )
  parser.add_argument(
    "--output", type=str, default=None, help="Output ONNX path (default: next to .pt)"
  )
  args = parser.parse_args()

  checkpoint_path = Path(args.checkpoint)
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

  # Determine output path.
  if args.output:
    onnx_path = Path(args.output)
  else:
    stem = checkpoint_path.stem  # e.g. model_32000
    iteration = stem.split("_")[-1] if "_" in stem else "0"
    onnx_path = checkpoint_path.parent / f"policy_{iteration}.onnx"

  # Load environment config to get observation dimensions.
  import mjlab.tasks  # noqa: F401 — register tasks
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  env_cfg = load_env_cfg(args.task, play=True)
  rl_cfg = load_rl_cfg(args.task)

  # Build a minimal env to instantiate the actor with correct obs structure.
  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.rl.amp_runner import MjlabAmpOnPolicyRunner

  env_cfg.scene.num_envs = 1
  device = "cpu"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env_wrapper = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

  # Build runner (this creates actor_critic with correct architecture).
  runner = MjlabAmpOnPolicyRunner(env_wrapper, asdict(rl_cfg), device=device)

  # Load checkpoint weights.
  runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True)

  # Extract actor from adapter.
  actor = runner.actor_critic.actor
  actor.eval()
  actor.to("cpu")

  # Create flat exporter.
  exporter = FlatActorExporter(actor)
  exporter.eval()

  # Create dummy input matching actor obs_dim.
  dummy_input = torch.zeros(1, actor.obs_dim)

  # Export.
  os.makedirs(onnx_path.parent, exist_ok=True)
  torch.onnx.export(
    exporter,
    dummy_input,
    str(onnx_path),
    export_params=True,
    opset_version=18,
    verbose=False,
    input_names=["obs"],
    output_names=["actions"],
    dynamic_axes={},
    dynamo=False,
  )

  print(f"✓ Exported ONNX policy to: {onnx_path}")
  print(f"  Input:  obs [{1}, {actor.obs_dim}]")
  print(f"  Output: actions [{1}, 12]")
  print()
  print("To test with gb-rl-locomotion sim2sim runner:")
  print(f'  Edit policy_config.toml: onnx_package = "{onnx_path.resolve()}"')
  print(
    "  Run: python deployment/sim2sim/run_policy.py"
    " --config deployment/sim2sim/config/robots/lowerbodyqdd/policy_config.toml"
  )

  env.close()


if __name__ == "__main__":
  main()
