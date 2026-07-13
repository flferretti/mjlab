"""Collect a design-randomized trajectory dataset for world-model training."""

from dataclasses import dataclass, field

import torch
import tyro

from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.utils.torch import configure_torch_backends
from mjlab.world_model import DESIGN_SPACES, CollectCfg
from mjlab.world_model.collect import collect_dataset


@dataclass(frozen=True)
class WmCollectConfig:
  task: str = "Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond"
  """Motor-conditioned task id used for collection."""
  out: str = "data/wm/qdd_v1"
  """Output dataset directory."""
  policy: str | None = None
  """Frozen policy (TorchScript or rsl_rl checkpoint); None = random policy."""
  design_space: str = "qdd_v1"
  """Design-space preset name (see mjlab.world_model.DESIGN_SPACES)."""
  collect: CollectCfg = field(default_factory=CollectCfg)
  contact_sensor: str = "feet_ground_contact"
  command_name: str = "twist"
  device: str | None = None
  """None selects cuda:0 if available, else cpu (Warp-CPU; slow)."""


def main() -> None:
  maybe_print_top_level_help("wm-collect")
  import mjlab.tasks  # noqa: F401  (registers tasks)

  cfg = tyro.cli(WmCollectConfig, config=mjlab.TYRO_FLAGS)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  if cfg.design_space not in DESIGN_SPACES:
    raise SystemExit(
      f"Unknown design space '{cfg.design_space}'; available: {sorted(DESIGN_SPACES)}"
    )
  meta = collect_dataset(
    task=cfg.task,
    cfg=cfg.collect,
    design_space=DESIGN_SPACES[cfg.design_space],
    out_dir=cfg.out,
    policy_path=cfg.policy,
    device=device,
    contact_sensor_name=cfg.contact_sensor,
    command_name=cfg.command_name,
  )
  total = cfg.collect.num_envs * cfg.collect.steps
  print(f"[wm-collect] Wrote {total} transitions to {cfg.out}")
  print(f"[wm-collect] Reward terms: {meta['reward_term_names']}")


if __name__ == "__main__":
  main()
