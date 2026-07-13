"""Train the contact-enhanced design-conditioned world model on a dataset.

Pure PyTorch: runs on CUDA, ROCm (torch-rocm), or CPU; no simulator needed.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import tyro

import mjlab
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab.world_model import (
  TrainCfg,
  TrajectoryDataset,
  WorldModelCfg,
  WorldModelTrainer,
)


@dataclass(frozen=True)
class WmTrainConfig:
  data: str = "data/wm/qdd_v1"
  """Dataset directory written by wm-collect."""
  out: str | None = None
  """Output directory; defaults to logs/world_model/<dataset>/<timestamp>."""
  model: WorldModelCfg = field(default_factory=WorldModelCfg)
  train: TrainCfg = field(default_factory=TrainCfg)


def main() -> None:
  maybe_print_top_level_help("wm-train")
  cfg = tyro.cli(WmTrainConfig, config=mjlab.TYRO_FLAGS)
  configure_torch_backends()

  dataset = TrajectoryDataset(cfg.data, val_fraction=cfg.train.val_fraction)
  out = cfg.out or str(
    Path("logs/world_model")
    / Path(cfg.data).name
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  )
  dump_yaml(Path(out) / "params" / "model.yaml", asdict(cfg.model))
  dump_yaml(Path(out) / "params" / "train.yaml", asdict(cfg.train))

  trainer = WorldModelTrainer(dataset, cfg.model, cfg.train, out)
  print(f"[wm-train] device={trainer.device} out={out}")
  final = trainer.train()
  print(f"[wm-train] Final checkpoint: {final}")


if __name__ == "__main__":
  main()
