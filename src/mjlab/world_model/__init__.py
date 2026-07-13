"""Contact-enhanced design-conditioned world models for rapid robot co-design.

Train a single latent world model on massively parallel simulation data in
which every environment carries a different actuator design, then evaluate,
rank, and optimize new designs in imagination — replacing physics rollouts
inside the NSGA-II co-design loop (``scripts/codesign_ga.py``).

CLI entry points: ``wm-collect``, ``wm-train``, ``wm-eval``, ``wm-codesign``.

This package root stays importable without the simulation/viewer stack
(pure PyTorch — wm-train/wm-eval run on ROCm or CPU-only machines); the
env-facing collector lives in :mod:`mjlab.world_model.collect`.
"""

from mjlab.world_model.codesign_backend import WorldModelBackend
from mjlab.world_model.config import (
  CollectCfg,
  ImaginationCfg,
  TrainCfg,
  WorldModelBackendCfg,
  WorldModelCfg,
)
from mjlab.world_model.design_space import (
  DESIGN_SPACES,
  GO1_DESIGN_SPACE_V1,
  QDD_DESIGN_SPACE_V1,
  DesignParamCfg,
  DesignSpaceCfg,
  design_space_from_dict,
  randomize_design,
)
from mjlab.world_model.imagination import DesignEvaluator, DesignScores
from mjlab.world_model.networks import WorldModel, WorldModelDims
from mjlab.world_model.replay import Normalizer, ShardWriter, TrajectoryDataset
from mjlab.world_model.trainer import WorldModelTrainer, load_world_model

__all__ = [
  "DESIGN_SPACES",
  "GO1_DESIGN_SPACE_V1",
  "QDD_DESIGN_SPACE_V1",
  "CollectCfg",
  "DesignEvaluator",
  "DesignParamCfg",
  "DesignScores",
  "DesignSpaceCfg",
  "ImaginationCfg",
  "Normalizer",
  "ShardWriter",
  "TrainCfg",
  "TrajectoryDataset",
  "WorldModel",
  "WorldModelBackend",
  "WorldModelBackendCfg",
  "WorldModelCfg",
  "WorldModelDims",
  "WorldModelTrainer",
  "design_space_from_dict",
  "load_world_model",
  "randomize_design",
]
