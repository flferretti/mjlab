"""End-to-end smoke test: collect -> train -> imagine on the QDD task."""

import pytest
import torch
from conftest import get_test_device

# The task registry transitively requires the private motion-prior package.
pytest.importorskip("gb_motion_prior_lupin")

import mjlab.tasks  # noqa: F401, E402  (registers tasks)
from mjlab.world_model import (
  QDD_DESIGN_SPACE_V1,
  CollectCfg,
  DesignEvaluator,
  TrainCfg,
  TrajectoryDataset,
  WorldModelCfg,
)
from mjlab.world_model.collect import RandomPolicy, collect_dataset
from mjlab.world_model.trainer import WorldModelTrainer

TASK = "Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond"


@pytest.mark.slow
def test_collect_train_imagine_smoke(tmp_path):
  device = get_test_device()
  data_dir = tmp_path / "data"
  cfg = CollectCfg(
    num_envs=4,
    steps=30,
    seed=0,
    flush_steps=8,
    shard_steps=16,
    store_dtype="float32",
  )
  meta = collect_dataset(
    task=TASK,
    cfg=cfg,
    design_space=QDD_DESIGN_SPACE_V1,
    out_dir=data_dir,
    policy_path=None,  # Random policy.
    device=device,
  )
  assert meta["tau_obs_joint_names"] is not None
  assert "motor_tau_max" in meta["obs_layout"]
  assert "track_linear_velocity" in meta["reward_term_names"]

  dataset = TrajectoryDataset(data_dir, val_fraction=0.25)
  assert dataset.num_steps == 30 and dataset.num_envs == 4
  # Designs were applied and are in bounds.
  theta = dataset.data["theta"].reshape(-1, QDD_DESIGN_SPACE_V1.dim).float()
  unit = QDD_DESIGN_SPACE_V1.normalize(theta)
  assert (unit >= 0).all() and (unit <= 1).all()
  assert theta.std() > 0  # Not a single constant design.
  # Feet touch the ground at least sometimes.
  assert dataset.data["contact_found"].float().mean() > 0.01
  assert dataset.start_obs is not None and dataset.start_obs.shape[0] >= 4

  model_cfg = WorldModelCfg(
    latent_dim=32,
    hidden_dim=32,
    num_layers=1,
    design_embed_dim=8,
    mode_embed_dim=4,
    ensemble_size=2,
    horizon=3,
  )
  train_cfg = TrainCfg(
    total_steps=3,
    batch_size=16,
    device="cpu",
    autocast=False,
    logger="none",
    log_interval=1000,
    val_interval=1000,
    save_interval=1000,
  )
  trainer = WorldModelTrainer(dataset, model_cfg, train_cfg, tmp_path / "out")
  checkpoint = trainer.train()

  evaluator = DesignEvaluator(
    checkpoint,
    RandomPolicy(dataset.data["action"].shape[-1]),
    "cpu",
    seed=0,
  )
  evaluator.cfg.rollout_steps = 5
  designs = QDD_DESIGN_SPACE_V1.holdout_designs(3, seed=0)
  scores = evaluator.evaluate_designs(designs, n_seeds=2)
  assert scores.performance.shape == (3,)
  assert scores.rms_torque.shape == (3, len(QDD_DESIGN_SPACE_V1.joint_names))
  assert scores.uncertainty.shape == (3,)
  assert torch.isfinite(scores.performance).all()
  assert (scores.rms_torque >= 0).all()
