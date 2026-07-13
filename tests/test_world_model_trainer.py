"""Trainer overfit test on a synthetic linear system (CPU, fast)."""

from dataclasses import asdict

import torch

from mjlab.world_model import TrainCfg, TrajectoryDataset, WorldModelCfg
from mjlab.world_model.design_space import DesignParamCfg, DesignSpaceCfg
from mjlab.world_model.replay import ShardWriter
from mjlab.world_model.trainer import WorldModelTrainer

SPACE = DesignSpaceCfg(
  name="toy",
  tau_obs_range=(5.0, 90.0),
  params=(
    DesignParamCfg("a", "tau_max", ("l_a", "r_a"), (10.0, 100.0)),
    DesignParamCfg("b", "tau_max", ("l_b", "r_b"), (5.0, 50.0)),
  ),
)

OBS_DIM, ACT_DIM, NUM_ENVS, STEPS = 6, 2, 8, 120


def write_linear_system(out_dir):
  """obs' = 0.9*obs + B a; reward = -|obs|^2; contact = sign of obs[0]."""
  torch.manual_seed(0)
  b_mat = torch.randn(ACT_DIM, OBS_DIM) * 0.3
  writer = ShardWriter(
    out_dir,
    NUM_ENVS,
    {
      "obs": (OBS_DIM,),
      "action": (ACT_DIM,),
      "reward": (),
      "reward_terms": (1,),
      "terminated": (),
      "done": (),
      "contact_found": (2,),
      "contact_force": (2, 3),
      "air_time": (2,),
      "contact_time": (2,),
      "torque": (2,),
      "theta": (2,),
      "command": (0,),
    },
    device="cpu",
    flush_steps=32,
    shard_steps=64,
    store_dtype=torch.float32,
  )
  obs = torch.randn(NUM_ENVS, OBS_DIM)
  theta = SPACE.sample(NUM_ENVS, seed=0, region="train")
  for _ in range(STEPS):
    action = torch.randn(NUM_ENVS, ACT_DIM)
    reward = -(obs**2).mean(dim=1)
    contact = torch.stack([(obs[:, 0] > 0).float(), (obs[:, 1] > 0).float()], dim=1)
    writer.add(
      {
        "obs": obs,
        "action": action,
        "reward": reward,
        "reward_terms": reward.unsqueeze(-1),
        "terminated": torch.zeros(NUM_ENVS),
        "done": torch.zeros(NUM_ENVS),
        "contact_found": contact,
        "contact_force": contact.unsqueeze(-1).expand(-1, -1, 3) * 10.0,
        "air_time": 1.0 - contact,
        "contact_time": contact,
        "torque": action.clone(),
        "theta": theta,
        "command": torch.zeros(NUM_ENVS, 0),
      }
    )
    obs = 0.9 * obs + action @ b_mat
  writer.save_extra("start_obs.pt", torch.randn(16, OBS_DIM))
  writer.close(meta={"task": "linear", "design_space": asdict(SPACE)})


def test_trainer_overfits_linear_system(tmp_path):
  write_linear_system(tmp_path / "data")
  dataset = TrajectoryDataset(tmp_path / "data", val_fraction=0.13)
  model_cfg = WorldModelCfg(
    latent_dim=32,
    hidden_dim=32,
    num_layers=2,
    design_embed_dim=8,
    mode_embed_dim=4,
    ensemble_size=2,
    horizon=3,
    teacher_force_anneal_steps=40,
  )
  train_cfg = TrainCfg(
    total_steps=80,
    batch_size=64,
    learning_rate=1e-3,
    device="cpu",
    autocast=False,
    log_interval=1000,
    val_interval=1000,
    save_interval=1000,
    logger="none",
  )
  trainer = WorldModelTrainer(dataset, model_cfg, train_cfg, tmp_path / "out")

  gen = torch.Generator().manual_seed(0)
  first_batch = dataset.sample(64, 4, "cpu", "train", gen)
  loss_before, _ = trainer.compute_loss(first_batch, train=False)
  final = trainer.train()
  loss_after, metrics = trainer.compute_loss(first_batch, train=False)

  assert float(loss_after) < 0.5 * float(loss_before)
  assert final.exists()
  assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def test_checkpoint_round_trip(tmp_path):
  from mjlab.world_model.trainer import load_world_model

  write_linear_system(tmp_path / "data")
  dataset = TrajectoryDataset(tmp_path / "data", val_fraction=0.13)
  model_cfg = WorldModelCfg(
    latent_dim=32,
    hidden_dim=16,
    num_layers=1,
    design_embed_dim=8,
    mode_embed_dim=4,
    ensemble_size=1,
    horizon=2,
  )
  train_cfg = TrainCfg(
    total_steps=1, batch_size=8, device="cpu", autocast=False, logger="none"
  )
  trainer = WorldModelTrainer(dataset, model_cfg, train_cfg, tmp_path / "out")
  path = tmp_path / "out" / "ckpt.pt"
  trainer.save(path)

  model, normalizer, ckpt = load_world_model(path, "cpu")
  obs = torch.randn(3, OBS_DIM)
  z = model.encode(normalizer.normalize_obs(obs))
  assert z.shape == (3, 32)
  assert ckpt["dataset_meta"]["task"] == "linear"
  assert ckpt["start_obs"].shape == (16, OBS_DIM)
