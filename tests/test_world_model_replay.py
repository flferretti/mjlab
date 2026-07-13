"""Tests for shard writing and sequence sampling."""

import json

import pytest
import torch

from mjlab.world_model.replay import ShardWriter, TrajectoryDataset

NUM_ENVS = 3
OBS_DIM = 4

FIELD_SHAPES = {
  "obs": (OBS_DIM,),
  "action": (2,),
  "reward": (),
  "reward_terms": (2,),
  "terminated": (),
  "done": (),
  "contact_found": (2,),
  "contact_force": (2, 3),
  "air_time": (2,),
  "contact_time": (2,),
  "torque": (2,),
  "theta": (2,),
  "command": (3,),
}


def write_dataset(out_dir, steps=40, done_every=13):
  """Synthetic dataset with periodic dones; obs encodes (step, env)."""
  writer = ShardWriter(
    out_dir, NUM_ENVS, FIELD_SHAPES, device="cpu", flush_steps=7, shard_steps=16
  )
  for t in range(steps):
    done = torch.zeros(NUM_ENVS)
    if (t + 1) % done_every == 0:
      done[:] = 1.0
    row = {
      "obs": torch.full((NUM_ENVS, OBS_DIM), float(t)),
      "action": torch.randn(NUM_ENVS, 2),
      "reward": torch.full((NUM_ENVS,), float(t)),
      "reward_terms": torch.randn(NUM_ENVS, 2),
      "terminated": done.clone(),
      "done": done,
      "contact_found": torch.randint(0, 2, (NUM_ENVS, 2)).float(),
      "contact_force": torch.randn(NUM_ENVS, 2, 3),
      "air_time": torch.rand(NUM_ENVS, 2),
      "contact_time": torch.rand(NUM_ENVS, 2),
      "torque": torch.randn(NUM_ENVS, 2),
      "theta": torch.rand(NUM_ENVS, 2),
      "command": torch.randn(NUM_ENVS, 3),
    }
    writer.add(row)
  writer.save_extra("start_obs.pt", torch.randn(8, OBS_DIM))
  writer.close(meta={"task": "synthetic", "design_space": {}})
  return out_dir


def test_shard_round_trip(tmp_path):
  write_dataset(tmp_path, steps=40)
  with open(tmp_path / "meta.json") as f:
    meta = json.load(f)
  assert meta["total_steps"] == 40
  assert len(list(tmp_path.glob("shard_*.pt"))) >= 2

  ds = TrajectoryDataset(tmp_path, val_fraction=0.34)
  assert ds.num_steps == 40
  assert ds.num_envs == NUM_ENVS
  assert ds.data["obs"].dtype == torch.float16
  assert ds.start_obs is not None and ds.start_obs.shape == (8, OBS_DIM)
  # Time-major ordering survives flush/shard boundaries.
  assert float(ds.data["obs"][17, 0, 0]) == 17.0


def test_windows_do_not_cross_episode_boundaries(tmp_path):
  write_dataset(tmp_path, steps=40, done_every=13)
  ds = TrajectoryDataset(tmp_path, val_fraction=0.34)
  seq_len = 6
  gen = torch.Generator().manual_seed(0)
  batch = ds.sample(64, seq_len, "cpu", "train", gen)
  assert batch["obs"].shape == (64, seq_len, OBS_DIM)
  assert batch["action"].shape == (64, seq_len - 1, 2)
  assert batch["theta"].shape == (64, 2)
  # A done may only appear on the final transition of a window.
  interior_done = batch["done"][:, :-1]
  assert interior_done.sum() == 0
  # valid_next masks exactly the terminal final transitions.
  assert torch.equal(batch["valid_next"], 1.0 - batch["done"])
  # Obs rows are consecutive within every window (no reset crossings).
  first = batch["obs"][:, :, 0]
  deltas = first[:, 1:] - first[:, :-1]
  assert torch.all(deltas == 1.0)


def test_terminal_windows_are_sampled(tmp_path):
  """Windows ending on a done transition must appear (termination labels)."""
  write_dataset(tmp_path, steps=40, done_every=13)
  ds = TrajectoryDataset(tmp_path, val_fraction=0.34)
  gen = torch.Generator().manual_seed(1)
  batch = ds.sample(256, 4, "cpu", "train", gen)
  assert batch["done"][:, -1].sum() > 0


def test_normalizer_stats(tmp_path):
  write_dataset(tmp_path, steps=40)
  ds = TrajectoryDataset(tmp_path, val_fraction=0.34)
  norm = ds.normalizer()
  obs = torch.randn(10, OBS_DIM)
  restored = norm.denormalize_obs(norm.normalize_obs(obs))
  assert torch.allclose(restored, obs, atol=1e-4)
  assert norm.stats["torque_std"].shape == (2,)
  assert (norm.stats["obs_std"] > 0).all()


def test_writer_rejects_mismatched_rows(tmp_path):
  writer = ShardWriter(
    tmp_path, NUM_ENVS, {"obs": (OBS_DIM,)}, device="cpu", flush_steps=4
  )
  with pytest.raises(KeyError):
    writer.add({"wrong": torch.zeros(NUM_ENVS, OBS_DIM)})
