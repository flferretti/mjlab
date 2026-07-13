"""On-disk trajectory shards and the sequence sampler for world-model training.

Data layout (written by the collector, consumed by the trainer):

  <out_dir>/
    meta.json        # schema, dims, obs layout, design space, provenance
    start_obs.pt     # bank of post-reset actor observations (imagination)
    shard_0000.pt    # dict[name -> tensor [T, num_envs, ...]] (time-major)
    shard_0001.pt
    ...

Row semantics (all fields at row ``t``):
  - ``obs``/``contact_*``/``theta``/``command``: state s_t (pre-step, aligned
    with the observation the action was computed from).
  - ``action``/``reward``/``reward_terms``/``torque``/``terminated``/``done``:
    the transition s_t -> s_{t+1} executed at step t (``torque`` is the
    applied joint torque during that step, matching the GA's RMS readout).

Sequences sampled for training never cross an episode boundary except that a
window may *end* on a terminal transition; per-step ``valid_next`` masks the
next-state targets (consistency/obs/contact) on that final transition while
keeping the reward/torque/termination targets, so the termination head sees
positive labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

_SCHEMA_VERSION = 1

# Fields with next-state (pre-step) semantics; everything else is transition
# data. Shapes are per-env trailing shapes filled in by the collector.
STATE_FIELDS = (
  "obs",
  "contact_found",
  "contact_force",
  "air_time",
  "contact_time",
  "theta",
  "command",
)
TRANSITION_FIELDS = (
  "action",
  "reward",
  "reward_terms",
  "torque",
  "terminated",
  "done",
)


class ShardWriter:
  """Buffers per-step batches on device and writes time-major shards."""

  def __init__(
    self,
    out_dir: str | Path,
    num_envs: int,
    field_shapes: dict[str, tuple[int, ...]],
    device: torch.device | str,
    flush_steps: int = 512,
    shard_steps: int = 2_048,
    store_dtype: torch.dtype = torch.float16,
  ) -> None:
    self.out_dir = Path(out_dir)
    self.out_dir.mkdir(parents=True, exist_ok=True)
    self.num_envs = num_envs
    self.field_shapes = dict(field_shapes)
    self.flush_steps = flush_steps
    self.shard_steps = shard_steps
    self.store_dtype = store_dtype
    self._staging = {
      name: torch.zeros((flush_steps, num_envs, *shape), device=device)
      for name, shape in field_shapes.items()
    }
    self._cursor = 0
    self._pending: list[dict[str, torch.Tensor]] = []
    self._pending_steps = 0
    self._shard_index = 0
    self.total_steps = 0

  def add(self, row: dict[str, torch.Tensor]) -> None:
    if set(row) != set(self.field_shapes):
      missing = set(self.field_shapes) ^ set(row)
      raise KeyError(f"Row fields mismatch: {sorted(missing)}")
    for name, value in row.items():
      self._staging[name][self._cursor] = value.detach()
    self._cursor += 1
    self.total_steps += 1
    if self._cursor == self.flush_steps:
      self._flush()

  def _flush(self) -> None:
    if self._cursor == 0:
      return
    chunk = {}
    for name, buf in self._staging.items():
      host = buf[: self._cursor].to("cpu")
      if host.is_floating_point():
        host = host.to(self.store_dtype)
      chunk[name] = host
    self._pending.append(chunk)
    self._pending_steps += self._cursor
    self._cursor = 0
    if self._pending_steps >= self.shard_steps:
      self._write_shard()

  def _write_shard(self) -> None:
    if not self._pending:
      return
    shard = {
      name: torch.cat([c[name] for c in self._pending], dim=0)
      for name in self.field_shapes
    }
    path = self.out_dir / f"shard_{self._shard_index:04d}.pt"
    torch.save(shard, path)
    self._shard_index += 1
    self._pending = []
    self._pending_steps = 0

  def save_extra(self, filename: str, obj: Any) -> None:
    torch.save(obj, self.out_dir / filename)

  def write_meta(self, meta: dict[str, Any]) -> None:
    meta = dict(meta)
    meta["schema_version"] = _SCHEMA_VERSION
    meta["num_envs"] = self.num_envs
    meta["total_steps"] = self.total_steps
    meta["field_shapes"] = {k: list(v) for k, v in self.field_shapes.items()}
    with open(self.out_dir / "meta.json", "w") as f:
      json.dump(meta, f, indent=2, default=str)

  def close(self, meta: dict[str, Any] | None = None) -> None:
    self._flush()
    self._write_shard()
    if meta is not None:
      self.write_meta(meta)


class Normalizer:
  """Per-field normalization statistics computed from the training split."""

  def __init__(self, stats: dict[str, torch.Tensor]) -> None:
    self.stats = stats

  @classmethod
  def from_data(
    cls,
    obs: torch.Tensor,
    reward: torch.Tensor,
    reward_terms: torch.Tensor,
    torque: torch.Tensor,
  ) -> "Normalizer":
    eps = 1e-6
    flat_obs = obs.reshape(-1, obs.shape[-1]).float()
    return cls(
      {
        "obs_mean": flat_obs.mean(dim=0),
        "obs_std": flat_obs.std(dim=0).clamp_min(eps),
        "reward_std": reward.float().std().clamp_min(eps),
        "reward_terms_std": reward_terms.reshape(-1, reward_terms.shape[-1])
        .float()
        .std(dim=0)
        .clamp_min(eps),
        "torque_std": torque.reshape(-1, torque.shape[-1])
        .float()
        .std(dim=0)
        .clamp_min(eps),
      }
    )

  def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
    return (obs - self.stats["obs_mean"]) / self.stats["obs_std"]

  def denormalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
    return obs * self.stats["obs_std"] + self.stats["obs_mean"]

  def to(self, device: torch.device | str) -> "Normalizer":
    self.stats = {k: v.to(device) for k, v in self.stats.items()}
    return self

  def state_dict(self) -> dict[str, torch.Tensor]:
    return self.stats

  @classmethod
  def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "Normalizer":
    return cls(dict(state))


class TrajectoryDataset:
  """Loads all shards into memory and samples fixed-length sequences."""

  def __init__(self, data_dir: str | Path, val_fraction: float = 0.05) -> None:
    self.data_dir = Path(data_dir)
    with open(self.data_dir / "meta.json") as f:
      self.meta = json.load(f)
    shard_paths = sorted(self.data_dir.glob("shard_*.pt"))
    if not shard_paths:
      raise FileNotFoundError(f"No shards found in {self.data_dir}")
    shards = [torch.load(p, map_location="cpu", weights_only=True) for p in shard_paths]
    self.data: dict[str, torch.Tensor] = {
      name: torch.cat([s[name] for s in shards], dim=0) for name in shards[0]
    }
    self.num_steps, self.num_envs = self.data["obs"].shape[:2]

    start_obs_path = self.data_dir / "start_obs.pt"
    self.start_obs: torch.Tensor | None = (
      torch.load(start_obs_path, map_location="cpu", weights_only=True)
      if start_obs_path.exists()
      else None
    )

    n_val = max(1, int(round(val_fraction * self.num_envs)))
    self._val_envs = torch.arange(n_val)
    self._train_envs = torch.arange(n_val, self.num_envs)
    if len(self._train_envs) == 0:
      raise ValueError("val_fraction leaves no training envs.")

  def normalizer(self) -> Normalizer:
    train = self._train_envs
    return Normalizer.from_data(
      self.data["obs"][:, train],
      self.data["reward"][:, train],
      self.data["reward_terms"][:, train],
      self.data["torque"][:, train],
    )

  def _valid_starts(self, seq_len: int, envs: torch.Tensor) -> torch.Tensor:
    """(K, 2) tensor of (start_row, env) pairs for windows of ``seq_len`` rows.

    A window covers rows [s, s + seq_len - 1] with transitions at rows
    [s, s + seq_len - 2]; no ``done`` may occur strictly inside, but the final
    transition may be terminal (masked downstream via ``valid_next``).
    """
    horizon = seq_len - 1
    done = self.data["done"][:, envs].bool()  # [T, E]
    t_max = self.num_steps - seq_len
    if t_max < 0:
      raise ValueError(f"Dataset too short for seq_len={seq_len}.")
    # interior_done[s, e] = any done in rows [s, s + horizon - 2].
    if horizon > 1:
      cum = torch.cumsum(done.int(), dim=0)  # [T, E]
      pad = torch.zeros(1, done.shape[1], dtype=cum.dtype)
      cum = torch.cat([pad, cum], dim=0)  # cum[t] = done[:t].sum()
      interior = cum[horizon - 1 :] - cum[: -(horizon - 1)]  # windows of h-1
      interior_done = interior[: t_max + 1] > 0
    else:
      interior_done = torch.zeros(t_max + 1, done.shape[1], dtype=torch.bool)
    starts, env_pos = torch.nonzero(~interior_done, as_tuple=True)
    return torch.stack([starts, envs[env_pos]], dim=1)

  def sample(
    self,
    batch_size: int,
    seq_len: int,
    device: torch.device | str,
    split: str = "train",
    generator: torch.Generator | None = None,
  ) -> dict[str, torch.Tensor]:
    """Sample sequences of ``seq_len`` rows (= ``seq_len - 1`` transitions)."""
    envs = self._train_envs if split == "train" else self._val_envs
    cache_key = (seq_len, split)
    cache = getattr(self, "_starts_cache", {})
    if cache_key not in cache:
      cache[cache_key] = self._valid_starts(seq_len, envs)
      self._starts_cache = cache
    starts = cache[cache_key]
    idx = torch.randint(starts.shape[0], (batch_size,), generator=generator)
    s, e = starts[idx, 0], starts[idx, 1]
    rows = s.unsqueeze(1) + torch.arange(seq_len).unsqueeze(0)  # [B, L+1]

    out: dict[str, torch.Tensor] = {}
    horizon = seq_len - 1
    for name, tensor in self.data.items():
      length = seq_len if name in STATE_FIELDS else horizon
      gathered = tensor[rows[:, :length], e.unsqueeze(1)]
      out[name] = gathered.to(device=device, dtype=torch.float32, non_blocking=True)
    # Design is constant within an episode; keep the window's first row.
    out["theta"] = out["theta"][:, 0]
    # Mask for next-state targets: False on a terminal final transition.
    out["valid_next"] = 1.0 - out["done"]
    return out
