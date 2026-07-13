"""Training loop for the contact-enhanced design-conditioned world model.

Losses over sampled sequences of ``horizon`` transitions:
  - latent consistency of the unrolled dynamics against an EMA target encoder
    (TD-MPC2 recipe), per-step discounted;
  - reward (total + decomposed terms), applied torque, termination;
  - contact supervision: next-contact BCE on each member's mode head and
    current contact state/force on the shared contact head;
  - actor-observation reconstruction (so the frozen policy can act in
    imagination).

Contact-mode teacher forcing anneals to fully self-predicted modes. Ensemble
members train on disjoint batch chunks to decorrelate. All code is
device-agnostic PyTorch and runs unchanged on CUDA, ROCm, and CPU.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from mjlab.world_model.config import TrainCfg, WorldModelCfg
from mjlab.world_model.design_space import DesignSpaceCfg, design_space_from_dict
from mjlab.world_model.networks import (
  WorldModel,
  WorldModelDims,
  contact_feature_vector,
)
from mjlab.world_model.replay import Normalizer, TrajectoryDataset


class WorldModelTrainer:
  def __init__(
    self,
    dataset: TrajectoryDataset,
    model_cfg: WorldModelCfg,
    train_cfg: TrainCfg,
    out_dir: str | Path,
  ) -> None:
    self.dataset = dataset
    self.model_cfg = model_cfg
    self.train_cfg = train_cfg
    self.out_dir = Path(out_dir)
    self.out_dir.mkdir(parents=True, exist_ok=True)

    device = train_cfg.device
    if device is None:
      device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.device = torch.device(device)

    torch.manual_seed(train_cfg.seed)
    self.design_space: DesignSpaceCfg = design_space_from_dict(
      dataset.meta["design_space"]
    )
    self.dims = WorldModelDims(
      obs_dim=dataset.data["obs"].shape[-1],
      action_dim=dataset.data["action"].shape[-1],
      command_dim=dataset.data["command"].shape[-1],
      num_contacts=dataset.data["contact_found"].shape[-1],
      num_joints=dataset.data["torque"].shape[-1],
      num_reward_terms=dataset.data["reward_terms"].shape[-1],
      design_dim=self.design_space.dim,
    )
    self.normalizer: Normalizer = dataset.normalizer().to(self.device)
    self.model = WorldModel(model_cfg, self.dims).to(self.device)
    self.opt = torch.optim.AdamW(
      self.model.parameters(),
      lr=train_cfg.learning_rate,
      weight_decay=train_cfg.weight_decay,
    )
    self.step = 0
    self._sampler = torch.Generator().manual_seed(train_cfg.seed)
    self._logger = _make_logger(train_cfg, self.out_dir)

  # -- Loss -------------------------------------------------------------------

  def compute_loss(
    self, batch: dict[str, torch.Tensor], train: bool = True
  ) -> tuple[torch.Tensor, dict[str, float]]:
    cfg = self.model_cfg
    model = self.model
    horizon = batch["action"].shape[1]

    obs_n = self.normalizer.normalize_obs(batch["obs"])  # [B, L+1, O]
    contact_vec = contact_feature_vector(
      batch["contact_found"],
      batch["contact_force"],
      batch["air_time"],
      batch["contact_time"],
    )
    theta_n = self.design_space.normalize(batch["theta"])
    e_all = model.embed_design(theta_n)  # [B, E]
    with torch.no_grad():
      z_bar = model.encode_target(
        obs_n, contact_vec if cfg.privileged_encoder else None
      )  # [B, L+1, Z]
    r_std = self.normalizer.stats["reward_std"]
    terms_std = self.normalizer.stats["reward_terms_std"]
    tau_std = self.normalizer.stats["torque_std"]
    force_log = torch.sign(batch["contact_force"]) * torch.log1p(
      batch["contact_force"].abs()
    )

    # Teacher-forcing probability for the contact-mode code.
    anneal = max(cfg.teacher_force_anneal_steps, 1)
    p_tf = max(0.0, 1.0 - self.step / anneal) if train else 0.0

    chunks = torch.arange(batch["obs"].shape[0]).chunk(cfg.ensemble_size)
    losses: dict[str, torch.Tensor] = {}

    def acc(name: str, value: torch.Tensor) -> None:
      losses[name] = losses.get(name, value.new_zeros(())) + value

    for member, c in enumerate(chunks):
      if c.numel() == 0:
        continue
      e = e_all[c]
      z = model.encode(
        obs_n[c, 0], contact_vec[c, 0] if cfg.privileged_encoder else None
      )
      # Losses anchored at the encoded first step.
      acc("obs", F.mse_loss(model.predict_obs(z, e), obs_n[c, 0]))
      if cfg.contact_heads:
        logits0, forces0 = model.contact(z, e)
        acc("contact_bce", _contact_bce(logits0, batch["contact_found"][c, 0]))
        acc(
          "contact_force",
          _masked_force_loss(forces0, force_log[c, 0], batch["contact_found"][c, 0]),
        )

      for t in range(horizon):
        rho = cfg.unroll_discount**t
        a = batch["action"][c, t]
        cmd = batch["command"][c, t] if self.dims.command_dim > 0 else None
        valid = batch["valid_next"][c, t]  # [b]
        true_next = batch["contact_found"][c, t + 1]

        teacher = bool(torch.rand((), generator=self._sampler) < p_tf)
        z_next, next_logits = model.step(
          member,
          z,
          a,
          cmd,
          e,
          true_next_contact=true_next if teacher else None,
          sample_mode=train,
        )

        acc(
          "consistency",
          rho * _masked_mse(z_next, z_bar[c, t + 1], valid),
        )
        pred_r = model.reward(member, z, a, cmd, e)
        target_r = torch.cat(
          [
            (batch["reward"][c, t] / r_std).unsqueeze(-1),
            batch["reward_terms"][c, t] / terms_std,
          ],
          dim=-1,
        )
        acc("reward", rho * F.mse_loss(pred_r, target_r))
        acc(
          "termination",
          rho
          * F.binary_cross_entropy_with_logits(
            model.termination(z_next, e), batch["terminated"][c, t]
          ),
        )
        acc(
          "torque",
          rho * F.mse_loss(model.torque(z, a, e), batch["torque"][c, t] / tau_std),
        )
        acc(
          "obs",
          rho * _masked_mse(model.predict_obs(z_next, e), obs_n[c, t + 1], valid),
        )
        if next_logits is not None:
          acc(
            "next_contact_bce",
            rho * _contact_bce(next_logits, true_next, valid),
          )
        if cfg.contact_heads:
          logits_t, forces_t = model.contact(z_next, e)
          acc(
            "contact_bce",
            rho * _contact_bce(logits_t, true_next, valid),
          )
          acc(
            "contact_force",
            rho
            * _masked_force_loss(
              forces_t, force_log[c, t + 1], true_next * valid.unsqueeze(-1)
            ),
          )
        z = z_next

    n_members = len([c for c in chunks if c.numel() > 0])
    weights = {
      "consistency": cfg.w_consistency,
      "reward": cfg.w_reward,
      "termination": cfg.w_termination,
      "torque": cfg.w_torque,
      "obs": cfg.w_obs,
      "contact_bce": cfg.w_contact_bce,
      "next_contact_bce": cfg.w_contact_bce,
      "contact_force": cfg.w_contact_force,
    }
    total = torch.zeros((), device=self.device)
    metrics: dict[str, float] = {"p_teacher_force": p_tf}
    for name, value in losses.items():
      value = value / n_members
      metrics[f"loss/{name}"] = float(value.detach())
      total = total + weights[name] * value
    metrics["loss/total"] = float(total.detach())
    return total, metrics

  # -- Loop -------------------------------------------------------------------

  def train(self) -> Path:
    cfg = self.train_cfg
    seq_len = self.model_cfg.horizon + 1
    autocast_dtype = torch.bfloat16
    use_autocast = cfg.autocast
    start = time.monotonic()

    while self.step < cfg.total_steps:
      batch = self.dataset.sample(
        cfg.batch_size, seq_len, self.device, "train", self._sampler
      )
      with torch.autocast(
        device_type=self.device.type, dtype=autocast_dtype, enabled=use_autocast
      ):
        loss, metrics = self.compute_loss(batch, train=True)
      self.opt.zero_grad(set_to_none=True)
      loss.backward()
      grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
      self.opt.step()
      self.model.update_target(cfg.ema_tau)
      self.step += 1

      if self.step % cfg.log_interval == 0:
        metrics["grad_norm"] = float(grad_norm)
        metrics["steps_per_s"] = self.step / (time.monotonic() - start)
        self._logger(metrics, self.step)
      if self.step % cfg.val_interval == 0:
        self._logger(self.validate(), self.step)
      if self.step % cfg.save_interval == 0:
        self.save(self.out_dir / f"wm_{self.step}.pt")

    final = self.out_dir / "wm_final.pt"
    self.save(final)
    return final

  @torch.no_grad()
  def validate(self, batch_size: int | None = None) -> dict[str, float]:
    self.model.eval()
    batch = self.dataset.sample(
      batch_size or min(self.train_cfg.batch_size, 256),
      self.model_cfg.horizon + 1,
      self.device,
      "val",
      self._sampler,
    )
    _, metrics = self.compute_loss(batch, train=False)
    self.model.train()
    return {f"val/{k}": v for k, v in metrics.items()}

  # -- Checkpointing ------------------------------------------------------------

  def save(self, path: str | Path) -> None:
    torch.save(
      {
        "model": self.model.state_dict(),
        "model_cfg": asdict(self.model_cfg),
        "dims": asdict(self.dims),
        "normalizer": self.normalizer.state_dict(),
        "dataset_meta": self.dataset.meta,
        "start_obs": self.dataset.start_obs,
        "step": self.step,
      },
      path,
    )


def load_world_model(
  path: str | Path, device: torch.device | str
) -> tuple[WorldModel, Normalizer, dict[str, Any]]:
  """Load a trained world model checkpoint (model, normalizer, metadata)."""
  ckpt = torch.load(path, map_location=device, weights_only=False)
  cfg = WorldModelCfg(**ckpt["model_cfg"])
  dims = WorldModelDims(**ckpt["dims"])
  model = WorldModel(cfg, dims).to(device)
  model.load_state_dict(ckpt["model"])
  model.eval()
  normalizer = Normalizer.from_state_dict(ckpt["normalizer"]).to(device)
  return model, normalizer, ckpt


def _masked_mse(
  pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
  per_row = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
  return (per_row * valid).sum() / valid.sum().clamp_min(1.0)


def _contact_bce(
  logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
  per_row = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(
    dim=-1
  )
  if valid is None:
    return per_row.mean()
  return (per_row * valid).sum() / valid.sum().clamp_min(1.0)


def _masked_force_loss(
  pred: torch.Tensor, target_log: torch.Tensor, in_contact: torch.Tensor
) -> torch.Tensor:
  """Huber on sign-log forces, only where ground-truth contact exists."""
  per_point = F.huber_loss(pred, target_log, reduction="none").mean(dim=-1)
  mask = in_contact
  return (per_point * mask).sum() / mask.sum().clamp_min(1.0)


def _make_logger(cfg: TrainCfg, out_dir: Path):
  if cfg.logger == "wandb":
    import wandb

    wandb.init(project=cfg.wandb_project, name=cfg.run_name, dir=str(out_dir))

    def log(metrics: dict[str, float], step: int) -> None:
      wandb.log(metrics, step=step)

    return log
  if cfg.logger == "tensorboard":
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    def log(metrics: dict[str, float], step: int) -> None:
      for k, v in metrics.items():
        writer.add_scalar(k, v, step)

    return log

  def log(metrics: dict[str, float], step: int) -> None:
    parts = "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
    print(f"[wm-train {step:>7d}] {parts}")

  return log
