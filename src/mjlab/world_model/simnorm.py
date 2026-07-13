"""Simplicial normalization (SimNorm) from TD-MPC2.

Partitions a latent vector into fixed-size groups and applies a softmax within
each group, so the latent lives on a product of simplices. This bounds the
latent space, which stabilizes long dynamics unrolls — particularly through
contact discontinuities where unbounded latents tend to blow up.

Reference: Hansen et al., "TD-MPC2: Scalable, Robust World Models for
Continuous Control" (ICLR 2024).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimNorm(nn.Module):
  """Softmax over contiguous groups of ``group_size`` latent dimensions."""

  def __init__(self, group_size: int = 8) -> None:
    super().__init__()
    if group_size < 2:
      raise ValueError(f"group_size must be >= 2, got {group_size}")
    self.group_size = group_size

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    if dim % self.group_size != 0:
      raise ValueError(
        f"Latent dim {dim} is not divisible by SimNorm group size {self.group_size}."
      )
    shape = x.shape
    x = x.view(*shape[:-1], dim // self.group_size, self.group_size)
    x = torch.softmax(x, dim=-1)
    return x.view(*shape)

  def extra_repr(self) -> str:
    return f"group_size={self.group_size}"
