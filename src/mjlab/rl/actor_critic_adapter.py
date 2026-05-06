"""Adapter bridging rsl_rl 5.x separate actor/critic to the combined
``ActorCritic`` interface expected by :class:`amp_rsl_rl.algorithms.AMP_PPO`.

rsl_rl 5.x split the monolithic ``ActorCritic`` into individual
``MLPModel`` (actor) and ``MLPModel`` (critic) modules. This thin wrapper
delegates to both and exposes the combined API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from tensordict import TensorDict

if TYPE_CHECKING:
  from rsl_rl.models.mlp_model import MLPModel


class ActorCriticAdapter(nn.Module):
  """Combine separate actor and critic ``MLPModel`` instances into a single
  ``ActorCritic``-like interface for ``AMP_PPO``.

  Parameters
  ----------
  actor : MLPModel
      Actor network.
  critic : MLPModel
      Critic network.
  """

  actor: MLPModel
  critic: MLPModel

  def __init__(self, actor: Any, critic: Any) -> None:
    super().__init__()
    self.actor = actor
    self.critic = critic

  # -- Properties expected by AMP_PPO --

  @property
  def is_recurrent(self) -> bool:
    return bool(getattr(self.actor, "is_recurrent", False))

  @property
  def action_mean(self) -> torch.Tensor:
    return self.actor.output_mean

  @property
  def action_std(self) -> torch.Tensor:
    return self.actor.output_std

  @property
  def entropy(self) -> torch.Tensor:
    return self.actor.output_entropy

  @property
  def std(self) -> torch.Tensor:
    return self.actor.output_std

  @property
  def log_std(self) -> torch.Tensor:
    return torch.log(self.actor.output_std)

  # -- Methods expected by AMP_PPO --

  def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
    return self.actor(obs, stochastic_output=True)

  def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
    return self.critic(obs)

  def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
    return self.actor.get_output_log_prob(actions)

  def act_inference(self, obs: TensorDict) -> torch.Tensor:
    return self.actor(obs, stochastic_output=False)

  def update_normalization(self, obs: TensorDict) -> None:
    self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)

  def reset(self, dones: torch.Tensor | None = None) -> None:
    self.actor.reset(dones)
    self.critic.reset(dones)

  def get_hidden_states(self) -> tuple:
    return (
      self.actor.get_hidden_state(),
      self.critic.get_hidden_state(),
    )
