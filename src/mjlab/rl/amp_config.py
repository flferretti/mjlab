"""AMP runner configuration for MJLab.

Extends the base RSL-RL config with discriminator and dataset fields required
by the AMP training loop.
"""

from dataclasses import dataclass, field

from mjlab.rl.config import RslRlBaseRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg


@dataclass
class AmpDiscriminatorCfg:
  """Discriminator network configuration."""

  hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
  reward_scale: float = 1.0
  loss_type: str = "BCEWithLogits"
  use_minibatch_std: bool = True
  empirical_normalization: bool = False


@dataclass
class AmpDatasetCfg:
  """AMP motion dataset configuration."""

  amp_data_path: str = ""
  datasets: dict[str, float] = field(default_factory=dict)
  slow_down_factor: float = 1.0


@dataclass
class AmpRunnerCfg(RslRlBaseRunnerCfg):
  """Runner config for AMP-based training."""

  class_name: str = "AMPOnPolicyRunner"

  actor: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      }
    )
  )
  critic: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
  discriminator: AmpDiscriminatorCfg = field(default_factory=AmpDiscriminatorCfg)
  dataset: AmpDatasetCfg = field(default_factory=AmpDatasetCfg)
  amp_joint_names: list[str] = field(default_factory=list)
  """Ordered joint names matching the AMP motion dataset."""
