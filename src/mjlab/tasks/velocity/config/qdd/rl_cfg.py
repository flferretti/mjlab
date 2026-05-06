"""RL configuration for Gbionics QDD velocity task with AMP."""

from gb_motion_prior_lupin import resolve_dataset_dir

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg
from mjlab.rl.amp_config import AmpDatasetCfg, AmpDiscriminatorCfg, AmpRunnerCfg
from mjlab.tasks.velocity.config.qdd.env_cfgs import AMP_JOINT_NAMES


def gbionics_qdd_ppo_runner_cfg() -> AmpRunnerCfg:
  """Create AMP RL runner configuration for Gbionics QDD velocity task.

  Network architecture and hyperparameters match gb-rl-locomotion's
  ``QDDRunnerCfg`` for checkpoint compatibility.
  """
  return AmpRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      class_name="AMP_PPO",
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=2.0e-3,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    discriminator=AmpDiscriminatorCfg(
      hidden_dims=[256, 128],
      reward_scale=1.0,
      loss_type="BCEWithLogits",
      use_minibatch_std=True,
      empirical_normalization=True,
    ),
    dataset=AmpDatasetCfg(
      amp_data_path=resolve_dataset_dir("lowerbodyqdd-mixamo"),
      datasets={
        "Happy Left Turn Slow": 0.5,
        "Happy Right Turn Slow": 0.5,
        "Happy Left Turn Fast": 0.5,
        "Happy Right Turn Fast": 0.5,
        "Standing": 0.2,
        "Start Walking": 1.0,
        "Stop Walking": 1.0,
        "Walking Left Turn": 1.0,
        "Walking Right Turn": 1.0,
        "Walking Backwards": 1.0,
      },
      slow_down_factor=1.0,
    ),
    amp_joint_names=list(AMP_JOINT_NAMES),
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
      "discriminator": ("amp",),
    },
    experiment_name="qdd_velocity",
    save_interval=2000,
    num_steps_per_env=32,
    max_iterations=64_000,
  )
