# SPDX-FileCopyrightText: Generative Bionics S.R.L.
# SPDX-License-Identifier: LicenseRef-GenerativeBionics-AllRightsReserved

"""Gene01 Nohands velocity task configuration."""

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import gbionics_gene01_nohands_flat_env_cfg
from .motor_cond_env_cfg import gbionics_gene01_nohands_motor_cond_env_cfg
from .rl_cfg import gbionics_gene01_nohands_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Gbionics-Gene01-Nohands",
  env_cfg=gbionics_gene01_nohands_flat_env_cfg(),
  play_env_cfg=gbionics_gene01_nohands_flat_env_cfg(play=True),
  rl_cfg=gbionics_gene01_nohands_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Gbionics-Gene01-Nohands-MotorCond",
  env_cfg=gbionics_gene01_nohands_motor_cond_env_cfg(),
  play_env_cfg=gbionics_gene01_nohands_motor_cond_env_cfg(play=True),
  rl_cfg=gbionics_gene01_nohands_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)

__all__ = [
  "gbionics_gene01_nohands_flat_env_cfg",
  "gbionics_gene01_nohands_motor_cond_env_cfg",
  "gbionics_gene01_nohands_ppo_runner_cfg",
]
