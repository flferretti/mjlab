from mjlab.rl.amp_runner import MjlabAmpOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .codesign_env_cfg import gbionics_qdd_codesign_env_cfg
from .env_cfgs import (
  gbionics_qdd_flat_env_cfg,
  gbionics_qdd_rough_env_cfg,
)
from .motor_cond_env_cfg import gbionics_qdd_motor_cond_env_cfg
from .rl_cfg import gbionics_qdd_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Gbionics-QDD",
  env_cfg=gbionics_qdd_rough_env_cfg(),
  play_env_cfg=gbionics_qdd_rough_env_cfg(play=True),
  rl_cfg=gbionics_qdd_ppo_runner_cfg(),
  runner_cls=MjlabAmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Gbionics-QDD",
  env_cfg=gbionics_qdd_flat_env_cfg(),
  play_env_cfg=gbionics_qdd_flat_env_cfg(play=True),
  rl_cfg=gbionics_qdd_ppo_runner_cfg(),
  runner_cls=MjlabAmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Gbionics-QDD-Codesign",
  env_cfg=gbionics_qdd_codesign_env_cfg(),
  play_env_cfg=gbionics_qdd_codesign_env_cfg(play=True),
  rl_cfg=gbionics_qdd_ppo_runner_cfg(),
  runner_cls=MjlabAmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond",
  env_cfg=gbionics_qdd_motor_cond_env_cfg(),
  play_env_cfg=gbionics_qdd_motor_cond_env_cfg(play=True),
  rl_cfg=gbionics_qdd_ppo_runner_cfg(),
  runner_cls=MjlabAmpOnPolicyRunner,
)
