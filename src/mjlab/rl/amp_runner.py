"""MJLab-native AMP on-policy runner.

Reuses :class:`amp_rsl_rl.algorithms.AMP_PPO`,
:class:`amp_rsl_rl.networks.Discriminator`, and
:class:`amp_rsl_rl.utils.AMPLoader` while keeping the MJLab runner
interface (``save``/``load``/``learn`` signatures, ONNX export, env-state
persistence, W&B integration).
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import amp_rsl_rl
import rsl_rl
import torch
from amp_rsl_rl.algorithms import AMP_PPO
from amp_rsl_rl.networks import Discriminator
from amp_rsl_rl.utils import AMPLoader
from rsl_rl.env import VecEnv
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from mjlab.rl.actor_critic_adapter import ActorCriticAdapter


class MjlabAmpOnPolicyRunner:
  """AMP training runner for MJLab environments.

  This runner mirrors the training loop of
  ``amp_rsl_rl.runners.AMPOnPolicyRunner`` but is adapted for MJLab's
  environment configuration and checkpoint format.

  Parameters
  ----------
  env : VecEnv
      Wrapped MJLab environment (``RslRlVecEnvWrapper``).
  train_cfg : dict
      Flat training configuration (produced by ``dataclasses.asdict``
      on :class:`AmpRunnerCfg`).
  log_dir : str | None
      Directory for checkpoints, ONNX exports, and videos.
  device : str
      Torch device string.
  """

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    self.cfg = train_cfg
    self.device = device
    self.env = env

    # Unpack sub-configs.
    alg_cfg = dict(train_cfg.get("algorithm", {}))
    disc_cfg = dict(train_cfg.get("discriminator", {}))
    dataset_cfg = dict(train_cfg.get("dataset", {}))

    # Resolve observation groups.
    observations = self.env.get_observations()
    default_sets = ["critic"]
    self.cfg["obs_groups"] = resolve_obs_groups(
      observations, self.cfg.get("obs_groups") or {}, default_sets
    )

    # Build actor-critic via rsl_rl 5.x MLPModel, wrapped in adapter.
    actor_build_cfg = dict(train_cfg.get("actor", {}))
    critic_build_cfg = dict(train_cfg.get("critic", {}))

    # Strip None-valued optional configs.
    for cfg_dict in (actor_build_cfg, critic_build_cfg):
      for opt in ("cnn_cfg", "distribution_cfg"):
        if cfg_dict.get(opt) is None:
          cfg_dict.pop(opt, None)
      if cfg_dict.get("rnn_type") is None:
        for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
          cfg_dict.pop(opt, None)

    actor_cls_name = actor_build_cfg.pop("class_name", "MLPModel")
    critic_cls_name = critic_build_cfg.pop("class_name", "MLPModel")

    actor_cls = resolve_callable(actor_cls_name)
    critic_cls = resolve_callable(critic_cls_name)

    actor = actor_cls(
      observations,
      self.cfg["obs_groups"],
      "actor",
      self.env.num_actions,
      **actor_build_cfg,
    ).to(self.device)
    critic = critic_cls(
      observations,
      self.cfg["obs_groups"],
      "critic",
      1,
      **critic_build_cfg,
    ).to(self.device)

    self.actor_critic = ActorCriticAdapter(actor, critic)

    # AMP joint names: extracted from the env config's "amp" obs group.
    amp_joint_names: list[str] = train_cfg.get("amp_joint_names", [])
    if not amp_joint_names:
      raise ValueError(
        "train_cfg must include 'amp_joint_names' — an ordered list of "
        "joint names matching the AMP motion dataset."
      )

    # AMP dataset.
    env_cfg = self.env.unwrapped.cfg  # type: ignore[union-attr]
    sim_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
    num_amp_obs = observations["amp"].shape[1]
    self.amp_data = AMPLoader(
      device=self.device,
      dataset_path_root=dataset_cfg["amp_data_path"],
      datasets=dataset_cfg["datasets"],
      simulation_dt=sim_dt,
      slow_down_factor=dataset_cfg.get("slow_down_factor", 1.0),
      expected_joint_names=amp_joint_names,
    )

    # Discriminator.
    self.discriminator = Discriminator(
      input_dim=num_amp_obs * 2,
      hidden_layer_sizes=disc_cfg.get("hidden_dims", [256, 128]),
      reward_scale=disc_cfg.get("reward_scale", 1.0),
      device=self.device,
      loss_type=disc_cfg.get("loss_type", "BCEWithLogits"),
      use_minibatch_std=disc_cfg.get("use_minibatch_std", True),
      empirical_normalization=disc_cfg.get("empirical_normalization", False),
    ).to(self.device)

    # AMP_PPO algorithm.
    alg_cfg.pop("class_name", None)
    # Remove keys not accepted by AMP_PPO.
    for key in list(alg_cfg.keys()):
      if key not in AMP_PPO.__init__.__code__.co_varnames:
        alg_cfg.pop(key)

    # rsl-rl v4+ requires separate actor/critic; v3 uses combined
    # actor_critic.
    try:
      from amp_rsl_rl.utils._compat import RSL_RL_V4_PLUS
    except ImportError:
      RSL_RL_V4_PLUS = False

    if RSL_RL_V4_PLUS:
      self.alg: AMP_PPO = AMP_PPO(
        actor=self.actor_critic.actor,
        critic=self.actor_critic.critic,
        discriminator=self.discriminator,
        amp_data=self.amp_data,
        device=self.device,
        **alg_cfg,
      )
    else:
      self.alg: AMP_PPO = AMP_PPO(
        actor_critic=self.actor_critic,
        discriminator=self.discriminator,
        amp_data=self.amp_data,
        device=self.device,
        **alg_cfg,
      )

    # Storage.
    self.num_steps_per_env: int = self.cfg["num_steps_per_env"]
    self.save_interval: int = self.cfg["save_interval"]
    obs_template = observations.clone().detach().to(self.device)
    self.alg.init_storage(
      self.env.num_envs,
      self.num_steps_per_env,
      obs_template,
      (self.env.num_actions,),
    )

    # Logging.
    self.log_dir = log_dir
    self.logger: Any = None
    self.tot_timesteps = 0
    self.tot_time = 0.0
    self.current_learning_iteration = 0
    self.git_status_repos: list[str] = [rsl_rl.__file__, amp_rsl_rl.__file__]
    self._export_policy_fn: Callable | None = None

    # Standalone codesign module (only active if env has codesign config).
    self._init_codesign()

  # ------------------------------------------------------------------
  # Training loop
  # ------------------------------------------------------------------

  def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
  ) -> None:
    if self.log_dir is not None and self.logger is None:
      self._init_logger()

    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf,
        high=int(self.env.max_episode_length),
      )

    obs = self.env.get_observations().to(self.device)
    amp_obs = obs["amp"].clone()
    self.train_mode()

    ep_infos: list[dict] = []
    rewbuffer: deque[float] = deque(maxlen=100)
    lenbuffer: deque[float] = deque(maxlen=100)
    cur_reward_sum = torch.zeros(
      self.env.num_envs, dtype=torch.float, device=self.device
    )
    cur_episode_length = torch.zeros(
      self.env.num_envs, dtype=torch.float, device=self.device
    )

    start_iter = self.current_learning_iteration
    tot_iter = start_iter + num_learning_iterations
    for it in range(start_iter, tot_iter):
      start = time.time()

      mean_style_reward_log = 0.0
      mean_task_reward_log = 0.0

      with torch.inference_mode():
        for _ in range(self.num_steps_per_env):
          actions = self.alg.act(obs)
          self.alg.act_amp(amp_obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          obs = obs.to(self.device)
          rewards = rewards.to(self.device)
          dones = dones.to(self.device)

          next_amp_obs = obs["amp"].clone()
          style_rewards = self.discriminator.predict_reward(amp_obs, next_amp_obs)

          mean_task_reward_log += rewards.mean().item()
          mean_style_reward_log += style_rewards.mean().item()

          rewards = 0.5 * rewards + 0.5 * style_rewards

          self.alg.process_env_step(obs, rewards, dones, extras)
          self.alg.process_amp_step(next_amp_obs)

          amp_obs = next_amp_obs

          if self.log_dir is not None:
            if "episode" in extras:
              ep_infos.append(extras["episode"])
            elif "log" in extras:
              ep_infos.append(extras["log"])
            cur_reward_sum += rewards
            cur_episode_length += 1
            new_ids = torch.nonzero(dones, as_tuple=False)
            if new_ids.numel() > 0:
              env_indices = new_ids.view(-1)
              rewbuffer.extend(cur_reward_sum[env_indices].cpu().tolist())
              lenbuffer.extend(cur_episode_length[env_indices].cpu().tolist())
              cur_reward_sum[env_indices] = 0
              cur_episode_length[env_indices] = 0

      stop = time.time()
      collection_time = stop - start

      start = stop
      self.alg.compute_returns(obs)

      mean_style_reward_log /= self.num_steps_per_env
      mean_task_reward_log /= self.num_steps_per_env

      update_results = self.alg.update()
      (
        mean_value_loss,
        mean_surrogate_loss,
        mean_amp_loss,
        mean_grad_pen_loss,
        mean_policy_pred,
        mean_expert_pred,
        mean_accuracy_policy,
        mean_accuracy_expert,
        mean_kl_divergence,
      ) = update_results[:9]
      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      # -- Codesign step (if codesign actuator is active) --
      self._codesign_step(it)

      if self.log_dir is not None:
        self._log(
          it=it,
          tot_iter=tot_iter,
          collection_time=collection_time,
          learn_time=learn_time,
          mean_value_loss=mean_value_loss,
          mean_surrogate_loss=mean_surrogate_loss,
          mean_amp_loss=mean_amp_loss,
          mean_grad_pen_loss=mean_grad_pen_loss,
          mean_policy_pred=mean_policy_pred,
          mean_expert_pred=mean_expert_pred,
          mean_accuracy_policy=mean_accuracy_policy,
          mean_accuracy_expert=mean_accuracy_expert,
          mean_kl_divergence=mean_kl_divergence,
          mean_style_reward_log=mean_style_reward_log,
          mean_task_reward_log=mean_task_reward_log,
          ep_infos=ep_infos,
          rewbuffer=rewbuffer,
          lenbuffer=lenbuffer,
        )

      if it % self.save_interval == 0:
        assert self.log_dir is not None
        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

      ep_infos.clear()

      if it == start_iter:
        try:
          from rsl_rl.utils import store_code_state  # type: ignore[attr-defined]

          git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
          if self.logger is not None and git_file_paths:
            for path in git_file_paths:
              self.logger.save_file(path)
        except ImportError:
          pass

    assert self.log_dir is not None
    self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
    self._print_codesign_summary()

  def _print_codesign_summary(self) -> None:
    """Print final codesign motor assignment if codesign module is active."""
    if self._codesign_module is None:
      return
    print("\n" + "=" * 60)
    print("CODESIGN FINAL MOTOR ASSIGNMENT")
    print("=" * 60)
    print(self._codesign_module.summary())
    print("=" * 60 + "\n")

  def _init_codesign(self) -> None:
    """Initialize standalone codesign module if env has codesign config."""
    self._codesign_module = None
    self._codesign_scheduler = None

    env_cfg = self.env.unwrapped.cfg  # type: ignore[union-attr]
    codesign_cfg_dict = getattr(env_cfg, "codesign", None)
    if codesign_cfg_dict is None:
      return

    from mjlab.actuator.gumbel_codesign import (
      CodesignConfig,
      CodesignScheduler,
      GumbelSoftmaxActuator,
      SymmetrySpec,
    )

    symmetry = SymmetrySpec(
      joint_names=codesign_cfg_dict["joint_names"],
      symmetry_pairs=codesign_cfg_dict["symmetry_pairs"],
    )
    cfg = CodesignConfig(
      n_types=codesign_cfg_dict["n_types"],
      init_tau_max=codesign_cfg_dict["init_tau_max"],
      temperature_init=codesign_cfg_dict["temperature_init"],
      temperature_min=codesign_cfg_dict["temperature_min"],
      temperature_decay=codesign_cfg_dict["temperature_decay"],
      lambda_types=codesign_cfg_dict["lambda_types"],
      lambda_balance=codesign_cfg_dict["lambda_balance"],
      lambda_tau=codesign_cfg_dict["lambda_tau"],
      lambda_saturation=codesign_cfg_dict["lambda_saturation"],
      lambda_rms=codesign_cfg_dict["lambda_rms"],
      lambda_peak=codesign_cfg_dict["lambda_peak"],
      min_tau=codesign_cfg_dict["min_tau"],
      max_tau=codesign_cfg_dict["max_tau"],
      codesign_lr=codesign_cfg_dict["codesign_lr"],
      codesign_interval=codesign_cfg_dict["codesign_interval"],
      freeze_tau=codesign_cfg_dict.get("freeze_tau", False),
    )
    self._codesign_module = GumbelSoftmaxActuator(cfg, symmetry).to(self.device)
    self._codesign_module.eval()
    self._codesign_scheduler = CodesignScheduler(
      self._codesign_module, device=self.device
    )
    self._codesign_warmup = codesign_cfg_dict.get("warmup_iters", 0)

    # Resolve the joint-to-actuator mapping for effort limit updates.
    self._codesign_joint_names = codesign_cfg_dict["joint_names"]
    print(
      f"[Codesign] Initialized: {cfg.n_types} types, "
      f"τ_max={codesign_cfg_dict['init_tau_max']}, "
      f"warmup={self._codesign_warmup} iters"
    )

  def _codesign_step(self, it: int) -> None:
    """Run standalone codesign optimizer step and update effort limits."""
    if self._codesign_module is None or self._codesign_scheduler is None:
      return

    # Skip codesign during warmup — let the policy learn to walk first.
    if it < self._codesign_warmup:
      if it == 0:
        print(
          f"[Codesign] Warmup active, codesign starts at iter {self._codesign_warmup}"
        )
      return

    # Collect torques from all actuators on the robot entity.
    robot = self.env.unwrapped.scene["robot"]
    tau = robot.data.actuator_force  # (num_envs, num_actuators)

    # Log torques into the codesign module.
    self._codesign_module.log_torques(tau.detach())

    # Run codesign optimization step.
    info = self._codesign_scheduler.step(it)
    if info is None:
      return

    # Update effort limits on the actual actuators based on learned τ_max.
    with torch.no_grad():
      tau_eff = self._codesign_module.tau_eff(use_gumbel=False)  # (n_joints,)
      for actuator in robot.actuators:
        if actuator.force_limit is None:
          continue
        for i, jname in enumerate(actuator.target_names):
          if jname in self._codesign_joint_names:
            j_idx = self._codesign_joint_names.index(jname)
            actuator.force_limit[:, i] = tau_eff[j_idx]

    self._log_codesign(info, it)

  def _log_codesign(
    self,
    info: dict[str, object],
    it: int,
  ) -> None:
    """Log detailed codesign metrics to W&B/TensorBoard."""
    import torch.nn.functional as F

    assert self._codesign_module is not None
    gumbel = self._codesign_module
    writer = self.logger
    if writer is None or not hasattr(writer, "add_scalar"):
      return

    # --- Scalar metrics ---
    writer.add_scalar("codesign/loss_total", info["codesign/total_loss"], it)
    writer.add_scalar("codesign/temperature", info["codesign/temperature"], it)
    writer.add_scalar("codesign/n_types_active", info["codesign/n_types_active"], it)

    # Per-type τ_max as separate curves.
    tau_max_list = info["codesign/tau_max"]
    for k, tau_val in enumerate(tau_max_list):
      writer.add_scalar(f"codesign_tau/type_{k}", tau_val, it)

    # τ_max spread (max - min): measures differentiation between types.
    if len(tau_max_list) > 1:
      tau_spread = max(tau_max_list) - min(tau_max_list)
      writer.add_scalar("codesign/tau_spread", tau_spread, it)

    # --- Assignment entropy (measures how decisive the assignment is) ---
    with torch.no_grad():
      p = F.softmax(gumbel.alpha / max(gumbel.temperature.item(), 0.01), dim=-1)
      entropy = -(p * (p + 1e-8).log()).sum(dim=-1)
      writer.add_scalar("codesign/assignment_entropy_mean", entropy.mean().item(), it)

      usage = p.mean(dim=0)
      for k in range(p.shape[-1]):
        writer.add_scalar(f"codesign_usage/type_{k}", usage[k].item(), it)

    # --- Per-joint assigned type ---
    if it % 50 == 0:
      assignment = gumbel.hard_assignment()
      for j_name in gumbel.symmetry.unique_joints:
        writer.add_scalar(f"codesign_joint/{j_name}", assignment[j_name], it)

      # Log assignment evolution table for W&B scatter plot.
      self._log_codesign_assignment_table(assignment, it)

    # --- Torque statistics from the logged buffer ---
    if gumbel._torque_log:
      with torch.no_grad():
        tau_all = torch.cat(gumbel._torque_log, dim=0)
        tau_abs = tau_all.abs()
        writer.add_scalar(
          "codesign_torque/rms", tau_all.pow(2).mean().sqrt().item(), it
        )
        writer.add_scalar("codesign_torque/peak", tau_abs.max().item(), it)
        joint_peaks = tau_abs.max(dim=0).values
        tau_eff = gumbel.tau_eff(use_gumbel=False)
        sat_ratios = joint_peaks / (tau_eff + 1e-6)
        writer.add_scalar(
          "codesign_torque/max_saturation_ratio",
          sat_ratios.max().item(),
          it,
        )
        writer.add_scalar(
          "codesign_torque/mean_saturation_ratio",
          sat_ratios.mean().item(),
          it,
        )

    # Print summary periodically.
    if it % 100 == 0:
      print(f"[Codesign iter {it}] {gumbel.summary()}")

  def _log_codesign_assignment_table(self, assignment: dict[str, int], it: int) -> None:
    """Log joint-type assignment as a W&B scatter plot.

    Produces a plot with: x=iteration, y=joint name, color=motor type.
    """
    if self._logger_type != "wandb":
      return

    if not hasattr(self, "_codesign_assignment_data"):
      self._codesign_assignment_data: list[list] = []

    assert self._codesign_module is not None
    tau_max_list = self._codesign_module.hard_tau_max()

    for j_name, type_idx in assignment.items():
      tau_val = tau_max_list[type_idx]
      self._codesign_assignment_data.append(
        [it, j_name, type_idx, f"Type {type_idx} ({tau_val:.0f} Nm)"]
      )

    # Log every 500 iterations to avoid excessive overhead.
    if it % 500 == 0 and self._codesign_assignment_data:
      self._plot_assignment_evolution(it)

  def _plot_assignment_evolution(self, it: int) -> None:
    """Render assignment evolution as a colored scatter and log to W&B."""
    import wandb

    try:
      import matplotlib

      matplotlib.use("Agg")
      import matplotlib.pyplot as plt
    except ImportError:
      return

    assert self._codesign_module is not None
    n_types = self._codesign_module.cfg.n_types
    tau_max_list = self._codesign_module.hard_tau_max()

    data = self._codesign_assignment_data
    iters = [row[0] for row in data]
    joints = [row[1] for row in data]
    types = [row[2] for row in data]

    # Map joint names to y-indices for plotting.
    unique_joints = list(dict.fromkeys(joints))  # preserve order
    joint_to_y = {j: i for i, j in enumerate(unique_joints)}
    y_vals = [joint_to_y[j] for j in joints]

    # Color map: one color per motor type.
    cmap = plt.cm.get_cmap("tab10", n_types)
    colors = [cmap(t) for t in types]

    fig, ax = plt.subplots(figsize=(10, max(4, len(unique_joints) * 0.4)))
    ax.scatter(iters, y_vals, c=colors, s=30, marker="s", edgecolors="none")

    ax.set_yticks(range(len(unique_joints)))
    ax.set_yticklabels(unique_joints, fontsize=9)
    ax.set_xlabel("Iteration")
    ax.set_title("Motor Type Assignment Evolution")
    ax.grid(axis="x", alpha=0.3)

    # Legend.
    handles = []
    for k in range(n_types):
      handles.append(
        plt.Line2D(
          [0],
          [0],
          marker="s",
          color="w",
          markerfacecolor=cmap(k),
          markersize=10,
          label=f"Type {k} ({tau_max_list[k]:.0f} Nm)",
        )
      )
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    plt.tight_layout()
    wandb.log({"codesign/assignment_evolution": wandb.Image(fig)}, step=it)
    plt.close(fig)

  # ------------------------------------------------------------------
  # Save / Load
  # ------------------------------------------------------------------

  def save(self, path: str, infos: dict | None = None) -> None:
    env_state = {
      "common_step_counter": self.env.unwrapped.common_step_counter  # type: ignore[union-attr]
    }
    infos = {**(infos or {}), "env_state": env_state}
    saved_dict = {
      "model_state_dict": self.actor_critic.state_dict(),
      "optimizer_state_dict": self.alg.optimizer.state_dict(),
      "discriminator_state_dict": self.discriminator.state_dict(),
      "iter": self.current_learning_iteration,
      "infos": infos,
    }
    # Save codesign state if active.
    if self._codesign_module is not None:
      saved_dict["codesign_state_dict"] = self._codesign_module.state_dict()
    torch.save(saved_dict, path)
    if self.cfg.get("upload_model", True) and self.logger is not None:
      self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
    self.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
    disc_state = loaded_dict.get("discriminator_state_dict")
    if disc_state is not None:
      self.discriminator.load_state_dict(disc_state, strict=False)

    amp_norm = loaded_dict.get("amp_normalizer")
    if amp_norm is not None and getattr(
      self.discriminator, "empirical_normalization", False
    ):
      self.discriminator.amp_normalizer.load_state_dict(amp_norm.state_dict())

    opt_state = loaded_dict.get("optimizer_state_dict")
    if opt_state is not None:
      try:
        self.alg.optimizer.load_state_dict(opt_state)
      except Exception:
        pass

    self.current_learning_iteration = loaded_dict.get("iter", 0)
    infos = loaded_dict.get("infos", {})
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"][  # type: ignore[union-attr]
        "common_step_counter"
      ]

    # Restore codesign state if available.
    codesign_state = loaded_dict.get("codesign_state_dict")
    if codesign_state is not None and self._codesign_module is not None:
      self._codesign_module.load_state_dict(codesign_state)
      print("[Codesign] Restored state from checkpoint.")
      print(f"[Codesign] {self._codesign_module.summary()}")

      # Apply learned effort limits to actuators (enables clipping during play).
      self._apply_codesign_effort_limits()

    return infos

  def _apply_codesign_effort_limits(self) -> None:
    """Set actuator force limits from the codesign module's learned τ_max."""
    if self._codesign_module is None:
      return
    robot = self.env.unwrapped.scene["robot"]
    with torch.no_grad():
      tau_eff = self._codesign_module.tau_eff(use_gumbel=False)
      for actuator in robot.actuators:
        if actuator.force_limit is None:
          continue
        for i, jname in enumerate(actuator.target_names):
          if jname in self._codesign_joint_names:
            j_idx = self._codesign_joint_names.index(jname)
            actuator.force_limit[:, i] = tau_eff[j_idx]

  # ------------------------------------------------------------------
  # Inference
  # ------------------------------------------------------------------

  def get_inference_policy(self, device: str | None = None):
    self.eval_mode()
    if device is not None:
      self.actor_critic.to(device)
    return self.actor_critic.act_inference

  def train_mode(self) -> None:
    self.actor_critic.train()
    self.discriminator.train()

  def eval_mode(self) -> None:
    self.actor_critic.eval()
    self.discriminator.eval()

  def add_git_repo_to_log(self, repo_file_path: str) -> None:
    self.git_status_repos.append(repo_file_path)

  # ------------------------------------------------------------------
  # Logging (private)
  # ------------------------------------------------------------------

  def _init_logger(self) -> None:
    assert self.log_dir is not None
    logger_type = self.cfg.get("logger", "tensorboard").lower()
    if logger_type == "wandb":
      from rsl_rl.utils.wandb_utils import WandbSummaryWriter

      self.logger = WandbSummaryWriter(
        log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
      )
    elif logger_type == "tensorboard":
      from torch.utils.tensorboard import SummaryWriter

      self.logger = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
    else:
      from torch.utils.tensorboard import SummaryWriter

      self.logger = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
    self._logger_type = logger_type

  def _log(  # noqa: C901
    self,
    *,
    it: int,
    tot_iter: int,
    collection_time: float,
    learn_time: float,
    mean_value_loss: float,
    mean_surrogate_loss: float,
    mean_amp_loss: float,
    mean_grad_pen_loss: float,
    mean_policy_pred: float,
    mean_expert_pred: float,
    mean_accuracy_policy: float,
    mean_accuracy_expert: float,
    mean_kl_divergence: float,
    mean_style_reward_log: float,
    mean_task_reward_log: float,
    ep_infos: list[dict],
    rewbuffer: deque[float],
    lenbuffer: deque[float],
  ) -> None:
    self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
    self.tot_time += collection_time + learn_time
    fps = int(
      self.num_steps_per_env * self.env.num_envs / (collection_time + learn_time)
    )

    if hasattr(self.logger, "add_scalar"):
      writer = self.logger
    else:
      return

    # Episode info.
    ep_string = ""
    if ep_infos:
      for key in ep_infos[0]:
        infotensor = torch.tensor([], device=self.device)
        for ep_info in ep_infos:
          if key not in ep_info:
            continue
          if not isinstance(ep_info[key], torch.Tensor):
            ep_info[key] = torch.Tensor([ep_info[key]])
          if len(ep_info[key].shape) == 0:
            ep_info[key] = ep_info[key].unsqueeze(0)
          infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
        value = torch.mean(infotensor).item()
        if "/" in key:
          writer.add_scalar(key, value, it)
        else:
          writer.add_scalar(f"Episode/{key}", value, it)
        ep_string += f"  {key}: {value:.4f}\n"

    # Standard losses.
    writer.add_scalar("Loss/value_function", mean_value_loss, it)
    writer.add_scalar("Loss/surrogate", mean_surrogate_loss, it)
    writer.add_scalar("Loss/amp_loss", mean_amp_loss, it)
    writer.add_scalar("Loss/grad_pen_loss", mean_grad_pen_loss, it)
    writer.add_scalar("Loss/policy_pred", mean_policy_pred, it)
    writer.add_scalar("Loss/expert_pred", mean_expert_pred, it)
    writer.add_scalar("Loss/accuracy_policy", mean_accuracy_policy, it)
    writer.add_scalar("Loss/accuracy_expert", mean_accuracy_expert, it)
    writer.add_scalar("Loss/mean_kl_divergence", mean_kl_divergence, it)
    writer.add_scalar("Perf/total_fps", fps, it)
    writer.add_scalar("Perf/collection time", collection_time, it)
    writer.add_scalar("Perf/learning_time", learn_time, it)

    if self._logger_type in ("wandb", "mlflow") and self.log_dir:
      if hasattr(writer, "save_video"):
        video_dir = Path(self.log_dir) / "videos" / "train"
        if video_dir.is_dir():
          for video_path in sorted(video_dir.glob("*.mp4")):
            writer.save_video(video_path, it)

    if len(rewbuffer) > 0:
      writer.add_scalar("Train/mean_reward", statistics.mean(rewbuffer), it)
      writer.add_scalar("Train/mean_episode_length", statistics.mean(lenbuffer), it)
      writer.add_scalar("Train/mean_style_reward", mean_style_reward_log, it)
      writer.add_scalar("Train/mean_task_reward", mean_task_reward_log, it)

    # Terminal output.
    if hasattr(self.actor_critic, "log_std"):
      mean_std = torch.exp(self.actor_critic.log_std).mean().item()
    else:
      mean_std = self.actor_critic.std.mean().item()

    pad = 35
    if len(rewbuffer) > 0:
      log_string = (
        f"{'#' * 80}\n"
        f" Learning iteration {it}/{tot_iter} \n\n"
        f"{'Computation:':>{pad}} {fps:.0f} steps/s "
        f"(collection: {collection_time:.3f}s, "
        f"learning {learn_time:.3f}s)\n"
        f"{'Value function loss:':>{pad}} {mean_value_loss:.4f}\n"
        f"{'Surrogate loss:':>{pad}} {mean_surrogate_loss:.4f}\n"
        f"{'AMP loss:':>{pad}} {mean_amp_loss:.4f}\n"
        f"{'Mean noise std:':>{pad}} {mean_std:.2f}\n"
        f"{'Mean reward:':>{pad}} {statistics.mean(rewbuffer):.2f}\n"
        f"{'Mean task reward:':>{pad}} {mean_task_reward_log:.4f}\n"
        f"{'Mean style reward:':>{pad}} {mean_style_reward_log:.4f}\n"
        f"{'Mean episode length:':>{pad}} "
        f"{statistics.mean(lenbuffer):.2f}\n"
      )
    else:
      log_string = (
        f"{'#' * 80}\n"
        f" Learning iteration {it}/{tot_iter} \n\n"
        f"{'Computation:':>{pad}} {fps:.0f} steps/s "
        f"(collection: {collection_time:.3f}s, "
        f"learning {learn_time:.3f}s)\n"
        f"{'Value function loss:':>{pad}} {mean_value_loss:.4f}\n"
        f"{'Surrogate loss:':>{pad}} {mean_surrogate_loss:.4f}\n"
        f"{'AMP loss:':>{pad}} {mean_amp_loss:.4f}\n"
        f"{'Mean noise std:':>{pad}} {mean_std:.2f}\n"
      )

    log_string += ep_string

    eta_seconds = self.tot_time / (it + 1) * (tot_iter - it)
    eta_h, rem = divmod(eta_seconds, 3600)
    eta_m, eta_s = divmod(rem, 60)
    log_string += (
      f"{'-' * 80}\n"
      f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
      f"{'Iteration time:':>{pad}} {collection_time + learn_time:.2f}s\n"
      f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
      f"{'ETA:':>{pad}} {int(eta_h)}h {int(eta_m)}m {int(eta_s)}s\n"
    )
    print(log_string)
