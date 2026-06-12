# Motor-conditioned policy: per-joint torque randomization in the observation

## Goal

Train **one** policy that can run with **any** per-joint max-torque (`τ_max`)
configuration, by randomizing `τ_max` every episode and feeding it to the policy
as an observation. A frozen motor-conditioned policy then lets a genetic
algorithm (Phase 2) search for the best torque combination — minimizing motor
cost / number of motor tiers — **without retraining per candidate**, and
supports different policies for different tasks.

Task: `Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond`.

## How it works

Three pieces in `src/mjlab/tasks/velocity/mdp/motor_randomization.py`:

1. **Reset event** `randomize_motor_tau_max` — on every reset, each env samples a
   new `τ_max` per joint and applies it via `set_motor_tau_max`.
2. **`set_motor_tau_max`** (single source of truth) — writes the value to:
   - a per-env buffer `env._motor_tau_max` of shape `(num_envs, 12)`, and
   - the actuators' `force_limit`, so the physics actually **clips** torque at
     `τ_max` (`IdealPdActuator` does `clamp(effort, -force_limit, force_limit)`).
3. **Observation** `motor_tau_max` — returns the buffer normalized to `[0, 1]`,
   added to **both actor and critic** groups so the policy conditions its gait on
   the available motors.

Wired up in `config/qdd/motor_cond_env_cfg.py`.

### Sampling

- **Range:** single absolute `5–90 Nm`, uniform for all joints.
- **Symmetry:** L/R pairs share a value — 6 unique samples mirrored to 12.

```
sample (n_reset, 6) ~ U(5, 90)  ->  mirror to (n_reset, 12)  ->
  write buffer + actuator force_limit
obs = (buffer - 5) / (90 - 5)        # normalized [0,1], shape (num_envs, 12)
```

## Why these choices

- **`τ_max` in the obs (actor + critic):** the policy must *know* how strong each
  motor is to adapt its gait; the critic needs it because episode difficulty
  depends on the sampled config (lower value-function variance).
- **Hard clamp in physics + value in obs:** consistent train/deploy — what the
  policy is told matches what it physically gets.
- **Per-episode (reset) randomization:** exposes the policy to many configs so it
  genuinely conditions, rather than learning a single average gait.
- **Observation normalization:** training already uses running obs normalization
  (`obs_normalization` / `empirical_normalization`), so `[0,1]` is fine.

## Phase 2 hook (GA)

`set_motor_tau_max(env, env_ids, tau_values, ...)` is the same entry point the GA
will use to inject a fixed genome per env, roll out the frozen policy, and score
**task performance − motor cost**. Each evaluation is a rollout, not a training
run.

## Notes / open considerations

- Uniform `5–90 Nm` for every joint means some configs are physically infeasible
  (e.g. a hip at 5 Nm can't hold the robot up), so a fraction of episodes are
  unwalkable. If training is dominated by these failures, add a **curriculum**
  (start near nominal, widen). The range is configurable via `MOTOR_TAU_RANGE`.
- Nominal QDD limits for reference: hips/knees 80, hip_yaw 40, ankle_pitch 50,
  ankle_roll 17 Nm.
- Flat-QDD torque/power penalties are keyed to *realized* torque (fine); the
  nominal-keyed codesign penalties are **not** in this config.

## Verified

Actor obs `168 → 180`, critic `180 → 192` (+12 each); L/R symmetry holds; sampled
values in `[5, 90]`; actuator `force_limit` exactly matches the buffer (physics
clips correctly); the `set_motor_tau_max` injection path drives both actuators
and observation.

## Try it

```sh
uv run train Mjlab-Velocity-Flat-Gbionics-QDD-MotorCond
```
