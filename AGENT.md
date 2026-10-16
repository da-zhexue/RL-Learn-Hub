# AGENT.md

> **How this file differs from README.md**: README is a concise, human-facing
> guide to conclusions and how to run the project. This file records **why**,
> **past pitfalls**, **which numbers were measured**, and **what must not change**.
> Read this entire file before modifying the repository; follow it when writing code.
>
> Maintenance rule: **add every newly discovered pitfall to the appropriate category**
> (symptom / root cause / fix / measured results). Recording these findings matters
> more than adding code: most losses in this repository have come from silent
> failures, not from an inability to implement algorithms.

---

## 1. Repository Structure

```text
/home/lin/unitree/
├── go2_common/           Algorithm-independent shared layer for SAC / PPO (ONE implementation)
│   ├── config.py         Constants, EnvCfg/RewardCfg, quat_to_rpy/projected_gravity (no mujoco import)
│   ├── terrain.py        Curriculum variants (build_model), terrain heights (TerrainHeight), criteria
│   ├── env.py            Go2TerrainEnv (Gymnasium) + run_episode
│   ├── reward.py         One function per reward term + compute_reward() -> (total, parts)
│   └── train_utils.py    make_env / EvalMetricsCallback / load_resume_configs / resolve_model_path
├── go2_sac/              SAC-specific functionality only
│   ├── config.py         SAC TrainCfg
│   ├── train.py          SAC training entry point (trot_action for diagnostics, demonstration seeding)
│   ├── play.py           Playback/deployment (--mode viewer / dds)
│   └── diag.py           Diagnostic subcommands: terrain/stance/lift/frontier/policy/why/reward
├── go2_ppo/              PPO equivalent; no seeding, otherwise matches go2_sac
│   ├── config.py         PPO TrainCfg
│   ├── train.py          PPO training entry point
│   ├── play.py           Playback/deployment
│   └── diag.py           Thin wrapper: replace go2_sac.diag.load_policy with PPO.load, then delegate
├── go2_issac/            Same task ported to Isaac Sim (rsl_rl, self-converted USD assets)
│   ├── course.py         Single source of terrain geometry, pure numpy (no mujoco / isaaclab imports)
│   ├── core.py           Observations / 14 reward terms / termination criteria, pure torch (same restriction)
│   ├── tests/            Local element-by-element parity checks: course_parity / core_parity
│   ├── assets/           Asset and resting-state reference data from compiled MjModel (generated locally, committed)
│   ├── convert_assets.py MJCF -> USD (TARGET MACHINE ONLY)
│   ├── env_cfg.py mdp/ agents/ train.py play.py smoke.py   (TARGET MACHINE ONLY)
│   └── RUNBOOK.md        Target-machine guide: installation, conversion, checks, training
├── ctrl/                 Code maintained by this repository
│   ├── go2_ctrl.py       Go2Ctrl: DDS I/O + independent 500 Hz thread + interpolation/smoothing + reset
│   └── ctrl_test.py      Manual DDS tests
├── run_curriculum.sh     Terrain curriculum (flat -> two low obstacles), A=0.6, increasing heights
├── run_stairs.sh         Stair stage, A=0.7, seven scales from 0.56 to 1.00
├── README.md             Concise human-facing guide
├── AGENT.md              This file
├── models/               Training outputs (gitignored, tens of GB)
├── runs/                 TensorBoard (gitignored)
├── logs/                 nohup logs (gitignored)
├── unitree_mujoco/       Upstream clone: DO NOT MODIFY
├── unitree_sdk2_python/  Upstream clone: DO NOT MODIFY
└── unitree_ros2/         Upstream clone: DO NOT MODIFY
```

**Never modify any of the three upstream directories.** Changes to simulation
behavior must go through runtime `MjSpec` modifications in `go2_common/terrain.py`
(remove/scale geometry), the stepping logic in `go2_common/env.py`, or parameters
in `go2_common/config.py`. **Modify rewards and environments only in the shared
`go2_common/` implementation**, so both algorithm packages inherit the changes.
Do not copy environments or rewards into `go2_sac/` or `go2_ppo/`.
**Do not modify the user's `scene.xml`.** Curriculum terrain variants are compiled
at runtime with `MjSpec` and are not written to disk.

---

## 2. Running the Project

Environment: conda environment **`sim`** (Python 3.10.12). CPU only, 16 cores, no GPU.

Both invocation styles are supported. At the top of `train.py` / `play.py`,
`if __package__ in (None, ""): sys.path.insert(...)` provides the fallback:

```bash
python3 -m go2_sac.train ...     # From the repository root; recommended
python3 go2_sac/train.py ...     # Also supported
```

Common commands:

```bash
# Train from scratch (automatically seed the replay buffer with an open-loop trot)
python3 -m go2_sac.train --terrain full --terrain-scale 0.35 --action-scale 1.0 \
    --steps 2000000 --gradient-steps 1 --eval-freq 50000 --eval-episodes 20 \
    --checkpoint-freq 100000 --save-dir models/amp1_s035

# Curriculum / stairs
./run_curriculum.sh models/go2_sac_flat_xxx/checkpoints/rl_800000_steps.zip
./run_stairs.sh models/st_s056_full/best/best_model.zip

# Playback
python3 -m go2_sac.play --mode viewer --model models/xxx/best/best_model.zip
python3 -m go2_sac.play --mode dds --model models/xxx/model.zip   # Start unitree_mujoco.py separately
```

```bash
# PPO: same curriculum and argument names, a different module
python3 -m go2_ppo.train --terrain flat --steps 200000 --checkpoint-freq 40000
python3 -m go2_ppo.play  --mode viewer --model models/go2_ppo_flat_xxx/model.zip
```

```bash
# Isaac Sim: same task, rsl_rl, self-converted assets. Runtime commands require an NVIDIA GPU.
python3 -m go2_issac.tests.course_parity     # These two parity checks also run locally
python3 -m go2_issac.tests.core_parity
python3 go2_issac/convert_assets.py
python3 go2_issac/smoke.py --all --num-envs 4
python3 go2_issac/train.py --terrain flat --steps 200_000 --num-envs 512 --visualizer none
python3 go2_issac/play.py --resume models/go2_issac_flat_xxx
```

The CLI for `go2_issac/train.py` is **much smaller** than those of the other two:
`--terrain --terrain-scale --action-scale --steps --num-envs --seed --save-dir
--resume --privileged --mini-batches --max-iterations`.
PPO hyperparameters are not exposed through that CLI; they all live in
`agents/rsl_rl_ppo_cfg.py`, with a single set of defaults that does not drift
against `go2_ppo/config.py`. See `go2_issac/RUNBOOK.md` for target-machine setup.

CLI for `go2_sac.train`:
`--terrain --scene --steps --n-envs --seed --save-dir --resume --privileged
--reset-x LO HI --terrain-scale --action-scale --learning-rate --target-entropy
--gradient-steps --fresh-reward --eval-freq --eval-episodes --checkpoint-freq
--torch-threads --seed-trot`.

CLI for `go2_ppo.train`: **remove** `--target-entropy --gradient-steps --seed-trot`
from the above (PPO has no target entropy, no UTD control, and no replay buffer to
seed), and **add** `--n-steps --batch-size --n-epochs --ent-coef --target-kl`.
All other arguments have exactly the same semantics, including "on resume,
`--steps` means additional steps for this stage" and "changing stages requires
`--fresh-reward`."

**SAC throughput baseline** (8 environments, CPU):
- `gradient_steps=1`: **about 440 environment steps/s** (about 18 ms per update,
  one update every 8 steps).
- `gradient_steps=2`: about 220 steps/s; `-1` (UTD=1): **about 55 steps/s**.
- Estimated duration: 2M steps at `gradient_steps=1` takes about 75 minutes.

**PPO throughput has not been properly measured.** Only 512/1024-step validation
runs have been performed; they take seconds and are dominated by startup overhead.
Use the runtime `time/fps` output to estimate duration. Do not reuse SAC's numbers.

---

## 3. Coding Rules (Mandatory for Code Changes)

1. **`config.py` must not import mujoco.** It depends only on numpy. `play.py`
   needs to read `config.json`, and `reward.py` must support standalone checks
   without simulation; neither should be forced to construct a model.
2. **Reward term name = function name in `reward.py` = field name in `RewardCfg`.**
   Maintain the one-to-one mapping through `TERMS = (("progress", _progress), ...)`
   and `getattr(cfg, name)`. **When adding/changing rewards, change weights in
   `RewardCfg`, not existing function bodies.** For a new term, add a function in
   `reward.py`, register it in `TERMS`, and add the same-named field in `RewardCfg`.
3. **Before adding a reward term, check whether it can be exploited.** Rejected
   examples in this repository:
   - `feet_air_time`: without a velocity-command gate, stepping in place and
     running off a cliff are classic reward exploits.
   - `Δbase_z` as a climbing signal: raising the hindquarters or bouncing can
     exploit it, so use `Δterrain_h` instead.
   - `base_height` in the form `exp(-err²)`: paying the full reward for the correct
     standing height creates guaranteed income for standing still with zero action.
4. **Comments must explain why and include measured numbers.** The high comment
   density is deliberate: every conclusion must be traceable to a measurement.
   Do not write comments that merely restate "this code does X."
5. **Document failed attempts too.** Explain how hypotheses such as "critic
   collapse caused policy collapse" were falsified; otherwise the next person,
   or your next session, will repeat them.
6. **Do not add third-party dependencies.** Available dependencies are mujoco
   3.10.0, stable-baselines3 2.9.0, gymnasium, torch, and numpy 1.26.4. Training
   must remain runnable on CPU.
7. **Do not relax success criteria to make a demo look better.** Success means
   reaching the 0.92 m top platform. Report the highest stair level reached as a
   secondary progress metric, without lowering the completion standard.

---

## 4. Hard Invariants (Violations Often Fail Silently)

| # | Invariant | Consequence of violating it |
|---|---|---|
| 1 | **Joint order**: `qpos` follows the body tree, FL,FR,RL,RR; actuators / DDS `LowCmd`/`LowState` / sensors in `go2.xml` use FR,FL,RR,RL. | PD feedback is connected to the wrong joints; the robot collapses as soon as it is powered. |
| 2 | Build address maps (`env.qadr` / `env.vadr`) from `m.actuator_trnid[:,0]` + `m.jnt_qposadr` / `m.jnt_dofadr`. | Hardcoding `qpos[7:19]` is wrong. Measured mapping: `[10,11,12, 7,8,9, 16,17,18, 13,14,15]`. |
| 3 | 47-dimensional **proprioceptive** observation = angular velocity 3×0.25 + projected gravity 3 + **heading (sin,cos) 2** + command 3×2 + (q−q_default) 12 + dq 12×0.05 + previous action 12; clip to ±10. The heading block **must follow projected gravity** (indices 6:8; added 2026-10-10; see §5-B7). Appended **after** these 47 (and after the optional privileged dim) come the optional extras: `privileged=True` → 1 dim (base height above the local support); `EnvCfg.height_scan=True` → `len(scan_x)×len(scan_y)` dims of terrain height scan (default 5×3=15; value = terrain height under the sample point − base height, clip ±1; sample points are rotated into the world with the full base quaternion, see `env._height_scan`). So the observation is 47 / 48 / 62 / 63 dims. | Changing dimensions or ordering invalidates all old models (`config.json` stores the layout, but `play.py` does not validate it). Inserting heading elsewhere breaks the projected-gravity slice `obs[0,3:6]` in `smoke.py`. **Flipping `height_scan` changes the dimension → you cannot warm-start across it** (SB3 fails loudly at load; there is no padding). |
| 4 | **Every observation dimension must be reproducible from DDS.** The 47 proprioceptive dims must come from `LowState` alone — do not read `rpy`, base height, or `foot_force` in the observation code. The privileged dim uses `rt/sportmodestate`. `height_scan` is an explicit exception: those dims come from `rt/utlidar/height_map` (`unitree_go/HeightMap_`), not from `LowState`. | `--mode dds` and viewer would run different policies, invalidating deployment on the real robot. The DDS bridge always reports 0 for `rpy`/`foot_force`. For a `height_scan` model, `go2_ppo/play.py --mode dds` raises instead of silently assembling a 47-dim observation (2026-10-16) — the HeightMap path is not wired up yet. |
| 5 | Timing: `SIM_DT=0.005` (= `SIMULATE_DT` in `simulate_python/config.py`), `DECIMATION=4`, giving a 50 Hz policy. | Other values increase the sim-to-sim gap; the 50 Hz frame assembly in `play.py --mode dds` must change accordingly. |
| 6 | PD at every physics substep: `tau = kp*(q_target − q) − kd*dq`, clipped per actuator to `ctrlrange`; kp=60 / kd=3.5. | **kd ≥ 4.5 excites high-frequency oscillation** (measured `|dq|` reaches 40 rad/s; the robot shakes in place and moves backward). |
| 7 | Default stance: `DEFAULT_Q = [0, 0.9, −1.8] * 4`, in FR,FL,RR,RL order, from the `home` keyframe in `go2.xml`. | Incorrect ordering causes the same failure as invariant 1. |
| 8 | Timeouts must use `truncated`, not `terminated`. | SB3 can then bootstrap correctly; `terminated` cuts off Q at timeout boundaries. |

---

## 5. Past Pitfalls

### A. Silent Failures (Most Dangerous: No Error, Apparent Success, No Actual Effect)

**A1: CLI hyperparameters were silently ignored during resumed training.**
- Symptom: resuming the same model for 50k steps with `--learning-rate 2e-4` and
  `1e-4` produced **bit-for-bit identical evaluation results** (177.2 / 62.8 / 138.4).
- Root cause: `SAC.load()` follows `cls(...) → model.__dict__.update(data) →
  model.__dict__.update(kwargs)`. `update(data)` restores all checkpoint
  hyperparameters, and calling `SAC.load()` **without kwargs** provides no overrides.
- Fix in the resume branch of `train.py`: explicitly compare and use `setattr` to
  override `learning_rate / batch_size / tau / gamma / gradient_steps /
  target_entropy / train_freq`. **Assigning `learning_rate` alone is not enough.**
  It is **not a property** in SB3. Each gradient step uses
  `_update_learning_rate([actor.optimizer, critic.optimizer, ent_coef_optimizer])`
  in `SAC.train()`, which reads `self.lr_schedule(...)`. Also replace
  `model.lr_schedule = get_schedule_fn(lr)`; otherwise none of the three optimizers
  changes its learning rate.
- Impact: every curriculum stage specified `LR=2e-4`, but training actually kept
  the model's initial 3e-4. That was the worst measured fine-tuning rate: 62.8 return
  / 1 stair level, versus 122.5 / 6 levels at 1e-4. **This was one of the main
  reasons further training made the model worse.**

**A2: Old checkpoint reward weights override changes in `config.py` on resume.**
- Symptom: changing the default `climb` weight and adding a `base_height` dead zone
  produced no behavioral change after resuming.
- Root cause: as in A1, `load_resume_configs` restores reward weights from the
  checkpoint's `config.json`; **it recognizes field names, not code changes**.
- Fix: **changing curriculum stages requires `--fresh-reward`**, which uses current
  `RewardCfg` defaults instead of checkpoint weights.
- Related rule: **explicitly pass `--terrain` when changing stages**, or the
  resumed model's terrain is retained.

**A3: On resume, `--steps` means additional steps for this stage, not a cumulative target.**
- Root cause: with `reset_num_timesteps=False`, SB3 executes
  `total_timesteps += self.num_timesteps` (`base_class.py:416`).
- Impact: interpreting it as a cumulative target inflated every stage to millions
  of steps and prevented the curriculum from reaching its final stage. The first
  curriculum attempt failed this way: intended 250k steps, actually trained 1M.

**A4: `gradient_steps` does not mean "steps per update."**
- Actual meaning: updates per collection round, where **one round =
  `train_freq × n_envs` environment steps** (`collect_rollouts` increments
  `self.num_timesteps += env.num_envs`).
- Measured old checkpoint: `_n_updates=412,498 / num_timesteps=3,300,024 = 0.1250`.
  With `1` and 8 environments, UTD is only **1/8**: 3.3M environment steps yielded
  only about 410k updates.
- `-1` gives SB3's standard UTD=1 (`gradient_steps = rollout.episode_timesteps = 8`),
  at approximately eight times the wall-clock cost.
- **Warning: whether UTD=1 helps has not been established.** In measured runs it
  actually accelerated collapse: rw4 at 1/8 lasted 40k steps, while rw5 at 1 lasted
  only 20k. Those runs also differed in other variables. **Do not recommend it as
  known to be better.** The historical curriculum used 1 throughout.

**A5: All TensorBoard runs were written into a single `runs/` directory.**
- Root cause: SB3 only increments `SAC_<n>`. After dozens of curriculum runs, one
  directory contained dozens of event files with overlapping step ranges.
- Impact: curves from different learning rates and terrains were mixed into one.
  This caused an actual misdiagnosis: at 3.35M, -11.98 / 122.5 / 50.61 / 219.4 all
  appeared as the "same point."
- Fix: separate directories by `--save-dir` name (implemented in `train.py`).

**A6: The seeded buffer disabled timeout handling, and resumed training inherited it
(diagnosed 2026-10-11; fixed 2026-10-12).**
- Root cause: `go2_sac.train.seed_replay_buffer` constructed `ReplayBuffer` with
  `handle_timeout_termination=False`, then replaced `model.replay_buffer`. Loading
  the `.pkl` on resume retained False. Although the environment correctly returned
  `truncated`, the buffer did not record `TimeLimit.truncated`, so timeout critic
  targets were cut off at `r` as if they were genuine terminations, violating
  invariant 8 in §4.
- Measurements: in the `go2_sac_steps_s1.00/best` buffer, 459 of 578 complete
  episodes lasted exactly 1000 steps, but all `timeouts` were zero. The same flag
  was False for `full_s0.56/rl_1750016`, with 417 of 597 episodes at exactly 1000
  steps. That duration strongly suggests timeout, but old arrays alone cannot
  exclude a genuine termination on that same step.
- Repair requirements: correctly save and handle timeout labels for new data.
  **Setting the old buffer's flag to True does not recover lost labels.** Plan
  data refresh/recollection. No training ablation has been performed, so this
  defect cannot be claimed to explain all regression by itself.
- Fix: seeding records `done=terminated or truncated` and `TimeLimit.truncated`;
  timeout handling is enabled. Resume rejects old buffers whose flag is False;
  only explicit `--reset-replay-buffer` skips the old `.pkl`. When resuming with
  an empty/missing buffer, set `learning_starts = cumulative steps + 10_000` to
  prevent updates from starting immediately after restoration with no data.
  Do not modify source files or invent labels from "exactly 1000 steps."
  `collect_seed_transitions` now returns a sixth item, the timeout array; seeding
  retains the terminal observation rather than the next episode's reset observation.

### B. Reward Design

**B1: "Standing still" must yield exactly 0 reward. This is the one invariant the
entire reward system must preserve.**
It took three revisions to fix completely, each exposing a different version of
the same root cause:

- First revision: `base_height` was a `0.5*exp(-err²/0.01)` reward, with
  `track_sigma=0.25`. **Standing with zero action earned +674 per episode for
  free** (height +0.50/step, velocity tracking +0.20/step), while `progress` was
  -0.0001. SAC quickly found this risk-free local optimum. After 1e6 steps,
  average vx was only 1/10 of the command, feet lifted at most 7 cm (insufficient
  for an 8 cm obstacle), yet return exceeded that of actual climbing (790 vs 456).
  Fix: use the deviation penalty `max(0, |err| − 0.05)`, which is exactly 0 at
  the correct standing height.
- Second revision: reducing `track_sigma` to 0.1 only solved half the problem.
  Gaussian velocity tracking at vx=0 is `exp(-cmd²/σ²)`: 0.002 for command 0.8,
  but **0.19/step for command 0.4**. Commands are uniformly sampled from (0.4,
  0.8), leaving the lower half uncovered. Standing with zero action for 1000
  steps still earned **+148.7**. The ordering was completely wrong: climbing
  and getting stuck **-125** < approaching the obstacle and stopping **-1.6**
  < standing still **+148.7**. Fix: **subtract the vx=0 floor**:
  `max(0, exp(-(vx−cmd)²/σ²) − exp(-cmd²/σ²))`. At vx=0 the reward is exactly 0,
  independent of σ and cmd.
- Third revision: `progress` itself was **recurring income**, paid every step
  for more than two hundred steps. Measured component totals over 5 episodes:
  a six-level completion earned +321, including `progress` +258.2, while
  stair-specific income was only 55 (`climb` +5.1, `level_bonus` +30, `success`
  +20). **Climbing the entire staircase paid less than brisk walking on flat
  ground.** Fix: change to the **potential increment** `ΔΦ` (`Φ = x`, within
  the corridor and before `x_cap`), and change the weight from 2.0 to 100.0;
  the units change from reward/step to reward/meter.

**B2: γ in potential-based shaping is a trap.**
- The textbook form `γΦ(s') − Φ(s)` is safe only when Φ is **bounded**. With
  `Φ = x` increasing along the course, discounting becomes **a tax on time spent**.
  Measured at 50 Hz, with Φ≈1.3 and the robot **moving forward**
  (Δx = 0.0008 m/step), reward was still
  `0.995×1.2717 − 1.2725 = −0.0071`, because the leakage term
  `(1−γ)Φ = 0.0064` was an order of magnitude larger. Breaking even requires
  `0.005·Φ` meters per step; at Φ=3.5 this means 0.875 m/s, **above the maximum
  commanded speed**.
- A result such as `progress = −664` is not a separate implementation bug; this
  is the cause. Standing for 1000 steps costs 650 points for no reason.
- **Fix: use `ΔΦ` directly.** Neither `_progress` nor `_climb` multiplies by γ;
  `climb_gamma = 1.0`. The expanded sum telescopes exactly, standing yields 0,
  and moving back and forth yields a net 0, preventing that exploit.

**B3: Corridor weight `w(y)` must multiply the increment, not the accumulated potential.**
- The first version used a hard gate: set Φ directly to 0 when `|y|>1.25`.
  Crossing it at x=3.0 caused a ±300 one-step discontinuity. Measured
  `progress = −376 / −664 / −654`, as though the robot had moved backward
  6 meters in a course only 4 m wide.
- The second version used continuous decay, `1 − (|y|/gate)³`. It removed the
  one-step jump, **not the final accounting problem**. In an episode reaching
  x=3.83 and drifting to |y|=1.42, w became zero and all accumulated potential
  was returned (`progress = −174`), despite a net 2.2 m of forward travel.
  Climbing and then falling scored -211, while standing still scored only -52;
  **standing still became optimal again**.
- Final version: `progress = Δx · w(y)`. Outside the corridor, Δx earns 0
  (still preventing free rewards for bypassing terrain on flat ground), but
  drifting out does not revoke income already earned. Inside the corridor,
  the sum still telescopes.

**B4: Penalties need dead zones and a force-noise allowance; otherwise they punish
the act of climbing.**
Climbing necessarily involves a straddling stance, with front feet above and rear
feet below, leaving the base lower than its target. Sliding a calf against a
stair edge produces sustained contact forces of tens of newtons: that is climbing,
not a collision. Without allowances, getting stuck on an obstacle cost 125 points
per episode, teaching the policy **not to approach stairs at all**. Measured
behavior stopped at x≈0.94, with the front of the body against the obstacle face
at x=1.1, and made no progress for 200k steps.
Fix: a 0.05 m dead zone for `base_height`, and subtract a 25 N noise allowance
from `collision`.

**B5: The progress term must become zero for `|y| > 1.25` and `x > 3.5`.**
Terrain blocks extend only across y∈[−2, 2]; beyond them is **infinite flat ground**.
Without a gate, the policy can go around the terrain and collect free forward
reward as x increases without limit. This is the easiest exploit in this course.

**B6: Demonstration rewards must be relabeled (`POSTURE_TERMS`).**
- Symptom: after the reward changes, the seeded buffer averaged **-0.286
  reward/step** (A=1.0, 8 episodes, 8/8 falls). Components: `yaw_rate −0.181`,
  `yaw −0.162`, `lateral −0.142`, versus `progress +0.156` and `track_lin_vel +0.150`.
- Root cause: `trot_action` **drives only thighs and calves** (`a[1::3]`,
  `a[2::3]`), **not hips**. Lateral motion and heading are physically uncontrolled,
  inevitably causing rotation and drift. Meanwhile, the new reward makes
  standing still worth **exactly 0**. Seeding therefore teaches the critic that
  **small hops are worse than standing**, the opposite of its purpose. The old
  standing reward of +0.19 had hidden this reversal.
- Fix: in `collect_seed_transitions`, zero the posture/energy terms
  (`base_height lateral orientation yaw yaw_rate action_rate torques collision`)
  and retain only task terms. The result is **+0.238 reward/step** (hopping
  segments +0.283 vs standing 0).
- This is still only **an initial starting point** for the critic. Real
  transitions gradually correct it afterward, without affecting correctness.

**B7: Penalties must be observable to the policy: add heading observations and
replace `lateral` (y²) with `lateral_vel` (world-frame vy²).**
- Symptom, measured while investigating `日志.md`: an 800k-step model dragged
  one leg and hopped slowly in 11 of 30 episodes with randomized commands.
  Those episodes ended with `|y| = 2.50` (corridor-boundary termination) and
  `|yaw| = 1~3 rad`.
- First root cause: `projected_gravity` **cannot mathematically reveal yaw**,
  since rotating a gravity vector around the vertical axis does not change it.
  Thus, the 45-dimensional observation provided only the third gyro component
  (yaw rate) for heading. But `yaw = −1.0·(1−cos yaw)` and
  `lateral = −0.3·y²` penalized **absolute quantities** the policy could not
  observe or correct. In 6 flat-ground episodes with rl_800000, per-step
  accounting showed `lateral −0.510` (largest expense) and `yaw −0.093` (second).
  Together they made up 73% of per-step expenses, while the largest income was
  only `track_lin_vel +0.669`.
- Second root cause: lateral drift is **heading misalignment, not sideways
  slipping**. Across 10 manually tuned open-loop trot episodes, world-frame
  `vy²=0.0462`, `|vy|=0.186`, versus body-frame `vy²` of only `0.0022`, a
  **21-fold difference**. Use world-frame lateral velocity
  (`RewardInput.base_v[1]`, the time derivative associated with the old y²
  penalty, directly related to drifting out of the corridor). Copying the
  Isaac convention of penalizing body-frame motion misses this effect. With
  heading observed, the term becomes **inferable**
  (known yaw ⇒ `vy_world ≈ −sin(yaw)·vx`), turning an unobservable tax into a
  controllable penalty.
- Fix: (1) add `(sin yaw, cos yaw)` to observations, from
  `imu_state.quaternion`, at no extra real-robot cost and satisfying invariant
  4 in §4; (2) add `lateral_vel = −5.0·vy_world²` and set the old `lateral`
  weight to 0, preserving its field and function because old `config.json`
  files and `POSTURE_TERMS` refer to its name. Position constraints remain
  in progress's corridor weight `w(y)` (§5-B5), tied to forward movement.
- Weight calibration: manually tuned trot `vy²=0.0462 → −0.217/step`
  (8 measured episodes); clean straight walking ~0.0025 → ~-0.01/step.
  Maximum per-step income is +1.32, so a bad gait pays about 17% of income,
  while a good gait pays almost nothing, with about 20-fold discrimination.
  Overweighting risks low-command policies that crouch in place (vy=0,
  zero cost, zero income); if that happens, reduce the weight to -3.0.
- **Cost**: observations changed from 45 to 47 dimensions and reward terms
  from 14 to 15, **invalidating all old models and replay buffers**. `models/`
  and `runs/` had already been cleared that day, so there was no actual loss.

**B8: `level_bonus` was repeatable rather than once per new level per episode
(diagnosed 2026-10-11; fixed 2026-10-12).**
- Root cause: `_level_bonus` compared current and preceding-step levels, and
  `env.step` updated `prev_terrain_level` to the current level each step.
  Descending and climbing again paid the same level bonus a second time.
- Measurement: `full_s0.56/rl_1750016`, seed 10015, reached at most level 2
  but collected 19 positive level increments, worth +95, in 1000 steps.
  True once-per-new-level accounting would pay at most +10.
- Repair requirement: compare against the episode's highest historical level,
  initialized on reset; repeated ascent/descent is not new progress. This is
  a definite accounting defect, but later models earned less from this term,
  so it cannot be the sole main cause of this regression.
- Fix: retain the field name `prev_terrain_level` but define it as the
  **episode maximum before the current step**, updated with max each step
  and initialized to the spawn level on reset. `1→2→1→2→3` pays only
  `5/5/0/0/5`. The climb term still uses consecutive-step height differences
  and penalizes descent. Isaac's `core.advance` also takes per-environment
  maxima; reset_joints clears level history for resetting environments, and
  post-reset observations initialize their spawn levels. Old buffers' repeated
  rewards are not automatically relabeled.

### C. Simulation / Geometry

**C1: Spawn height must use the maximum terrain height over the robot's whole
footprint, not its body center.**
Use `TerrainHeight.top_footprint`, not `TerrainHeight.top`.
- Root cause: the body is about 0.4 m long. Across a height discontinuity
  (obstacle edge or stair riser), **the center can still be above flat ground
  while front feet already extend over the raised terrain**. Center-based
  spawning puts the front feet and calves **directly inside the terrain**.
- Measurement: at scale=0.62, **17 of 24 spawn positions buried at least one
  foot, up to 0.0915 m deep**. The solver immediately applied huge contact
  forces and flipped the robot within 0.3 seconds. **Penetration depth is
  approximately the height difference at the edge, so higher terrain is worse**:
  only 3 cm at scale 0.35, but 15 cm at 1.0.
- This is a real cause of worsening training at higher curriculum stages,
  **independent of reward, learning rate, and action_scale**.
- Also, feet in the `home` keyframe already penetrate the ground by about
  18 mm. Raising the robot above terrain on reset and letting it settle is correct.

**C2: Call `mj_forward` once after `reset`.**
Otherwise contact and sensor data still belong to the preceding episode.
**The same applies to `geom_xpos`, the four foot positions used for support height.**
It updates only during forward computation/stepping. Read `prev_terrain_h`
**after** `mj_forward`, or it uses the preceding episode's landing location,
potentially a whole level away when spawning near stairs. `_pose()` in
`core_parity` must also run forward computation; otherwise synthetic state
measurements use terrain under an unrelated previous pose.

**C3: Fall criteria need measured clearance margins, not intuition.**
Fully collapsed, the base is about 0.077 m above ground. During stair crossing,
with the body over the next level but feet below, the minimum clearance is about
0.12 m. Therefore `fall_clearance = 0.10`: **values of 0.12 or above falsely
classify normal crossing as a fall**. Use `flip_rad = 1.0`; crossing a 0.15 m
step can legitimately produce pitch of about 0.4 rad, so 0.8 is too restrictive.

**C4: Measure `goal_z` from terrain; do not hardcode it.**
Use `goal_z = terrain.top(goal_x) + goal_clearance`. Hardcoded 1.05 fails when
terrain is scaled: the platform at scale=0.3 is only 0.28 m high, making success
impossible. For terrain without stairs (`flat`/`steps`), `goal_z` must be `inf`;
otherwise reaching x≥3.3 on `flat` counts as success and ends the episode early,
reducing measured flat-ground return from 715 to 281.

**C5: Measure foot lift only after skipping the initial landing phase.**
Reset places the robot 0.30~0.33 m above terrain so it can settle. **All four
feet are airborne while it falls.** Counting that instant yields "foot lift"
up to 0.15 m, twice the actual walking value, falsely suggesting sufficient
foot clearance. This is why `diag lift` has a `skip` parameter.

**C6: "Local terrain height" must come from the four feet (`support_height`),
not `top()` at the body center.**
C1 concerns spawning; this concerns the height used by rewards and termination.
Both arise from terrain discontinuities at edges. `climb` / `level_bonus` /
`base_height` / `is_fallen` **all** consume this value: one wrong definition
breaks all four.
- Root cause: when the body center crosses a stair riser, the robot **can still
  have every foot on flat ground**, but `top()` already returns the next level.
- Measurement for the s0.86 model, stair height 0.1462 m: `climb` awarded
  0.1462×100 = **+14.6** and `level_bonus` **+5.0** with only one foot on the
  stair and roll already -0.66 as the robot tipped over. Of 10 evaluation
  episodes, 7 ended at 25~59 steps with roll≈-1.0. **Extending the head across
  the stair and then falling was optimal under that reward.**
- This also explains maximum x stuck at 2.1 regardless of curriculum height:
  2.1 is a geometric location, the first stair riser, not a difficulty threshold.
  Actually moving the front feet onto levels 1/2 at a larger x earned
  **exactly the same reward**, leaving no gradient incentive to bring up the
  rear legs and complete the motion.
- Definition: use the **median of the terrain heights under the four feet**.
  Stair tread depth, 0.19~0.28 m, is shorter than the 0.4 m footprint, so two
  legs are on one level and two on the adjacent level. The median represents
  **the level supporting the robot**: flat=0, two front legs on the first
  level=half a level, spanning two levels=the lower level, reaching the top
  platform=its top height. Min requires all four feet to ascend, impossible
  to fit on a tread and too sparse as a signal. Max lets one airborne foot
  above a stair collect the full reward, reproducing the exploit.
- **Do not use `torch.median` on the torch side.** For an even sample count it
  returns the lower middle value, whereas `np.median` averages the two middle
  values. With feet spanning adjacent levels, this differs by half a step,
  exactly the most common case, explicitly covered in `core_parity`.
- Related changes: `is_fallen` takes support height + position instead of
  terrain object + position. For highest-level diagnostics, use
  `env.support_level()`. `terrain.level(x, y)` reports one level too high,
  which caused the old inflated measurements.
- **Isaac has an additional integration pitfall**: `FL_foot` is an **empty
  body** exactly at the foot-sphere center (`0 0 -0.213` in the calf frame).
  USD conversion may merge it into `*_calf`, causing `body_pos_w` to point
  to the knee and shifting sampling upward by 0.21 m without an error.
  `smoke.py` checks this: standing foot-sphere center height is approximately
  0.022, while knee height is approximately 0.235.

### D. Evaluation / Model Selection

**D1: `model_of` must select the best model at each stage, not the final model.**
Once difficulty reaches the policy's limit, training starts pulling it toward
avoiding risk: a fall terminates the episode and forfeits the remaining nine
tenths of return. Thus **the final checkpoint is precisely the worst one**.
Measured `st_s056`: `model.zip` return 111 / success 1/5, versus `best_model.zip`
return 269 / success 4/5. Selecting the final model feeds a regressed policy into
each subsequent stage, compounding the problem.
Implication: if a stage never improves on its starting point, `best_model`
**inherits the preceding stage's model unchanged**. Compare best models across
stages: measured `st2_s056` and `st2_s062` had bit-for-bit identical `|mu|` in
every dimension. They were the same model; **the entire curriculum had been
making no progress since scale 0.56**.

**D2: Use enough `eval_episodes` (default 20), or `best_model` selects noise.**
With 5 episodes, the success-rate standard error is **±22 percentage points**.
Since `best_model` takes **the maximum across evaluations**, it inevitably
selects a lucky sample. Measured results for the same model:

| Evaluation episodes | Reported success rate |
|---|---|
| 5 (old default) | 4/5 = **80%** |
| 20 | 8/20 = 40% |
| 40 | 12/40 = **30%** |

This directly creates the illusion that every curriculum stage makes the
model worse: comparing a lucky old evaluation with a new evaluation will
inevitably show a decline. **When training appears to degrade the model,
check the number of evaluation episodes first.**

**D3: Behavior is bimodal; count completions rather than averaging final levels.**
The policy either completes all 6 levels or falls very early. For the same model
over 20 episodes: deterministic 9/20, stochastic 7/20. Averaging final levels
obscures this bimodal distribution.

**D4: `eval/vx_ratio` can silently become NaN when the policy collapses.**
`np.mean([])` returns NaN. If all 5 episodes end within 100 steps, the velocity
list is empty. This looks like simulation divergence but actually means the
policy falls immediately. `train.py` now falls back to whole-episode averages
for empty lists and adds a warning marker.

**D5: Evaluate `eval/vx_ratio` (actual speed / commanded speed), not just return.**
Return is uninformative when it includes guaranteed income: a stationary policy
once scored 771, more than an actual obstacle-climbing policy's 456.

**Understand how the metric is calculated.** `EvalMetricsCallback` uses only
steps with `step_count > 100`, intended to skip the first 2 s of acceleration.
For short episodes, this leaves only the **tail**, precisely the period of
standing still in front of stairs. A `vx_ratio` of 0.1× then looks like an
inability to walk, although the robot actually traversed the first 1.8 m
(`mean_max_x` increased from spawn x=0.2 to 2.0). To determine whether it got
moving, examine **both** `mean_max_x` and episode length. `vx_ratio` represents
steady-state speed only when episodes approach 1000 steps, the full 20 s.

**D6: `eval/max_level` measured the maximum final level across episodes, not
the trajectory peak (diagnosed 2026-10-11; fixed 2026-10-12).**
- Root cause: `EvalMetricsCallback` read `info["level"]` only after `run_episode`
  returned, tracking only maximum x along the way. Climbing and subsequently
  falling erased progress from the logs.
- Measurement: `full_s0.56/rl_1750016`, seeds 10000–10019, mean trajectory
  peak 1.25 versus mean final level only 0.35.
- Repair requirement: accumulate the peak using `support_level()` every step
  and log final levels separately. The peak uses the existing support-surface
  criterion; it does not prove stable standing and cannot replace completion rate.
- Fix: accumulate trajectory peaks during evaluation. Environment
  `info["max_level"]` includes the spawn level. Existing `eval/mean_level` /
  `eval/max_level` now report trajectory statistics; final levels are stored
  as `eval/mean_final_level` / `eval/max_final_level`. Old logs are not
  backfilled. This change does not alter success criteria or reward-only selection.
- Regression validation: all 17 tests in
  `python -m unittest go2_common.tests.test_training_fixes -v` pass, covering
  seeded/online timeouts, rejection of legacy buffers, fresh-data collection
  after reset, checkpoint save/resume, repeated ascent/descent, and spawn levels.
  `go2_issac.tests.course_parity` and `go2_issac.tests.core_parity` also pass;
  maximum reward parity difference is 7.105e-15. These validate CPU semantics
  and the reset adapter, not Isaac Sim execution or long-training comparisons.

**D7: Reward-only best-model selection can favor approaching stairs quickly and
then falling; D1 does not guarantee preservation of climbing ability.**
- Same-seed backtest on 2026-10-11: fixed
  `full scale=0.56 / action_scale=0.7 / reset_x=(-0.3,0.75)`, with 20 training
  evaluation seeds and 20 held-out seeds. Early `rl_1750016`: return 89.25,
  mean length 789 steps, reached ≥2 levels in 10/40. `full_s0.56/best`
  (2450024 steps): return 228.73, 289 steps, 4/40. `full_s1.00/best`
  (3200072 steps), backtested on low stairs: return 240.85, 369 steps, 0/40.
  All three completed the course in 0/40.
- Reward accounting: early positive income +403, penalties -314. The 0.56
  best's positive income fell to +324, but penalties shrank to -96. Climb
  income changed only from +6.04 to +7.43; higher return mostly came from
  reducing ongoing penalties, not completing more stairs.
- Repair direction: prioritize task success, then reliable climbing progress,
  then return. Add lower-difficulty regression evaluations and curriculum
  advancement thresholds. **The historical §7 conclusion that reward ordering
  was correct applies only to the policies tested then, not every reward version.**
  No training ablation was performed in this diagnosis; weight changes still
  require single-variable comparisons. Raw data and full analysis are in
  `logs/full_diagnosis_20261011/REPORT.md` (the logs directory is gitignored).

### E. Tools / Operations (Agent-Side Precautions)

**E1: `pkill -f 'go2_sac.train'` can kill its own command.**
`pkill -f` matches the **entire command line**, and the bash command line itself
contains that literal string. Use `pkill -f 'go2_sac[.]train'`: the brackets
prevent the pattern from matching itself. **Run it as a separate command**;
do not put it in the same `bash -c` as commands containing that literal string.

**E2: The Bash tool's default timeout is 120 s.** Commands such as `sleep 540`
require an explicit `timeout` (milliseconds, maximum 600000) or `run_in_background`.

**E3: Do not delete shared scratch files, such as those in `/tmp`, using wildcards.**
Commands such as `rm -rf /tmp/lrtest_*` are blocked by the permission system
("Shared Scratch Sweep"). Use **new, explicitly named directories** and delete
them individually when cleanup is needed.

**E4: Pitfalls in SB3's internal APIs when writing diagnostic scripts.**
- `m.policy.actor.get_action_dist_params(obs)` returns **3 values** in SB3 2.9,
  not 2 or 4.
- **Do not compute `mu/log_std` inside `torch.no_grad()` and then differentiate
  them.** The tensors lack `grad_fn`, producing
  `element 0 of tensors does not require grad`.
- Backpropagating twice through the same parameter computation raises
  `Trying to backward through the graph a second time`. Use `retain_graph=True`
  or repeat the forward pass each time.
- `m.policy.actor.log_std` **is a network layer, not a tensor**. `.detach()`
  raises `'Linear' object has no attribute 'detach'`. Read σ through
  `get_action_dist_params`.
- `m.policy.actor(obs)` **returns actions directly**, not a tuple; `[0]` selects
  the first row.
- `float(m.ent_coef)` raises `could not convert string to float: 'auto'`.
  Use `float(m.log_ent_coef.exp())`.
- Read TensorBoard `ScalarEvent` objects with `for e in scalars: e.value`.
  `for _, v in scalars` raises `cannot unpack non-iterable ScalarEvent`.

**E5: `glfwTerminate()` segfaults when `play.py` exits.**
On this machine's Wayland setup, glfw cleanup consistently crashes; **this is
unrelated to project code**. `play.py` works around it by not calling
`glfwTerminate` after viewer playback, explicitly exiting with a normal status.
Do not investigate that exit-time segfault as a simulation or policy defect.

**E6: The `ctrl/go2_ctrl.py` constructor reads `sys.argv[1]` as the network interface.**
The original `ChannelFactoryInitialize(0, sys.argv[1])` took the first CLI
argument directly. Importing it from `play.py --model foo.zip` therefore called
`ChannelFactoryInitialize(0, "foo.zip")`. The constructor now accepts optional
`domain_id` / `interface` arguments, with **explicit values taking precedence
over `sys.argv`**; omitted arguments preserve the old behavior. Whenever another
script imports `Go2Ctrl`, rather than directly running
`python3 ctrl_test.py <network-interface>`, pass both arguments explicitly.
Do not rely on `sys.argv`.

### F. PPO / On-Policy Specific (Not Present in the SAC Implementation)

**F1: Never call `model._setup_model()` to change `n_steps` / `batch_size`.**
- Symptom: it **rebuilds the policy network**.
  `OnPolicyAlgorithm._setup_model()` unconditionally executes
  `self.policy = self.policy_class(...)`, replacing weights with random initialization.
- Root cause: SAC is off-policy and `_setup_model` manages networks and the
  buffer; PPO reuses that function, which also constructs the policy network.
- Fix in the resume branch of `go2_ppo/train.py`: **rebuild only the rollout
  buffer**, using the corresponding `_setup_model` construction:
  `rollout_buffer_class(n_steps, obs_space, act_space, device, gamma, gae_lambda, n_envs)`.
  Do this after setting gamma/gae_lambda.
- Why safe: `rollout_buffer` is in `_excluded_save_params` and **is not saved
  in checkpoints**. `load()` already rebuilds it, so reconstruction loses nothing.
- Measurement: before/after resume with `--n-steps 256→128 --batch-size 256→64`,
  the `policy.state_dict()` sha256 was **identical** (`ac4f1709…`), while the
  buffer changed to `(128, 2, 45)`.

**F2: `clip_range` and `learning_rate` have different forms after `load()`.**
- `clip_range` is a `FloatSchedule`, unconditionally wrapped by
  `PPO._setup_model`; `PPO.train()` calls
  `self.clip_range(self._current_progress_remaining)`. Directly assigning
  `setattr(model, "clip_range", 0.2)` causes the first update to raise
  **`TypeError: 'float' object is not callable`**.
- Conversely, `BaseAlgorithm.__init__` merely stores `learning_rate`; after
  loading it **is a plain float**. `model.learning_rate(1.0)` therefore also
  raises `'float' object is not callable`. **This was encountered in practice**
  when code incorrectly assumed both were schedules. `model.lr_schedule`
  must still be replaced too, as in A1.
- Fix: read either form with
  `_lr_or_clip_value(v) = float(v(1.0)) if callable(v) else float(v)`, and
  rewrap `clip_range` with `FloatSchedule(...)`.

**F3: On-policy `log_interval` counts training iterations, not episodes.**
SAC's `log_interval=4` logs after every 4 finished episodes. PPO's default `1`
logs **after every collected rollout (`n_steps × n_envs` steps)**. Do not copy
these values between algorithms.

**F4: `train/*` curves lag `rollout/*` by one logging point; a single iteration
produces no `train/*` data.**
- Root cause: `OnPolicyAlgorithm.learn()` calls `dump_logs()` **before**
  `self.train()`. `logger.dump()` performs the actual TensorBoard write.
  Therefore, iteration N's `train/approx_kl` and other metrics are written
  only at iteration N+1's dump, with iteration N+1's step count.
- Measurement: 512 steps (1 iteration) produced only 11
  `rollout/* / eval/* / time/*` tags, **no `train/*` tags**. After 1024 steps
  (2 iterations), they appeared, all at the single point `step=1024`.
- Implication: **when `train/*` data is missing, first check whether the run
  was too short; do not assume TensorBoard is broken.** The last iteration's
  `train/*` values are not written either.

**F5: Do not copy SB3's default PPO `n_steps=2048`.**
At a 50 Hz policy rate, an episode is limited to 20 s = 1000 steps; 2048 steps
is **41 s**, longer than an entire episode. Stale data from the same trajectories
is repeatedly used for advantage estimation. The default was changed to **256**,
or 5.12 s per environment. Choose `batch_size=256` as a divisor of
`n_steps × n_envs` to avoid partial minibatches when changing `--n-envs`.

---

### G. Isaac Sim Migration (`go2_issac/`)

**G0: The development machine cannot run Isaac Sim; this is hardware, not configuration.**
`nvidia-smi` is unavailable. The machine has only an AMD HawkPoint1 integrated
GPU, 27 GB RAM (requires ≥32), 21 GB free on `/` (needs ~50), and Python 3.10.12
(requires 3.11). **Isaac Sim has no CPU fallback.** `--cpu` only runs tensors on
CPU; physics and rendering still require a GPU. Do not try to install it here.
Thus `go2_issac/` deliberately has two layers: semantics (provable locally) and
runtime integration (target machine only). This is the only way to reduce risk
without a GPU. **Do not move the semantic layer into `mdp/` for "consistency"**;
that would make it depend on isaaclab and prevent local testing.

**G1: Gains are not in USD; the MJCF converter does not handle them.**
`default_drive_stiffness` is a **URDF** converter field, absent from the MJCF
converter. `<motor>` in `go2.xml` becomes only a USD DriveAPI; `ImplicitActuatorCfg`
must write kp/kd at instantiation. Set `damping` to **3.6, not kd=3.5**:
MuJoCo's 12 hinge joints each have `damping=0.1`, while Isaac's implicit actuator
has only one damping parameter, so add them. Reference:
`joints[].damping` in `assets/go2_reference.json`, measured as 0.1.

**G2: There are three joint orders; identify joints by name, not index.**
MJCF `qpos` follows the body tree (FL,FR,RL,RR), actuators/DDS use FR,FL,RR,RL,
and USD uses import order. `JOINT_NAMES` in `mdp/state.py` explicitly specifies
DDS order, then resolves indices by name and asserts all 12 names match on startup.
**Reorder actions by name too** (`raw_action`): action terms use `find_joints`
(USD order), which may differ from the DDS order used by `State.q/dq/tau`.
Getting this wrong produces no error, just failure to learn. Likewise, do not
depend on `preserve_order`, which older versions do not have.

**G3: `prev_action` must contain the clipped action.**
`env.py:194` does `action = clip(action); self.prev_action = action`: clip first,
then store. Isaac's `ActionTerm.raw_actions` is the raw network output; clipping
only changes `processed_actions`. Therefore **clip again** in `state.raw_action()`.
This is not a corner case: initial policy σ=1.0 means about 1/3 of actions have
`|a|>1`. Without clipping, the previous-action observation systematically differs
from MuJoCo.

**G4: Terrain coordinate offsets must be calibrated on the target machine and
affect only one constant.**
Isaac Lab writes the patch **center** into `env.scene.env_origins`, but the
meaning of "center" has changed between versions. Rather than guessing, infer
the mapping with `smoke.py --check-terrain`; if it disagrees, change only
`course.py:COURSE_ORIGIN_FROM_ENV_ORIGIN`. Also check heightfield
`horizontal_scale`/`vertical_scale`: at scale 0.7, stair heights are exact
multiples of `vertical_scale`, with quantization error ≤0.5 mm.

**G5: Training cannot be resumed across simulators because their physics differ.**
MuJoCo uses an `elliptic` friction cone + `impratio=100`, foot `condim=6`
(torsional/rolling friction), and `priority=1` controlling the contact pair.
PhysX uses isotropic Coulomb friction and has no priority.
**These are different physics**, not two implementations of the same physics.
Thus `go2_ppo` `.zip` files and Isaac `.pt` files are not interchangeable;
`train.py --resume` explicitly rejects `.zip`. Compare curve trends, not raw values.
`go2_common/` still has only one implementation: the simulator changes, not
the task definition.

**G6: rsl_rl hyperparameter names differ from SB3; the mapping table is the sole reference.**
See the module comment in `agents/rsl_rl_ppo_cfg.py`. Easily confused but
**semantically** equivalent mappings:
`gae_lambda`→`lam`, `ent_coef`→`entropy_coef`, `batch_size`→`num_mini_batches`
(sample count to partition count), `log_std_init=0` ⇔ `init_noise_std=1.0`,
and `target_kl` early stopping ⇔ `schedule="adaptive"+desired_kl`.
Also, rsl_rl's `use_clipped_value_loss` **defaults to True**, while this project
does not clip the value loss. Explicitly disable it: opposite defaults are
particularly prone to silent mismatch.

**G7: Isaac represents timeout truncation with `time_out=True`.**
The termination term must be **named** `time_out` and have `time_out=True`,
so rsl_rl puts it in `extras["time_outs"]` for bootstrapping. A normal
termination cuts off value at timeouts, opposite to the `truncated` semantics
in `env.py` (invariant 8 in §4).

---

## 6. Measured Results (How Difficult a Task the Robot Can Handle)

### 6.1 `action_scale` Is a Geometric Constraint, Not Just a Tuning Parameter

Joint target = default stance + `action_scale` × action. This directly determines
the height of obstacles the robot can cross.

**Do not estimate this by fixing the body and sweeping the joint workspace.**
That measures how high a foot can be posed while stationary, not whether the
robot can walk into an obstacle and push itself up through friction. It
**seriously overestimates capability**: at 0.40, static reach is 15.8 cm,
apparently ample for an 8 cm obstacle, but the moving robot cannot get onto it.

The only credible measurement is a manually tuned open-loop hop, without
feedback (`train.trot_action`), directly attempting obstacles (`diag frontier`):

| action_scale | 0.06 m | 0.08 m | 0.10 m | 0.12 m |
|---|---|---|---|---|
| 0.40 | Stuck | Stuck | Stuck | Stuck |
| 0.60 | Crossed | Stuck | Stuck | Stuck |
| 0.80 | Crossed | Crossed | Crossed | Stuck |

A feedback policy performs about one height increment better than this open-loop
trot: at 0.40, open loop crosses only 0.04, while a flat-trained policy crosses
0.06. The scene's two obstacles are 0.08 m, so **A=0.4 cannot cross them**.
The policy stops at x≈0.94 and makes no progress for 200k steps. **This is not
a reward or exploration problem**, although reward functions were initially
blamed.

`diag lift` directly measures foot clearance. The foot is a sphere of radius
0.022; the table gives **the foot's bottom surface** above ground. Rows are
action_scale; columns are action amplitude:

| | Amplitude 0.5 | 0.7 | 1.0 (near saturation) |
|---|---|---|---|
| A=0.6 front feet | 0.053 | 0.086 | 0.098~0.139 |
| A=0.6 rear feet | 0.034 | 0.065 | 0.104~0.128 |
| **A=0.8 front feet** | 0.077 | 0.103~0.124 | Open loop cannot stay stable |
| **A=0.8 rear feet** | 0.059 | **0.100~0.113** | Open loop cannot stay stable |

Key points: (1) at A=0.6, rear feet need near-saturated actions to cross 0.08;
at amplitude 0.7 their bottom surfaces reach only 0.065, short by 1.5 cm.
(2) Maximum amplitude makes the **open-loop** trot fall, but that does not
mean a closed-loop policy cannot use it.

### 6.2 The Trained Policy's Real Limit Is Approximately 0.10~0.11 m

Sweep terrain_scale with an A=0.7-trained policy (`st2_s056_full/best`, 20 episodes):

| terrain_scale | First stair height | Successes/20 | Highest level |
|---|---|---|---|
| 0.56 | 0.095 m | 8/20 (40%) | 6/6 |
| 0.60 | 0.102 m | 7/20 | 6/6 |
| 0.62 | 0.105 m | 5/20 | 6/6 |
| 0.65 | 0.111 m | 2/20 | 6/6 |
| 0.70 | 0.119 m | 1/20 | 6/6 |
| 0.80 | 0.136 m | 0/20 | 4/6 |
| 1.00 | 0.170 m | 0/20 | 1/6 |

**Degradation is gradual, not a cliff**, so declaring a stage a success or
failure based on 5 evaluation episodes is completely unreliable (see D2).

Sweep action_scale with the same policy, without retraining:

| action_scale | Terrain 0.56 successes/20 | Terrain 1.00 successes/20 |
|---|---|---|
| 0.5 | 0/20 (highest 1/6) | Not measured |
| **0.7 (training value)** | **8/20** | 0/20 (1/6) |
| 0.9 | 4/20 | 0/20 (1/6) |
| 1.1 | 1/20 | 0/20 (1/6) |
| 1.3 | 0/20 (3/6) | 0/20 (1/6) |

Conclusion: **the policy is tightly bound to its training action_scale**.
Increasing or decreasing it worsens performance. On full-height terrain,
every amplitude gives 0/20, reaches only 1/6 levels, and stops at x≈2.2.
**Amplitude changes cannot raise this limit; a higher limit requires training
a new gait.**

### 6.3 Terrain-Height Curriculum Stages

The `run_stairs.sh` progression (`--terrain-scale` → first stair height):

```text
0.56 (0.095) → 0.62 (0.105) → 0.70 (0.119) → 0.78 (0.133)
→ 0.86 (0.146) → 0.93 (0.158) → 1.00 (0.170)
```

Scaling affects only geometry height (`size[2]`/`pos[2]` multiplied together),
not the x/y footprint. Thus **tread depth is unchanged; the stairs simply
become lower**, exactly the intended simplification.

### 6.4 Demonstration Seeding Really Works

SAC with a feed-forward MLP and no phase input must discover a periodic gait
from independent white noise in each action dimension. Automatic entropy
adaptation compresses noise to ±0.06 rad, while a walking diagonal trot needs
**coherent oscillations of ±0.25 rad**. The buffer then contains only standing
transitions, Q learns that standing is best, and the actor never steps.
Approach: collect 30k transitions with `trot_action` and seed the buffer
(SACfD), **giving Q only a starting point**. After 200k steps, measured average
vx improved from **0.00 to 0.79 times the commanded speed**.

Use `log_std_init=0.0` / `target_entropy=-6.0`, not SB3's -1.0 / -12.
With -1.0, α quickly falls to 0.024 and σ is only 0.25 after 200k steps.

---

## 7. Current Status and Open Questions

**Completed**: flat ground; two 0.08 m obstacles; six stair levels at 0.095 m/level.

**PPO path** (`go2_ppo/`): implemented, all static checks pass, with one
512-step smoke run (collect one rollout, perform one update, evaluation callback
works, TensorBoard records values), **but no meaningful training**. Therefore:
- All conclusions above about climbing limits and hyperparameter tuning
  **come from SAC**, not PPO.
- PPO has no demonstration seeding and starts from a random policy with
  σ=1.0. The smoke run measured -20.8 episode return and almost immediate
  falls in place. There is **no evidence** yet that it can learn walking
  from scratch like SAC; start with flat terrain if trying it.

**Isaac Sim path** (`go2_issac/`): **never executed**, because this machine
has no NVIDIA GPU; hardware prevents it (see §5-G0). Separate guarantees
from untested behavior explicitly:

- **Validated against real MuJoCo**, locally runnable and all passing:
  `course.py` terrain geometry; `core.py` 47/48-dimensional observations,
  15 reward terms, termination and success criteria. Element-wise differences
  are around 1e-15, due to floating-point summation order. Asset reference
  values (mass 15.206408 kg, different front/rear joint limits, `damping=0.1`,
  foot-sphere `priority=1`/`friction=0.4`) were exported directly from the
  compiled `MjModel`.
- **Entirely untested**: PhysX itself, Isaac API integration, and training
  convergence. These require the target machine. `smoke.py --all` is
  designed for them, with five sections and a conclusion per section.
- Therefore, **if all `smoke.py` checks pass on the target machine but
  training does not converge, the question is not incorrectly written code,
  but whether these contact dynamics require retuning rewards/hyperparameters**.
  This is the same kind of problem encountered when first starting from
  scratch with MuJoCo.

**Not completed**: **full-height 0.15~0.17 m stairs**. The existing A=0.7 gait
cannot reach them at any amplitude (see 6.2).

**A=1.0 large-amplitude retraining from scratch: completed, and it failed.**
Run: `models/amp1_s035`, terrain_scale 0.35, 2M steps, `gradient_steps=1`,
finished during 21:03–22:26.

- Training itself worked: from 750k, `max_level` remained 6/6; success rate
  ranged from 55%~90% between 1.3–1.75M steps.
- But **the final 250k steps diverged**: success fluctuated between
  0.90 / 0.20 / 0.90 / 0.10 (mean_reward 325 → 19 → 300 → 34).
  Final `model.zip` achieved only **4/20**, whereas `best_model.zip`,
  saved at 1.0M, achieved **16/20**. **Another D1 case: the final checkpoint
  was the worst, by a factor of 4.**
- **Across terrain_scale, A=1.0 lost to A=0.7 at every setting**, using
  the same script and the same 20 seeds:

  | scale | First stair height | A=0.7 (trained at 0.56) | A=1.0 (trained at 0.35) |
  |---|---|---|---|
  | 0.35 | 0.059 | 14/20 | 16/20 |
  | 0.45 | 0.077 | **12/20** | 8/20 |
  | 0.56 | 0.095 | **10/20** | 2/20 |
  | 0.62 | 0.105 | **4/20** (6/6) | 0/20 (2/6) |
  | 0.70 | 0.119 | **1/20** (6/6) | 0/20 (1/6) |
  | 1.00 | 0.170 | 0/20 (1/6) | 0/20 (1/6) |

  A=0.7 had never trained at scale 0.35, only at the higher 0.56, yet still
  achieved 14/20 there. The extra 2M steps for A=1.0 bought nothing.
  Leg swings were larger (0.23 vs 0.17), but **higher swings do not mean
  better climbing**.

**Conclusion: changing `action_scale` does not raise the limit, and neither
does retraining a larger-amplitude gait.** Section 6.2 shows the policy is
tied to its training action_scale. This is a stronger result: **even training
a new larger-amplitude policy from scratch leaves the limit unchanged**.
Do not spend more compute on this direction.

**Next step, an unverified hypothesis**: the A=0.7 curriculum was trained
with the spawn-penetration bug. `st2_s056_full/best_model.zip` was saved at
19:20, while the `top_footprint` fix entered `env.py` only at 20:10.
Penetration depth scales with terrain height, from 3 cm at 0.35 to 15 cm
at 1.0, **worst precisely from the 0.62 scale where progress stalls**.
Despite this bug, it reached 10/20@0.56. Rerunning the progression with
the fixed environment is currently the best-supported next attempt.

**Falsified hypotheses; do not repeat them** (all measured):
- ~~"Reward ordering is reversed."~~ With the same seeds and weights:
  successful policy +140.5/episode, mid-course fall +53.3, worse policy +1.4.
  The ordering was correct.
- ~~"The critic becomes pessimistic, causing actor collapse."~~ Reducing
  `target_entropy` from -6 to -12 changed α from 0.102 to 0.038 and σ
  from 0.411 to 0.252; Q stayed at -5.91, starting from -4.92.
  **The policy still collapsed to 0/20 in 20k steps.** Behavior failed
  with a healthy Q, so the proposed causal chain does not hold.
- ~~"The critic has no slope near policy actions."~~ Measured
  `|dQ/da| ≈ 5.1`; a small step along the gradient raised Q by +0.220.
  Also, `|g_Q| = 3.47 > |g_ent| = 2.49`, so the Q term dominated
  the actor gradient.
- ~~"Fine-tuning inevitably destroys the policy."~~ The 0.62-stage policy
  did climb successfully: 6/6 levels, 25% success rate. The earlier failure
  conclusion was an artifact of 5-episode evaluation.
- ~~"Increasing amplitude raises the climbing limit."~~ See the table:
  A=1.0 trained from scratch for 2M steps and performed worse than A=0.7
  at every setting.

**Still unexplained**: fine-tuning at scale 0.56 takes a policy from 40%
success to **0/20 in just 20k steps**. This is **reproducible** at UTD=1/8
and UTD=1, with old and new rewards, and with `target_entropy` -6 and -12.
The only persistent common factor in `sac` was "warm start + 500k stale
buffer transitions + changed reward function."

---

## 8. Working Discipline in This Repository

1. **Before changing any rewards or hyperparameters, read the relevant
   category in §5.** Most losses here come from category A, silent failures.
2. **Before claiming an improvement, compare two configurations on the same
   seeds over 20 episodes each.** Differences from 5 episodes are noise (D2).
3. **Change only one variable at a time**, and record comparison numbers
   in comments.
4. **Do not relax success criteria or shorten evaluation to improve metrics**
   (coding rule 7 in §3).
5. **Run long training in the background** (`nohup ... &` + `run_in_background`),
   but **never run two training jobs concurrently**: on 16 cores, two jobs
   each using 8 environments + 8 torch threads interfere severely.
6. **After training, inspect `best/best_model.zip` first, not `model.zip`** (D1).
7. **Report failures with numbers.** Comments and README contain raw results
   such as measured -376 / -664 / -654; these are the only reliable way
   to avoid repeating the same mistakes.
