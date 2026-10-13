"""Run with: python -m unittest go2_common.tests.test_training_fixes"""

from dataclasses import replace
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from go2_common.config import EnvCfg, RewardCfg, obs_dim
from go2_sac.config import TrainCfg
from go2_common.env import Go2TerrainEnv
from go2_common.reward import RewardInput
from go2_common.reward import _climb as _climb_impl
from go2_common.reward import _level_income as _level_income_impl
from go2_common.train_utils import (
    EvalMetricsCallback,
    checkpoint_obs_dim,
    make_env,
    pad_replay_buffer,
    task_key,
    warm_start_scan,
    worker_scale,
)
from go2_issac import core
from go2_issac.tests.core_parity import synthetic_state, to_core_state
from go2_sac.train import (
    collect_seed_transitions,
    main,
    parse_args,
    prepare_resume_buffer,
    seed_replay_buffer,
)
from go2_sac import train as sac_train


class TimeoutTests(unittest.TestCase):
    def test_environment_timeout_is_not_termination(self):
        env = Go2TerrainEnv(EnvCfg(terrain="flat", max_episode_s=0.02))
        try:
            env.reset(seed=0)
            _, _, terminated, truncated, _ = env.step(np.zeros(12))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
        finally:
            env.close()

    def test_seed_collection_keeps_timeout_and_terminal_labels(self):
        class SeedEnv:
            max_steps = 2

            def __init__(self, *args):
                self.closed = False

            def reset(self, seed):
                self.episode = seed
                self.step_count = 0
                return np.zeros(47), {}

            def step(self, action):
                self.step_count += 1
                terminated = self.episode == 1
                truncated = self.step_count == 2 or terminated
                return np.full(47, self.step_count), 1.0, terminated, truncated, {}

            def close(self):
                self.closed = True

        env = SeedEnv()
        with patch("go2_sac.train.Go2TerrainEnv", return_value=env):
            obs, next_obs, act, rew, done, timeouts = collect_seed_transitions(
                EnvCfg(), RewardCfg(), n_trot=0, n_rand=2
            )
        np.testing.assert_array_equal(done, [0, 1, 1])
        np.testing.assert_array_equal(timeouts, [False, True, False])
        np.testing.assert_array_equal(next_obs[:, 0], [1, 2, 1])
        self.assertEqual(obs.shape, (3, 47))
        self.assertEqual(act.shape, (3, 12))
        self.assertEqual(len(rew), 3)
        self.assertTrue(env.closed)

    def test_seed_buffer_masks_timeouts_but_not_real_terminations(self):
        obs_space = spaces.Box(-10, 10, (47,), dtype=np.float32)
        act_space = spaces.Box(-1, 1, (12,), dtype=np.float32)
        model = SimpleNamespace(buffer_size=8, device="cpu")
        vec_env = SimpleNamespace(num_envs=2, observation_space=obs_space, action_space=act_space)
        transitions = (
            np.zeros((4, 47), dtype=np.float32),
            np.arange(4, dtype=np.float32)[:, None] * np.ones((4, 47), dtype=np.float32),
            np.zeros((4, 12), dtype=np.float32),
            np.ones(4, dtype=np.float32),
            np.array([0, 1, 1, 0], dtype=np.float32),
            np.array([False, True, False, False]),
        )
        with patch("go2_sac.train.collect_seed_transitions", return_value=transitions):
            n, reward = seed_replay_buffer(model, vec_env, EnvCfg(), RewardCfg(), 1, 0)
        rb = model.replay_buffer
        self.assertEqual((n, reward), (4, 1.0))
        self.assertTrue(rb.handle_timeout_termination)
        np.testing.assert_array_equal(rb.dones[:2], [[0, 1], [1, 0]])
        np.testing.assert_array_equal(rb.timeouts[:2], [[0, 1], [0, 0]])
        with patch("numpy.random.randint", return_value=np.array([1, 0])):
            batch = rb._get_samples(np.array([0, 1]))
        np.testing.assert_array_equal(batch.dones.numpy().ravel(), [0, 1])
        np.testing.assert_array_equal(batch.next_observations.numpy()[:, 0], [1, 2])
        targets = batch.rewards + 0.995 * (1 - batch.dones) * 10
        np.testing.assert_allclose(targets.numpy().ravel(), [10.95, 1], rtol=1e-6)

    def make_model(self, cfg=None):
        env = Go2TerrainEnv(cfg or EnvCfg(terrain="flat"))
        self.addCleanup(env.close)
        model = SAC("MlpPolicy", env, buffer_size=8, learning_starts=10,
                    policy_kwargs={"net_arch": [8, 8]}, device="cpu")
        model.num_timesteps = 500
        return model

    def test_online_rollouts_keep_timeout_handling_after_real_seeding(self):
        cfg = EnvCfg(terrain="flat", max_episode_s=0.02)
        model = self.make_model(cfg)
        model.num_timesteps = 0
        seed_replay_buffer(model, model.get_env(), cfg, RewardCfg(), 0, 2)
        model.learn(total_timesteps=4)
        rb = model.replay_buffer
        self.assertEqual(rb.size(), 6)
        self.assertTrue(rb.handle_timeout_termination)
        np.testing.assert_array_equal(rb.dones[:6, 0], np.ones(6))
        np.testing.assert_array_equal(rb.timeouts[:6, 0], np.ones(6))

    def fill_buffer(self, rb):
        rb.add(np.zeros((1, 47)), np.ones((1, 47)), np.zeros((1, 12)),
               np.array([1]), np.array([1]), [{"TimeLimit.truncated": True}])

    def test_legacy_buffer_is_rejected_without_mutating_source(self):
        model = self.make_model()
        model.replay_buffer.handle_timeout_termination = False
        self.fill_buffer(model.replay_buffer)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pkl"
            model.save_replay_buffer(path)
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "--reset-replay-buffer"):
                prepare_resume_buffer(model, path, False, 10)
            self.assertEqual(path.read_bytes(), original)

    def test_explicit_reset_preserves_weights_and_waits_for_fresh_data(self):
        model = self.make_model()
        self.fill_buffer(model.replay_buffer)
        params = {k: v.clone() for k, v in model.policy.state_dict().items()}
        with patch.object(model, "load_replay_buffer") as load:
            prepare_resume_buffer(model, Path("legacy.pkl"), True, 10)
            load.assert_not_called()
        self.assertEqual(model.replay_buffer.size(), 0)
        self.assertTrue(model.replay_buffer.handle_timeout_termination)
        self.assertFalse(model.replay_buffer.timeouts.any())
        self.assertEqual(model.learning_starts, 510)
        self.assertEqual(model.num_timesteps, 500)
        for key, value in model.policy.state_dict().items():
            self.assertTrue(torch.equal(params[key], value))

    def test_valid_buffer_preserves_timeout_labels(self):
        model = self.make_model()
        self.fill_buffer(model.replay_buffer)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "valid.pkl"
            model.save_replay_buffer(path)
            prepare_resume_buffer(model, path, False, 10)
        self.assertEqual(model.replay_buffer.size(), 1)
        self.assertEqual(model.replay_buffer.timeouts[0, 0], 1)
        self.assertEqual(model.learning_starts, 10)

    def test_missing_buffer_uses_cumulative_warmup(self):
        model = self.make_model()
        prepare_resume_buffer(model, None, False, 10)
        self.assertEqual(model.learning_starts, 510)

    def test_resumed_learning_collects_before_updating(self):
        model = self.make_model()
        prepare_resume_buffer(model, None, True, 8)
        model.learn(total_timesteps=8, reset_num_timesteps=False)
        self.assertEqual(model.num_timesteps, 508)
        self.assertEqual(model._n_updates, 0)
        model.learn(total_timesteps=8, reset_num_timesteps=False)
        self.assertEqual(model.num_timesteps, 516)
        self.assertGreater(model._n_updates, 0)

    def test_reset_flag_requires_resume(self):
        self.assertTrue(parse_args(["--resume", "model", "--reset-replay-buffer"]).reset_replay_buffer)
        with self.assertRaisesRegex(ValueError, "--resume"):
            main(["--reset-replay-buffer"])


class LevelBonusTests(unittest.TestCase):
    def test_descent_does_not_rearm_bonus_and_reset_does(self):
        env = Go2TerrainEnv(EnvCfg(terrain="full"), RewardCfg())
        try:
            env.reset(seed=0)
            tops = env.terrain.stair_tops
            heights = [tops[0], tops[1], tops[0], tops[1], tops[2]]
            env.data.qpos[2] = 1.2
            with patch.object(env, "_support_height", side_effect=heights), \
                    patch("go2_common.env.mujoco.mj_step"):
                infos = [env.step(np.zeros(12))[-1] for _ in heights]
            bonus = RewardCfg().level_bonus  # 权重会被调（2026-10-12 climb/level_bonus 都提过），判据本身不变
            self.assertEqual([i["level_bonus"] for i in infos], [bonus, bonus, 0, 0, bonus])
            self.assertEqual([i["level"] for i in infos], [1, 2, 1, 2, 3])
            self.assertEqual([i["max_level"] for i in infos], [1, 2, 2, 2, 3])
            self.assertEqual(env.prev_terrain_level, 3)
            self.assertLess(infos[2]["climb"], 0)
            env.reset(seed=0)
            self.assertEqual(env.prev_terrain_level, 0)
            env.data.qpos[2] = 1.2
            with patch.object(env, "_support_height", return_value=tops[0]), \
                    patch("go2_common.env.mujoco.mj_step"):
                self.assertEqual(env.step(np.zeros(12))[-1]["level_bonus"], RewardCfg().level_bonus)
        finally:
            env.close()

    def test_spawn_level_is_not_rewarded(self):
        env = Go2TerrainEnv(EnvCfg(terrain="full", reset_x=(3.3, 3.3)))
        try:
            env.reset(seed=0)
            spawn_level = env.support_level()
            self.assertGreater(spawn_level, 0)
            self.assertEqual(env.prev_terrain_level, spawn_level)
            with patch.object(env, "_support_height", return_value=env.prev_terrain_h), \
                    patch("go2_common.env.mujoco.mj_step"):
                self.assertEqual(env.step(np.zeros(12))[-1]["level_bonus"], 0)
        finally:
            env.close()

    def test_torch_history_is_independent_per_environment(self):
        env = Go2TerrainEnv(EnvCfg(terrain="full"))
        try:
            f = {k: v[:2] for k, v in synthetic_state(env, 512, np.random.default_rng(0)).items()}
            f["prev_terrain_h"] = np.zeros(2)
            f["prev_terrain_level"] = np.zeros(2)
            st = to_core_state(f, [0, 0], [0, 0])
            st = replace(st, prev_terrain_level=torch.tensor([2, 0]),
                         terrain_level=torch.tensor([1, 3]))
            np.testing.assert_array_equal(core.term_level_bonus(st, RewardCfg()).numpy(), [0, 3])
            advanced = core.advance(st)
            np.testing.assert_array_equal(advanced["prev_terrain_level"].numpy(), [2, 3])
            st = replace(st, prev_terrain_level=advanced["prev_terrain_level"],
                         terrain_level=torch.tensor([2, 3]))
            np.testing.assert_array_equal(core.term_level_bonus(st, RewardCfg()).numpy(), [0, 0])
        finally:
            env.close()

    def test_isaac_reset_clears_only_selected_episode_history(self):
        robot = SimpleNamespace(write_joint_state_to_sim=lambda *args, **kwargs: None)
        env = SimpleNamespace(device="cpu", scene={"robot": robot},
                              _course_prev={"prev_terrain_level": torch.tensor([3, 4, 5])})
        ctx = SimpleNamespace(q_default=torch.zeros(12), joint_ids=list(range(12)))
        path = Path(__file__).resolve().parents[2] / "go2_issac/mdp/events.py"
        spec = importlib.util.spec_from_file_location("test_course_events", path)
        events = importlib.util.module_from_spec(spec)
        # Load only the reset adapter; the full mdp package requires Isaac Lab.
        fake_mdp = SimpleNamespace(state=SimpleNamespace(_ctx=lambda env: ctx))
        with patch.dict("sys.modules", {"go2_issac.mdp": fake_mdp}):
            spec.loader.exec_module(events)
            events.reset_joints(env, env_ids=torch.tensor([0, 2]))
        np.testing.assert_array_equal(env._course_prev["prev_terrain_level"].numpy(), [0, 4, 0])


class StairPricingTests(unittest.TestCase):
    """把"老实爬"和"放弃"两条路的差价钉在这里（2026-10-13）。

    背景：0.78~0.80 上七次训练（R5/R6/R7/R10/R11/R12/R13）最后都退化成
    "冲几步就翻滚"。`python -m go2_sac.diag reward` 逐项拆开后发现
    **这不是优化失败**，而是两条路的**折现**差价塌了——见下面 _discounted 的注释。

    数据是**实测的"去权重"每回合分项原始量**（weight=1 时的量），来源是
    `go2_sac.diag reward <模型> <scale>`，各 6 回合，取自 `s076/best`（老实爬）
    与 `s080`（退化后的产物）。**存原始量而不是加权分**，是为了让这条测试
    **跟着 RewardCfg 的权重走**：谁把权重调回塌缩带的老值，这里就会红。
    """

    # (每回合步数, 去权重的每回合分项)。s076/best @0.80：前进 1.505 m、升 0.389 m、4.33 级。
    HONEST_080 = (537, {
        "progress": 1.5048,
        "track_lin_vel": 97.53,
        "climb": 0.38867,
        "level_bonus": 4.33333,
        # 这一项存的是**登顶次数**（不是 reward 分项，见 env.py:326-329 的名字冲突）。
        # 6 回合样本里恰好 0 次登顶，n 太小（p≈4.7%）；改用 20 局扫描的 8/20 = 0.4。
        "success": 0.4,
        "base_height": 5.77778,
        "lateral": 0.0,
        "lateral_vel": 15.48667,
        "orientation": 31.9,
        "yaw": 27.0,
        "yaw_rate": 148.1333,
        "action_rate": 478.6667,
        "torques": 402500.0,
        "collision": 12458.33,
        "fall": 0.5,
    })
    # s080 退化后 @0.80：前进 1.212 m、升 0.163 m、1.17 级、**从不登顶**。
    GIVEUP_080 = (153, {
        "progress": 1.212,
        "track_lin_vel": 72.11667,
        "climb": 0.162667,
        "level_bonus": 1.166667,
        "success": 0.0,
        "base_height": 3.05556,
        "lateral": 0.0,
        "lateral_vel": 4.38333,
        "orientation": 15.06667,
        "yaw": 3.56667,
        "yaw_rate": 61.66667,
        "action_rate": 202.0,
        "torques": 144166.7,
        "collision": 3875.0,
        "fall": 1.0,
    })

    @staticmethod
    def _discounted(arc, cfg: RewardCfg, gamma: float) -> float:
        """把"每回合净分"按 `gamma` 折现成 SAC 真正在优化的那个量。

        **这一步不能省。** 只比"每回合净分"会得出错误结论：老实爬挣得多，
        但它要 537 步才挣完，而放弃只要 153 步。SAC 优化的是折现回报
        G = r·(1−γ^N)/(1−γ)（每步 r 恒定、末步价值 0），
        `gamma=0.995` 的有效视野只有 200 步——**老实爬后半段的收入基本被打折打没了**。

        这条判据被**唯一已知"学得动"的那一档**验证过：0.76 旧定价下
        老实爬 +153.1 对放弃 +146.8（仅 +4.3%），而那一档 R4/R8/R9 都学会了。
        所以"折现后略微领先"就是够的，判据不用留太大余量。
        """
        steps, raw = arc
        total = sum(getattr(cfg, name) * value for name, value in raw.items())
        rate = total / steps
        return rate * (1.0 - gamma**steps) / (1.0 - gamma)

    def test_honest_climb_beats_giving_up_at_the_collapse_band_scale(self):
        cfg, gamma = RewardCfg(), TrainCfg.gamma
        honest = self._discounted(self.HONEST_080, cfg, gamma)
        giveup = self._discounted(self.GIVEUP_080, cfg, gamma)
        self.assertGreater(
            honest, giveup,
            f"折现后：老实爬 {honest:.1f} vs 放弃 {giveup:.1f}"
            f"（{honest / giveup:.2f} 倍，{(honest / giveup - 1) * 100:+.1f}%）——"
            f"跌回 1 倍以下就是把策略推向'少活一会儿'，调权重时要盯住这条",
        )

    def test_a_failed_attempt_still_beats_giving_up(self):
        """一次都不登顶时也必须还是老实爬划算——否则"敢去试"会被掐掉。

        失败尝试的摔倒率实测 45~95%，所以这条不能只靠 `success` 一项撑着。
        """
        cfg, gamma = RewardCfg(), TrainCfg.gamma
        honest_no_success = (self.HONEST_080[0], dict(self.HONEST_080[1], success=0.0))
        # 登顶次数置 0 后老实爬会输（这就是 `success` 权重必须给足的理由），
        # 所以这里比的是**每回合净分**：它是"敢去试"的下限判据。
        self.assertGreater(
            sum(getattr(cfg, k) * v for k, v in honest_no_success[1].items()),
            sum(getattr(cfg, k) * v for k, v in self.GIVEUP_080[1].items()),
        )


class EvaluationTests(unittest.TestCase):
    def test_peak_level_survives_descent_and_is_reset_per_episode(self):
        with TemporaryDirectory() as tmp:
            callback = EvalMetricsCallback(EnvCfg(terrain="full"), RewardCfg(),
                                           eval_freq=1, n_episodes=2, save_dir=tmp)
            self.addCleanup(callback.eval_env.close)
            callback.num_timesteps = 1
            callback.best_key = (10**9,) * 4  # 大到不可能被超越：这些用例只验指标，不验存盘
            logger = configure(folder=tmp, format_strings=[])
            callback.model = SimpleNamespace(logger=logger)
            episode_levels = iter(([0, 2, 1, 0], [0, 1, 0]))

            def fake_episode(env, act, seed, on_step):
                last = 0
                max_level = 0
                for step, level in enumerate(next(episode_levels), 1):
                    last = level
                    max_level = max(max_level, level)
                    fake = SimpleNamespace(
                        data=SimpleNamespace(qpos=np.array([step, 0, 0]), qvel=np.zeros(6)),
                        cmd_vx=0.6, step_count=step,
                        support_level=lambda level=level: level,
                    )
                    on_step(fake)
                return 1.0, {"level": last, "max_level": max_level, "success": False}

            with patch("go2_common.train_utils.run_episode", side_effect=fake_episode):
                self.assertTrue(callback._on_step())
            values = logger.name_to_value
            self.assertEqual(values["eval/max_level"], 2)
            self.assertEqual(values["eval/mean_level"], 1.5)
            self.assertEqual(values["eval/max_final_level"], 0)
            self.assertEqual(values["eval/mean_final_level"], 0)
            self.assertEqual(values["eval/success_rate"], 0)

    def test_spawn_peak_is_counted_after_immediate_descent(self):
        with TemporaryDirectory() as tmp:
            callback = EvalMetricsCallback(EnvCfg(terrain="full"), RewardCfg(),
                                           eval_freq=1, n_episodes=1, save_dir=tmp)
            self.addCleanup(callback.eval_env.close)
            callback.num_timesteps = 1
            callback.best_key = (10**9,) * 4  # 大到不可能被超越：这些用例只验指标，不验存盘
            logger = configure(folder=tmp, format_strings=[])
            callback.model = SimpleNamespace(logger=logger)

            def fake_episode(env, act, seed, on_step):
                fake = SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(3), qvel=np.zeros(6)),
                                       cmd_vx=0.6, step_count=1, support_level=lambda: 0)
                on_step(fake)
                return 1.0, {"level": 0, "max_level": 3, "success": False}

            with patch("go2_common.train_utils.run_episode", side_effect=fake_episode):
                callback._on_step()
            self.assertEqual(logger.name_to_value["eval/max_level"], 3)
            self.assertEqual(logger.name_to_value["eval/max_final_level"], 0)


class TrainingEntryTests(unittest.TestCase):
    def test_save_resume_reject_legacy_and_explicit_reset(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            cfg = EnvCfg(terrain="flat", max_episode_s=0.02)
            common = ["--steps", "16", "--n-envs", "2", "--torch-threads", "1",
                      "--seed-trot", "0", "--eval-freq", "8", "--eval-episodes", "2",
                      "--checkpoint-freq", "8"]
            first = root / "first"
            with patch.object(sac_train, "RUNS_DIR", root / "runs"), \
                    patch.object(sac_train, "EnvCfg", return_value=cfg):
                main([*common, "--terrain", "flat", "--save-dir", str(first)])
            source = SAC.load(first / "model.zip", device="cpu")
            source.load_replay_buffer(first / "model.pkl")
            self.assertTrue(source.replay_buffer.handle_timeout_termination)
            self.assertEqual(source.num_timesteps, 16)
            self.assertTrue(source.replay_buffer.timeouts[:8].all())
            for name in ("model.zip", "model.pkl", "best/best_model.zip", "best/best_model.pkl",
                         "checkpoints/rl_8_steps.zip", "checkpoints/rl_replay_buffer_8_steps.pkl"):
                self.assertTrue((first / name).exists(), name)

            resumed = root / "resumed"
            with patch.object(sac_train, "RUNS_DIR", root / "runs"):
                main([*common, "--resume", str(first / "model"), "--save-dir", str(resumed)])
            continued = SAC.load(resumed / "model.zip", device="cpu")
            continued.load_replay_buffer(resumed / "model.pkl")
            self.assertEqual(continued.num_timesteps, 32)
            self.assertEqual(continued.replay_buffer.size(), 16)
            self.assertTrue(continued.replay_buffer.timeouts[:16].all())

            source.replay_buffer.handle_timeout_termination = False
            source.replay_buffer.timeouts.fill(0)
            source.save_replay_buffer(first / "model.pkl")
            legacy_bytes = (first / "model.pkl").read_bytes()
            reset = root / "reset"
            with patch.object(sac_train, "RUNS_DIR", root / "runs"):
                with self.assertRaisesRegex(ValueError, "--reset-replay-buffer"):
                    main([*common, "--resume", str(first / "model"), "--save-dir", str(reset)])
                main([*common, "--resume", str(first / "model"), "--save-dir", str(reset),
                      "--reset-replay-buffer"])
            repaired = SAC.load(reset / "model.zip", device="cpu")
            repaired.load_replay_buffer(reset / "model.pkl")
            self.assertEqual(repaired.num_timesteps, 32)
            self.assertEqual(repaired.learning_starts, 10016)
            self.assertEqual(repaired.replay_buffer.size(), 8)
            self.assertTrue(repaired.replay_buffer.handle_timeout_termination)
            self.assertTrue(repaired.replay_buffer.timeouts[:8].all())
            self.assertEqual((first / "model.pkl").read_bytes(), legacy_bytes)
            for key, value in source.policy.state_dict().items():
                self.assertTrue(torch.equal(value, repaired.policy.state_dict()[key]))


class ScanTests(unittest.TestCase):
    """前方地形高度扫描（`--scan`）和它的暖启动。

    这里的核心断言只有一条：**扩维之后策略和扩维之前逐位一致**
    （见 `train_utils.warm_start_scan`）。没有这一条，暖启动就等于随机重新初始化，
    那还不如从头训——整个"给观测加信息"的方案就不成立了。
    """

    def test_heights_matches_pointwise_top(self):
        """向量化的 heights() 必须和逐点 top() 完全相同（这是它唯一的安全性论据）。"""
        env = Go2TerrainEnv(EnvCfg(terrain="full", terrain_scale=0.76))
        try:
            rng = np.random.default_rng(0)
            xs = rng.uniform(-0.5, 4.0, 800)
            ys = rng.uniform(-0.5, 0.5, 800)
            ref = np.array([env.terrain.top(float(x), float(y)) for x, y in zip(xs, ys)])
            np.testing.assert_array_equal(ref, env.terrain.heights(xs, ys))
        finally:
            env.close()

    def test_obs_dim_combinations(self):
        self.assertEqual(obs_dim(), 47)
        self.assertEqual(obs_dim(True), 48)
        self.assertEqual(obs_dim(False, True), 63)
        self.assertEqual(obs_dim(True, True), 64)

    def test_scan_obs_is_zero_on_flat_ground(self):
        """出生在 x=-0.8（前方连 0.06 m 的槛都没有）时，扫描必须逐点全 0。

        注意不能随便挑个"离台阶远"的点：场景里 x=1.10~1.70 还有两个门槛
        （见 terrain.py 开头），x=0.3 就能扫到它们。
        """
        env = Go2TerrainEnv(EnvCfg(terrain="full", terrain_scale=0.76, scan=True))
        try:
            self.assertEqual(env.observation_space.shape, (63,))
            env.cfg.reset_x = (-0.8, -0.8)
            env.cfg.reset_frac_stair = 0.0
            obs, _ = env.reset(seed=7)
            self.assertEqual(obs.shape, (63,))
            np.testing.assert_array_equal(obs[47:], np.zeros(16, np.float32))
        finally:
            env.close()

    def test_warm_start_scan_keeps_policy_identical(self):
        """47 维 checkpoint 扩成 63 维后，新 16 维填 0 时输出必须与原来一致。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_path = root / "old"
            old = SAC(
                "MlpPolicy",
                Go2TerrainEnv(EnvCfg(terrain="flat", max_episode_s=0.02)),
                device="cpu",
                seed=0,
                policy_kwargs=dict(net_arch=[64, 64]),
            )
            old.save(str(old_path))
            self.assertEqual(checkpoint_obs_dim(old_path), 47)

            new_path = warm_start_scan(old_path, root / "new.zip", 63)
            self.assertEqual(checkpoint_obs_dim(new_path), 63)

            env63 = Go2TerrainEnv(EnvCfg(terrain="flat", max_episode_s=0.02, scan=True))
            try:
                warmed = SAC.load(str(new_path), env=env63, device="cpu")
            finally:
                env63.close()

            # 权重层面是**精确**的：新增列就是 0，旧列原封不动
            old_sd, new_sd = old.policy.state_dict(), warmed.policy.state_dict()
            self.assertTrue(
                torch.equal(new_sd["actor.latent_pi.0.weight"][:, :47],
                            old_sd["actor.latent_pi.0.weight"])
            )
            self.assertTrue(
                torch.equal(new_sd["actor.latent_pi.0.weight"][:, 47:],
                            torch.zeros_like(new_sd["actor.latent_pi.0.weight"][:, 47:]))
            )
            # critic 的新列插在**动作列之前**（SB3 吃的是 [obs, action] 拼接）
            for q in ("critic.qf0.0.weight", "critic.qf1.0.weight", "critic_target.qf0.0.weight"):
                self.assertTrue(torch.equal(new_sd[q][:, :47], old_sd[q][:, :47]), q)
                self.assertTrue(torch.equal(new_sd[q][:, 63:], old_sd[q][:, 47:]), q)
                self.assertTrue(torch.equal(new_sd[q][:, 47:63], torch.zeros_like(new_sd[q][:, 47:63])), q)

            # 输出层面允许 1e-6 的浮点噪声：数学恒等是精确的，但多出来的 16 个零项
            # 会改变 BLAS 的累加顺序（实测这台机器上差值是 0.0，但不能依赖它）
            obs47 = np.random.default_rng(0).normal(size=(32, 47)).astype(np.float32)
            obs63 = np.concatenate([obs47, np.zeros((32, 16), np.float32)], axis=1)
            a47 = np.array([old.predict(o, deterministic=True)[0] for o in obs47])
            a63 = np.array([warmed.predict(o, deterministic=True)[0] for o in obs63])
            self.assertLess(float(np.abs(a47 - a63).max()), 1e-6)
            with torch.no_grad():
                q47 = old.critic(torch.as_tensor(obs47), torch.as_tensor(a47))[0]
                q63 = warmed.critic(torch.as_tensor(obs63), torch.as_tensor(a47))[0]
            self.assertLess(float((q47 - q63).abs().max()), 1e-6)

    def test_warm_start_scan_rejects_wrong_dims(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = SAC(
                "MlpPolicy",
                Go2TerrainEnv(EnvCfg(terrain="flat", max_episode_s=0.02)),
                device="cpu",
                seed=0,
                policy_kwargs=dict(net_arch=[64, 64]),
            )
            old.save(str(root / "old"))
            with self.assertRaisesRegex(ValueError, "不需要暖启动"):
                warm_start_scan(root / "old", root / "a.zip", 47)
            with self.assertRaisesRegex(ValueError, "方向反了"):
                warm_start_scan(root / "old", root / "b.zip", 32)

    def test_pad_replay_buffer_keeps_experience_and_zero_fills(self):
        """补齐 buffer：旧数据一条不丢，新增列全 0，超时标签保留。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = Go2TerrainEnv(EnvCfg(terrain="flat", max_episode_s=0.02))
            try:
                model = SAC("MlpPolicy", env, device="cpu", seed=0,
                            buffer_size=32, learning_starts=0,
                            policy_kwargs=dict(net_arch=[64, 64]))
                model.learn(16, progress_bar=False)
            finally:
                env.close()
            old_path = root / "old.pkl"
            model.save_replay_buffer(old_path)
            old = model.replay_buffer
            self.assertEqual(old.observations.shape[-1], 47)
            n = old.size()
            self.assertGreater(n, 0)

            new_path = pad_replay_buffer(old_path, root / "new.pkl", 63)
            with open(new_path, "rb") as f:
                import pickle

                new = pickle.load(f)
            self.assertEqual(new.observations.shape[-1], 63)
            self.assertEqual(new.size(), n)
            self.assertTrue(new.handle_timeout_termination)
            self.assertEqual(new.observations.shape[:-1], old.observations.shape[:-1])
            # 旧列原样、新列全 0（observations 和 next_observations 都要）
            for attr in ("observations", "next_observations"):
                np.testing.assert_array_equal(getattr(new, attr)[..., :47], getattr(old, attr))
                np.testing.assert_array_equal(
                    getattr(new, attr)[..., 47:], np.zeros(new.observations.shape[:-1] + (16,), np.float32)
                )
            for attr in ("actions", "rewards", "dones", "timeouts"):
                np.testing.assert_array_equal(getattr(new, attr), getattr(old, attr))
            with self.assertRaisesRegex(ValueError, "不需要补齐"):
                pad_replay_buffer(old_path, root / "x.pkl", 47)

    def test_resume_with_scan_warm_starts_and_keeps_old_buffer(self):
        """走一遍真正的 `main()`：47 维续训 + `--scan` 要自动扩到 63 维。

        buffer 是**补齐**不是清空（理由见 `pad_replay_buffer`：清空实测会塌），
        所以续训之后 buffer 里应当同时有旧数据和新采的数据。
        """
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            cfg = EnvCfg(terrain="flat", max_episode_s=0.02)
            common = ["--steps", "16", "--n-envs", "2", "--torch-threads", "1",
                      "--seed-trot", "0", "--eval-freq", "8", "--eval-episodes", "2",
                      "--checkpoint-freq", "8"]
            first = root / "first"
            with patch.object(sac_train, "RUNS_DIR", root / "runs"), \
                    patch.object(sac_train, "EnvCfg", return_value=cfg):
                main([*common, "--terrain", "flat", "--save-dir", str(first)])
            self.assertEqual(checkpoint_obs_dim(first / "model"), 47)

            second = root / "second"
            with patch.object(sac_train, "RUNS_DIR", root / "runs"):
                main([*common, "--resume", str(first / "model"), "--scan",
                      "--save-dir", str(second)])

            self.assertEqual(checkpoint_obs_dim(second / "model"), 63)
            # 种进 best/ 的必须是**扩维后**的模型，否则评估回调加载它会维度不符
            self.assertEqual(checkpoint_obs_dim(second / "best" / "best_model"), 63)
            self.assertTrue((second / "warm_start" / "best_model.zip").exists())
            self.assertTrue((second / "warm_start" / "replay_buffer_padded.pkl").exists())
            # 旧经验必须还在。注意 `size()` 数的是**每个环境的 add 次数**（SB3 的口径），
            # 16 步 / 2 个环境 = 8，所以"补齐"应当比"清空"多出第一轮那 8 次。
            warmed = SAC.load(second / "model.zip", device="cpu")
            warmed.load_replay_buffer(second / "model.pkl")
            self.assertEqual(warmed.replay_buffer.observations.shape[-1], 63)
            padded_size = warmed.replay_buffer.size()

            # 显式 --reset-replay-buffer 仍然要能强制清空（逃生舱口）
            third = root / "third"
            with patch.object(sac_train, "RUNS_DIR", root / "runs"):
                main([*common, "--resume", str(first / "model"), "--scan",
                      "--reset-replay-buffer", "--save-dir", str(third)])
            cleared = SAC.load(third / "model.zip", device="cpu")
            cleared.load_replay_buffer(third / "model.pkl")
            cleared_size = cleared.replay_buffer.size()
            self.assertEqual(cleared_size, 8)
            self.assertEqual(padded_size, 16)


class BestModelKeyTests(unittest.TestCase):
    """选模判据的回归测试。实测背景：full 阶段按 eval/mean_reward 取极大值，
    选中的是"冲 1.8 m 就地翻滚"（20/20 摔倒，+172.3），而同一批评估里真有进展的
    那次（峰值台阶 0.15）回报只有 -29.92，永远选不上 —— 于是 best_model 忠实地
    记录下了退化。这几条断言就是那个 bug 的护栏。
    """

    def test_success_beats_everything(self):
        goal = 3.30
        completer = task_key(1, 0.0, 3.30, -500.0, goal)
        flipper = task_key(0, 5.0, 3.30, 1e6, goal)
        self.assertGreater(completer, flipper)

    def test_reward_is_only_the_last_tiebreak(self):
        goal = 3.30
        climber = task_key(0, 1.5, 2.4, -100.0, goal)
        sprinter = task_key(0, 0.0, 2.4, 1e6, goal)
        self.assertGreater(climber, sprinter)

    def test_max_x_is_clamped_at_goal_x(self):
        # 绕到 |y|≈2.2 的平地上一路向前能刷出很大的 x，但不该因此赢（§5-B5）
        goal = 3.30
        far = task_key(0, 0.0, 99.0, 0.0, goal)
        at_goal = task_key(0, 0.0, goal, 0.0, goal)
        self.assertEqual(far[2], 3.30)
        self.assertEqual(far, at_goal)

    def test_ties_do_not_overwrite(self):
        goal = 3.30
        a = task_key(0, 1.0, 2.0, 10.0, goal)
        b = task_key(0, 1.0, 2.0, 10.04, goal)  # 第四项 round 到 0.1 后相同
        self.assertEqual(a, b)  # 严格 > 比较下不会反复覆盖 best_model


class TerrainScaleMixTests(unittest.TestCase):
    """多尺度混合（--terrain-scale-range）：分配规则、旧行为不变、评估尺子不被污染。"""

    def test_none_range_is_the_old_value_bit_for_bit(self):
        # 旧行为必须逐位不变：range=None 时连浮点运算都不能做，否则历史 run 的可复现性没了
        cfg = EnvCfg(terrain="full", terrain_scale=0.8)
        for seed in (0, 1, 7, 12345):
            self.assertEqual(worker_scale(cfg, seed), 0.8)
        self.assertIsNone(cfg.terrain_scale_range)

    def test_range_spreads_workers_over_the_whole_interval(self):
        lo, hi = 0.76, 0.90
        cfg = EnvCfg(terrain="full", terrain_scale=0.80, terrain_scale_range=(lo, hi))
        scales = [worker_scale(cfg, seed + i) for i, seed in enumerate([0] * 8)]
        self.assertTrue(all(lo <= s <= hi for s in scales), scales)
        # 黄金比低差异铺开：8 个点必须覆盖到区间的 80% 以上（随机抽签经常只覆盖一半）
        self.assertGreater((max(scales) - min(scales)) / (hi - lo), 0.80)

    def test_same_seed_same_scale(self):
        cfg = EnvCfg(terrain="full", terrain_scale=0.80, terrain_scale_range=(0.76, 0.90))
        self.assertEqual(worker_scale(cfg, 12345), worker_scale(cfg, 12345))
        self.assertNotEqual(worker_scale(cfg, 12345), worker_scale(cfg, 12346))

    def test_rejects_inverted_or_negative_range(self):
        for bad in ((0.9, 0.8), (-0.1, 0.5)):
            cfg = EnvCfg(terrain="full", terrain_scale_range=bad)
            with self.assertRaises(ValueError):
                worker_scale(cfg, 0)

    def test_list_from_config_json_still_works(self):
        # asdict/JSON 往返会把 tuple 变成 list（reset_x 有同样的先例），必须照样能吃
        cfg = EnvCfg(terrain="full", terrain_scale=0.80, terrain_scale_range=[0.76, 0.90])
        self.assertTrue(0.76 <= worker_scale(cfg, 3) <= 0.90)

    def test_make_env_builds_the_assigned_scale(self):
        cfg = EnvCfg(terrain="full", terrain_scale=0.80, terrain_scale_range=(0.76, 0.90))
        env = make_env(cfg, RewardCfg(), seed=5)
        self.addCleanup(env.close)
        want = worker_scale(cfg, 5)
        # cfg 被 make_env 复制过（goal_z 会被 env 改写），所以原 cfg 不受影响
        self.assertEqual(cfg.terrain_scale, 0.80)
        got = env.unwrapped.cfg.terrain_scale
        self.assertAlmostEqual(got, want, places=12)
        # 地形真的按这个 scale 建出来了，不是只改了 cfg 字段
        tops = [round(t, 4) for t in env.unwrapped.terrain.stair_tops]
        self.assertEqual(tops, [round(scale * got, 4) for scale in (0.17, 0.32, 0.47, 0.62, 0.77, 0.92)])

    def test_eval_ruler_is_pinned_to_a_single_scale(self):
        # 训练侧有混合时，评估环境绝不能被带着跑 —— 否则选模 key 跨 run 不可比
        cfg = EnvCfg(terrain="full", terrain_scale=0.80, terrain_scale_range=(0.76, 0.90))
        with TemporaryDirectory() as tmp:
            callback = EvalMetricsCallback(cfg, RewardCfg(), eval_freq=1, n_episodes=1, save_dir=tmp)
            self.addCleanup(callback.eval_env.close)
            got = callback.eval_env.cfg
            self.assertIsNone(got.terrain_scale_range)
            self.assertEqual(got.terrain_scale, 0.80)
            self.assertEqual(callback._protocol()["terrain_scale"], 0.80)



class LevelIncomeTests(unittest.TestCase):
    """`level_income` 必须真的是"非望远镜位势"—— 这正是 §7.4 那个病的药。

    背景（2026-10-13，修复记录 §7.3~§7.8）：`climb` 一局的合计是 `h_末 − h_始`。
    在爬得上去的高度上没问题；但在能力边界之上，"爬上去→掉下来"的净高度变化恰好是 0，
    于是 `climb` 项**整项为 0**（实测 1.00 档 20 局），`success_bonus` 也是 0，
    而"去试"要付 collision+orientation。→ "不试"成了严格最优，且因为观测里没有地形信息，
    这个"别试"会无条件地扩散到邻档 —— R9~R19 每次坍缩、以及"当档特化"的统一机制。

    按**级数**计价而不是按**米**：第一版写的是 `height_income × terrain_h`，
    evalx 实测 1.00 档上 20 局白拿 +172/局——钱大多是在 x=1.10~1.70 那两个
    0.08 m 的**槛**上蹲着领的（狗本来就爱蹲槛）。`level_at_height(0.08) = 0`
    是定义使然（第一个 top 是 0.170），所以按级数计门槛严格发 0，
    且"从槛爬上第 1 级"的差价不被摊薄、也不需要任何魔法常数。
    """

    @staticmethod
    def _input(h: float, prev_h: float, lvl: int = 0, prev_lvl: int = 0):
        z = np.zeros(12)
        return RewardInput(
            action=z, prev_action=z, q=z, dq=z, tau=z,
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            base_v=np.zeros(3), base_w=np.zeros(3),
            base_pos=np.array([0.0, 0.0, h + 0.27]),
            prev_x=0.0, terrain_h=float(h), prev_terrain_h=float(prev_h),
            terrain_level=int(lvl), prev_terrain_level=int(prev_lvl),
            cmd_vx=0.0, contact_force=0.0, terminated=False, success=False,
        )

    @staticmethod
    def _walk(steps):
        """`steps` = [(支撑面高度, 级数), ...] 逐帧走一遍，返回 (Σ climb, Σ level_income)。"""
        cfg = RewardCfg()
        prev_h = steps[0][0]
        sc = si = 0.0
        for h, lvl in steps:
            r = LevelIncomeTests._input(h, prev_h, lvl)
            sc += _climb_impl(r, cfg)
            si += _level_income_impl(r, cfg)
            prev_h = h
        return sc, si

    #: 平地 100 步 → 第 1 级(0.170) 300 步 → 掉回平地 100 步。
    #: 这正是"爬上去又掉下来"，测得 1.00 档上每局都在发生的形态。
    _FAILED_ATTEMPT = [(0.0, 0)] * 100 + [(0.170, 1)] * 300 + [(0.0, 0)] * 100

    def test_climb_is_telescoping_so_a_failed_attempt_nets_zero(self):
        s_climb, _ = self._walk(self._FAILED_ATTEMPT)
        # 这正是"病"：爬上去再掉下来，climb 一分钱没挣到（h_末 − h_始 = 0）
        self.assertAlmostEqual(s_climb, 0.0, places=9)

    def test_level_income_makes_the_same_attempt_pay(self):
        _, s_income = self._walk(self._FAILED_ATTEMPT)
        # 300 步 × 1 级 × 权重（与级数线性，平地严格为 0）
        self.assertAlmostEqual(s_income, float(RewardCfg().level_income) * 300, places=9)
        self.assertGreater(s_income, 0.0)

    def test_menkan_pays_nothing(self):
        """0.08 m 的槛必须是 0 级、收入必须为 0 —— 这是弃用"按米计价"的唯一理由。"""
        env = Go2TerrainEnv(EnvCfg(terrain="full", terrain_scale=1.00), RewardCfg())
        self.addCleanup(env.close)
        menkan = 0.08 * 1.00
        self.assertEqual(env.terrain.level_at_height(menkan), 0)
        # 第一级台阶(0.170)才是 1 级，所以"从槛爬到第 1 级"的差价是满额 1 级
        self.assertEqual(env.terrain.level_at_height(0.170), 1)
        cfg = RewardCfg()
        got = _level_income_impl(self._input(menkan, menkan, 0), cfg)
        self.assertEqual(got, 0.0)

    def test_income_is_zero_when_the_knob_is_off(self):
        # 旋钮归零 = 旧行为：两条路都恒为 0，差价仍然是塌的
        cfg = RewardCfg(level_income=0.0)
        total = 0.0
        for h, lvl in self._FAILED_ATTEMPT:
            total += _level_income_impl(LevelIncomeTests._input(h, h, lvl), cfg)
        self.assertEqual(total, 0.0)

    def test_income_is_linear_in_level_and_never_negative(self):
        cfg = RewardCfg(level_income=0.4)
        for lvl in (0, 1, 2, 3, 4, 5, 6):
            got = _level_income_impl(LevelIncomeTests._input(0.92, 0.92, lvl), cfg)
            self.assertAlmostEqual(got, 0.4 * lvl, places=12)
            self.assertGreaterEqual(got, 0.0)

    def test_term_is_registered_in_the_shared_pricing(self):
        from go2_common.reward import TERMS
        names = [n for n, _ in TERMS]
        self.assertIn("level_income", names)
        # 紧挨着 climb —— 三项名字/顺序的唯一真源是这里，Isaac 侧靠断言同步
        self.assertEqual(names.index("level_income"), names.index("climb") + 1)

    def test_default_pricing_is_actually_on(self):
        # 默认值必须非 0，否则"修好了"只是错觉
        self.assertGreater(RewardCfg().level_income, 0.0)

    def test_env_emits_the_term_every_step(self):
        env = Go2TerrainEnv(EnvCfg(terrain="full", terrain_scale=0.80), RewardCfg())
        self.addCleanup(env.close)
        env.reset(seed=0)
        obs, r, term, trunc, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        self.assertIn("level_income", info)
        self.assertAlmostEqual(info["level_income"],
                               RewardCfg().level_income * env.support_level(), places=9)

if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
