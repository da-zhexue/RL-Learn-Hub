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

from go2_common.config import EnvCfg, RewardCfg
from go2_common.env import Go2TerrainEnv
from go2_common.train_utils import EvalMetricsCallback
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
            self.assertEqual([i["level_bonus"] for i in infos], [5, 5, 0, 0, 5])
            self.assertEqual([i["level"] for i in infos], [1, 2, 1, 2, 3])
            self.assertEqual([i["max_level"] for i in infos], [1, 2, 2, 2, 3])
            self.assertEqual(env.prev_terrain_level, 3)
            self.assertLess(infos[2]["climb"], 0)
            env.reset(seed=0)
            self.assertEqual(env.prev_terrain_level, 0)
            env.data.qpos[2] = 1.2
            with patch.object(env, "_support_height", return_value=tops[0]), \
                    patch("go2_common.env.mujoco.mj_step"):
                self.assertEqual(env.step(np.zeros(12))[-1]["level_bonus"], 5)
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


class EvaluationTests(unittest.TestCase):
    def test_peak_level_survives_descent_and_is_reset_per_episode(self):
        with TemporaryDirectory() as tmp:
            callback = EvalMetricsCallback(EnvCfg(terrain="full"), RewardCfg(),
                                           eval_freq=1, n_episodes=2, save_dir=tmp)
            self.addCleanup(callback.eval_env.close)
            callback.num_timesteps = 1
            callback.best_reward = np.inf
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
            callback.best_reward = np.inf
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


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
