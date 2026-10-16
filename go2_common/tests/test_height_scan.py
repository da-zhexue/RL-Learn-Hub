
"""地形高度扫描（height scan）观测的测试。

不变量：
  1. height_scan=False 时观测仍是 47 维，而且**逐位不变**（老模型/Isaac 都不受影响）；
  2. 打开后 = 47 + n_scan，值 = 采样点下方的地形高度 − 机身高度，clip ±1；
  3. 平地上每个采样点都等于 −base_z；台阶前能看到地面在抬高。
"""

from __future__ import annotations

import pathlib
import sys
import unittest
from dataclasses import fields

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402

from go2_common.config import (  # noqa: E402
    EnvCfg,
    RewardCfg,
    obs_dim,
    quat_rotate,
    scan_dim,
    to_jsonable,
)
from go2_common.env import Go2TerrainEnv  # noqa: E402


class HeightScanConfigTests(unittest.TestCase):
    def test_obs_dim(self):
        self.assertEqual(obs_dim(), 47)
        self.assertEqual(obs_dim(True), 48)
        self.assertEqual(obs_dim(False, 15), 62)
        self.assertEqual(obs_dim(True, 15), 63)

    def test_scan_dim(self):
        self.assertEqual(scan_dim(False, (0.0, 0.2), (0.0,)), 0)
        self.assertEqual(scan_dim(True, (0.0, 0.2), (0.0,)), 2)

    def test_default_grid_is_15(self):
        self.assertFalse(EnvCfg().height_scan)
        self.assertEqual(EnvCfg().n_scan, 0)
        self.assertEqual(EnvCfg(height_scan=True).n_scan, 15)

    def test_config_roundtrip_keeps_scan(self):
        cfg = EnvCfg(height_scan=True, scan_x=(0.0, 0.3), scan_y=(-0.1, 0.1))
        saved = to_jsonable(cfg)
        self.assertNotIn("n_scan", saved)  # property 不该进 config.json
        names = {f.name for f in fields(EnvCfg)}
        back = EnvCfg(**{k: v for k, v in saved.items() if k in names})
        self.assertTrue(back.height_scan)
        self.assertEqual(tuple(back.scan_x), (0.0, 0.3))
        self.assertEqual(back.n_scan, 4)

    def test_quat_rotate(self):
        np.testing.assert_allclose(quat_rotate([1, 0, 0, 0], [1, 0, 0]), [1, 0, 0], atol=1e-12)
        # 绕 y 轴 90°：+x -> -z
        q = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]
        np.testing.assert_allclose(quat_rotate(q, [1, 0, 0]), [0, 0, -1], atol=1e-9)
        # 一组向量
        out = quat_rotate(q, [[1, 0, 0], [0, 0, 1]])
        self.assertEqual(out.shape, (2, 3))
        np.testing.assert_allclose(out[1], [1, 0, 0], atol=1e-9)


class HeightScanEnvTests(unittest.TestCase):
    def test_backward_compatible_when_off(self):
        """关掉扫描时，观测和带扫描 env 的前 47 维逐位相同（同一 seed、同一动作）。"""
        plain = Go2TerrainEnv(EnvCfg(terrain="full", terrain_scale=1.0), RewardCfg())
        scan = Go2TerrainEnv(EnvCfg(terrain="full", terrain_scale=1.0, height_scan=True), RewardCfg())
        try:
            self.assertEqual(plain.observation_space.shape, (47,))
            self.assertEqual(scan.observation_space.shape, (62,))
            self.assertEqual(scan.n_scan, 15)
            obs_a, _ = plain.reset(seed=0)
            obs_b, _ = scan.reset(seed=0)
            np.testing.assert_array_equal(obs_a, obs_b[:47])
            action = np.zeros(12, dtype=np.float32)
            for _ in range(25):
                obs_a, _, ta, tra, _ = plain.step(action)
                obs_b, _, tb, trb, _ = scan.step(action)
                np.testing.assert_array_equal(obs_a, obs_b[:47])
                self.assertEqual(obs_b.shape, (62,))
                self.assertLessEqual(float(np.abs(obs_b[47:]).max()), 1.0)
                if ta or tra or tb or trb:
                    break
        finally:
            plain.close()
            scan.close()

    def test_flat_ground_scan_equals_minus_base_height(self):
        env = Go2TerrainEnv(EnvCfg(terrain="flat", height_scan=True), RewardCfg())
        try:
            obs, _ = env.reset(seed=0)
            scan = obs[47:]
            self.assertEqual(scan.shape, (15,))
            np.testing.assert_allclose(scan, -env.data.qpos[2], atol=1e-6)
        finally:
            env.close()

    def test_scan_sees_the_staircase_rising(self):
        env = Go2TerrainEnv(EnvCfg(terrain="full", terrain_scale=1.0, height_scan=True), RewardCfg())
        try:
            env.reset(seed=0)
            env.data.qpos[:] = 0.0
            env.data.qpos[0:3] = [1.9, 0.0, 0.28]  # 第一级立面在 x=2.1
            env.data.qpos[3] = 1.0
            env.data.qpos[env.qadr] = env.q_default
            mujoco.mj_forward(env.model, env.data)
            scan = env._height_scan()
            self.assertEqual(scan.shape, (15,))
            # 采样点 x = 1.9 ~ 2.7，y = -0.2/0/0.2；z=0.28
            self.assertAlmostEqual(float(scan.min()), -0.28, places=6)  # 脚下是平地和 -0.28
            self.assertGreater(float(scan.max()), 0.0)  # 前方地形已经高过机身
            rows = scan.reshape(len(env.cfg.scan_x), len(env.cfg.scan_y)).max(axis=1)
            self.assertGreater(float(rows[-1] - rows[0]), 0.4)  # 由近到远明显升高
            self.assertAlmostEqual(float(rows[0]), -0.28, places=6)
        finally:
            env.close()

    def test_scan_empty_grid_means_no_extra_dims(self):
        env = Go2TerrainEnv(EnvCfg(terrain="flat", height_scan=True, scan_x=(0.0,), scan_y=(0.0,)),
                            RewardCfg())
        try:
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs.shape, (48,))
            self.assertEqual(env.n_scan, 1)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
