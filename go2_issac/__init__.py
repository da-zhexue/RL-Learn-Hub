"""go2_issac：把 go2 过地形这套（地形 / 观测 / 奖励 / 课程）搬到 Isaac Sim。

本机没有 NVIDIA 显卡、装不了 Isaac Sim，所以这个包**刻意分两层**：

* 本机可跑（纯 numpy/torch，不 import mujoco / isaaclab）：
  `course.py`（地形几何）、`core.py`（观测 / 奖励 / 终止判据）、`tests/*`。
  语义由对拍保证——和真 MuJoCo 环境逐位比。
* 只能上机跑（import isaaclab）：`env_cfg.py`、`mdp/`、`agents/`、`train.py`、`play.py`、
  `smoke.py`、`convert_assets.py`。见 `RUNBOOK.md`。

**这里不做 gym 注册**：注册要 import isaaclab，一旦写进 `__init__.py`，
本机连 `import go2_issac.course` 都会炸。环境由 `env_cfg.build_env(cfg)` 直接实例化，
不走 `gym.make`——没有外部工具按字符串 id 找任务，注册只多一层版本差异。
"""
