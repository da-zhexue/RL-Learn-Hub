"""本机（无显卡）就能跑的对拍测试。

`course_parity.py` 把 `course.py` 和真 MuJoCo 地形逐点对拍；
`core_parity.py` 把 `core.py` 和真 `Go2TerrainEnv` 的观测/奖励/终止逐位对拍。
两个都用 `python3 -m go2_issac.tests.<名字>` 跑，全绿才算这一层可信。
"""
