"""把 `course.py` 的盒子表变成 Isaac Lab 的 sub-terrain（高程图 patch）。

Isaac Lab 的地形是**按 patch 铺的高程图**：`TerrainGenerator` 给每个 patch 一个
`difficulty`（0~1），sub-terrain 负责把 `difficulty` 变成一块 `(H, W)` 的 int16 高度场。
我们这里的"难度"就是 `EnvCfg.terrain_scale`（0.3~1.0，只压地形高度），
所以 `difficulty` **就是** `terrain_scale`，不用自己写 curriculum：
`TerrainGeneratorCfg(difficulty_range=(s, s), curriculum=False)` 就固定成单一档位，
想要自动课程时把 `curriculum` 打开、`difficulty_range` 放宽即可（见 RUNBOOK 的"可选后续"）。

高度场只从 `course.heightfield_for` 来（`course.py` 是唯一真源，本机对着 MuJoCo 逐位对拍过），
这里只做两件事：量化成 int16、按 Isaac 的惯例在边缘加一圈平地缓冲。

**量化**：`vertical_scale=0.001`（1 mm）。为什么够用：0.7 档的台阶顶面是
0.056/0.119/0.224/... 全都是整毫米，量化误差 ≤0.5 mm；再细就要把 int16 的行程浪费掉。
`horizontal_scale=0.02`（2 cm/像素）对 0.19 m 的踏面没影响，台阶立面被拉成 2 cm 的斜面
——MuJoCo 是垂直面，这点差异只体现在"蹭着台阶沿往上爬"的接触上，属于跨仿真器消不掉的那一类。
"""
from __future__ import annotations

import numpy as np

from go2_issac import course

try:  # Isaac Lab 2.x
    from isaaclab.terrains import SubTerrainBaseCfg
except ImportError:  # Isaac Lab 1.x
    from isaaclab.terrain import SubTerrainBaseCfg


class CourseTerrain:
    """一个 patch = 一条跑道（5×5 m），里面按课程档位摆好槛和台阶。

    `difficulty` 直接当 `terrain_scale` 用，所以同一个类既能跑单档位
    （`difficulty_range=(0.7, 0.7)`），也能直接接自动课程（`difficulty_range=(0.3, 1.0)`）。
    """

    def __init__(self, cfg: "CourseTerrainCfg") -> None:
        self.cfg = cfg
        self.variant = cfg.variant
        self._size = (cfg.size[0], cfg.size[1])
        self._h_scale = cfg.horizontal_scale
        self._v_scale = cfg.vertical_scale
        self._border = cfg.border_width

    def __call__(self, difficulty: float) -> tuple[np.ndarray, np.ndarray]:
        """返回 `(height_field(int16), origin(np.ndarray(3)))`。

        `origin` 是 patch 左下角在 patch 局部坐标系里的位置——Isaac Lab 用它给机器人
        出生点定位，**约定是"以 patch 中心为原点"**，也就是返回 `(-size_x/2, -size_y/2, 0)`。
        这一条正是 `COURSE_ORIGIN_FROM_ENV_ORIGIN` 的校准对象（`smoke.py --check-terrain`）。
        """
        return self._height_field(difficulty), np.array(
            [-self._size[0] / 2.0, -self._size[1] / 2.0, 0.0], dtype=np.float64)

    def _height_field(self, difficulty: float) -> np.ndarray:
        """按课程档位生成高度场（米 -> int16）。

        `difficulty` 就是 `terrain_scale`：0.3 时两个槛只有 2.4 cm、六级台阶每级升 4.5 cm。
        """
        scale = float(difficulty)
        hf = course.heightfield_for(self.variant, scale, size=self._size,
                                    horizontal_scale=self._h_scale)

        if self._border > 0:
            # 边缘留一圈平地：Isaac Lab 的 patch 是紧挨着拼的，相邻跑道的台阶如果顶到
            # patch 边界，狗一走出去就直接踩到隔壁跑道的高度上。`course.heightfield_for`
            # 已经让课程只占 patch 的左边 5×5 里的一块（x∈[-1,4]、y∈[-2.5,2.5]），
            # 这里再把最外圈压平，双保险。
            b = int(round(self._border / self._h_scale))
            hf[:b, :] = 0.0
            hf[-b:, :] = 0.0
            hf[:, :b] = 0.0
            hf[:, -b:] = 0.0

        q = np.rint(hf / self._v_scale).astype(np.int16)
        return q


class CourseTerrainCfg(SubTerrainBaseCfg):
    """`env_cfg.py` 里挂在 `sub_terrains={"course": CourseTerrainCfg(...)}`。

    这几个字段（`size`/`horizontal_scale`/`vertical_scale`/`border_width`/`slope_threshold`）
    **全部显式声明**，哪怕其中几个 `SubTerrainBaseCfg` 本身也有：`TerrainGenerator` 建地形时
    会往 sub-terrain 的 cfg 上直接赋值（`sub_terrain_cfg.size = self.cfg.size` 这种），
    而 `@configclass` 对"给没声明的字段赋值"是不客气的。声明出来两条路都走得通，
    代价只是多几行、多一个用不到的 `slope_threshold`。
    """

    function: type = CourseTerrain

    #: flat / steps / full，语义见 `course.py`
    variant: str = "full"

    size: tuple[float, float] = course.PATCH_SIZE
    horizontal_scale: float = course.HORIZONTAL_SCALE
    vertical_scale: float = course.VERTICAL_SCALE
    #: 边缘留一圈平地的宽度（米）。0 = 不留（`course.heightfield_for` 已经保证跑道不贴边）
    border_width: float = 0.0
    #: Isaac 用不到（那是 HfTerrainBaseCfg 的斜坡生成参数），声明只是为了能接住赋值
    slope_threshold: float = 0.75
