"""把 `course.py` 的盒子表变成 Isaac Lab 的 sub-terrain（一块 5×5 m 的三角网 patch）。

`difficulty`（0~1）直接当 `terrain_scale` 用——我们的"难度"就是 `EnvCfg.terrain_scale`
（0.3~1.0，只压地形高度），所以 `difficulty_range=(s, s)` 固定单一档位，
`curriculum=False`。想要自动加难就把 `curriculum` 打开、`difficulty_range` 放宽
（见 RUNBOOK 的"可选后续"），函数本身不用改。

------------------------------------------------------------------------------
Isaac Lab 要的协议（`SubTerrainBaseCfg.function`，2.3.0）
------------------------------------------------------------------------------

    function(difficulty: float, cfg) -> (list[trimesh.Trimesh], origin: np.ndarray(3))

两条容易写错的约定：

**1. mesh 要铺满 patch 局部 (0, 0) → (size_x, size_y)，原点在左下角，不是中心。**
   基类 docstring 原话："the terrain must extend from :math:`(0, 0)` to
   :math:`(size[0], size[1])`"。返回的 `origin` 才是 **patch 中心**，即
   `[size_x/2, size_y/2, origin_z]`。`TerrainGenerator` 随后做两次平移：
   `_get_terrain_mesh` 把 mesh 挪 `-size/2`，`_add_sub_terrain` 把 origin 挪
   `+((row+0.5)*size_x, (col+0.5)*size_y, 0)`。结果 `terrain_origins[row, col]`
   = 这块 patch **中心的世界坐标**——`course.COURSE_ORIGIN_FROM_ENV_ORIGIN`
   正是照这个口径推的。把 origin 写成 `(-size/2, -size/2, 0)` 会让整套课程坐标
   映射整体平移半格，而且是**静默**的。

**2. 顶点数比像素数多 1。** `course.heightfield_for` 给 `int(size/h)=250` 个采样点、
   只铺到 x=4.98，而 mesh 要在 x=5.00 上有顶点，相邻 patch 才不留缝（差 2 cm，
   狗一脚踩空）。所以补到 251 个顶点。官方那个 `height_field_to_mesh` 装饰器
   **不能用**：它无条件垫 `border_pixels = int(border_width/h) + 1 >= 1` 个像素的
   边框（`border_width=0` 时也垫），还会临时改 `cfg.size` 再改回来——正好把
   课程坐标整体推偏一格。这里直接自己拼，只用底层的 `convert_height_field_to_mesh`。

补出来的那一行/一列落在 patch 最外圈（课程坐标 x=4.00、y=±2.48），而方块只到
课程 x=3.66 / |y|≤2，那儿一定是平地，所以 `mode="edge"` 复制一格是精确值不是近似。
本机验过：`heightfield_for` 的末两行/末两列在 3 种 variant × 3 档 scale 下恒为 0。

------------------------------------------------------------------------------
高度只有 `course.py` 一个真源
------------------------------------------------------------------------------
`course.heightfield_for` 是对着 MuJoCo 逐位对拍过的，这里只做两件事：量化成 int16、
补到 251 个顶点。

**量化**：`vertical_scale=0.001`（1 mm）。0.7 档的台阶顶面 0.056/0.119/0.224/...
全是整毫米，量化误差 ≤0.5 mm；再细就要把 int16 的行程浪费掉。`horizontal_scale=0.02`
（2 cm/像素）对 0.19 m 的踏面没影响，台阶立面被拉成 2 cm 的斜面——MuJoCo 是垂直面，
这点差异只体现在"蹭着台阶沿往上爬"的接触上，属于跨仿真器消不掉的那一类。

**`slope_threshold` 默认 `None`（不生效）。** Isaac 用它把陡面拉成垂直面（把立面
**底**顶点往前挪一格），代价是网格顶点不再等于 `heightfield_for` 的采样点，
`tests/course_parity.py` 对拍的就是那套采样点。想要"立面真的垂直"（更贴 MuJoCo）
就把它设成 0.75——两种都只和 MuJoCo 差 2 cm，选哪个都不算错，默认取"能和
`course.py` 逐点对上"的那一种。
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import trimesh

from go2_issac import course

from isaaclab.utils import configclass

try:  # Isaac Lab 2.x
    from isaaclab.terrains import SubTerrainBaseCfg
except ImportError:  # Isaac Lab 1.x
    from isaaclab.terrain import SubTerrainBaseCfg

try:  # Isaac Lab 2.x
    from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
except ImportError:  # Isaac Lab 1.x
    from isaaclab.terrain.height_field.utils import convert_height_field_to_mesh


def course_terrain(difficulty: float,
                   cfg: "CourseTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """一个 patch = 一条跑道：按课程档位把槛和台阶摆进 5×5 m 的高程图。

    `difficulty` 就是 `terrain_scale`（理由见模块 docstring 开头）。
    """
    h, v = cfg.horizontal_scale, cfg.vertical_scale

    # 顶点数 = 像素数 + 1（理由见模块 docstring 第 2 条）
    nx = int(round(cfg.size[0] / h)) + 1
    ny = int(round(cfg.size[1] / h)) + 1
    hf = course.heightfield_for(cfg.variant, float(difficulty), size=cfg.size,
                                horizontal_scale=h)
    if hf.shape != (nx - 1, ny - 1):
        # 尺寸对不上就别往下走了：patch 大小/采样间隔一变，这里的假设就失效了，
        # 硬凑出来的 mesh 会以"地形莫名其妙错位"的形式出现，比直接报错难查得多。
        raise ValueError(f"heightfield_for 给了 {hf.shape}，按 size/h 应该是 "
                         f"{(nx - 1, ny - 1)}")
    hf = np.pad(hf, ((0, 1), (0, 1)), mode="edge")

    if cfg.border_width > 0:
        # 边缘留一圈平地：patch 是紧挨着拼的，相邻跑道的台阶顶到边界时狗一走出去
        # 就踩到隔壁的高度上。`course.heightfield_for` 已经保证跑道不贴边
        # （方块只占 patch 局部 x∈[2.10, 4.66]、y∈[0.5, 4.5]），这里是双保险。
        b = max(1, int(round(cfg.border_width / h)))
        hf[:b, :] = 0.0
        hf[-b:, :] = 0.0
        hf[:, :b] = 0.0
        hf[:, -b:] = 0.0

    q = np.rint(hf / v).astype(np.int16)
    vertices, triangles = convert_height_field_to_mesh(q, h, v, cfg.slope_threshold)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    # origin 的 z 用官方那条规则：patch 正中 2×2 m 里的最高点
    # （官方 `height_field_to_mesh` 装饰器里同样是这两行）。狗出生在这个高度上，
    # 往高处出生时不至于一上来就陷在台阶里。
    x1, x2 = int((cfg.size[0] * 0.5 - 1.0) / h), int((cfg.size[0] * 0.5 + 1.0) / h)
    y1, y2 = int((cfg.size[1] * 0.5 - 1.0) / h), int((cfg.size[1] * 0.5 + 1.0) / h)
    origin_z = float(np.max(q[x1:x2, y1:y2])) * v

    # origin 是 patch 中心（局部坐标），不是左下角（理由见模块 docstring 第 1 条）
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], origin_z], dtype=np.float64)
    return [mesh], origin


@configclass
class CourseTerrainCfg(SubTerrainBaseCfg):
    """`env_cfg.py` 里挂在 `sub_terrains={"course": CourseTerrainCfg(...)}`。

    **`@configclass` 不能少**：它才是那个把本类新加的字段（`variant` 等）注册进
    dataclass 的装饰器。光继承一个 `@configclass` 基类，子类拿到的是**基类那份**
    `__init__`，多出来的字段全是"unexpected keyword argument"——
    这一条就是在 `CourseTerrainCfg(variant=...)` 上撞出来的。

    这几个字段全部显式声明，但**理由分两种**：
    - `size`：`TerrainGenerator.__init__` 会无条件地写 `sub_cfg.size = self.cfg.size`，
      声明出来两边同值，不声明就是它说了算（结果一样，但别指望能在这层定制）。
    - `horizontal_scale` / `vertical_scale`：**只有 `HfTerrainBaseCfg` 的子类才会被
      `TerrainGenerator` 覆盖**（那句赋值在 `isinstance(sub_cfg, HfTerrainBaseCfg)` 里面），
      本类不是，所以这里的值就是最终值。必须和 `TerrainGeneratorCfg` 上的一致，
      否则高程图的采样间隔对不上。
    """

    function: Callable = course_terrain

    #: flat / steps / full，语义见 `course.py`
    variant: str = "full"

    size: tuple[float, float] = course.PATCH_SIZE
    horizontal_scale: float = course.HORIZONTAL_SCALE
    vertical_scale: float = course.VERTICAL_SCALE
    #: 边缘留一圈平地的宽度（米）。0 = 不留（`course.heightfield_for` 已经保证跑道不贴边）
    border_width: float = 0.0
    #: 陡面转垂直面的阈值；`None` = 不转，保持网格顶点 = `course.py` 的采样点（理由见模块 docstring）
    slope_threshold: float | None = None
