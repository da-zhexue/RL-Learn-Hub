"""地形几何的唯一真源：把 `scene.xml` 里那张桌子搬成纯 numpy。

为什么要有这个文件（而不是直接用 `go2_common/terrain.py`）：那一位顶层 `import mujoco`，
只能在装了 MuJoCo 的进程里用；而 Isaac 侧的地形是一张 **heightfield（高程图）**——
必须在生成 terrain **之前**就算出每个像素的高度，所以几何得能用纯 numpy 重算一遍。

两份实现（`terrain.py` 读编译后的 MjModel / 这里读盒子表）靠 `tests/course_parity.py`
逐点对拍，不靠"看起来一样"。**本文件不 import mujoco / isaaclab**，本机无卡也能 import 和测试。

地形（`scene.xml` 的 worldbody）：

    x=1.2 / 1.6   两个 0.08 m 的槛（坐在地面上）
    x=2.3 ~ 3.4   六级台阶，顶面 0.17 / 0.32 / 0.47 / 0.62 / 0.77 / 0.92 m（埋进地里）
    x=3.14~3.66   顶平台（0.92 m），再往前是悬崖

台阶是**互相重叠**的（`pos_z` 都是 0.02、没有一块"平台"），所以每级**踏面只有约 0.19 m**。
"""

from __future__ import annotations

import numpy as np

TERRAINS = ("flat", "steps", "full")

# 槛（半高 0.04）与台阶（半高 0.15 起）靠半高区分：场景里的方块都没有 name，只能按 size[2] 筛。
# 必须在**缩放之前**用原尺寸筛，否则 scale < 0.5 时矮台阶会被当成槛留下（terrain.py 同款坑）。
THRESHOLD_HALF_Z = 0.1

# scene.xml worldbody 里的 8 个 box，逐行照抄（pos 三个数在前、size 三个数在后，方便对着 xml 看）：
#   (x, y, z,      half_x, half_y, half_z)
BOXES: tuple[tuple[float, float, float, float, float, float], ...] = (
    (1.2, 0.0, 0.04, 0.10, 2.0, 0.04),  # 槛 1，顶面 0.08
    (1.6, 0.0, 0.04, 0.10, 2.0, 0.04),  # 槛 2，顶面 0.08
    (2.3, 0.0, 0.02, 0.20, 2.0, 0.15),  # 第 1 级，顶面 0.17，x∈[2.10, 2.50]
    (2.6, 0.0, 0.02, 0.22, 2.0, 0.30),  # 第 2 级，顶面 0.32，x∈[2.38, 2.82]
    (2.8, 0.0, 0.02, 0.23, 2.0, 0.45),  # 第 3 级，顶面 0.47
    (3.0, 0.0, 0.02, 0.24, 2.0, 0.60),  # 第 4 级，顶面 0.62
    (3.2, 0.0, 0.02, 0.25, 2.0, 0.75),  # 第 5 级，顶面 0.77
    (3.4, 0.0, 0.02, 0.26, 2.0, 0.90),  # 第 6 级 = 顶平台，顶面 0.92，x∈[3.14, 3.66]
)

# 地形高度 ≤ 这个值的方块不当回事（terrain.py 的 TerrainHeight 同款过滤）：
# 极小的 terrain_scale 下槛会矮到 1 cm 以下，落在这一档里。
_MIN_TOP = 0.01

# ------------------------------------------------------------------ patch 布局
#
# 一条跑道放在一个 5×5 m 的 patch 里（Isaac Lab 的 TerrainGenerator 以 patch 为单位铺）。
# 为什么是 5 m：课程地形在 x 方向要 [-1.0, 3.7]（出生点能到 x=-0.3、出界判据 out_x_back=-1.0、
# 顶平台到 x=3.66），y 方向要 [-2, 2]（方块宽度）。5×5 刚好把 [-1.0, 4.0] × [-2.5, 2.5] 装下，
# 剩下的边角是平地——和 MuJoCo 里"方块只有 |y|≤2、外面是无限平地"一致，
# 所以 out_y=2.5 的终止语义不变，相邻跑道的台阶之间也自然隔开 1 m 平地。
PATCH_SIZE = (5.0, 5.0)

# 高程图分辨率：2 cm/像素。x 方向 2 cm 对 0.19 m 的踏面没有影响（台阶立面会被拉成 2 cm 斜面，
# MuJoCo 是垂直面，这点差异只体现在蹭着台阶沿往上爬的接触上）。
HORIZONTAL_SCALE = 0.02
# 高程量化 1 mm。0.7 档的顶面 0.056/0.119/0.224/... 恰好是整毫米，量化误差 ≤0.5 mm。
VERTICAL_SCALE = 0.001

# 课程坐标原点 (x=0, y=0) 在 patch 局部坐标里的位置：patch 的 (0,0) 角对应课程 (-1.0, -2.5)。
COURSE_ORIGIN_PATCH = (1.0, 2.5)

# Isaac Lab 把每个 patch 的**中心**世界坐标存进 `env.scene.env_origins`（terrain_origins 的约定）。
# patch 中心在 patch 局部坐标里是 (2.5, 2.5)，换成课程坐标就是 (2.5, 2.5) - COURSE_ORIGIN_PATCH
# = (1.5, 0.0)。于是：
#
#     course_xy = world_xy - env_origin - COURSE_ORIGIN_FROM_ENV_ORIGIN
#
# 自检：机器人站在 patch 中心时 course_x = 1.5、course_y = 0.0（见文件末尾的 __main__）。
# **这一条是上机第一个要确认的事**（smoke.py --check-terrain 用射线打网格反推），
# 万一那套 Isaac Lab 是"以 patch 角为 origin"，改这一个常量即可。
COURSE_ORIGIN_FROM_ENV_ORIGIN = (-1.5, 0.0)


# ------------------------------------------------------------------ 盒子表


def _kept_boxes(variant: str, scale: float):
    """筛出保留的地形方块并按 scale 压矮，产出 `(x_lo, x_hi, y_lo, y_hi, top, bottom)`。

    逐行复刻 `terrain.build_model()`：先按**原** size[2] 筛（flat 全删 / steps 只留两个槛 /
    full 全留），再把留下的 `size[2]` 和 `pos[2]` 同乘 scale —— x/y 脚印不动，
    所以台阶的踏面深度不变、只是变矮，这正是想要的"变简单"。

    顺序不能反：先缩放再筛的话，`steps` 档在 scale<0.5 时会把台阶也留下来。
    """
    if variant not in TERRAINS:
        raise ValueError(f"variant 只能是 {TERRAINS}，收到 {variant!r}")
    for x, y, z, sx, sy, sz in BOXES:
        if variant == "flat" or (variant == "steps" and sz > THRESHOLD_HALF_Z):
            continue
        if scale != 1.0:
            z *= scale
            sz *= scale
        top = z + sz
        if top <= _MIN_TOP:
            continue
        yield (x - sx, x + sx, y - sy, y + sy, float(top), float(z - sz))


def boxes_for(variant: str = "full", scale: float = 1.0) -> tuple[tuple[float, ...], ...]:
    """按课程阶段筛出地形方块，返回 `(x_lo, x_hi, y_lo, y_hi, top)`，按 x_lo 排序。"""
    out = [b[:5] for b in _kept_boxes(variant, scale)]
    out.sort(key=lambda b: b[0])
    return tuple(out)


def stair_tops_for(variant: str = "full", scale: float = 1.0) -> tuple[float, ...]:
    """台阶各级的顶面高度（槛不算）。

    判据是"底面在不在水平地面以下"：槛坐在地面上（`pos_z == size_z` -> `bottom == 0`），
    台阶是埋进地里的（`pos_z=0.02 < size_z` -> `bottom < 0`）。这个判据对 scale 免疫
    （同乘正数不改变符号），而按顶面高度筛（`top > 0.1` 之类）在课程压矮后会把台阶全漏掉。
    阈值和 `round(..., 4)` 都跟 `terrain.TerrainHeight` 一字不差。
    """
    return tuple(sorted({round(b[4], 4) for b in _kept_boxes(variant, scale) if b[5] < -1e-6}))


class CourseHeight:
    """地形高度查询：`terrain.TerrainHeight` 的纯 numpy 版本，语义逐条对齐。

    `top(x, y)` 取覆盖该点的**最高**一块方块（没有方块盖住就是地面 z=0）；x/y 可以是标量，
    也可以是任意形状的数组（一次算 N 个环境）。奖励/终止/出生点都从这里取"当地地形高度"，
    绝不硬编码方块坐标——这样课程变体自动生效。
    """

    def __init__(self, variant: str = "full", scale: float = 1.0):
        self.variant = variant
        self.scale = float(scale)
        self.boxes = boxes_for(variant, scale)
        self.stair_tops = stair_tops_for(variant, scale)

    # -------------------------------------------------------------- 查询

    def top(self, x, y=0.0):
        """点 (x, y) 处的地形高度。标量进标量出，数组进数组出。"""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        h = np.zeros(np.broadcast(x, y).shape, dtype=np.float64)
        for x_lo, x_hi, y_lo, y_hi, top in self.boxes:
            hit = (x_lo <= x) & (x <= x_hi) & (y_lo <= y) & (y <= y_hi)
            h = np.where(hit, np.maximum(h, top), h)
        return h if h.ndim else float(h)

    def top_footprint(self, x, y=0.0, half_x: float = 0.24, half_y: float = 0.12):
        """身体脚印范围内的**最大**地形高度。出生点必须用这个，不能用中心点的高度。

        狗身长约 0.4 m，横跨高度不连续处（槛的棱、台阶的立面）时中心还在平地、
        前脚已探到台阶上方。9×3 个采样点和默认半长宽与 `terrain.TerrainHeight` 完全一致
        （按偏航 ±0.20 rad 时的投影取的）。
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xs = np.linspace(x - half_x, x + half_x, 9)
        ys = np.stack([y - half_y, y, y + half_y])
        # xs[:, None] 是 (9, 1, ...)、ys[None, :] 是 (1, 3, ...)，广播成 (9, 3, ...)，
        # 对全部 27 个采样点取 max。标量进出标量、数组进出数组。
        h = self.top(xs[:, None], ys[None, :])
        return float(h.max()) if h.ndim == 2 else np.max(h, axis=(0, 1))

    def level(self, x, y=0.0):
        """已经站上第几级台阶，0 表示还在平地/槛上，满级 = len(stair_tops)。"""
        h = np.asarray(self.top(x, y), dtype=np.float64)
        if not self.stair_tops:
            return np.zeros(h.shape, dtype=np.int64) if h.ndim else 0
        tops = np.asarray(self.stair_tops, dtype=np.float64)
        # 写成 `t <= h + 1e-6` 而不是 `h >= t - 1e-6`：和 terrain.py 一字不差，
        # 免得在边界值上因为浮点舍入方向不同而对不上。
        n = np.sum(tops <= h[..., None] + 1e-6, axis=-1)
        return n if h.ndim else int(n)


# ------------------------------------------------------------------ heightfield


def heightfield_for(variant: str = "full", scale: float = 1.0,
                    size: tuple[float, float] = PATCH_SIZE,
                    horizontal_scale: float = HORIZONTAL_SCALE) -> np.ndarray:
    """生成一个 patch 的高程图（**单位：米**，float64）。

    形状 `(nx, ny)`，`arr[i, j]` 对应 patch 局部坐标 `(i*h, j*h)`；把 patch 局部坐标
    减去 `COURSE_ORIGIN_PATCH` 就是课程坐标。Isaac Lab 的 `TerrainGenerator` 会
    按 `vertical_scale` 量化成 int16 后拼进整张地形。

    每个 patch = `int(size/h)` 个像素（Isaac Lab 的口径），patch 之间在缓冲区里是**连续**的，
    所以跑道的边界不需要额外留缝。
    """
    nx = int(round(size[0] / horizontal_scale))
    ny = int(round(size[1] / horizontal_scale))
    px = np.arange(nx, dtype=np.float64) * horizontal_scale - COURSE_ORIGIN_PATCH[0]
    py = np.arange(ny, dtype=np.float64) * horizontal_scale - COURSE_ORIGIN_PATCH[1]
    return np.asarray(CourseHeight(variant, scale).top(px[:, None], py[None, :]))


def world_to_course(xy_world, env_origins):
    """世界坐标 -> 课程坐标。

    `env_origins` 是 `env.scene.env_origins`（每个环境所属 patch 中心的世界坐标）。
    x/y 都支持 `(N, 2)` 或广播。
    """
    return (np.asarray(xy_world, dtype=np.float64)
            - np.asarray(env_origins, dtype=np.float64)
            - np.asarray(COURSE_ORIGIN_FROM_ENV_ORIGIN, dtype=np.float64))


# ------------------------------------------------------------------ 自检

if __name__ == "__main__":
    # 这个文件自带一份"打印地形"的自检：它不依赖 mujoco，所以本机（无卡）随时能看。
    print(f"patch {PATCH_SIZE[0]}×{PATCH_SIZE[1]} m，{HORIZONTAL_SCALE*100:g} cm/像素，"
          f"量化 {VERTICAL_SCALE*1000:g} mm")
    print(f"课程原点在 patch 局部坐标 {COURSE_ORIGIN_PATCH}，"
          f"patch 中心对应课程坐标 "
          f"{(-COURSE_ORIGIN_FROM_ENV_ORIGIN[0], -COURSE_ORIGIN_FROM_ENV_ORIGIN[1])}")
    for variant in TERRAINS:
        for scale in (1.0, 0.7):
            t = CourseHeight(variant, scale)
            tops = [f"{b[4]:.3f}" for b in t.boxes]
            print(f"\n{variant:5s} scale={scale:g}: 方块 {len(t.boxes)} 个 "
                  f"顶面 {tops}")
            print(f"                台阶顶面 {t.stair_tops}")
    print("\n沿 x 走一条线（y=0，scale=1.0）:")
    t = CourseHeight("full")
    for x in (-1.0, -0.3, 0.0, 1.0, 1.15, 1.2, 1.5, 1.6, 2.0, 2.3, 3.0, 3.3, 3.4, 3.66, 3.8):
        print(f"  x={x:+.2f}  top={t.top(x, 0.0):.3f}  "
              f"level={t.level(x, 0.0)}  footprint={t.top_footprint(x, 0.0):.3f}")
    hf = heightfield_for("full", 1.0)
    print(f"\n高程图 {hf.shape}，max={hf.max():.3f} m，非零点 {int((hf > 0).sum())}")
