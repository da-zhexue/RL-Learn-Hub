"""地形：课程变体、地形高度查询、成功/摔倒判据。

地形本身**不新建 xml**，直接用 unitree_mujoco 自带的那份
`unitree_robots/go2/scene.xml` —— 它正好就是"平地 -> 两个 0.08 m 的槛 -> 六级 0.15 m 台阶"：

    x=1.2 / 1.6 : box size[2]=0.04 -> 高 0.08 m 的槛
    x=2.3 ~ 3.4 : box size[2]=0.15 ~ 0.9，顶面 0.17 / 0.32 / 0.47 / 0.62 / 0.77 / 0.92 m
                  再往前 x>3.66 是 0.92 m 的悬崖

而且 simulate_python/config.py 加载的就是这个文件，所以训练地形 == 用户已经在跑的仿真地形。

课程变体用 MuJoCo 的 MjSpec 在内存里删几何体得到，不写临时文件、不改原 xml。
"""

from __future__ import annotations

import mujoco
import numpy as np

TERRAINS = ("flat", "steps", "full")

# 槛（0.08 m）与台阶（0.15 m 起）靠半高区分：地形方块都没有 name，只能按 size[2] 筛。
_THRESHOLD_HALF_Z = 0.1


def build_model(scene: str, terrain: str = "full", scale: float = 1.0) -> mujoco.MjModel:
    """按课程阶段编译 MuJoCo 模型。

    flat  : 删掉全部地形方块，只剩地面
    steps : 只留两个 0.08 m 的槛
    full  : 原样

    scale 把剩下的方块**等比压矮**（size[2] 和 pos[2] 同乘），x/y 脚印不动。
    场景里的方块都是埋在土里的（槛 pos_z=0.04/size_z=0.04，台阶 pos_z=0.02），
    所以这样缩放后每个方块仍然是"从地面长到 scale 倍原高度"，踏面深度也保持不变。
    """
    if terrain not in TERRAINS:
        raise ValueError(f"terrain 只能是 {TERRAINS}，收到 {terrain!r}")

    spec = mujoco.MjSpec.from_file(scene)
    for geom in list(spec.worldbody.geoms):
        if geom.type != mujoco.mjtGeom.mjGEOM_BOX:
            continue  # 地面是 plane，永远保留
        # 必须在缩放**之前**按原尺寸筛，否则 scale<0.5 时矮台阶会被当成槛留下来
        if terrain == "flat" or (terrain == "steps" and geom.size[2] > _THRESHOLD_HALF_Z):
            spec.delete(geom)
            continue
        if scale != 1.0:
            geom.size[2] *= scale
            geom.pos[2] *= scale
    return spec.compile()


class TerrainHeight:
    """地形高度查询表。

    构造时把 worldbody 里的方块缓存成 [x_lo, x_hi, y_lo, y_hi, top]，
    `top(x, y)` 返回该点最高的一块（没有方块盖住就是地面 z=0）。
    奖励和终止判据都从这里取"当地地形高度"，绝不硬编码方块坐标
    —— 这样课程变体（删掉方块）自动生效。
    """

    def __init__(self, model: mujoco.MjModel):
        self.boxes: list[tuple[float, float, float, float, float]] = []
        stair_tops: set[float] = set()
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != 0:
                continue  # 只认 worldbody
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            cx, cy, cz = model.geom_pos[g]
            sx, sy, sz = model.geom_size[g]
            top = float(cz + sz)
            if top <= 0.01:
                continue
            self.boxes.append(
                (float(cx - sx), float(cx + sx), float(cy - sy), float(cy + sy), top)
            )
            # 台阶和槛靠"底面在不在水平地面"区分：槛坐在地面上（pos_z == size_z），
            # 台阶是埋进地里的（pos_z=0.02 < size_z）。这个判据对 terrain_scale 缩放免疫，
            # 而按顶面高度筛（旧的 size[2] > 0.1 写法）在课程压矮后会把台阶全漏掉。
            if cz - sz < -1e-6:
                stair_tops.add(round(top, 4))
        self.boxes.sort(key=lambda b: b[0])

        # 台阶顶面，用来统计"爬到了第几级"
        self.stair_tops = sorted(stair_tops)

    def top(self, x: float, y: float = 0.0) -> float:
        """点 (x, y) 处的地形高度。"""
        h = 0.0
        for x_lo, x_hi, y_lo, y_hi, top in self.boxes:
            if x_lo <= x <= x_hi and y_lo <= y <= y_hi and top > h:
                h = top
        return h

    def top_footprint(self, x: float, y: float = 0.0,
                      half_x: float = 0.24, half_y: float = 0.12) -> float:
        """身体脚印范围内的**最大**地形高度。出生点必须用这个，不能用中心点的高度。

        狗身长约 0.4 m（前脚中心在 base 前方 ~0.16，后脚在后方 ~0.16，加上偏航
        最大 ±0.20 rad，x 方向投影到约 ±0.19）。一旦横跨高度不连续处——槛的棱、
        台阶的立面——中心还在平地上，前脚已经探到台阶上方，按中心高度出生就会把
        前脚**直接生成在台阶内部**。实测 scale=0.62 时 24 个出生点里 17 个有脚被埋，
        最深 0.0915 m。穿模会触发巨大的接触力，狗在头 0.3 秒（17~31 步）就被弹翻：
        那次 diag why 的 8 个回合里有 3 个是这么死的，而且 x 只有 1.3~2.1、还在平地上。

        穿深≈那条棱的高度差，所以**地形越高越严重**（scale 0.35 时只有 3 cm，
        到 1.0 就是 15 cm）——这正是"课程越往上、训练反而越差"的根源，
        和 reward、学习率、action_scale 都无关。
        """
        ys = (y - half_y, y, y + half_y)
        return max(
            self.top(float(xi), float(yi))
            for xi in np.linspace(x - half_x, x + half_x, 9)
            for yi in ys
        )

    def level(self, x: float, y: float = 0.0) -> int:
        """已经站上第几级台阶，0 表示还在平地/槛上，满级 = len(stair_tops)。"""
        h = self.top(x, y)
        return sum(1 for t in self.stair_tops if t <= h + 1e-6)


# ------------------------------------------------------------------ 判据


def is_success(cfg, base_pos) -> bool:
    """到达顶平台。高度条件不能省：否则在平地往前走同样能白拿这个奖励。"""
    x, y, z = float(base_pos[0]), float(base_pos[1]), float(base_pos[2])
    return x >= cfg.goal_x and z >= cfg.goal_z and abs(y) <= cfg.goal_y


def is_fallen(cfg, terrain: TerrainHeight, base_pos) -> bool:
    """机身贴着当地地形了（摔倒）。

    用"相对当地地形"而不是绝对高度：在顶平台上 base 有 1.19 m，在平地上只有 0.27 m，
    绝对阈值在台阶上永远触发不了。
    """
    x, y, z = float(base_pos[0]), float(base_pos[1]), float(base_pos[2])
    return (z - terrain.top(x, y)) < cfg.fall_clearance


def is_out_of_course(cfg, base_pos) -> bool:
    """横向跑出地形、或者倒退太多。地形只有 y∈[-2,2] 宽，外面全是平地。"""
    x, y = float(base_pos[0]), float(base_pos[1])
    return abs(y) > cfg.out_y or x < cfg.out_x_back


def is_flipped(cfg, rpy) -> bool:
    """机身翻得太厉害。"""
    return bool(abs(rpy[0]) > cfg.flip_rad or abs(rpy[1]) > cfg.flip_rad)
