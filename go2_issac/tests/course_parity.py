"""把 `course.py`（纯 numpy）和真 MuJoCo 地形逐点对拍。

    python3 -m go2_issac.tests.course_parity

这是"本机没有显卡也能证明 Isaac 侧地形是对的"那一步：`course.py` 是 Isaac 侧
heightfield 的唯一几何来源，而 `go2_common/terrain.py` 是 MuJoCo 侧训练/回放时用的那一份。
两边只要有一个数不一样，上机之后就是"狗在 Isaac 里踩的高度和 MuJoCo 对不上"这种
极难查的错，所以这里**要求逐位相等**（不是近似），并且把边界点单独拿出来测。

为什么能做到逐位相等：两份实现读的是同一批十进制字面量（scene.xml 的 pos/size），
浮点解析和 `*= scale` 的运算顺序也一样，比较也用的是同一串表达式。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from go2_common import terrain as mj_terrain  # noqa: E402
from go2_common.config import DEFAULT_SCENE  # noqa: E402
from go2_issac import course  # noqa: E402

# 课程档位：0.7 / 1.0 是实际用到的两级（flat 那级 scale 无意义但也要测）。
CASES = tuple(
    (variant, scale) for variant in course.TERRAINS for scale in (0.7, 1.0)
)

_failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        _failures.append(msg)
        print(f"  FAIL {msg}")


def _report_diff(tag: str, got, want, tol: float = 0.0) -> None:
    """收集不一致的点，只打印前几例，避免刷屏。"""
    got = np.asarray(got, dtype=np.float64).ravel()
    want = np.asarray(want, dtype=np.float64).ravel()
    bad = np.nonzero(np.abs(got - want) > tol)[0]
    if bad.size:
        worst = bad[np.argmax(np.abs(got[bad] - want[bad]))]
        check(False, f"{tag}: {bad.size}/{got.size} 个点不一致，"
                     f"最大差 {abs(got[worst] - want[worst]):.3e} @ idx {worst} "
                     f"(course={got[worst]!r} mujoco={want[worst]!r})")


def check_box_table(variant: str, scale: float, th, ch) -> None:
    """盒子表与台阶顶面逐位相等（表抄错一个数，这里立刻炸）。"""
    check(len(th.boxes) == len(ch.boxes),
          f"{variant}/{scale}: 方块个数 {len(ch.boxes)} != MuJoCo {len(th.boxes)}")
    if len(th.boxes) == len(ch.boxes):
        _report_diff(f"{variant}/{scale}: 方块表", ch.boxes, th.boxes)
    check(list(th.stair_tops) == list(ch.stair_tops),
          f"{variant}/{scale}: 台阶顶面 {ch.stair_tops} != MuJoCo {th.stair_tops}")


def check_grid(variant: str, scale: float, th, ch) -> None:
    """均匀网格 + 边界点上，top / level / top_footprint 与 MuJoCo 完全一致。"""
    xs = np.arange(-2.5, 4.51, 0.02)
    ys = np.array([-2.2, -2.0, -1.0, 0.0, 1.0, 2.0, 2.2])

    got = np.asarray(ch.top(xs[:, None], ys[None, :]))
    want = np.array([[th.top(float(x), float(y)) for y in ys] for x in xs])
    _report_diff(f"{variant}/{scale}: top", got, want)

    # level 只在有台阶的地形上非零，但还是全测（flat 两侧都该恒为 0）
    got_l = np.asarray(ch.level(xs[:, None], ys[None, :]))
    want_l = np.array([[th.level(float(x), float(y)) for y in ys] for x in xs])
    check(bool(np.array_equal(got_l, want_l)),
          f"{variant}/{scale}: level 有 {int((got_l != want_l).sum())} 个点不一致")

    got_f = np.asarray(ch.top_footprint(xs[:, None], ys[None, :]))
    want_f = np.array([[th.top_footprint(float(x), float(y)) for y in ys] for x in xs])
    _report_diff(f"{variant}/{scale}: top_footprint", got_f, want_f)

    # 边界点：每个盒子的 x/y 边界 ± 几个极小偏移
    eps = np.array([0.0, 1e-9, -1e-9, 1e-6, -1e-6])
    ex = np.concatenate([np.concatenate([np.asarray([b[0], b[1]]) + e for e in eps])
                         for b in th.boxes]) if th.boxes else np.array([0.0])
    ey = np.concatenate([np.concatenate([np.asarray([b[2], b[3]]) + e for e in eps])
                         for b in th.boxes]) if th.boxes else np.array([0.0])
    got_e = np.asarray(ch.top(ex[:, None], ey[None, :]))
    want_e = np.array([[th.top(float(x), float(y)) for y in ey] for x in ex])
    _report_diff(f"{variant}/{scale}: top@边界", got_e, want_e)
    got_ef = np.asarray(ch.top_footprint(ex[:, None], ey[None, :]))
    want_ef = np.array([[th.top_footprint(float(x), float(y)) for y in ey] for x in ex])
    _report_diff(f"{variant}/{scale}: top_footprint@边界", got_ef, want_ef)


def check_heightfield(variant: str, scale: float, ch) -> None:
    """高程图：形状、覆盖范围、峰值。防的是 patch 原点符号写反这类错。"""
    hf = course.heightfield_for(variant, scale)
    nx = int(round(course.PATCH_SIZE[0] / course.HORIZONTAL_SCALE))
    check(hf.shape == (nx, nx), f"{variant}/{scale}: 高程图形状 {hf.shape} != {(nx, nx)}")
    check(hf.dtype == np.float64, f"{variant}/{scale}: 高程图 dtype {hf.dtype} != float64")

    px = np.arange(nx) * course.HORIZONTAL_SCALE - course.COURSE_ORIGIN_PATCH[0]
    py = np.arange(nx) * course.HORIZONTAL_SCALE - course.COURSE_ORIGIN_PATCH[1]

    if ch.boxes:
        top_max = max(b[4] for b in ch.boxes)
        check(abs(hf.max() - top_max) < 1e-12,
              f"{variant}/{scale}: 高程图峰值 {hf.max()} != 方块最高顶面 {top_max}")
        # 抬高区域必须落在所有方块脚印的并集里（±1 像素），否则就是 patch 原点算错了
        i, j = np.nonzero(hf > 0)
        x_lo = min(b[0] for b in ch.boxes) - course.HORIZONTAL_SCALE
        x_hi = max(b[1] for b in ch.boxes) + course.HORIZONTAL_SCALE
        y_lo = min(b[2] for b in ch.boxes) - course.HORIZONTAL_SCALE
        y_hi = max(b[3] for b in ch.boxes) + course.HORIZONTAL_SCALE
        check(x_lo <= px[i].min() and px[i].max() <= x_hi,
              f"{variant}/{scale}: 高程图 x 范围 [{px[i].min():.3f}, {px[i].max():.3f}] "
              f"超出方块脚印 [{x_lo:.3f}, {x_hi:.3f}]")
        check(y_lo <= py[j].min() and py[j].max() <= y_hi,
              f"{variant}/{scale}: 高程图 y 范围 [{py[j].min():.3f}, {py[j].max():.3f}] "
              f"超出方块脚印 [{y_lo:.3f}, {y_hi:.3f}]")
    else:
        check(float(hf.max()) == 0.0, f"{variant}/{scale}: flat 的高程图应该恒为 0")

    # 出生点、终点这些真会用到的位置上，高程图（带量化）和平地几何一致到 1 个 vertical_scale
    for x in (-0.3, 0.0, 1.2, 2.3, 3.3):
        i = int(round((x + course.COURSE_ORIGIN_PATCH[0]) / course.HORIZONTAL_SCALE))
        if 0 <= i < nx:
            check(abs(hf[i, nx // 2] - ch.top(x, 0.0)) <= course.VERTICAL_SCALE,
                  f"{variant}/{scale}: 高程图 x={x} 处 {hf[i, nx // 2]} "
                  f"与几何 {ch.top(x, 0.0)} 差超过 1 mm")


def check_patch_mapping() -> None:
    """patch/课程坐标映射：上机第一个要确认的常量，这里先把它的语义钉死。"""
    env_origin = np.array([12.5, -7.5])  # 随便一个 patch 中心
    # 环境原点处应该是课程坐标 (1.5, 0)：patch 中心在 patch 局部是 (2.5, 2.5)，
    # 减去课程原点在 patch 里的偏移 (1.0, 2.5) -> (1.5, 0.0)
    got = course.world_to_course(env_origin, env_origin)
    check(np.allclose(got, [1.5, 0.0], atol=1e-12),
          f"patch 中心应映射到课程 (1.5, 0)，实际 {got}")
    # 出生点 x∈[-0.3, 0.3] 落在 patch 内、且离 patch 边界有富余（要放得下 0.24 m 的脚印）
    for x in (-1.0, -0.3, 0.3, 3.66, 4.0):
        patch_local = x + course.COURSE_ORIGIN_PATCH[0]
        check(0.0 <= patch_local <= course.PATCH_SIZE[0],
              f"课程 x={x} 落在 patch 外（patch 局部 x={patch_local}）")


def main() -> int:
    print(f"对拍 course.py <-> go2_common/terrain.py（scene: {pathlib.Path(DEFAULT_SCENE).name}）")
    for variant, scale in CASES:
        model = mj_terrain.build_model(DEFAULT_SCENE, variant, scale)
        th = mj_terrain.TerrainHeight(model)
        ch = course.CourseHeight(variant, scale)
        print(f"\n[{variant} scale={scale:g}] 方块 {len(ch.boxes)} 个，"
              f"台阶顶面 {len(ch.stair_tops)} 级")
        check_box_table(variant, scale, th, ch)
        if not _failures:
            check_grid(variant, scale, th, ch)
        check_heightfield(variant, scale, ch)
        del model

    print("\n[patch 映射]")
    check_patch_mapping()

    print("\n" + ("全部通过" if not _failures else f"{len(_failures)} 项失败:"))
    for f in _failures:
        print(f"  - {f}")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
