"""MJCF → USD：把仓库里自己的 `go2.xml` 转成 Isaac Sim 能加载的 USD。

    python3 go2_issac/convert_assets.py                     # 转狗
    python3 go2_issac/convert_assets.py --scene             # 顺便把 scene.xml（地形）也转一份，纯学习用
    python3 go2_issac/convert_assets.py --strip-collision   # 默认开；见下面第 2 条

**只能在有 NVIDIA 显卡的机器上跑**（要 Kit）。跑之前先 `python3 go2_issac/smoke.py --check-env`。

用官方入口，不自己写转换器：`isaaclab.sim.converters.MjcfConverter`，它底层调的是
`isaacsim.asset.importer.mjcf`（Isaac Sim 5.0 起换成 `mujoco-usd-converter` 的
`MJCFImporter(config).import_mjcf()`）。这里只传两代 API 都有的四个字段，避开版本差异
（见 RUNBOOK.md 的版本差异表）。

------------------------------------------------------------------------------
三条必须记住的事（错了会以"训不动"的形式出现，极难查）
------------------------------------------------------------------------------

**1. 增益不在这步烤进 USD。** MJCF importer **没有** `default_drive_stiffness`（那是 URDF
   转换器才有的字段）。`go2.xml` 里的 `<motor>` 会变成 USD 的 DriveAPI，但 `kp`/`kd` 是
   Isaac Lab 在实例化时用 `ImplicitActuatorCfg` 写回去的：
       stiffness=60.0, damping=3.6, friction=0.2, effort_limit_sim=…
   **`damping` 是 3.6 不是 3.5**：MuJoCo 侧 12 个铰链自己带 `damping=0.1`，而 kd=3.5 是
   纯 PD 的那一半，Isaac 的隐式执行器只有 `damping` 这一个旋钮，两个要加起来
   （推导见 AGENT.md §5-G，实测依据在 `assets/go2_reference.json` 的 joints[].damping）。
   所以**别在这里找 kp/kd 的开关**——找不到。

**2. `contype=0 conaffinity=0` 的 33 个视觉网格不能当碰撞体。** `go2.xml` 里腿和机身各有
   一份高质量 STL，全部设成不参与碰撞（真正的碰撞几何是另外 23 个 box/cylinder/sphere）。
   转换器**可能**把网格也建成碰撞体——那样狗会被自己的网格卡住、`base_link` 和腿互相顶，
   表现为关节疯狂抖。所以默认走 `--strip-collision`，把 mesh 上的 CollisionAPI 摘掉。
   转完拿 `assets/go2_reference.json` 对：狗的碰撞几何应该正好 **23** 个。

**3. 不传 `fix_base` / `make_instanceable` / `self_collision`。** `go2.xml` 有 freejoint，
   保持默认就是浮动基座（我们要的）。`make_instanceable` 从 Isaac Sim 5.0 起被忽略；
   `fix_base`/`import_sites` 在老版本有、新版移除；`self_collision` 改名成
   `allow_self_collision`。跨版本能用的就那四个字段，别贪。

转换是**幂等**的：`force_usd_conversion=True` 每次都重转（USD 是派生物，不该手工改）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
GO2_XML = REPO / "unitree_mujoco" / "unitree_robots" / "go2" / "go2.xml"
SCENE_XML = REPO / "unitree_mujoco" / "unitree_robots" / "go2" / "scene.xml"
ASSETS = HERE / "assets"


def _launch_kit(headless: bool = True):
    """先引导 Kit，再 import isaaclab —— 顺序不能反。

    `isaaclab` 的很多模块在 import 时就会去问 Kit 要 stage/物理接口，没有 app 就报一堆
    "omni.* not found"。`AppLauncher` 必须是最先执行的那一步。
    """
    from isaaclab.app import AppLauncher

    ap = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(ap)
    args, _ = ap.parse_known_args()
    args.headless = headless
    app_launcher = AppLauncher(args)
    return app_launcher.app


def convert(asset_path: pathlib.Path, usd_file_name: str, out_dir: pathlib.Path) -> str:
    """跑一次转换，返回 USD 路径。"""
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    cfg = MjcfConverterCfg(
        # 只用跨版本稳定的四个字段（理由见模块 docstring 第 3 条）
        asset_path=str(asset_path),
        usd_dir=str(out_dir),
        usd_file_name=usd_file_name,
        force_usd_conversion=True,
    )
    converter = MjcfConverter(cfg)
    return converter.usd_path


def write_manifest() -> dict:
    """把"这份 USD 是在什么环境下转出来的"记下来。

    转换器跨 Isaac Sim 版本改过好几轮（模块 docstring 第 3 条那张表），同一份 go2.xml 在
    5.1 和 6.x 上转出来的 USD 不保证一样。这份 manifest 是以后"为什么当时是好的"的唯一线索，
    也是 RUNBOOK 里让人抄版本号的地方。
    """
    import subprocess

    info: dict = {"repo_commit": "unknown", "isaaclab": "unknown", "isaacsim": "unknown"}
    try:
        info["repo_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
            timeout=10).stdout.strip() or "unknown"
        info["repo_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True,
            timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass

    for mod in ("isaaclab", "isaacsim"):
        try:
            m = __import__(mod)
            info[mod] = str(getattr(m, "__version__", "unknown"))
        except ImportError:
            pass

    try:
        from pxr import Usd
        info["usd"] = str(Usd.GetVersion())
    except ImportError:
        pass

    info["mjcf_source"] = str(GO2_XML)
    info["mjcf_sha256"] = json.loads((ASSETS / "go2_reference.json").read_text())["source_sha256"]

    out = ASSETS / "manifest.json"
    out.write_text(json.dumps(info, indent=1) + "\n")
    print(f"已写出 {out}: " + "  ".join(f"{k}={v}" for k, v in info.items()
                                        if k in ("repo_commit", "isaaclab", "isaacsim")))
    return info


def strip_visual_collision(usd_path: str, mesh_names: set[str]) -> int:
    """摘掉视觉网格上的碰撞 API。返回被摘掉的原语数。

    `go2.xml` 的 mesh geom 都带 `contype=0 conaffinity=0`，但转换器不一定照着办。
    这里按名字精确匹配（名字来自 `go2_reference.json` 的 geoms，不猜），只动那些确定是
    "纯视觉"的；真碰撞几何（box/cylinder/sphere）一个都不碰。
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    removed = 0
    for prim in stage.Traverse():
        if prim.GetName() not in mesh_names:
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
        removed += 1
    if removed:
        stage.Save()
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", action="store_true",
                    help="把 scene.xml（地形）也转一份；纯学习用，训练不加载它")
    ap.add_argument("--strip-collision", dest="strip", default=True,
                    action=argparse.BooleanOptionalAction,
                    help="摘掉视觉网格的碰撞 API（默认开，理由见模块 docstring）")
    ap.add_argument("--gui", action="store_true", help="不 headless（调试用）")
    args = ap.parse_args()

    app = _launch_kit(headless=not args.gui)
    try:
        ASSETS.mkdir(parents=True, exist_ok=True)

        # 视觉网格的名字从真值 JSON 里拿，不在这里重新解析 XML：
        # "哪些 geom 不参与碰撞"这件事应该只有一份定义。
        ref_path = ASSETS / "go2_reference.json"
        if not ref_path.exists():
            print(f"缺 {ref_path}，先在**本机**跑 "
                  f"`python3 -m go2_issac.assets.make_mjcf_reference`")
            return 1
        ref = json.loads(ref_path.read_text())
        visual = {g["name"] for g in ref["geoms"]
                  if g["body_id"] != 0 and not g["contype"] and not g["conaffinity"]}
        print(f"真值 JSON: 总质量 {ref['total_mass']:.4f} kg，"
              f"视觉网格 {len(visual)} 个，碰撞几何 "
              f"{sum(1 for g in ref['geoms'] if g['body_id'] and (g['contype'] or g['conaffinity']))} 个")

        usd = convert(GO2_XML, "go2.usd", ASSETS)
        print(f"\n已生成 {usd}")
        if args.strip:
            n = strip_visual_collision(usd, visual)
            print(f"从 {n} 个视觉网格上摘掉了碰撞 API")

        if args.scene:
            scene_usd = convert(SCENE_XML, "scene.usd", ASSETS)
            print(f"已生成 {scene_usd}（地形，仅供目视对比）")

        write_manifest()

        print("\n下一步：python3 go2_issac/smoke.py --check-asset --check-order --drop-test")
        print("     —— 会拿 USD 实例化出来的 articulation 和 assets/go2_reference.json 逐项比。")
        return 0
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    sys.exit(main())
