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

**3. `fix_base` 必须显式传，而且只能传 False。** 这是唯一一个"得按版本试探"的字段：
   Isaac Lab 2.3.0 / Isaac Sim 5.1 的 `MjcfConverterCfg.fix_base` 默认值是 `MISSING`，
   **不传就死在 `cfg.validate()` 的 `TypeError`**；3.x 又把它整个删了。所以 `convert()`
   用 `dataclasses.fields()` 探测它存不存在，存在才传。
   **值恒为 False**：`go2.xml` 有 freejoint，要的就是浮动基座；钉在世界系上这仓库就不用训了。
   其余字段（`make_instanceable` / `import_sites` / `self_collision`）照默认走，别传。

**4. `finally: app.close()` 会把 traceback 一起带走。** Kit 的 `close()` 直接结束进程，
   实测**退出码 0、一个字符都不打**。所以 `main()` 里必须先 `traceback.print_exc()` 再收尾；
   没有这层保护，转换失败的表现就是"打印完真值 JSON 就安静地退出了"——第 3 条那个
   `TypeError` 就是这么被藏起来的。

**5. headless 的 experience 文件里没有 MJCF importer，得自己 `--enable`。**
   `isaaclab.python.kit`（GUI 版）的 `[dependencies]` 里有 `isaacsim.asset.importer.mjcf`，
   `isaaclab.python.headless.kit` 里**没有**（`grep -c asset.importer` = 0）。不启用就没有
   `MJCFCreateImportConfig` 这条命令，`omni.kit.commands.execute()` 返回 `(None, None)`，
   然后 `MjcfConverter` 在 `import_config.set_import_sites(True)` 上炸成
   `AttributeError: 'NoneType' object has no attribute 'set_import_sites'`。所以 `_launch_kit()`
   走 AppLauncher 官方的 `--kit_args "--enable isaacsim.asset.importer.mjcf"`。

**6. 脚底球要从小腿搬到 `*_foot` 下，`*_foot` 要写一个显式质量。** 这步是转换之后
   `retarget_foot_colliders()` 做的，理由一句话：**Isaac 的接触力按 body 归属，MuJoCo 按 geom。**
   - `go2.xml` 里每只脚底有个 `r=0.022` 的球 geom，它属于 `*_calf`；真正会蹭地的"小腿圆柱"
     也在 `*_calf`。转换器把两者都挂在 `*_calf/collisions` 下，于是**站着不动**时 Isaac 报
     "非脚部件接触地形 188 N"（= 整机重量），而 MuJoCo 那边这个数只有脚底球以外的接触才算。
     把球挪到 `*_foot`（脚 body 的坐标原点正好就是球心）之后，两边的切分才一致：
     球→`*_foot`（`state.FOOT_BODY_NAMES`，不罚），圆柱→`*_calf`（罚）。
   - `*_foot` 在 MuJoCo 里是零质量（球的质量早被算进 `*_calf` 的 0.241352 里了），
     **PhysX 不接受零质量**，会自己补成 1.0 kg/个——4 条腿 +4 kg，整机 15.20641→19.20641 kg，
     而 smoke 的总质量容差只有 1e-3。所以给 4 个 `*_foot` 各写一个 `FOOT_MASS = 1e-4`。
   两个数（半径、球心位置）都从 USD 里**读**出来，不写死。
   **这刀落在 `configuration/go2_physics.usd` 那一层上，不是根层 `go2.usd`**：
   `/collisions` 只活在 instance 的 prototype 里，在组合后的 stage 上根本查不到，
   往根层写这些路径的意见进不了 prototype（理由和实测现象见 `retarget_foot_colliders`）。

转换是**幂等**的：`force_usd_conversion=True` 每次都重转（USD 是派生物，不该手工改），
上面这刀每次都在新产物上重来一遍。
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import traceback

# Kit 的 `app.close()` 会直接结束进程（理由见 docstring 第 4 条），**缓冲区里的 stdout
# 一个字符都留不下**：在终端里跑是行缓冲、看着没事，一旦 `> log.txt` 重定向就成了整块缓冲，
# 所有 print（真值 JSON、路径、这一节的新日志）全部消失，只剩 traceback。
sys.stdout.reconfigure(line_buffering=True)

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
    # 见模块 docstring 第 5 条：headless 的 experience 文件没列 MJCF importer，得自己 --enable。
    args.kit_args = "--enable isaacsim.asset.importer.mjcf"
    app_launcher = AppLauncher(args)
    return app_launcher.app


def convert(asset_path: pathlib.Path, usd_file_name: str, out_dir: pathlib.Path) -> str:
    """跑一次转换，返回 USD 路径。"""
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    kwargs = dict(
        # 跨版本稳定的四个字段（理由见模块 docstring 第 3 条）
        asset_path=str(asset_path),
        usd_dir=str(out_dir),
        usd_file_name=usd_file_name,
        force_usd_conversion=True,
    )
    # fix_base 在 2.3 是 MISSING（不传就 TypeError），在 3.x 又没了。探测着传。
    # 恒为 False：go2.xml 带 freejoint，要的是浮动基座。
    if "fix_base" in {f.name for f in dataclasses.fields(MjcfConverterCfg)}:
        kwargs["fix_base"] = False

    converter = MjcfConverter(MjcfConverterCfg(**kwargs))
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


#: 每条腿，按转换器产出的顺序（`/collisions/FL_*` 在前）。这里只用于遍历，
#: 顺序不影响结果——**别和 `state.LEGS`（DDS 顺序）混**，两个顺序不一样。
LEGS = ("FL", "FR", "RL", "RR")

#: 写进每个 `*_foot` 的显式质量，kg。取值理由：MuJoCo 真值是 0（球的质量在小腿里），
#: PhysX 不接受 0 会补 1.0 kg；而 smoke 的总质量容差是 1e-3，4 个加起来必须小于它。
#: 1e-4 比容差小两个数量级，同时给 PhysX 一个正经的（非零）质量，让它别再自己补。
FOOT_MASS = 1e-4


def retarget_foot_colliders(usd_path: str) -> list[str]:
    """把 4 个脚底球从小腿挪到 `*_foot`，并给 `*_foot` 写显式质量。返回每腿一行日志。

    为什么这么接，见模块 docstring 第 6 条；这里只说**为什么这么写**。

    **只改 `configuration/go2_physics.usd` 这一层，根层 `go2.usd` 一个字节都不动。**
    转换器把每根连杆的碰撞几何集中放在 `/collisions/<连杆>` 下，再让
    `/go2/base_link/<连杆>/collisions` 用**内部引用**指过去，并给那个引用节点加
    `instanceable = true`。踩过的五个坑（都实测过）：

      1. **实例里面写不进去**：往 `instanceable` 节点的子树里写，要么报
         `Cannot copy unknown spec`，要么静默写到一个跟谁都不相干的地方。
      2. **`/collisions` 不是组合后 stage 上的 prim**：`stage.GetPrimAtPath("/collisions/…")`
         全是 invalid——它只活在 instance 的 prototype 里。所以在根层写 `/collisions/…`
         的意见进不了 prototype，等于没写。
      3. **一个 prim 的 spec 分散在多层、而且路径还不一样**：物理层是
         `/collisions/<LEG>_calf/...`（球只是个空 Over），真正的 `radius`/平移在基础层的
         `/collisions/<LEG>/...`（按 **geom 名**组织，不是按 body 名）。想"搬家"就得在两套
         命名里各搬一次，还得给目标层补父节点——拼不齐，别走这条路。
      4. **属性类型必须和 schema 一致，否则 PhysX 会整条忽略它**：`UsdGeomSphere.radius`
         是 **`double`**，写成 `float`（`Sdf.ValueTypeNames.Float`）不报任何错，USD 组合后
         `GetRadiusAttr().Get()` 也照样返回 0.022，一路都对——**只有 PhysX 不认**，直接
         退回 schema 默认半径 **1.0 m**。表现：四只脚各顶一个 1 m 的球，静置时机身飘在
         1.25 m、还一路打滑（球在滚）、接触力冲到 4 倍体重。所以在下面**类型名一律从
         schema 上抄**（`GetTypeName()`），别手写 `ValueTypeNames.*`——这一条是实测踩出来的。
      5. **instance proxy 上的 `Usd.Attribute` 会随改层立刻过期**：读出来的类型名要当场
         存成字符串，不能留到"改完层再问"——那会抛 `Accessed invalid attribute …
         on expired … instance proxy prim`（`active = False` 一改，stage 就重组合了）。

    所以做法不是搬，而是**新建 + 停用**：在物理层的 `/collisions/<LEG>_foot`（那层里本来
    就建了这个空 scope）下把球重新建一个（半径从真身上**读**出来，不写死），再把小腿那份
    `active = False` 掉。`active` 是组合属性——最上面那层说不活跃，这棵子树在所有层里
    一起失效（实测：组合后小腿底下 `children=[]`、脚下是一个半径正确的球 ✓）。

    **为什么就写物理层**：`go2_physics.subLayers = [go2_base]`、
    `go2_base.subLayers = [go2_robot]`，是一条链，所以物理层 > 基础层 > 机器人层，
    写在那儿就是最终意见（实测 ✓）。

    **球心用世界变换反算，不手工减偏移。** 球搬完必须待在世界的同一个点上，而
    `/go2/base_link/<连杆>_foot/collisions` 这个引用节点的 xform 未必和 `*_calf` 那个一样，
    手工减 `foot_t - calf_t` 会漏掉这一级。这里从**组合后的 stage**（能读，只是不能写）
    用 `XformCache` 取两边的世界变换相除，多出来的 xform 自动被算进去。

    做完会**重新打开组合一遍**核对（见函数末尾）：小腿底下没有球、脚下有球、半径对得上、
    而且半径属性的**类型**和 schema 一致。这一步不是装饰——"属性留在原层""类型写错"
    这两类错，不查的话在 smoke 那边只表现为一些奇怪的数（半径写成 float 那次就是
    机身静置时飘在 1.25 m）。
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    physics = pathlib.Path(usd_path).resolve().parent / "configuration" / "go2_physics.usd"
    layer = Sdf.Layer.FindOrOpen(str(physics))
    if layer is None:
        raise RuntimeError(f"打不开 {physics}（转换器的产物结构和预期不符）")
    Sdf.Layer.Reload(layer, force=True)       # 这层刚被 importer 写过：磁盘上的才是产物

    stage = Usd.Stage.Open(usd_path)          # 只用来读半径/世界变换/类型名
    dflt = stage.GetDefaultPrim().GetPath()

    lines = []
    want_radius = {}                          # 后置检查用：每腿从源上读到的半径
    for leg in LEGS:
        # ---- 读（走 instance proxy，读得到）
        src = f"{dflt}/base_link/{leg}_calf/collisions/{leg}"
        src_wrap = stage.GetPrimAtPath(src)
        src_ball = stage.GetPrimAtPath(f"{src}/{leg}")
        if not (src_wrap.IsValid() and src_ball.IsValid()):
            raise RuntimeError(
                f"USD 里找不到 {src} 和它里面的 Sphere——产物结构和预期不符。"
                f"先去 {physics.parent.name}/ 里看清楚 /collisions 长什么样再改这里，别猜")
        radius_attr = UsdGeom.Sphere(src_ball).GetRadiusAttr()
        extent_attr = UsdGeom.Boundable(src_ball).GetExtentAttr()
        radius = radius_attr.Get()
        # 类型名必须**在这一拍就取出来存成字符串**：下面 `old.active = False` 一改层，
        # stage 重组合，挂在 instance proxy 上的 `Usd.Attribute` 立刻"过期"——之后再问
        # 它的类型会抛 `Accessed invalid attribute ... on expired ... instance proxy prim`。
        radius_type = radius_attr.GetTypeName()
        extent_type = extent_attr.GetTypeName()
        want_radius[leg] = radius
        site = stage.GetPrimAtPath(f"{dflt}/base_link/{leg}_foot/collisions")
        if not site.IsValid():
            raise RuntimeError(f"USD 里找不到 {site.GetPath()}，没法把球挂过去")
        # 球的局部平移 = inv(新家父节点的世界变换) × 球现在的世界变换
        cache = UsdGeom.XformCache()          # 每腿新建：上一腿改完层，缓存可能已经过时
        t = (cache.GetLocalToWorldTransform(src_wrap)
             * cache.GetLocalToWorldTransform(site).GetInverse()).ExtractTranslation()
        # translate 的类型名从源上读（基础层里是 double3），别写死——原理同 docstring 第 4 条
        tr_op = next((o for o in UsdGeom.Xformable(src_wrap).GetOrderedXformOps()
                      if o.GetOpName() == "xformOp:translate"), None)
        tr_type = tr_op.GetAttr().GetTypeName() if tr_op else Sdf.ValueTypeNames.Double3

        # ---- 旧的停用：`active` 是组合属性，这一条就够整棵子树在所有层里失效
        src_path = f"/collisions/{leg}_calf/{leg}"
        old = layer.GetPrimAtPath(src_path)
        if old is None:
            raise RuntimeError(f"{physics.name} 里没有 {src_path}——产物结构和预期不符")
        old.active = False

        # ---- 新的建在 *_foot 底下，位置照着"球心世界坐标不变"算
        dst_path = f"/collisions/{leg}_foot/{leg}"
        scope = layer.GetPrimAtPath(f"/collisions/{leg}_foot")
        if scope is None:
            raise RuntimeError(f"{physics.name} 里没有 /collisions/{leg}_foot 这个空 scope——"
                               f"产物结构和预期不符")
        wrap = Sdf.PrimSpec(scope, leg, Sdf.SpecifierDef, "Xform")
        _set_attr(wrap, "xformOp:translate", tr_type, Gf.Vec3d(t))
        _set_attr(wrap, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, ["xformOp:translate"])
        ball = Sdf.PrimSpec(wrap, leg, Sdf.SpecifierDef, "Sphere")
        # 类型名照抄源 prim（`radius` 是 double、`extent` 是 float3[]），理由见 docstring 第 4 条
        _set_attr(ball, "radius", radius_type, radius)
        _set_attr(ball, "extent", extent_type,
                  [Gf.Vec3f(-radius, -radius, -radius), Gf.Vec3f(radius, radius, radius)])
        # 和真身一样：CollisionAPI 挂在里层那个球上（不是外面那层 Xform）
        ball.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(["PhysicsCollisionAPI"]))

        # ---- 质量：PhysX 见到 0 会按"密度×碰撞体积"补一个出来（实测 1.0 kg/个）
        body_path = f"{dflt}/base_link/{leg}_foot"
        body = stage.GetPrimAtPath(body_path)
        if not body.IsValid():
            raise RuntimeError(f"USD 里找不到 {body_path}")
        mass_api = UsdPhysics.MassAPI(body)
        # 惯量：0.1 g 的 r=0.022 小球的真实量级（2/5·m·r² ≈ 2e-8）。
        # **全 0 会被 PhysX 当成"没写"**，那就又回去补 1.0 kg 了。
        inertia = 0.4 * FOOT_MASS * radius * radius
        attrs = (
            ("physics:mass", mass_api.GetMassAttr(), FOOT_MASS),
            ("physics:centerOfMass", mass_api.GetCenterOfMassAttr(), Gf.Vec3f(0.0, 0.0, 0.0)),
            ("physics:diagonalInertia", mass_api.GetDiagonalInertiaAttr(),
             Gf.Vec3f(inertia, inertia, inertia)),
            ("physics:principalAxes", mass_api.GetPrincipalAxesAttr(),
             Gf.Quatf(1.0, 0.0, 0.0, 0.0)),
        )
        body_spec = layer.GetPrimAtPath(body_path)
        if body_spec is None:
            raise RuntimeError(f"{physics.name} 里没有 {body_path}，质量写不进去")
        # 类型名照着同名 USD 属性抄（不写死），换了 Isaac 版本也不会悄悄写错类型
        for name, attr, value in attrs:
            _set_attr(body_spec, name, attr.GetTypeName(), value)

        lines.append(f"{leg}: 球 r={radius:.4f} {src_path} 停用 → {dst_path}"
                     f"（局部平移 ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})），"
                     f"{leg}_foot {FOOT_MASS:g} kg、惯量 {inertia:.2e}")

    if not layer.dirty:
        raise RuntimeError(f"{physics.name} 一个字节都没动过——遍历没走到该改的地方")
    layer.Save()

    # 后置检查：**重新打开组合一遍**，不信内存里那份。这一步抓的是"新建的球没进
    # prototype""停用的不是真身""属性类型和 schema 不符"这类静默失效——不查的话
    # 在 smoke 那边只表现为一些奇怪的数（半径写成 float 那次就是机身飘在 1.25 m）。
    Sdf.Layer.Reload(layer, force=True)
    check = Usd.Stage.Open(usd_path)
    for leg in LEGS:
        gone = check.GetPrimAtPath(f"{dflt}/base_link/{leg}_calf/collisions/{leg}")
        if gone.IsValid() and gone.IsActive():
            raise RuntimeError(f"{leg}_calf 底下那个球没停用掉，PhysX 会当成小腿的碰撞体")
        new = check.GetPrimAtPath(f"{dflt}/base_link/{leg}_foot/collisions/{leg}/{leg}")
        if not new.IsValid():
            raise RuntimeError(f"{leg}_foot 底下没长出球来，组合结果是错的")
        got = UsdGeom.Sphere(new).GetRadiusAttr().Get()
        if got is None or abs(got - want_radius[leg]) > 1e-9:
            raise RuntimeError(f"{leg} 球的半径读回来是 {got}，期望 {want_radius[leg]}")
        # **类型也要对**：USD 读得回来、PhysX 却会因为类型不符整条忽略（docstring 第 4 条）
        authored = new.GetAttribute("radius")
        want_type = UsdGeom.Sphere(new).GetRadiusAttr().GetTypeName()
        if authored.GetTypeName() != want_type:
            raise RuntimeError(f"{leg} 球的 radius 类型是 {authored.GetTypeName()}，"
                               f"schema 要的是 {want_type}——PhysX 会忽略它，"
                               f"退回默认半径 1.0 m")
    return lines


def _set_attr(spec, name: str, type_name, value) -> None:
    """在 Sdf 的 prim spec 上把 `name` 设成 `value`，属性不存在就建一个。

    **必须走层级 Sdf 操作**：这些 prim 在组合后的 stage 上落在 instance prototype 里，
    `Usd.Attribute.Set()` 写不进去（理由见 `retarget_foot_colliders` 的 docstring 第 1 条）。
    属性已有时**只改 default、不建新的**，免得把别处写的 metadata 冲掉。
    """
    from pxr import Sdf

    a = spec.attributes.get(name)
    if a is None:
        a = Sdf.AttributeSpec(spec, name, type_name)
    a.default = value


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

        # 转换器不会做这两件事，而它们错了在 Isaac 里看不出来（只会表现为
        # "站着不动也一直在吃碰撞惩罚"和"重了 4 kg"）。理由见模块 docstring 第 6 条。
        print("\n脚底球归到 *_foot 下 + *_foot 写显式质量：")
        for line in retarget_foot_colliders(usd):
            print(f"   {line}")

        if args.scene:
            scene_usd = convert(SCENE_XML, "scene.usd", ASSETS)
            print(f"已生成 {scene_usd}（地形，仅供目视对比）")

        write_manifest()

        print("\n下一步：python3 go2_issac/smoke.py --check-asset --check-order --drop-test")
        print("     —— 会拿 USD 实例化出来的 articulation 和 assets/go2_reference.json 逐项比。")
        return 0
    except BaseException:
        # 必须在这里打：下面 finally 里的 app.close() 会把进程直接结束掉（实测退出码 0、
        # 无任何输出），异常冒泡上去就没机会打印了。理由见模块 docstring 第 4 条。
        traceback.print_exc()
        sys.stderr.flush()
        return 1
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    sys.exit(main())
