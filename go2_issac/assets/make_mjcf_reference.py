"""从 `go2.xml` 导出资产真值 JSON（本机可跑，只 import mujoco）。

    python3 -m go2_issac.assets.make_mjcf_reference          # 写 go2_reference.json
    python3 -m go2_issac.assets.make_mjcf_reference --print  # 顺便把摘要打到屏幕上

**为什么要这个东西**：Isaac 侧用的是自己转出来的 USD，不是官方 Nucleus 资产。
"转对了没有"这个问题，如果没有一份独立于转换器的真值，就只能靠肉眼在 viewport 里看狗长得像不像——
而质量、惯量、关节限位、驱动上限这些**看不出来**的东西错了，训练只会表现为"训不动"，
查起来极痛。所以这里把 MuJoCo 编译后的 `MjModel` 里的资产参数**全部**摊平成 JSON，
上机后 `smoke.py --check-asset` 拿 USD 实例化出来的 articulation 逐项比。

真值来自 `MjModel`（编译后的结果）而不是 XML 文本：XML 里有 default class、有 `<include>`、
有单位换算和轴角拆解，读文本等于自己再实现一遍 MuJoCo 的解析器，不如直接问编译结果。

**这份 JSON 里最要命的三件事**（上机后错了最难查）：
  1. 12 个关节的**名字和顺序**：MJCF 的 qpos 跟身体树（FL,FR,RL,RR），执行器/DDS 报文是
     FR,FL,RR,RL，USD 里又按导入顺序。名字是唯一可靠的锚点，下标不是。
  2. **前/后腿的关节限位不一样**（`thigh` 前腿上限 3.8、后腿 4.2 之类），限位抄成一样的会
     让策略在前腿学到"再抬一点"的动作直接被 Isaac clip 掉。
  3. **质量/惯量/COM**：总质量 15.2064 kg 是后面所有 kp/kd 能不能直接搬的前提。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from go2_common.config import DEFAULT_SCENE  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "go2_reference.json"


def _name(model, objtype, idx: int) -> str:
    """`mj_id2name` 返回 None 时给一个能读的占位（匿名 geom 很常见）。"""
    import mujoco

    got = mujoco.mj_id2name(model, objtype, idx)
    return got if got else f"<{idx}>"


def geom_type_names() -> dict[int, str]:
    """几何类型：数值 -> 名字。用 mujoco 的枚举常量取值，不写魔数。"""
    import mujoco

    g = mujoco.mjtGeom
    return {
        g.mjGEOM_PLANE: "plane", g.mjGEOM_HFIELD: "hfield", g.mjGEOM_SPHERE: "sphere",
        g.mjGEOM_CAPSULE: "capsule", g.mjGEOM_ELLIPSOID: "ellipsoid",
        g.mjGEOM_CYLINDER: "cylinder", g.mjGEOM_BOX: "box", g.mjGEOM_MESH: "mesh",
    }


def joint_type_names() -> dict[int, str]:
    import mujoco

    j = mujoco.mjtJoint
    return {j.mjJNT_FREE: "free", j.mjJNT_BALL: "ball",
            j.mjJNT_SLIDE: "slide", j.mjJNT_HINGE: "hinge"}


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _arr(a) -> list:
    """numpy -> 嵌套 list（JSON 可写；float64 的 repr 是精确往返的）。"""
    return np.asarray(a).tolist()


def build(path: str = DEFAULT_SCENE) -> dict:
    import mujoco

    from go2_common.config import DECIMATION, KD, KP, EnvCfg

    model = mujoco.MjModel.from_xml_path(path)
    mj = mujoco.mjtObj
    GEOM_TYPES = geom_type_names()
    JNT_TYPES = joint_type_names()

    # ---------------------------------------------------------- 关节
    joints = []
    for jid in range(model.njnt):
        jnt = {
            "id": jid,
            "name": _name(model, mj.mjOBJ_JOINT, jid),
            "type": JNT_TYPES[int(model.jnt_type[jid])],
            "body_id": int(model.jnt_bodyid[jid]),
            "body": _name(model, mj.mjOBJ_BODY, int(model.jnt_bodyid[jid])),
            "axis": _arr(model.jnt_axis[jid]),
            "pos": _arr(model.jnt_pos[jid]),
            "qposadr": int(model.jnt_qposadr[jid]),
            "dofadr": int(model.jnt_dofadr[jid]),
            "limited": bool(model.jnt_limited[jid]),
            "range": _arr(model.jnt_range[jid]),
        }
        # 铰链才有 dof 级的阻尼/电枢/摩擦（freejoint 的 dof 是机身的平移/转动，不带这些）
        if jnt["type"] in ("hinge", "slide"):
            d = int(model.jnt_dofadr[jid])
            jnt.update(
                damping=float(model.dof_damping[d]),
                armature=float(model.dof_armature[d]),
                frictionloss=float(model.dof_frictionloss[d]),
            )
        joints.append(jnt)

    # ---------------------------------------------------------- 身体树
    bodies = []
    for bid in range(model.nbody):
        bodies.append({
            "id": bid,
            "name": _name(model, mj.mjOBJ_BODY, bid),
            "parent_id": int(model.body_parentid[bid]),
            "parent": _name(model, mj.mjOBJ_BODY, int(model.body_parentid[bid])),
            "pos": _arr(model.body_pos[bid]),
            "quat": _arr(model.body_quat[bid]),
            "mass": float(model.body_mass[bid]),
            "subtree_mass": float(model.body_subtreemass[bid]),
            "ipos": _arr(model.body_ipos[bid]),        # 质心（body 系）
            "iquat": _arr(model.body_iquat[bid]),      # 惯量主轴（body 系）
            "inertia": _arr(model.body_inertia[bid]),  # 主轴惯量（对角）
        })

    # ---------------------------------------------------------- 几何
    geoms = []
    for gid in range(model.ngeom):
        body_id = int(model.geom_bodyid[gid])
        g = {
            "id": gid,
            "name": _name(model, mj.mjOBJ_GEOM, gid),
            "body_id": body_id,
            "body": _name(model, mj.mjOBJ_BODY, body_id),
            "type": GEOM_TYPES.get(int(model.geom_type[gid]), str(int(model.geom_type[gid]))),
            "size": _arr(model.geom_size[gid]),
            "pos": _arr(model.geom_pos[gid]),
            "quat": _arr(model.geom_quat[gid]),
            "contype": int(model.geom_contype[gid]),
            "conaffinity": int(model.geom_conaffinity[gid]),
            "condim": int(model.geom_condim[gid]),
            "priority": int(model.geom_priority[gid]),
            "friction": _arr(model.geom_friction[gid]),
            "group": int(model.geom_group[gid]),
            "mesh": None,
        }
        if g["type"] == "mesh":
            g["mesh"] = _name(model, mj.mjOBJ_MESH, int(model.geom_dataid[gid]))
        geoms.append(g)

    # ---------------------------------------------------------- 执行器
    actuators = []
    for aid in range(model.nu):
        trnid = int(model.actuator_trnid[aid][0])
        actuators.append({
            "id": aid,
            "name": _name(model, mj.mjOBJ_ACTUATOR, aid),
            "joint_id": trnid,
            "joint": _name(model, mj.mjOBJ_JOINT, trnid),
            "gear": _arr(model.actuator_gear[aid]),
            "ctrllimited": bool(model.actuator_ctrllimited[aid]),
            "ctrlrange": _arr(model.actuator_ctrlrange[aid]),
            "forcelimited": bool(model.actuator_forcelimited[aid]),
            "forcerange": _arr(model.actuator_forcerange[aid]),
        })

    # ---------------------------------------------------------- 网格
    # 网格名字 + 文件 sha256：转 USD 时用的是不是同一批 STL，靠这个确认。
    # （`MjModel` 不暴露网格文件路径，所以文件清单直接扫 assets 目录——反正 go2.xml 的
    #  meshdir 就指着那儿，编译器读的也是那几个文件。）
    assets_dir = pathlib.Path(path).parent / "assets"
    mesh_files = []
    for f in sorted(assets_dir.glob("*")):
        if f.is_file():
            mesh_files.append({"name": f.name, "size": f.stat().st_size,
                               "sha256": _sha256(f)})

    meshes = []
    for mid in range(model.nmesh):
        meshes.append({
            "id": mid,
            "name": _name(model, mj.mjOBJ_MESH, mid),
            "scale": _arr(model.mesh_scale[mid]),
            "nvert": int(model.mesh_vertnum[mid]),
            "nface": int(model.mesh_facenum[mid]),
        })

    # ---------------------------------------------------------- 关键帧 / 求解器选项
    keyframes = []
    for kid in range(model.nkey):
        keyframes.append({
            "id": kid,
            "name": _name(model, mj.mjOBJ_KEY, kid),
            "qpos": _arr(model.key_qpos[kid]),
            "ctrl": _arr(model.key_ctrl[kid]),
        })

    opt = {
        # scene.xml 没写 <option timestep>，所以编译出来是 MuJoCo 的默认 0.002；
        # **训练时用的是 0.005**——`env.py:50` 在建模之后覆盖成 `cfg.sim_dt`。
        # Isaac 侧的 `sim.dt` 要跟着 0.005 走（对拍/回放的依据），别用这个 0.002。
        "xml_default_timestep": float(model.opt.timestep),
        "gravity": _arr(model.opt.gravity),
        "cone": int(model.opt.cone),      # 0=pyr 1=elliptic
        "impratio": float(model.opt.impratio),
        "iterations": int(model.opt.iterations),
        "solver": int(model.opt.solver),
        "integrator": int(model.opt.integrator),
        "tolerance": float(model.opt.tolerance),
    }

    cfg = EnvCfg()
    return {
        "source": str(path),
        "source_sha256": _sha256(pathlib.Path(path)),
        "mujoco_version": mujoco.__version__,
        "counts": {
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
            "njnt": int(model.njnt), "nbody": int(model.nbody),
            "ngeom": int(model.ngeom), "nmesh": int(model.nmesh),
        },
        "total_mass": float(model.body_mass.sum()),
        "opt": opt,
        # 训练时的环境参数（来源是 go2_common/config.py，不是资产；放这儿是为了让
        # Isaac 侧一个文件就能查全"MuJoCo 那边到底用的什么数"）
        "env": {
            "sim_dt": float(cfg.sim_dt),
            "decimation": int(cfg.decimation),
            "policy_dt": float(cfg.sim_dt) * cfg.decimation,
            "policy_hz": 1.0 / (float(cfg.sim_dt) * cfg.decimation),
            "kp": float(KP), "kd": float(KD),
            "action_scale": float(cfg.action_scale),
            "joint_margin": float(cfg.joint_margin),
            "max_episode_s": float(cfg.max_episode_s),
            "sim_dt_source": "env.py:50 覆盖 model.opt.timestep；scene.xml 本身没写",
        },
        "joints": joints,
        "bodies": bodies,
        "geoms": geoms,
        "actuators": actuators,
        "meshes": meshes,
        "mesh_files": mesh_files,
        "keyframes": keyframes,
    }


# ---------------------------------------------------------------- 自检

def checks(ref: dict) -> list[str]:
    """把"已知的真值"断言一遍。这些数是实测出来的，改了就说明资产被换了。"""
    bad: list[str] = []

    def eq(tag, got, want, tol=1e-12):
        if abs(float(got) - float(want)) > tol:
            bad.append(f"{tag}: {got} != {want}")

    # 15.206408：12 条腿 + 机身。这个数一旦变了，说明 URDF/MJCF 被换过，
    # 后面所有"kp/kd 能直接搬"的假设都要重来。
    eq("总质量", ref["total_mass"], 15.206408, 1e-6)
    eq("nq", ref["counts"]["nq"], 19)
    eq("nv", ref["counts"]["nv"], 18)
    eq("nu", ref["counts"]["nu"], 12)
    eq("njnt", ref["counts"]["njnt"], 13)     # 1 free + 12 hinge
    eq("几何总数", ref["counts"]["ngeom"], 65)
    eq("impratio", ref["opt"]["impratio"], 100.0)
    eq("cone(elliptic)", ref["opt"]["cone"], 1)
    eq("时间步(sim_dt)", ref["env"]["sim_dt"], 0.005)
    eq("decimation", ref["env"]["decimation"], 4)

    # 狗自己的碰撞几何 23 个（地形那 9 个在 body 0 上，不算）：
    # base 3（box+cylinder+sphere）+ 髋 4 + 大腿 4 + 小腿 8 + 脚底 4。
    # 转 USD 之后这 23 个必须一个不少地带上 CollisionAPI，否则狗会穿地或卡住。
    robot_coll = [g for g in ref["geoms"]
                  if g["body_id"] != 0 and (g["contype"] or g["conaffinity"])]
    if len(robot_coll) != 23:
        bad.append(f"狗的碰撞几何 {len(robot_coll)} 个 != 23")
    visual = [g for g in ref["geoms"] if not g["contype"] and not g["conaffinity"]]
    if len(visual) != 33:
        bad.append(f"纯视觉几何 {len(visual)} 个 != 33")
    if any(g["type"] != "mesh" for g in visual):
        bad.append("纯视觉几何里有非 mesh 的（33 个应该全是 STL）")

    hinges = [j for j in ref["joints"] if j["type"] == "hinge"]
    if len(hinges) != 12:
        bad.append(f"铰链数 {len(hinges)} != 12")
    for j in hinges:
        eq(f"{j['name']} damping", j["damping"], 0.1)
        eq(f"{j['name']} armature", j["armature"], 0.01)
        eq(f"{j['name']} frictionloss", j["frictionloss"], 0.2)

    # 脚底球：半径 0.022、priority=1（priority 高的独占该接触对 → 脚-地摩擦由它决定）
    feet = [g for g in ref["geoms"] if g["type"] == "sphere" and abs(g["size"][0] - 0.022) < 1e-12]
    if len(feet) != 4:
        bad.append(f"半径 0.022 的脚底球有 {len(feet)} 个 != 4")
    for g in feet:
        eq(f"脚底球 friction[0]", g["friction"][0], 0.4)
        eq(f"脚底球 priority", g["priority"], 1)

    # 执行器力矩上限：hip/thigh ±23.7、calf ±45.43
    for a in ref["actuators"]:
        want = 45.43 if a["joint"].endswith("calf_joint") else 23.7
        eq(f"{a['joint']} ctrlrange", max(abs(v) for v in a["ctrlrange"]), want, 1e-9)

    return bad


def summarize(ref: dict) -> None:
    print(f"源: {ref['source']}  (sha256 {ref['source_sha256'][:12]}…)")
    print(f"mujoco {ref['mujoco_version']}  " +
          "  ".join(f"{k}={v}" for k, v in ref["counts"].items()))
    env = ref["env"]
    print(f"总质量 {ref['total_mass']:.4f} kg   "
          f"cone={'elliptic' if ref['opt']['cone'] == 1 else 'pyramidal'} "
          f"impratio={ref['opt']['impratio']:g}")
    print(f"仿真：dt {env['sim_dt']} s × decimation {env['decimation']} "
          f"→ 策略 {env['policy_hz']:.0f} Hz；kp {env['kp']:g} / kd {env['kd']:g}；"
          f"action_scale {env['action_scale']:g}")
    print(f"       （scene.xml 编译出来的 dt 是 {ref['opt']['xml_default_timestep']}，"
          f"训练时被 env.py 覆盖成 {env['sim_dt']}）")

    print("\n关节（顺序 = MJCF 身体树顺序，**不是** DDS 报文的顺序）：")
    for j in ref["joints"]:
        if j["type"] != "hinge":
            print(f"  [{j['id']:2d}] {j['name']:<24} {j['type']:<5} (机身自由关节)")
            continue
        print(f"  [{j['id']:2d}] {j['name']:<24} body={j['body']:<14} "
              f"axis={np.asarray(j['axis'])} "
              f"range=[{j['range'][0]:+.3f}, {j['range'][1]:+.3f}] "
              f"dof={j['dofadr']:2d}")

    print("\n执行器：")
    for a in ref["actuators"]:
        print(f"  [{a['id']:2d}] {a['joint']:<24} ctrl=[{a['ctrlrange'][0]:+.2f}, "
              f"{a['ctrlrange'][1]:+.2f}] N·m")

    counts: dict[str, int] = {}
    for g in ref["geoms"]:
        if g["body_id"] == 0:
            continue  # 地形，不算狗
        key = f"{g['type']}{'(碰撞)' if g['contype'] or g['conaffinity'] else '(纯视觉)'}"
        counts[key] = counts.get(key, 0) + 1
    print("\n狗的几何：" + "  ".join(f"{k}×{v}" for k, v in sorted(counts.items())))
    feet = [g for g in ref["geoms"] if g["type"] == "sphere" and g["size"][0] == 0.022]
    print(f"  脚底球 {len(feet)} 个：r={feet[0]['size'][0]} "
          f"friction={feet[0]['friction']} priority={feet[0]['priority']} "
          f"condim={feet[0]['condim']}（priority 高者独占接触对 → 脚-地 μ 就是它）")
    print(f"网格文件 {ref['counts']['nmesh']} 个（{len(ref['mesh_files'])} 个文件），"
          f"合计 {sum(m['nvert'] for m in ref['meshes'])} 顶点")

    total = sum(b["mass"] for b in ref["bodies"] if b["id"] != 0)
    print(f"\n身体树：{ref['counts']['nbody']} 个（含 world），连杆质量合计 {total:.4f} kg")
    for b in ref["bodies"]:
        if b["id"] == 0:
            continue
        print(f"  {b['name']:<16} parent={b['parent']:<16} m={b['mass']:.4f} kg  "
              f"ipos={np.asarray(b['ipos'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="把摘要打到屏幕上")
    args = ap.parse_args()

    ref = build(args.scene)
    bad = checks(ref)
    if bad:
        print("自检失败（资产和预期不符，别把 JSON 当真值用）：")
        for b in bad:
            print(f"  - {b}")
        return 1

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(ref, f, indent=1, sort_keys=False)
        f.write("\n")
    print(f"已写出 {out}（{out.stat().st_size / 1024:.0f} KB），自检通过")
    if args.do_print:
        print()
        summarize(ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
