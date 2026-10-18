"""CartPole robot constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# 通过 MJLAB_SRC_PATH 拼出 XML 的绝对路径，保证从任意工作目录运行都能找到模型文件
CARTPOLE_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "selfcartpole" / "xmls" / "selfcartpole.xml"
)
assert CARTPOLE_XML.exists(), f"XML not found: {CARTPOLE_XML}"


def get_spec() -> mujoco.MjSpec:
  # 将 XML 编译为 MuJoCo 的 MjSpec（后续由 scene 编译成 MjModel/MjData）
  return mujoco.MjSpec.from_file(str(CARTPOLE_XML))


def get_selfcartpole_robot_cfg() -> EntityCfg:
  """Get a fresh SelfCartPole robot configuration instance."""
  # SelfCartPole 的动作来自 XML 里的 <velocity ... joint="slide" .../> 执行器
  # 这里用 XmlActuatorCfg 将该执行器绑定到名为 "slide" 的关节
  actuators = (XmlActuatorCfg(target_names_expr=("slide",)),)
  articulation = EntityArticulationInfoCfg(actuators=actuators)
  return EntityCfg(spec_fn=get_spec, articulation=articulation)
