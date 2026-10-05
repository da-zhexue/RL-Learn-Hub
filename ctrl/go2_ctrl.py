import time
import sys
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

DEFAULT_Q = [0.0, 0.9, -1.8] * 4

# 关节限位，取自 go2.xml，顺序与 DDS 报文一致：FR, FL, RR, RL 各 [hip, thigh, calf]。
# 注意前腿大腿(front_hip)和后腿大腿(back_hip)的限位不同，所以不是 4 段简单重复。
Q_MIN = np.array([-1.0472, -1.5708, -2.7227] * 2 + [-1.0472, -0.5236, -2.7227] * 2, dtype=np.float32)
Q_MAX = np.array([ 1.0472,  3.4907, -0.83776] * 2 + [ 1.0472,  4.5379, -0.83776] * 2, dtype=np.float32)

class Go2Ctrl:
    # kd 不宜超过 4.0：实测 kd>=4.5 时微分项会激发高频振荡（|dq| 冲到 ~40 rad/s），
    # 狗原地抖动并向后走；kd<=4.0 时完全静止。参考 stand_go2.py 用的也是 3.5。
    def __init__(self, kp=60.0, kd=3.5, ctrl_dq_max=0.002, dt=0.002,
                 domain_id=None, interface=None):
        # 显式传 domain_id/interface 时优先用它们；都不传则沿用原来的行为（按 sys.argv[1] 判断）。
        # 显式参数是给 play.py 这类"自己也有命令行参数"的调用方用的——否则会把 --model 当网卡名。
        if domain_id is not None or interface is not None:
            ChannelFactoryInitialize(0 if domain_id is None else domain_id,
                                     "lo" if interface is None else interface)
        elif len(sys.argv) < 2:
            ChannelFactoryInitialize(1, "lo")
        else:
            ChannelFactoryInitialize(0, sys.argv[1])

        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

        self.kp = kp
        self.kd = kd
        self.ctrl_dq_max = ctrl_dq_max
        self.dt = dt
        self.crc = CRC()


        self.low_state = None
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._low_state_handler, 10)

        # 当前目标角度，由后台线程按 dt 持续下发；None 表示尚未设定
        self.q_target = None
        self.hold_thread = RecurrentThread(
            interval=self.dt, target=self._publish_target, name="go2_hold"
        )
        self.hold_thread.Start()

    def _publish_target(self):
        """后台线程体：每 dt 把 self.q_target 下发一次，不该被外部直接调用。

        仿真和真机都只在收到 LowCmd 时更新一次力矩，之后一直保持——指令流一断，
        力矩就冻结成开环常值，关节会被推到限位顶死。所以下发必须持续进行。
        """
        if self.q_target is None:
            # 还没设定过目标：先跟随当前姿态，避免一创建对象就把机器人拽向默认姿态
            if self.low_state is None:
                return
            self.q_target = np.array(
                [self.low_state.motor_state[i].q for i in range(12)], dtype=np.float32
            )

        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0
        for i in range(12):
            cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            cmd.motor_cmd[i].q = self.q_target[i]
            cmd.motor_cmd[i].kp = self.kp
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = self.kd
            cmd.motor_cmd[i].tau = 0.0

        cmd.crc = self.crc.Crc(cmd)
        self.pub.Write(cmd)

    @staticmethod
    def clamp_q(q):
        """把角度限幅到关节限位 [Q_MIN, Q_MAX] 内，返回新数组（不改原值）。

        超出限位的目标不会让关节转得更多，只会让电机持续顶在限位上（仿真里就是"别住"），
        所以下发前统一夹住。
        """
        return np.clip(np.asarray(q, dtype=np.float32), Q_MIN, Q_MAX)

    def send_cmd(self, q):
        """设定目标关节角度，q 为 12 维列表或数组，按顺序对应四条腿的 3 个关节角度。

        只更新目标值、不阻塞；实际下发由后台线程按 dt 持续进行，设定后会一直保持该姿态。
        超出关节限位的分量会被限幅。
        """
        if len(q) != 12:
            raise ValueError("Input q must be a list or array of length 12.")

        self.q_target = self.clamp_q(q)

    def stop(self):
        """停止后台下发线程（停止后不再有指令，机器人会失去该姿态保持）。"""
        self.hold_thread.Wait()

    def _low_state_handler(self, msg: LowState_):
        self.low_state = msg

    def read_state(self, timeout=2.0):
        """等待并返回最近一帧状态，超时抛 TimeoutError。返回 dict（numpy 数组），原始 LowState_ 报文在 self.low_state。
        """
        start_time = time.perf_counter()
        while self.low_state is None:
            if (time.perf_counter() - start_time) > timeout:
                raise TimeoutError("未收到 rt/lowstate")
            time.sleep(self.dt)

        s = self.low_state
        return {
            "q": np.array([s.motor_state[i].q for i in range(12)], dtype=np.float32),
            "dq": np.array([s.motor_state[i].dq for i in range(12)], dtype=np.float32),
            "tau_est": np.array([s.motor_state[i].tau_est for i in range(12)], dtype=np.float32),
            "quat": np.array(s.imu_state.quaternion, dtype=np.float32),  # w x y z
            "gyro": np.array(s.imu_state.gyroscope, dtype=np.float32),
            "acc": np.array(s.imu_state.accelerometer, dtype=np.float32),
            "rpy": np.array(s.imu_state.rpy, dtype=np.float32),
            "foot_force": np.array(s.foot_force, dtype=np.float32),
            "tick": s.tick,
        }

    def ctrl_smoothly(self, q_target):
        """平滑过渡到 q_target，返回过渡结束后的状态。

        步数由「当前角度与目标角度的最大差值」决定：max_diff / ctrl_dq_max 向上取整，
        因此单步每个关节的变化量都不超过 ctrl_dq_max。差值越大过渡越慢，不会突变。
        """
        if len(q_target) != 12:
            raise ValueError("Input q_target must be a list or array of length 12.")

        q_now = self.read_state()["q"]
        # 先限幅再算步数，保证斜坡的终点就是实际会到达的姿态
        q_target = self.clamp_q(q_target)

        max_diff = np.max(np.abs(q_target - q_now))
        steps = max(int(np.ceil(max_diff / self.ctrl_dq_max)), 1)

        t_start = time.perf_counter()
        for i in range(1, steps + 1):
            percent = min(i / steps, 1.0)
            self.send_cmd((1.0 - percent) * q_now + percent * q_target)
            # 按 dt 推进斜坡，补偿循环体本身耗时，避免整体被拖慢
            sleep_left = t_start + i * self.dt - time.perf_counter()
            if sleep_left > 0:
                time.sleep(sleep_left)

        return self.read_state()

    def reset(self, q_init=None):
        """从当前姿态平滑过渡到默认站立姿态，返回过渡完成后的状态。

        q_init: 目标姿态，默认 DEFAULT_Q（mujoco home 关键帧）
        过渡速度由 ctrl_dq_max 决定，与 ctrl_smoothly 一致。
        """
        target = DEFAULT_Q if q_init is None else q_init
        if len(target) != 12:
            raise ValueError("Input q_init must be a list or array of length 12.")

        self.ctrl_smoothly(target)

        return self.read_state()
