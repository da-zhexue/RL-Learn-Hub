from go2_ctrl import Go2Ctrl
import time

def main():
    ctrl = Go2Ctrl()
    state = ctrl.read_state()  # 等待并读取一帧状态，确保订阅器已启动
    print("当前关节角度:", state["q"])
    time.sleep(5)  # 等待一段时间，确保机器人有时间执行指令
    ctrl.reset()  # 重置机器人到默认站立姿态
    time.sleep(5)  # 等待一段时间，确保机器人有时间执行指令
    state = ctrl.read_state()  # 等待并读取一帧状态，确保订阅器已启动
    print("当前关节角度:", state["q"])

if __name__ == "__main__":
    main()