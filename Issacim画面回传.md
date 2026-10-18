# Isaac Sim 3D 训练画面回传：TurboVNC + VirtualGL

## 当前配置

2026-10-09 已在服务器 `n1`、用户 `u2024210044` 下配置常驻远程桌面。

| 项目 | 当前值 |
| --- | --- |
| 系统 / 服务管理 | Ubuntu 24.04 / runit，PID 1 是 `runsvdir` |
| GPU / 驱动 | NVIDIA RTX 4090 D 24 GB / 595.84 |
| TurboVNC / VirtualGL | 3.3.1 / 3.1.5，安装于 `/opt` |
| 桌面 / 分辨率 | XFCE / 1920 × 1080，24 位色深 |
| Isaac 环境 | conda `sim`，Python 3.12，Isaac Sim 6.1.0.0，Isaac Lab 3.0.0rc1 |
| 显示 / VNC 地址 | `DISPLAY=:1` / `127.0.0.1:5901` |
| runit 服务 | `/etc/service/turbovnc-u2024210044` |
| 启动辅助命令 | `~/.local/bin/isaac-vnc-run` |

画面路径：服务器 GPU 渲染 → TurboVNC 桌面 → SSH 隧道 → 本机 TurboVNC Viewer。

**Isaac Sim 的 RTX 渲染使用 Vulkan，VirtualGL 负责 OpenGL/GLX 应用，不能重定向 Vulkan。** 本次实测 NVIDIA 595.84 的 Vulkan 驱动能直接向 TurboVNC 窗口呈现，因此 Isaac Sim 直接运行在 `:1`；OpenGL 程序使用 `vglrun -d egl`。这仅是本文验证的驱动组合，不能推断所有旧驱动都支持。

VirtualGL 的 EGL 后端已能访问 GPU，无需启动额外的 Xorg，也无需执行 GLX 后端的 `vglserver_config`。旧 `/etc/X11/xorg.conf` 使用 `PCI:206:0:0`，与当前 GPU 的 `4B:00.0` 不符；本文方案不使用该文件。

## 1. 本机连接

本机安装 [TurboVNC Viewer](https://github.com/TurboVNC/turbovnc/releases)，无需安装 VirtualGL。在本机终端建立隧道，并保持终端打开：

```bash
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -L 15901:127.0.0.1:5901 -p 20465 u2024210044@10.160.17.102
```

服务器地址和 SSH 端口沿用原文档；平台重新分配地址后需替换。本机 `15901` 被占用时，换一个空闲端口，并同步修改 Viewer 地址。

在 TurboVNC Viewer 中填写 **`127.0.0.1::15901`**，双冒号表示直接指定 TCP 端口。在服务器查看初始密码：

```bash
cat ~/.vnc/connection-password
```

密码没有写入仓库，`~/.vnc/passwd` 和初始明文密码文件权限均为 `600`。改密码使用服务器上的 `/opt/TurboVNC/bin/vncpasswd`；更改后 `connection-password` 中的初始值不再有效。传统 VNC 密码只使用前 8 个字符。

VNC 仅监听服务器本机回环地址，经 SSH 加密传输，无需开放公网 5901，也无需 `ssh -X`、`vglconnect` 或 WebRTC UDP 端口。

## 2. 启动训练与回放

可以在 VNC 桌面终端、普通 SSH 或 VS Code 终端执行。辅助命令将窗口指向 `:1`，覆盖 SSH X11 转发的 `DISPLAY=localhost:10.0`。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sim
cd ~/RL-Learn-Hub

# 小规模观察画面；更多环境会增加 GPU 和渲染负载。
~/.local/bin/isaac-vnc-run python go2_issac/train.py \
    --terrain flat --steps 200000 --num-envs 16 --mini-batches 4 \
    --visualizer kit --livestream 0

# 回放：替换成已有的 Isaac 训练目录或 .pt checkpoint。
~/.local/bin/isaac-vnc-run python go2_issac/play.py \
    --resume models/你的Isaac训练目录 --num-envs 4 \
    --visualizer kit --livestream 0
```

当前 Isaac Lab 3.0 使用 `--visualizer kit` 显示桌面画面。`--visualizer none` 会关闭可视化，旧版的 `--headless` 不适用于当前版本；环境中也不能额外设置 `HEADLESS=1`。首次 Kit 初始化可能超过 1 分钟，等待 `app ready` 后再判断画面。

已经运行的无头进程不能通过修改另一终端的 `DISPLAY` 变成有窗口进程；需用带窗口参数重新启动，或回放 checkpoint。断开 Viewer / SSH 隧道不会关闭常驻桌面，训练生命周期仍由启动终端、tmux 等管理。

OpenGL 应用使用 VirtualGL EGL 后端：

```bash
~/.local/bin/isaac-vnc-run /opt/VirtualGL/bin/vglrun -d egl glxinfo -B
~/.local/bin/isaac-vnc-run /opt/VirtualGL/bin/vglrun -d egl \
    /opt/VirtualGL/bin/glxspheres64
```

应看到 `NVIDIA GeForce RTX 4090 D`，而非 `llvmpipe`。不要把 `vglrun` 加到 Isaac Sim 命令前来解决 Vulkan 问题。

## 3. 重建配置

当前机器已完成以下配置。此节用于同类机器或容器重建，不需要在现有服务上重复启动 `:1`。

### 软件前提

```bash
nvidia-smi
dpkg-query -W turbovnc virtualgl xfce4
sudo apt-get update
sudo apt-get install xfce4 xfce4-terminal dbus-x11 mesa-utils vulkan-tools
```

缺少 `/opt/TurboVNC` 或 `/opt/VirtualGL` 时，从 [TurboVNC 官方发布页](https://github.com/TurboVNC/turbovnc/releases) 和 [VirtualGL 官方发布页](https://github.com/VirtualGL/virtualgl/releases) 下载匹配架构的 `.deb`，用 `sudo apt-get install ./下载的包.deb` 安装。本文没有更换 NVIDIA 驱动；容器内的图形库必须与宿主机驱动匹配，不要用随意升级驱动代替检查。

当前 NVIDIA Vulkan ICD 是 `/etc/vulkan/icd.d/nvidia_icd.json`，其 `library_path` 为 `libGLX_nvidia.so.0`。其他机器应先确认实际路径，再修改辅助命令。

### 用户文件

以训练用户建立目录并设置密码：

```bash
mkdir -p ~/.vnc ~/.vnc/runit ~/.local/bin
chmod 700 ~/.vnc ~/.vnc/runit
/opt/TurboVNC/bin/vncpasswd
```

本机配置时生成了随机初始密码并保存为 `~/.vnc/connection-password`；手动重建时使用上述交互命令即可，无需建立明文密码文件。

`~/.vnc/xstartup.turbovnc`：

```sh
#!/bin/sh
unset SESSION_MANAGER DBUS_SESSION_BUS_ADDRESS
export XDG_SESSION_TYPE=x11
export XDG_CURRENT_DESKTOP=XFCE
export XDG_RUNTIME_DIR="$HOME/.vnc/runtime"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
exec dbus-run-session -- xfce4-session
```

`~/.local/bin/isaac-vnc-run`：

```bash
#!/bin/bash
set -euo pipefail
if [ "$#" -eq 0 ]; then
    echo "Usage: isaac-vnc-run COMMAND [ARGS...]" >&2
    exit 2
fi
export DISPLAY=:1
export XAUTHORITY="$HOME/.Xauthority"
export XDG_RUNTIME_DIR="$HOME/.vnc/runtime"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export VGL_DISPLAY=egl
ulimit -c 0
exec "$@"
```

```bash
chmod 700 ~/.vnc/xstartup.turbovnc ~/.local/bin/isaac-vnc-run
```

### runit 常驻服务

本机 PID 1 是 runit，使用 `sv` 管理。`systemctl get-default` 的输出不能证明 systemd 是当前服务管理器。

`~/.vnc/runit/run`，换机器时替换所有用户名和家目录：

```sh
#!/bin/sh
exec 2>&1
exec chpst -u u2024210044 env \
    HOME=/home/u2024210044 USER=u2024210044 LOGNAME=u2024210044 \
    PATH=/opt/TurboVNC/bin:/opt/VirtualGL/bin:/usr/local/bin:/usr/bin:/bin \
    /opt/TurboVNC/bin/vncserver :1 -fg -localhost \
    -geometry 1920x1080 -depth 24 -securitytypes VNC \
    -rfbauth /home/u2024210044/.vnc/passwd \
    -xstartup /home/u2024210044/.vnc/xstartup.turbovnc
```

`~/.vnc/runit/finish`：

```sh
#!/bin/sh
exec chpst -u u2024210044 env HOME=/home/u2024210044 \
    USER=u2024210044 PATH=/usr/bin:/bin \
    /opt/TurboVNC/bin/vncserver -kill :1
```

安装服务。家目录所在存储限制 root 读取私有文件，所以先由用户复制到临时目录，再由 root 安装：

```bash
chmod 700 ~/.vnc/runit/run ~/.vnc/runit/finish
staging_dir=$(mktemp -d /tmp/isaac-vnc-runit.XXXXXX)
chmod 755 "$staging_dir"
install -m 755 ~/.vnc/runit/run ~/.vnc/runit/finish "$staging_dir/"
sudo install -d -m 755 /etc/service/turbovnc-u2024210044
sudo install -m 755 "$staging_dir/run" "$staging_dir/finish" \
    /etc/service/turbovnc-u2024210044/
sudo sv -w 10 up /etc/service/turbovnc-u2024210044
```

runit 会监督桌面进程，容器重启后自动启动服务，但不会启动训练。平台销毁并重新创建容器时，是否保留 `/etc` 取决于平台，可能需要重新安装服务。

普通 systemd 机器可以先手动启动，另行按实际系统设置自启：

```bash
/opt/TurboVNC/bin/vncserver :1 -localhost -geometry 1920x1080 -depth 24 \
    -securitytypes VNC -rfbauth ~/.vnc/passwd \
    -xstartup ~/.vnc/xstartup.turbovnc
```

## 4. 验证与维护

```bash
sudo sv status /etc/service/turbovnc-u2024210044
ss -ltnp | grep 5901
tail -n 80 ~/.vnc/$(hostname):1.log

# OpenGL：VirtualGL EGL 后端。
~/.local/bin/isaac-vnc-run /opt/VirtualGL/bin/vglrun -d egl glxinfo -B

# Vulkan：单有设备枚举还不够，要验证窗口呈现。
~/.local/bin/isaac-vnc-run vulkaninfo --summary
~/.local/bin/isaac-vnc-run vkcube --c 120
```

服务应为 `run`，端口应为 `127.0.0.1:5901`。Vulkan 应识别 RTX 4090 D，并能在 Viewer 中显示旋转方块。

停止或重启桌面会关闭其中的图形应用，先结束或保存训练：

```bash
sudo sv down /etc/service/turbovnc-u2024210044
sudo sv up /etc/service/turbovnc-u2024210044
sudo sv -w 15 restart /etc/service/turbovnc-u2024210044
```

| 现象 | 检查方法 |
| --- | --- |
| Viewer 连接被拒绝 | 检查隧道是否存活、本机端口是否冲突，以及 `sv status` / 5901 监听 |
| 密码不正确 | 修改过密码后不要继续用初始密码文件；Viewer 地址使用双冒号 |
| 桌面有了但训练窗口没有 | 使用 `isaac-vnc-run`，指定 `--visualizer kit --livestream 0`，检查 `HEADLESS` |
| 狗在画面中凭空上飘、跳动或与物理位置不一致 | 用最新的 `train.py` / `play.py`；脚本已关闭 RTX geometry streaming 和 Fabric 动态变换读取，旧进程必须重启 |
| 启动时视口黑屏 | 等待 `app ready`；本次完整初始化约 113 秒，然后才加载测试场景 |
| NVIDIA Vulkan 设备不可见 | 检查 `nvidia-smi`、ICD 路径和图形库与宿主机驱动是否匹配 |
| OpenGL 变成 llvmpipe | 确认使用 `vglrun -d egl`；不要据此判断 Isaac Sim 的 Vulkan 状态 |
| 旧驱动能计算但无法向 VNC 呈现 | 用 `vkcube` 验证；VirtualGL 不会修复 Vulkan。需使用可呈现的 NVIDIA Xorg + 桌面抓取，或另行配置 Isaac WebRTC |
| `:1` 被占用 | 检查已有 Xvnc / 服务。用 `sv` 管理当前桌面，不要重复启动或盲删锁文件 |

## 5. 本次实测

- VirtualGL EGL 的 `glxinfo -B` 返回 NVIDIA、RTX 4090 D、OpenGL 4.6 / 595.84。
- 原生 Vulkan `vkcube --c 120` 正常退出，graphics queue 支持窗口呈现。
- 已通过 RFB 3.8 密码认证接收 `127.0.0.1:5901` 的 1920 × 1080 桌面帧。
- 已运行真实 Isaac Sim 6.1 窗口，创建旋转绿色方块、地面、灯光与相机；VNC 协议接收的截图中确认了 RTX 3D 场景，测试正常退出。
- `go2_issac/train.py --terrain flat --steps 2048 --num-envs 4 --mini-batches 4 --max-iterations 2 --visualizer kit --livestream 0` 正常完成 2 次 PPO 迭代、2048 步并保存 checkpoint；测试输出在 `/tmp/isaac-vnc-training-check`，日志在 `/tmp/isaac-vnc-training-check.log`。此短时测试验证运行与显示通道，不代表策略收敛。
- 已测试 `sv restart`，服务重新监听 `127.0.0.1:5901`，当前保持运行。
- Isaac Sim 6.1 的启动日志曾报告动态物体 geometry streaming 与 Fabric transform 读取同时开启，可能导致机器人渲染位置滞后。`go2_issac/train.py` 和 `play.py` 现在默认加入 `--/rtx/hydra/geometryStreaming/enabled=false` 与 `--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false`；这只影响视口同步，不改变物理。
- 本机到服务器的 SSH 隧道仍需在本机建立，没有从用户本机验证网络连通性。
