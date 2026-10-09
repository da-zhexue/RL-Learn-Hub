#ssh免密连接
# 查看本机的公钥
# cat ~/.ssh/id_rsa.pub # 本机运行
# 将输出的公钥内容复制到目标机器的 ~/.ssh/authorized_keys 文件中
# echo "AAA" >> ~/.ssh/authorized_keys # 目标机器运行

nvidia-smi
sudo apt update && sudo apt upgrade
sudo apt install nano

curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n sim python=3.12 -y    # Isaac Sim 6.1 只吃 3.12（requires-python ==3.12.*）
conda activate sim
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# Isaac Sim 6.1 = Kit 110：宿主机的 595.84 驱动上**只有 6.x 能渲染**
# （5.1 的 RTX renderer 在 595.xx 上必崩在 librtx.scenedb.plugin.so，官方验证驱动是 580.65.06，
#  而内核模块是宿主机的，本机改不了）。
# 用 uv 不用 pip：这颗依赖树上 pip 会 resolution-too-deep；而且官方文档那行
# `isaaclab[isaacsim,all]` 在 3.0.0rc1 上本身就是坏的——`[all]` 里的 ovphysx 要 packaging<24，
# isaacsim-core 要 packaging==26.0，装不到一起。所以不带 `all`（rsl_rl 在基础包里就有）。
export OMNI_KIT_ACCEPT_EULA=YES   # 非交互运行必须，否则 Kit 卡在 EULA 提示处退出
pip install -q uv
uv pip install "isaaclab[isaacsim]==3.0.0rc1" --extra-index-url https://pypi.nvidia.com

git clone https://github.com/da-zhexue/RL-Learn-Hub.git
cd RL-Learn-Hub
git clone https://github.com/unitreerobotics/unitree_mujoco.git

# 训练 + 画面回传
# `--visualizer none` 是 Isaac Lab 3.0 的无头模式（老版本的 `--headless` 已经删了）
ulimit -c 0   # 免得崩溃在仓库里写出 16GB 的 core.* 文件
PUBLIC_IP=10.160.17.102 python3 go2_issac/train.py \
    --terrain flat --steps 2000000 --visualizer none --livestream 1
# 看画面：Isaac Sim WebRTC Streaming Client 连 10.160.17.102:49100（TCP），媒体走 UDP 47998，两条都要通。
# 浏览器直接开 49100 会返回 501——那是 WebSocket 信令端点，属正常，不是网页。