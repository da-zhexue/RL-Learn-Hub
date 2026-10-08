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
conda create -n issac python=3.11 -y
conda activate issac
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install "setuptools<82" "pip<25" # Issac Lab 2.3.0 requires setuptools<82
pip install isaaclab[isaacsim,all]==2.3.0 --extra-index-url https://pypi.nvidia.com

git clone https://github.com/da-zhexue/RL-Learn-Hub.git
cd RL-Learn-Hub
git clone https://github.com/unitreerobotics/unitree_mujoco.git

# 画面回传
# 注意：本机驱动是宿主机的 595.84，而 Isaac Sim 5.1 的 RTX renderer 在 595.xx 分支上会崩
# (librtx.scenedb.plugin.so，NVIDIA 已知问题，官方验证驱动为 580.65.06)。
# 驱动没降到 580 之前，下面这条 --livestream 命令一定会 core dump，去掉 --livestream 1 可正常训练。
ulimit -c 0   # 免得每次崩溃在仓库里写出 16GB 的 core.* 文件
PUBLIC_IP=10.160.17.102 python3 go2_issac/train.py \
    --terrain flat --steps 2000000 --headless --livestream 1