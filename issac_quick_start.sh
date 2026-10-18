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

export OMNI_KIT_ACCEPT_EULA=YES   # 非交互运行必须，否则 Kit 卡在 EULA 提示处退出
pip install "isaacsim[all]==6.1.0" --extra-index-url https://pypi.nvidia.com
pip install -q uv
uv pip install "isaaclab[isaacsim]==3.0.0rc1" --extra-index-url https://pypi.nvidia.com

git clone https://github.com/da-zhexue/RL-Learn-Hub.git
cd RL-Learn-Hub
git clone https://github.com/unitreerobotics/unitree_mujoco.git
