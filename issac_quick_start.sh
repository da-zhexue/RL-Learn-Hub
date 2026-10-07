#ssh免密连接
# 查看本机的公钥
# cat ~/.ssh/id_rsa.pub # 本机运行
# 将输出的公钥内容复制到目标机器的 ~/.ssh/authorized_keys 文件中
# echo "AAA" >> ~/.ssh/authorized_keys # 目标机器运行

nvidia-smi
sudo apt update && sudo apt upgrade
sudo apt install nano
sudo apt install xauth xorg

curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n issac python=3.11 -y
conda activate issac
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install "setuptools<82" "pip<25" # Issac Lab 2.3.0 requires setuptools<82
pip install isaaclab[isaacsim,all]==2.3.0 --extra-index-url https://pypi.nvidia.com

git clone https://github.com/da-zhexue/RL-Learn-Hub.git