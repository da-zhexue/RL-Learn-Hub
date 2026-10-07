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