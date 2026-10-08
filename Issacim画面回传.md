服务器及本机均要下载VirtualGL
```bash
wget -q -O- https://packagecloud.io/dcommander/virtualgl/gpgkey | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/VirtualGL.gpg > /dev/null
sudo wget -O /etc/apt/sources.list.d/VirtualGL.list https://raw.githubusercontent.com/VirtualGL/repo/main/VirtualGL.list
sudo apt update
sudo apt install virtualgl
```
服务器端配置，根据服务器端启动项管理器选择sv还是systemctl
```bash
sudo sv stop gdm  # 停止图形界面,服务器使用sv而非systemctl
sudo /opt/VirtualGL/bin/vglserver_config # 选1
```

本机运行
```bash
# 使用 vglconnect 替代普通的 ssh -X，它会自动处理 VirtualGL 的隧道
/opt/VirtualGL/bin/vglconnect user@your_server_ip
/opt/VirtualGL/bin/vglconnect u2024210044@10.160.17.102 -p 20465
```

服务器运行训练程序时
```bash
# 这会确保 Isaac Sim 的渲染由服务器 GPU 处理
vglrun /path/to/isaac-sim.sh
```