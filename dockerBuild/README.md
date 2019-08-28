# How to build docker image
## 安装docker
更新apt包索引
```
sudo apt-get update
```
安装包以允许apt通过HTTPS使用存储库
```
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
```
添加Docker的官方GPG密钥：
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```
9DC8 5822 9FC7 DD38 854A E2D8 8D81 803C 0EBF CD88通过搜索指纹的最后8个字符，验证您现在拥有带指纹的密钥 。
```
sudo apt-key fingerprint 0EBFCD88
```
使用以下命令设置稳定存储库。要添加nightly或test存储库，请在下面的命令中的单词后添加单词nightly或test（或两者）stable。
```
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```
再次更新apt包索引
```
sudo apt-get update
```
安装最新版本的Docker Engine - 社区和容器
```
sudo apt-get install docker-ce docker-ce-cli containerd.io
```
最后，通过运行hello-world 映像验证是否正确安装了Docker Engine
```
sudo docker run hello-world
```
[参考网址](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-engine---community-1)
## 使用docker
### 解决docker加sudo方法
创建docker组
```
sudo groupadd docker
```
将当前用户加入docker组, 比如 {USER}=xwj
```
sudo gpasswd -a {USER} docker
```
重启服务
```
sudo service docker restart
```
刷新docker成员
```
newgrp - docker
```
### 列出本机正在运行的容器
```
docker container ls
```
### 列出本机所有容器，包括终止运行的容器
```
docker container ls --all
```
### 终止运行的容器文件
```
docker container rm [containerID]
```
### 编写Dockerfile
模板见Dockerfile, 缺什么加上什么。
### 创建image
在Dockerfile同目录下，输入下面命令
```bash
docker image build -t [imageName]:[tag] .
```
### 上传到dockerhub
登录
```bash
docker login
```
命名镜像和版本号
```bash
docker image tag [imageName] [username]/[repository]:[tag]
```
上传
```bash
docker image push [username]/[repository]:[tag]
```
### 本地运行镜像
```bash
docker container run --rm -p 8000:3000 -it [imageName]:[tag] /bin/bash
```
## 参考教程
[docker教程](http://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html)<br>
[anaconda_Dockerfile](https://github.com/Leinao/LeinaoPAI/tree/master/images/anaconda_3_tf_12_torch_10_cv_34_cuda_90)
## 遇见问题
### Q: Docker build “Could not resolve ‘archive.ubuntu.com’” apt-get fails to install anything
[解决办法](https://medium.com/@faithfulanere/solved-docker-build-could-not-resolve-archive-ubuntu-com-apt-get-fails-to-install-anything-9ea4dfdcdcf2)
如何检查DNS是否有问题：
```
docker run busybox nslookup google.com 
```
如果得到 connection timed out;......，说明无法解析DNS
### 解决这个问题
```
nmcli dev show | grep 'IP4.DNS'
IP4.DNS[1]:                             192.10.0.2
```
运行我们用于检查DNS是否正常工作的命令
```
docker run --dns 192.10.0.2 busybox nslookup google.com 
Server:    192.10.0.2
Address 1: 192.10.0.2
Name:      google.com
Address 1: 2a00:1450:4009:811::200e lhr26s02-in-x200e.1e100.net
Address 2: 216.58.198.174 lhr25s10-in-f14.1e100.net
```
### 永久性的系统范围修复
创建/etc/docker/daemon.json
```
{ 
    “dns”：[“192.10.0.2”，“8.8.8.8”] 
}
```
重启docker
```
sudo service docker restart
```