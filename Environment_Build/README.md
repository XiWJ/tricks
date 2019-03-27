# Environment-Build
Build Environment for DeepLearing
# Ubuntu Build
## get ubuntu-desktop-amd64.iso
Access [中科大镜像源](http://mirrors.ustc.edu.cn/ubuntu-releases/) in **ubuntu releases** to get *ubuntu * desktop amd64.iso*<br>
## installation
Follow [this Tutorial](https://blog.csdn.net/qq_31192383/article/details/78876905) to install ubuntu.<br>
Be careful of **disk partition**<br>
## install nvidia driver
Access [Nvidia Driver](https://www.geforce.cn/drivers) to get driver<br>
Follow [Driver Tutorial](https://blog.csdn.net/fu6543210/article/details/79746624) to install nvidia driver on ubuntu.Or follow this [Driver Tutorial2](https://github.com/joseph-zhang/QuickSolver/blob/0bf1c63bdfceffc6330b372af15925183423e6bf/setting/set-env.sh#L37)<br>
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-* nvidia-settings
reboot
nvidia-smi
```
# CUDA and Cudnn
Follow [CUDA官方网站](https://developer.nvidia.com/cuda-downloads) to get CUDA. For latest CUDA, it also has Installation Instructions.
```bash
sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
sudo gedit ~/.bashrc
export CUDA_HOME=/usr/local/cuda-10.0
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source ~/.bashrc
nvcc -V 
reboot
```
Follow [Cudnn 官方网站](https://developer.nvidia.com/rdp/cudnn-download) to get Cudnn. For latest Cudnn, it also has Installation Guide
```bash
tar -xzvf cudnn-10.0-linux-x64-v7.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda-10.0/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64
sudo chmod a+r /usr/local/cuda-10.0/include/cudnn.h /usr/local/cuda-10.0/lib64/libcudnn*
```
后续的，可以实现CUDA-8.0、CUDA-9.0、CUDA-10.0的切换。首先去官网上下载CUDA.run文件，注意是.run文件。具体查看[CUDA8.0和9.0版本切换](https://blog.csdn.net/u010821666/article/details/79957071)
```bash
sudo sh cuda_8.0.61_375.26_linux.run
sudo gedit ~/.bashrc
#export CUDA_HOME=/usr/local/cuda-10.0
#export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda
source ~/.bashrc
tar -xzvf cudnn-8.0-linux-x64-v7.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda-8.0/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64
sudo chmod a+r /usr/local/cuda-8.0/include/cudnn.h /usr/local/cuda-8.0/lib64/libcudnn*
sudo rm –rf /usr/local/cuda
sudo ln -s /usr/local/cuda-8.0 /usr/local/cuda
nvcc -V 
```
如果在import tensorflow报错，可以尝试运行下面命令：
```bash
if 
ImportError: libcublas.so.9.0: cannot open shared object file: No such file or directory
then
sudo ldconfig /usr/local/cuda/lib64
```

具体步骤可以参考[keras中文文档](https://keras-cn-docs.readthedocs.io/zh_CN/latest/getting_started/keras_linux/)、[keras中文docs](https://keras-cn.readthedocs.io/en/latest/for_beginners/keras_linux/#3-cudacpu)，也不知道为啥这两个写的Cudnn不完全一样。

# pip3 
首先安装pip3，并升级<br>
```bash
sudo apt-get install python3-pip
sudo pip3 install --upgrade pip
```
更改镜像源<br>
```bash
pip3 install pip -U
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
# OpenCV
首先在[OpenCV下载网站](https://opencv.org/releases.html)，下载OpenCV.zip，解压。
```
cd ~/opencv
mkdir build
cd build
```
如果下载ippicv_2017u2_lnx_intel64_20170418.tgz太慢，可以从网盘下载，放在本地路径下。具体参考[手动安装OpenCV下的IPP加速库](https://www.cnblogs.com/yongy1030/p/10293178.html).
```
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
```
测试是否安装成功
```
pkg-config --modversion opencv
```
## error
```
error: ‘nullptr’ was not declared in this scope
Makefile:160: recipe for target 'all' failed
make: *** [all] Error 2
```
这个问题是C++11被禁了，开启之后从新cmake.具体参考[opencv3.4.2 ubuntu16.04安装 error: ‘nullptr’ was not dec](https://www.cnblogs.com/blueridge/p/9510177.html)
