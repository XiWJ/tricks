# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (conda)
# pytorch       1.0.1  (conda)
# point-mvsnet  1.0    (pip)
# ==================================================================
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple" && \
    GIT_CLONE="git clone --depth 10" && \
    CONDA_INSTALL="conda install -y" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        apt-utils \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
        htop \
        tmux \
        openssh-client \
        openssh-server \
        libboost-dev \
        libboost-thread-dev \
        libboost-filesystem-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        && \
# ==================================================================
# anaconda
# ------------------------------------------------------------------
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
        /bin/bash ~/miniconda.sh -b -p /opt/conda && \
        rm ~/miniconda.sh && \
        #echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc && \
# ==================================================================
# conda
# ------------------------------------------------------------------
    conda config --set show_channel_urls yes && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
    $CONDA_INSTALL \
        python=3.6 && \
    $CONDA_INSTALL \
        pillow \
        pytorch=1.0.1 \
        torchvision \
        cudatoolkit=9.0 \
        && \
    $PIP_INSTALL \
        progressbar2>=3.0 \
        numpy>=1.13 \
        opencv_python>=3.2 \
        scikit-learn>=0.18 \
        scipy>=0.18 \
        matplotlib>=1.5 \
        Pillow>=3.1.2 \
        yacs \
        tqdm \
        tensorboardX \
        && \
    conda clean -y --all && \
## ==================================================================
## PointMVSNet
## ------------------------------------------------------------------
#    $GIT_CLONE --recursive git@github.com:callmeray/PointMVSNet.git /usr/local/PointMVSNet && \
#    cd /usr/local/PointMVSNet && \
#    bash compile.sh && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
EXPOSE 6006