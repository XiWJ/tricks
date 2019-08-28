# How to build docker image
## 编写Dockerfile
模板见Dockerfile, 缺什么加上什么。
## 创建image
在Dockerfile同目录下，输入下面命令
```bash
docker image build -t [imageName]:[tag] .
```
## 上传到dockerhub
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
## 本地运行镜像
```bash
docker container run --rm -p 8000:3000 -it [imageName]:[tag] /bin/bash
```
## 参考教程
[docker教程](http://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html)<br>
[anaconda_Dockerfile](https://github.com/Leinao/LeinaoPAI/tree/master/images/anaconda_3_tf_12_torch_10_cv_34_cuda_90)