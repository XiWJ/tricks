# How to use git
## 建立git仓库
```bash
git init
```
## 将项目的所有文件添加到仓库中
```bash
git add .
```
## 将add的文件commit到仓库
```bash
git commit -m "注释语句"
```
## 将本地的仓库关联到github上
```bash
git remote add origin https://github.com/XiWJ/tricks
```
## 上传github之前，要先pull一下
```bash
git pull origin master
```
## 最后一步，上传代码到github远程仓库
```bash
git push -u origin master
```
