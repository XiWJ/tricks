# Hexo搭建和使用
## 搭建hexo+Next
[参考博客](https://www.jianshu.com/p/21c94eb7bcd1)，按照步骤一步一步搭建hexo+Next.
## 使用
hexo 使用命令，在hexo文件下，新建一个文件，并部署上传
```bash
hexo new <layout> "title"
hexo generate
hexo deploy
```
简写成
```bash
hexo n "title"
hexo g && hexo deploy
```