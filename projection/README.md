# Projection

Give A and B camera pose intrinsic, project B feature map to A feature map using pytorch. 
## 理论
投影部分的数学原理是像素坐标到世界坐标的转换，具体可以参考[相机成像原理](https://blog.csdn.net/chentravelling/article/details/53558096)<br>
**注意：**[project.py](https://github.com/XiWJ/tricks/blob/master/projection/project.py)采用的是矩阵乘而未采用for循环操作，矩阵乘比for循环操作效率高了500倍。投影过程是反向投影，即A投影到B，双线性插值找到在B中对应四个坐标，在拿回到A中，保证连续，实现可导。
## warp
warp.py是用纯numpy实现warp，其中坐标变换采用的是cv2.remap. pytorch里面对应函数是F.grid_sample.

## warp2
warp.py中cv2.remap处理图片大小受限制，warp2.py采用torch函数grid_sample，效果更好.
