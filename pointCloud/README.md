## Ubuntu16.04+python3.5+PCL1.8.1 图像到点云
### 编程环境搭建
[编程环境搭建](https://medium.com/@ss4365gg/%E6%88%90%E5%8A%9F%E5%9C%A8ubuntu-16-04%E7%92%B0%E5%A2%83%E4%B8%AD%E5%AE%89%E8%A3%9D-pcl-1-8-1-python-pcl-a016b711bc4)<br>
### 图像到点云
[编程部分python](https://elody-07.github.io/%E4%BB%8E%E5%9B%BE%E5%83%8F%E5%88%B0%E7%82%B9%E4%BA%91/#2-%E7%BC%96%E7%A8%8B%E9%83%A8%E5%88%86)<br>
[理论部分1图像到点云](http://www.cnblogs.com/gaoxiang12/p/4652478.html)<br>
[理论部分2世界坐标](https://blog.csdn.net/chentravelling/article/details/53558096)
### 结果
结果保存在.pcd文件中，在bash下运行pcl_viewer \*.pcd文件
### new 代码和结果
见pointCloudv2.py，采用相机外参和内参，直接转成世界坐标，结果保存成 \*.ply，用meshLab打开。
### error
```
import pcl.pcl_visualization
ImportError: No module named pcl_visualization
```
[应对办法](https://github.com/strawlab/python-pcl/issues/127#issuecomment-379522531)
### 矩阵操作生成点云
[frame_to_world.py](https://github.com/XiWJ/tricks/blob/master/pointCloud/frame_to_world.py)和study.py功能一样，从图像像素坐标生成点云，但是采用矩阵操作，弃用for循环，效率高500倍，适合大尺度图像
