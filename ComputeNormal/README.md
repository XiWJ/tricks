# Compute normal vector from depth
## matlab code
代码参考[Data-driven 3d primitives for single image understanding](https://web.eecs.umich.edu/~fouhey/2013/3dp/index.html)

使用matlab库函数surfnorm

运行
```
matlab -nodesktop -nosplash -r generate_normal_gt_demon
or
bash run.sh
```
## pytorch code
理论参考[GeoNet](https://github.com/xjqi/GeoNet)，自己采用pytorch实现。

Note: pytorch>=1.3.1 for torch.det()

运行
```
python ComputeNormal.py
```

# Normal to depth
理论参考[GeoNet](https://github.com/xjqi/GeoNet)，自己采用pytorch实现。

Note: pytorch>=1.3.1 for torch.det()

运行
```
python Normal2Depth.py
```

[init_depth, normal] -> refined depth
