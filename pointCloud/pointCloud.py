import cv2
import pcl
import numpy as np

depth = cv2.imread('699.png', -1)
cloud = pcl.PointCloud()
rows = depth.shape[0]
cols = depth.shape[1]

pointcloud = []
instrinsic = []

with open('instrinsic.txt', 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        eles = line.split()
        for ele in eles:
            instrinsic.append(ele)

instrinsic = np.array(instrinsic).reshape([4, 4])

fx = instrinsic[0, 0]
fy = instrinsic[1, 1]

for i in range(rows):
    for j in range(cols):
        d = depth[i, j]
        if d == 0:
            pass
        else:
            z = np.float32(d)
            x = i * z / np.float32(fx)
            y = j * z / np.float32(fy)
            points = [x, y, z]
            pointcloud.append(points)

pointcloud = np.array(pointcloud, dtype=np.float32)
cloud.from_array(pointcloud)
pcl.save(cloud, '699cloud.pcd', format='pcd')