import cv2
import pcl
import os
import numpy as np
import argparse

def main(cfg):

    instrinsic = []
    with open(os.path.join(cfg.intrinsic_seg), 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            eles = line.split()
            for ele in eles:
                instrinsic.append(ele)
    instrinsic = np.array(instrinsic, dtype=np.float32).reshape([4, 4])
    instrinsic_inv = np.linalg.inv(instrinsic)

    depth_files = os.listdir(cfg.inputdepthdir)
    depth_files.sort()
    seg_files = os.listdir(cfg.inputsegdir)
    seg_files.sort()

    for depth_file, seg_file in zip(depth_files, seg_files):

        trajectory = []
        with open(os.path.join(cfg.trajectorydir, depth_file[:-4]+'.txt'), 'r') as ft:
            while True:
                line = ft.readline()
                if not line:
                    break
                eles = line.split()
                for ele in eles:
                    trajectory.append(ele)
        trajectory = np.array(trajectory, dtype=np.float32).reshape([4, 4])

        depth = cv2.imread(os.path.join(cfg.inputdepthdir, depth_file), -1)
        seg = cv2.imread(os.path.join(cfg.inputsegdir, seg_file))
        cloud = pcl.PointCloud_PointXYZRGBA()
        rows = seg.shape[0]
        cols = seg.shape[1]
        depth = cv2.resize(depth, (cols, rows))

        points = np.zeros([rows*cols, 4])
        camera = np.zeros([4, 1])

        for i in range(rows):
            for j in range(cols):
                df = depth[i, j]
                if df == 0:
                    pass
                else:
                    camera[0, 0] = df*j
                    camera[1, 0] = df*i
                    camera[2, 0] = df
                    camera[3, 0] = 1

                    camera_cor = np.dot(instrinsic_inv, camera)
                    world = np.dot(trajectory, camera_cor)

                    x = np.int((world[0, 0]))
                    y = np.int((world[1, 0]))
                    z = np.int((world[2, 0]))
                    r = (0xff & seg[i, j, 2]) << 16
                    g = (0xff & seg[i, j, 1]) << 8
                    b = (0xff & seg[i, j, 0])
                    rgb = r | g | b
                    # print(rgb)
                    points[i*rows+j, :] = [x, y, z, rgb]

        points = np.array(points, dtype=np.float32)
        cloud.from_array(points)
        pcl.save(cloud, os.path.join(cfg.saveplydir, depth_file[:-4]+'.ply'), format='ply')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointCloud')
    parser.add_argument('--inputdepthdir',
                        help='depth data folder',
                        default='/home/xwj/github/data_image/ScanNet-RGBD/depth', type=str)
    parser.add_argument('--inputsegdir',
                        help='seg data folder',
                        default='/home/xwj/github/data_image/ScanNet-RGBD/seg', type=str)
    parser.add_argument('--saveplydir',
                        help='ply data folder',
                        default='/home/xwj/github/data_image/ScanNet-RGBD/ply', type=str)
    parser.add_argument('--intrinsic_seg',
                        help='intrinsic_seg data folder',
                        default='/home/xwj/pointCloud-study/intrinsic_seg.txt', type=str)
    parser.add_argument('--trajectorydir',
                        help='trajectorydir data folder',
                        default='/home/xwj/github/data_image/ScanNet-RGBD/pose', type=str)
    args = parser.parse_args()
    main(args)