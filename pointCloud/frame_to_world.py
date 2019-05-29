import scipy.misc as misc
import pcl
import torch
import os
import numpy as np
import argparse

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    """
    adjust intrinsic in need size
    :param intrinsic: original intrinsic
    :param intrinsic_image_dim: original image size
    :param image_dim: need to resize the image size
    :return: modify intrinsic
    """
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = np.int(np.floor(image_dim[1] * np.float32(intrinsic_image_dim[0]) / np.float32(intrinsic_image_dim[1])))
    intrinsic[0,0] *= np.float32(resize_width) / np.float32(intrinsic_image_dim[0])
    intrinsic[1,1] *= np.float32(image_dim[1]) / np.float32(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0,2] *= np.float32(image_dim[0]-1) / np.float32(intrinsic_image_dim[0]-1)
    intrinsic[1,2] *= np.float32(image_dim[1]-1) / np.float32(intrinsic_image_dim[1]-1)
    return intrinsic

def frame_to_world(opt):
    """
    project pixel frame to world using matrix operation, for pointCloud generation
    input depth, color, intrinsic, camera pose
    :param opt: opt has depth color intrinsic camera pose path
    :return: pointCloud save in .ply
    """
    # input depth, color
    depth = misc.imread(os.path.join(opt.depth_path, opt.frame + '.png'))
    height, width = depth.shape
    color = misc.imread(os.path.join(opt.color_path, opt.frame + '.jpg'))
    color = misc.imresize(color, (height, width)).astype(np.int64)
    # input intrinsic
    intrinsic = []
    with open(opt.intrinsic, 'r') as f:
        line = f.readline()
        while line:
            intrinsic.append(line.split())
            line = f.readline()
    intrinsic = np.array(intrinsic, dtype=np.float32)
    intrinsic = adjust_intrinsic(intrinsic, [640, 480], [height, width])
    intrinsic_inv = np.linalg.inv(intrinsic)
    # camera pose = extrinsic_inv
    pose = []
    with open(os.path.join(opt.pose_path, opt.frame + '.txt'), 'r') as f:
        line = f.readline()
        while line:
            pose.append(line.split())
            line = f.readline()
    extrinsic_inv = np.array(pose, dtype=np.float32)
    # extrinsic = np.linalg.inv(extrinsic_inv)
    # reshape depth color to [height * width,]
    depth = depth.reshape(-1)
    color = color.reshape(-1, 3)
    # row -> i col -> j
    row = torch.arange(height).view(height, 1).repeat(1, width).view(-1).float().cpu().numpy()
    col = torch.arange(width).view(1, width).repeat(1, height).view(-1).float().cpu().numpy()
    # pixel frame [depth * j, depth * i, depth, 1]
    pixel_frame = np.zeros((4, depth.shape[0]))
    pixel_frame[0, :][depth != 0] = depth[depth != 0] * col[depth != 0]
    pixel_frame[1, :][depth != 0] = depth[depth != 0] * row[depth != 0]
    pixel_frame[2, :][depth != 0] = depth[depth != 0]
    pixel_frame[3, :][depth != 0] = 1
    # camera coordinate [Xc, Yc, Zc, 1]
    camera_coor = np.dot(intrinsic_inv, pixel_frame)
    # world coordinate [X, Y, Z, 1]
    world_coor = np.dot(extrinsic_inv, camera_coor)
    # RGBA for pointCloud
    RGBA = np.zeros((1, height * width)).astype(np.float64)
    RGBA[0, :][depth != 0] = (0xff & color[:, 0][depth != 0]) << 16 | (0xff & color[:, 1][depth != 0]) << 8 | (
                0xff & color[:, 0][depth != 0])
    # generate points [X, Y, Z, RGBA]
    points = np.concatenate((world_coor[0:3], RGBA), axis=0).T
    points = np.array(points, dtype=np.float32)
    # PCL Cloud
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_array(points)
    pcl.save(cloud, opt.frame + '.ply', format='ply')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointCloud')
    parser.add_argument('--depth_path',
                        help='depth data folder',
                        default='/media/xwj/Planar/ScanNet/2d_survey/scene0000_00/pred_depth/', type=str)
    parser.add_argument('--color_path',
                        help='depth data folder',
                        default='/media/xwj/Planar/ScanNet/2d_survey/scene0000_00/pred_seg/', type=str)
    parser.add_argument('--pose_path',
                        help='depth data folder',
                        default='/media/xwj/Planar/ScanNet/2d_survey/scene0000_00/pose/', type=str)
    parser.add_argument('--intrinsic',
                        help='intrinsic data file',
                        default='/home/xwj/pointCloud-study/intrinsic_depth.txt', type=str)
    parser.add_argument('--frame',
                        help='intrinsic data file',
                        default='100', type=str)
    args = parser.parse_args()
    frame_to_world(args)
