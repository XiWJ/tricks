# warp by numpy
import numpy as np
import cv2
import os
import re

def read_cam_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    return intrinsics, extrinsics

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def warp(src_img, ref_depth, ref_intrinsic, ref_extrinsic, src_intrinsic, src_extrinsic):
    # step 1. get homo coordinate [x, y, 1] -> [X, Y, Z_c]
    height, width = ref_depth.shape
    x, y = np.meshgrid(np.arange(0, width, dtype=np.float32), np.arange(0, height, dtype=np.float32))
    x, y = x.reshape(-1), y.reshape(-1)
    xy1 = np.stack((x,y,np.ones_like(x, dtype=np.float32)))
    xyz = xy1 * ref_depth.reshape(-1)

    # step 2. project reference camera space
    ref_camera = np.matmul(np.linalg.inv(ref_intrinsic), xyz)

    # step 3. rotation & transformation to world space
    ref_R = ref_extrinsic[:3, :3]
    ref_t = ref_extrinsic[:3, 3:]
    world = np.matmul(np.linalg.inv(ref_R), ref_camera-ref_t)

    # step 4. project to source camera space
    src_R = src_extrinsic[:3, :3]
    src_t = src_extrinsic[:3, 3:]
    src_camera = np.matmul(src_R, world) + src_t

    # step 5. project to source pixel coordinate to get pixel -> pixel map
    xyz_src = np.matmul(src_intrinsic, src_camera)
    xy1_src = xyz_src / xyz_src[2:, :]
    y_src, x_src = xy1_src[1, :], xy1_src[0, :]
    warped_src_img = cv2.remap(src_img, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    return warped_src_img.reshape((height, width, 3))

if __name__ == '__main__':
    image1 = cv2.imread("/home/xiweijie/github/tricks/projection/rect_001_0_r5000.png").astype(np.float32)
    image2 = cv2.imread("/home/xiweijie/github/tricks/projection/rect_002_0_r5000.png").astype(np.float32)
    intrinsics1, extrinsics1 = read_cam_file("/home/xiweijie/github/tricks/projection/00000000_cam.txt")
    intrinsics2, extrinsics2 = read_cam_file("/home/xiweijie/github/tricks/projection/00000001_cam.txt")
    depth1 = np.array(read_pfm("/home/xiweijie/github/tricks/projection/depth_map_0000.pfm")[0], dtype=np.float32)
    depth2 = np.array(read_pfm("/home/xiweijie/github/tricks/projection/depth_map_0001.pfm")[0], dtype=np.float32)
    resized_image1 = cv2.resize(image1, depth1.shape)
    resized_image2 = cv2.resize(image2, depth2.shape)
    warpped_src_img = warp(resized_image2, depth1, intrinsics1, extrinsics1, intrinsics2, extrinsics2)
    cv2.imwrite("/home/xiweijie/github/tricks/projection/warp.png", np.uint8(warpped_src_img))