import os
import cv2
import numpy as np

def write_ply(rgb, depth, ply_file, intrinsic, extrinsic=None, scale=1.0):
    """
    generate point cloud of input color image and depth map
    :param rgb: color image
    :param depth: depth map
    :param ply_file: save path
    :param intrinsic: intrinsic 3x3
    :param extrinsic: extrinsic 4x4
    :param scale: depth scale
    :return: ply
    """
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    if extrinsic is not None:
        R, t = extrinsic[:3, :3], extrinsic[:3, 3:]
    else:
        R, t = np.eye(3), np.zeros((3,1))
    points = []
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            color = rgb[v, u]  # rgb.getpixel((u, v))
            Z_c = depth[v, u] / scale
            if Z_c == 0: continue
            X_c = (u - cx) * Z_c / fx
            Y_c = (v - cy) * Z_c / fy
            Q_w = np.matmul(np.linalg.inv(R), np.array([X_c, Y_c, Z_c]).reshape((3, 1)) - t)
            X_w, Y_w, Z_w = Q_w[0, 0], Q_w[1, 0], Q_w[2, 0]
            points.append("%f %f %f %d %d %d 0\n" % (X_w, Y_w, Z_w, color[2], color[1], color[0]))

    file = open(ply_file, "w")
    file.write('''ply
                format ascii 1.0
                element vertex %d
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                property uchar alpha
                end_header
                %s
                ''' % (len(points), "".join(points)))
    file.close()
    print("save ply, fx:{}, fy:{}, cx:{}, cy:{}".format(fx, fy, cx, cy))


if __name__ == '__main__':
    rgb = cv2.resize(cv2.imread("/home/xiweijie/github/tricks/pointCloud/0.jpg"), (640, 480))
    intrinsic4x4 = np.array(open('/home/xiweijie/github/tricks/pointCloud/instrinsic.txt', 'r').read().rsplit(), dtype=np.float32).reshape((4,4))
    intrinsic3x3 = intrinsic4x4[:3, :3]
    extrinsic_inv = np.array(open('/home/xiweijie/github/tricks/pointCloud/00000000.txt', 'r').read().rsplit(), dtype=np.float32).reshape((4,4))
    extrinsic = np.linalg.inv(extrinsic_inv)
    depth = cv2.imread("/home/xiweijie/github/tricks/pointCloud/0.png", 2).astype(np.float32) / 1000
    ply_file = os.path.join("/home/xiweijie/github/tricks/pointCloud/", "{}.ply".format(0))
    write_ply(rgb, depth, ply_file, intrinsic3x3)