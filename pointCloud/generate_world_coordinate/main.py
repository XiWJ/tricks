import os, cv2, torch
import numpy as np
import matplotlib.pyplot as plt
plt.set_cmap('jet')
import torch.nn.functional as F

def gengerate_world_coordinate(ref_proj, shape, depth_values, device='cpu'):
    batch, height, width = shape[0], shape[2], shape[3]
    num_depth = depth_values.shape[1]
    with torch.no_grad():
        proj = torch.inverse(ref_proj)
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                               torch.arange(0, width, dtype=torch.float32, device=device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * \
                        depth_values.view(batch, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]

    return proj_xyz.view(batch, 3, num_depth, height, width)

if __name__ == '__main__':
    # step.1 input image & camera parameters
    tgt_img_path = "/home/xiweijie/github/x-plane/tmp/tgt_img.png"
    tgt_depth_path = "/home/xiweijie/github/x-plane/tmp/tgt_depth.npy"

    tgt_img = cv2.imread(tgt_img_path)
    tgt_depth = np.load(tgt_depth_path)

    intrinsics_path = "/home/xiweijie/github/x-plane/tmp/intrinsics.npy"
    extrinsic0_path = "/home/xiweijie/github/x-plane/tmp/extrinsic0.npy"

    intrinsics = np.load(intrinsics_path)
    extrinsic0 = np.load(extrinsic0_path)

    image0_tensor = torch.from_numpy(np.transpose(tgt_img, (2, 0, 1))).type(torch.float32).unsqueeze(0)
    intrinsics_tensor = torch.from_numpy(intrinsics).type(torch.float32)
    extrinsic0_tensor = torch.from_numpy(extrinsic0).type(torch.float32)

    ## step 2. compute project matrix
    ref_proj_new = extrinsic0_tensor.clone()
    ref_proj_new[:3, :4] = torch.matmul(intrinsics_tensor[:3, :3], extrinsic0_tensor[:3, :4])

    depth_range = torch.from_numpy(tgt_depth).type(torch.float32)
    depth_range = depth_range.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

    ## step 3. gengerate_world_coordinate
    world_coordinate = gengerate_world_coordinate(ref_proj_new.unsqueeze(0), image0_tensor.shape, depth_range)
    world_coordinate_np = world_coordinate[0, :, 0].permute(1,2,0).cpu().numpy()

    points = []
    for v in range(tgt_img.shape[0]):
        for u in range(tgt_img.shape[1]):
            color = tgt_img[v, u]  # rgb.getpixel((u, v))
            Z_c = tgt_depth[v, u] / 1.0
            if Z_c == 0: continue

            X_w, Y_w, Z_w = world_coordinate_np[v, u]
            points.append("%f %f %f %d %d %d 0\n" % (X_w, Y_w, Z_w, color[2], color[1], color[0]))

    ply_file = "/home/xiweijie/github/x-plane/tmp/point3D.ply"

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
