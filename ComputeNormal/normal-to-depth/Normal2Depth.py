import torch
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
plt.set_cmap('jet')

def get_points_coordinate(depth, instrinsic_inv, device="cuda"):
    B, height, width, C = depth.size()
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                           torch.arange(0, width, dtype=torch.float32, device=device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
    xyz = torch.matmul(instrinsic_inv, xyz) # [B, 3, H*W]
    depth_xyz = xyz * depth.view(1, 1, -1)  # [B, 3, Ndepth, H*W]

    return depth_xyz.view(B, 3, height, width)

def get_grid(shape, instrinsic_inv, device="cuda"):
    B, height, width, C = shape
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                           torch.arange(0, width, dtype=torch.float32, device=device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
    xyz = torch.matmul(instrinsic_inv, xyz) # [B, 3, H*W]

    return xyz.view(B, 3, height, width)

if __name__ == '__main__':
    ## step.1 input
    # normal & init_depth & intrinsic path
    normal_path = "/home/xiweijie/github/tricks/ComputeNormal/normal-to-depth/6_normal.npy"
    intrinsic_path = "/home/xiweijie/github/tricks/ComputeNormal/normal-to-depth/intrinsic.npy"
    init_depth_path = "/home/xiweijie/github/tricks/ComputeNormal/normal-to-depth/6_gt_depth.npy"

    # load normal & init_depth & grid
    # i. normal
    normal_np = np.load(normal_path)
    normal_torch = torch.from_numpy(normal_np).unsqueeze(0) # (B, h, w, 3)

    # ii. init_depth
    init_depth_np = np.load(init_depth_path)
    init_depth_torch = torch.from_numpy(init_depth_np).unsqueeze(0).unsqueeze(-1)  # (B, h, w, 1)
    valid_depth = init_depth_np > 0.0

    # iii. intrinsic -> grid
    intrinsic_np = np.load(intrinsic_path)
    intrinsic_inv_np = np.linalg.inv(intrinsic_np)
    intrinsic_inv_torch = torch.from_numpy(intrinsic_inv_np).unsqueeze(0) # (B, 4, 4)

    # iv. get grid
    grid = get_grid(normal_torch.size(), intrinsic_inv_torch[:, :3, :3], device="cpu")  # (B, 3, h, w)
    grid_patch = F.unfold(grid, kernel_size=5, stride=1, padding=4, dilation=2)
    grid_patch = grid_patch.view(1, 3, 25, 192, 256)

    ## step.2 compute matrix A from init depth -> depth_data
    # points [X_c, Y_c, Z_c]
    points = get_points_coordinate(init_depth_torch, intrinsic_inv_torch[:, :3, :3], "cpu")
    point_matrix = F.unfold(points, kernel_size=5, stride=1, padding=4, dilation=2)

    # 0, 1, depth_data = matrix_a[depth_X_c, depth_Y_c, depth_Z_c]
    matrix_a = point_matrix.view(1, 3, 25, 192, 256)  # (B, 3, 25, H, W)
    matrix_a = matrix_a.permute(0, 3, 4, 2, 1)  # (B, H, W, 25, 3)
    _, _, depth_data = torch.chunk(matrix_a, chunks=3, dim=4)

    ## step.3 compute Z_ji from Equ.7
    # i. normal neighbourhood matrix
    norm_matrix = F.unfold(normal_torch.permute(0, 3, 1, 2), kernel_size=5, stride=1, padding=4, dilation=2)
    matrix_c = norm_matrix.view(1, 3, 25, 192, 256)
    matrix_c = matrix_c.permute(0, 3, 4, 2, 1)  # (B, H, W, 25, 3)
    normal_torch_expand = normal_torch.unsqueeze(-1)

    # ii. angle dot(n_j^T, n_i) > \alpha
    angle = torch.matmul(matrix_c, normal_torch_expand)
    valid_condition = torch.gt(angle, 1e-5)
    valid_condition_all = valid_condition.repeat(1, 1, 1, 1, 3)
    tmp_matrix_zero = torch.zeros_like(angle)
    valid_angle = torch.where(valid_condition, angle, tmp_matrix_zero)

    # iii. Equ.7 lower \frac{1}{(ui-cx)/fx + (vi-cy)/fy + niz}
    lower_matrix = torch.matmul(matrix_c, grid.permute(0, 2, 3, 1).unsqueeze(-1))
    condition = torch.gt(lower_matrix, 1e-5)
    tmp_matrix = torch.ones_like(lower_matrix)
    lower_matrix = torch.where(condition, lower_matrix, tmp_matrix)
    lower = torch.reciprocal(lower_matrix)

    # iv. Equ.7 upper nix Xj + niy Yj + niz Zj
    valid_angle = torch.where(condition, valid_angle, tmp_matrix_zero)
    upper = torch.sum(torch.mul(matrix_c, grid_patch.permute(0, 3, 4, 2, 1)), dim=4)
    ratio = torch.mul(lower, upper.unsqueeze(-1))
    estimate_depth = torch.mul(ratio, depth_data)

    valid_angle = torch.mul(valid_angle, torch.reciprocal((valid_angle.sum(dim=(3, 4), keepdim=True)+1e-5).repeat(1, 1, 1, 25, 1)))
    depth_stage1 = torch.mul(estimate_depth, valid_angle).sum(dim=(3, 4))
    depth_stage1 = depth_stage1.squeeze().unsqueeze(2)
    depth_stage1 = torch.clamp(depth_stage1, 0, 10.0)

    ## step.4 save
    # torch->np
    depth_stage1_np = depth_stage1.squeeze().cpu().numpy()
    np.save(normal_path.replace("_normal.npy", "_est_depth.npy"), depth_stage1_np)
    plt.imsave(normal_path.replace("_normal.npy", "_est_depth.png"), depth_stage1_np, cmap='jet')
