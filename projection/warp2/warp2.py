import os, cv2, torch
import numpy as np
import matplotlib.pyplot as plt
plt.set_cmap('jet')
import torch.nn.functional as F

def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        ## for demon
        # proj_xy = proj_xyz[:, :2, :, :] / torch.where(proj_xyz[:, 2:3, :, :]==0.0, 1e-6*torch.ones_like(proj_xyz[:, 2:3, :, :]), proj_xyz[:, 2:3, :, :])  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

if __name__ == '__main__':
    # step.1 input image & camera parameters
    tgt_img_path = "/home/xiweijie/github/x-plane/tmp/tgt_img.png"
    ref_img1_path = "/home/xiweijie/github/x-plane/tmp/ref_img1.png"
    tgt_depth_path = "/home/xiweijie/github/x-plane/tmp/tgt_depth.npy"
    
    ref_img1 = cv2.imread(ref_img1_path)
    tgt_img = cv2.imread(tgt_img_path)
    tgt_depth = np.load(tgt_depth_path)
    
    intrinsics_path = "/home/xiweijie/github/x-plane/tmp/intrinsics.npy"    
    extrinsic0_path = "/home/xiweijie/github/x-plane/tmp/extrinsic0.npy"
    extrinsic1_path = "/home/xiweijie/github/x-plane/tmp/extrinsic1.npy"
    
    intrinsics = np.load(intrinsics_path)
    extrinsic0 = np.load(extrinsic0_path)
    extrinsic1 = np.load(extrinsic1_path)
    
    image0_tensor = torch.from_numpy(np.transpose(tgt_img, (2, 0, 1))).type(torch.float32).unsqueeze(0)
    image1_tensor = torch.from_numpy(np.transpose(ref_img1, (2, 0, 1))).type(torch.float32).unsqueeze(0)
    intrinsics_tensor = torch.from_numpy(intrinsics).type(torch.float32)
    extrinsic0_tensor = torch.from_numpy(extrinsic0).type(torch.float32)
    extrinsic1_tensor = torch.from_numpy(extrinsic1).type(torch.float32)
    
    ## step 2. compute project matrix
    ref_proj_new = extrinsic0_tensor.clone()
    ref_proj_new[:3, :4] = torch.matmul(intrinsics_tensor[:3, :3], extrinsic0_tensor[:3, :4])
    src_proj_new = extrinsic1_tensor.clone()
    src_proj_new[:3, :4] = torch.matmul(intrinsics_tensor[:3, :3], extrinsic1_tensor[:3, :4])
    
    depth_range = torch.from_numpy(tgt_depth).type(torch.float32)
    depth_range = depth_range.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    
    ## step 3. homograph warping
    warped_imgs = homo_warping(image1_tensor, src_proj_new.unsqueeze(0), ref_proj_new.unsqueeze(0), depth_range)
    warped_imgs_np = warped_imgs[0, :, 0].permute(1,2,0).cpu().numpy().astype(np.uint8)
    cv2.imwrite("/home/xiweijie/github/x-plane/tmp/warped_imgs.png", warped_imgs_np)
