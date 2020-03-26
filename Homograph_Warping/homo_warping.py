import torch
import torch.nn.functional as F
import numpy as np
import cv2

def read_cam_file(filename):
    intrinsics = np.zeros((4, 4))
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics[:3, :3] = np.genfromtxt(filename).astype(np.float32).reshape((3, 3))
    intrinsics[3, 3] = 1.0
    return intrinsics

def read_pose_file(filename):
    extrinsics = np.genfromtxt(filename).astype(np.float32)
    extrinsic1 = np.concatenate((extrinsics[0, :].reshape((3, 4)), np.array([[0, 0, 0, 1]], dtype=np.float32)), axis=0)
    extrinsic2 = np.concatenate((extrinsics[1, :].reshape((3, 4)), np.array([[0, 0, 0, 1]], dtype=np.float32)), axis=0)
    return extrinsic1, extrinsic2

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
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
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
    ## step 1. data load
    image0_path = "0000.jpg"
    image1_path = "0001.jpg"
    intrinsics = read_cam_file("cam.txt")
    extrinsic1, extrinsic2 = read_pose_file("poses.txt")
    image0 = cv2.imread(image0_path)
    image1 = cv2.imread(image1_path)
    image0_tensor = torch.from_numpy(np.transpose(image0, (2, 0, 1))).type(torch.float32).unsqueeze(0)
    image1_tensor = torch.from_numpy(np.transpose(image1, (2, 0, 1))).type(torch.float32).unsqueeze(0)
    intrinsics_tensor = torch.from_numpy(intrinsics).type(torch.float32)
    extrinsic1_tensor = torch.from_numpy(extrinsic1).type(torch.float32)
    extrinsic2_tensor = torch.from_numpy(extrinsic2).type(torch.float32)

    ## step 2. compute project matrix
    ref_proj_new = intrinsics_tensor.clone()
    ref_proj_new[:3, :4] = torch.matmul(intrinsics_tensor[:3, :3], extrinsic1_tensor[:3, :4])
    src_proj_new = intrinsics_tensor.clone()
    src_proj_new[:3, :4] = torch.matmul(intrinsics_tensor[:3, :3], extrinsic2_tensor[:3, :4])
    depth_range = 0.5 + 0.1 * torch.arange(32, dtype=torch.float32).reshape(1, -1, 1, 1).repeat(1, 1, image0_tensor.size()[-2], image0_tensor.size()[-1])

    ## step 3. homograph warping
    warped_imgs = homo_warping(image1_tensor, src_proj_new.unsqueeze(0), ref_proj_new.unsqueeze(0), depth_range)

    ## step 4. save image
    for i in range(warped_imgs.shape[2]):
        warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
        img_np = warped_img[0].detach().cpu().numpy()
        img_np = img_np[:, :, ::-1]

        alpha = 0.5
        beta = 1 - alpha
        gamma = 0
        img_np = np.uint8(img_np)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_add = cv2.addWeighted(image0, alpha, img_np, beta, gamma)
        cv2.imwrite('tmp/tmp{}.png'.format(i), np.hstack([image0, img_np, img_add]))  # * ratio + img_np*(1-ratio)]))