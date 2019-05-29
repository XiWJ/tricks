import torch

def projection(feature_1, extrinsic_0, extrinsic_1, depth_0, intrinsic, device='cuda'):
    """
    project source fearure to reference feature
    :param feature_1: source feature
    :param extrinsic_0: reference feature extrinsic
    :param extrinsic_1: source feature extrinsic
    :param depth_0: reference image depth
    :param intrinsic: intrinsic, reference = source
    :return: projection feature
    """
    extrinsic_0_inv = torch.inverse(extrinsic_0)
    intrinsic_inv = torch.inverse(intrinsic)
    batch_size, channel, height, width = feature_1.shape
    # row and colume of source/reference feature
    row = torch.arange(height).to(device).view(height, 1).repeat(batch_size, 1, width).view(batch_size, -1).float()
    col = torch.arange(width).to(device).view(1, width).repeat(batch_size, 1, height).view(batch_size, -1).float()

    depth_0 = depth_0.view(batch_size, -1)

    pixel_frame_0 = torch.zeros((batch_size, 4, depth_0.shape[1])).to(device)
    pixel_frame_0[:, 0, :] = depth_0 * col
    pixel_frame_0[:, 1, :] = depth_0 * row
    pixel_frame_0[:, 2, :] = depth_0
    pixel_frame_0[:, 3, :] = 1

    camera_coor_0 = torch.matmul(intrinsic_inv, pixel_frame_0)
    world_coor = torch.matmul(extrinsic_0_inv, camera_coor_0)

    camera_coor_1 = torch.matmul(extrinsic_1, world_coor)
    pixel_frame_1 = torch.matmul(intrinsic, camera_coor_1)

    d = pixel_frame_1[:, 2, :]
    pixel_frame_1[:, 0:2, :] /= d.unsqueeze(dim=1)

    u_1 = -torch.ones(depth_0.shape).to(device)
    v_1 = -torch.ones(depth_0.shape).to(device)
    u_2 = height * torch.ones(depth_0.shape).to(device)
    v_2 = width * torch.ones(depth_0.shape).to(device)

    u_1[depth_0!=0] = torch.floor(pixel_frame_1[:, 1, :][depth_0!=0])
    v_1[depth_0!=0] = torch.floor(pixel_frame_1[:, 0, :][depth_0!=0])
    u_2[depth_0!=0] = torch.ceil(pixel_frame_1[:, 1, :][depth_0!=0])
    v_2[depth_0!=0] = torch.ceil(pixel_frame_1[:, 0, :][depth_0!=0])

    valid_coor = torch.ones(v_2.shape).to(device)
    valid_coor[u_1 < 0] = 0
    valid_coor[v_1 < 0] = 0
    valid_coor[u_2 > height-1] = 0
    valid_coor[v_2 > width-1] = 0

    coor1 = torch.zeros(valid_coor.shape).to(device).long()
    coor2 = torch.zeros(valid_coor.shape).to(device).long()
    coor3 = torch.zeros(valid_coor.shape).to(device).long()
    coor4 = torch.zeros(valid_coor.shape).to(device).long()

    coor1[valid_coor!=0] = (u_1[valid_coor!=0] * width + v_1[valid_coor!=0]).long()
    coor2[valid_coor!=0] = (u_1[valid_coor!=0] * width + v_2[valid_coor!=0]).long()
    coor3[valid_coor!=0] = (u_2[valid_coor!=0] * width + v_1[valid_coor!=0]).long()
    coor4[valid_coor!=0] = (u_2[valid_coor!=0] * width + v_2[valid_coor!=0]).long()

    b = height * width * torch.arange(batch_size).to(device).view(-1, 1).repeat(1, height * width).long()
    coor1 = torch.where(valid_coor != 0, coor1 + b, coor1)
    coor2 = torch.where(valid_coor != 0, coor2 + b, coor2)
    coor3 = torch.where(valid_coor != 0, coor3 + b, coor3)
    coor4 = torch.where(valid_coor != 0, coor4 + b, coor4)

    feature_1 = feature_1.permute(1, 0, 2, 3).contiguous().view(channel, -1)
    projection_img = torch.zeros((channel, batch_size, height * width)).to(device)
    w = 0.25
    projection_img[:, valid_coor!=0] = w * feature_1[:, coor1[valid_coor != 0]] + w * feature_1[:, coor2[valid_coor != 0]] + \
                                       w * feature_1[:, coor3[valid_coor != 0]] + w * feature_1[:, coor4[valid_coor != 0]]
    return projection_img.contiguous().view(channel, batch_size, height, width).permute(1, 0, 2, 3)
