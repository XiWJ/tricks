import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels=512, out_channels=256):
        super(ASPP, self).__init__()
        # convolution with different Atrous dilation
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.conv3x3_3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=18, dilation=18),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )

        # image pooling
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )

        # output convolution
        self.out_conv = nn.Sequential(
            nn.Conv2d((in_channels // 2) * 5, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        N, C, H, W = x.size()

        # Atrous Spatial Pyramid Pooling
        out_1x1 = self.conv1x1_1(x)     # (N, C/2, H, W)
        out_3x3_1 = self.conv3x3_1(x)   # (N, C/2, H, W)
        out_3x3_2 = self.conv3x3_2(x)   # (N, C/2, H, W)
        out_3x3_3 = self.conv3x3_3(x)   # (N, C/2, H, W)
        # Image Pooling
        out_img = self.avg_pool(x)      # (N, C/2, 1, 1)
        out_img = F.interpolate(out_img, size=(H, W), mode='bilinear', align_corners=True)  # (N, C/2, H, W)

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], dim=1) # (N, 5 * C/2, H, W)
        out = self.out_conv(out)    # (N, C, H, W)

        return out

class BasicModule(nn.Module):
    def __init__(self, in_channels=512, out_channels=256):
        super(BasicModule, self).__init__()
        self.aspp = ASPP(in_channels, out_channels)

    def forward(self, x):
        output = self.aspp(x)

        return output


if __name__ == '__main__':
    basic = BasicModule(in_channels=64, out_channels=32)
    basic.eval()
    input = torch.rand(4, 64, 320, 240)
    output = basic(input)
    print('s')