import torch
import torch.nn as nn
import torch.nn.functional as F

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(reduction_dim),
                    nn.ReLU(inplace=True)
                )
            )
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()   # (N, C, H, W)
        out = [x]

        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))

        return torch.cat(out, 1)

class BasicModule(nn.Module):
    def __init__(self, in_dim, ratio=8):
        super(BasicModule, self).__init__()
        self.ppm = PPM(in_dim, in_dim // ratio, bins=(1, 2, 3, 6))

    def forward(self, x):
        output = self.ppm(x)

        return output

if __name__ == '__main__':
    basic = BasicModule(64)
    basic.eval()
    input = torch.rand(4, 64, 256, 192)
    output = basic(input)
    print('s')