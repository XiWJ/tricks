""" PyConv network for semantic segmentation  as presented in our paper:
    Duta et al. "Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition"
    https://arxiv.org/pdf/2006.11538.pdf
"""
import torch
from torch import nn
import torch.nn.functional as F

from model.build_backbone_layers import build_backbone_layers


class PyConv2d(nn.Module):
    """PyConv2d with padding (general case). Applies a 2D PyConv over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (list): Number of channels for each pyramid level produced by the convolution
        pyconv_kernels (list): Spatial size of the kernel for each pyramid level
        pyconv_groups (list): Number of blocked connections from input channels to output channels for each pyramid level
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``

    Example::

        >>> # PyConv with two pyramid levels, kernels: 3x3, 5x5
        >>> m = PyConv2d(in_channels=64, out_channels=[32, 32], pyconv_kernels=[3, 5], pyconv_groups=[1, 4])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)

        >>> # PyConv with three pyramid levels, kernels: 3x3, 5x5, 7x7
        >>> m = PyConv2d(in_channels=64, out_channels=[16, 16, 32], pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)
    """
    def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, stride=1, dilation=1, bias=False):
        super(PyConv2d, self).__init__()

        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)

        self.pyconv_levels = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels[i] = nn.Conv2d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                                              stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                                              dilation=dilation, bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))

        return torch.cat(out, 1)


class PyConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()

        self.conv2_1 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[0], stride=stride,
                                 padding=pyconv_kernels[0]//2, dilation=1, groups=pyconv_groups[0], bias=False)
        self.conv2_2 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[1], stride=stride,
                                 padding=pyconv_kernels[1] // 2, dilation=1, groups=pyconv_groups[1], bias=False)
        self.conv2_3 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[2], stride=stride,
                                 padding=pyconv_kernels[2] // 2, dilation=1, groups=pyconv_groups[2], bias=False)
        self.conv2_4 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[3], stride=stride,
                                 padding=pyconv_kernels[3] // 2, dilation=1, groups=pyconv_groups[3], bias=False)

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)


class GlobalPyConvBlock(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(GlobalPyConvBlock, self).__init__()
        self.features = nn.Sequential(
                nn.AdaptiveAvgPool2d(bins),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True),
                PyConv4(reduction_dim, reduction_dim),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduction_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_size = x.size()
        x = F.interpolate(self.features(x), x_size[2:], mode='bilinear', align_corners=True)
        return x


class LocalPyConvBlock(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=4):
        super(LocalPyConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes//reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True),
            PyConv4(inplanes // reduction1, inplanes // reduction1),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // reduction1, planes, kernel_size=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.layers(x)


class MergeLocalGlobal(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(MergeLocalGlobal, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(inplanes, planes,  kernel_size=3, padding=1, groups=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, local_context, global_context):
        x = torch.cat((local_context, global_context), dim=1)
        x = self.features(x)
        return x


class PyConvHead(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm):
        super(PyConvHead, self).__init__()

        out_size_local_context = 512
        out_size_global_context = 512

        self.local_context = LocalPyConvBlock(inplanes, out_size_local_context, BatchNorm, reduction1=4)
        self.global_context = GlobalPyConvBlock(inplanes, out_size_global_context, 9, BatchNorm)

        self.merge_context = MergeLocalGlobal(out_size_local_context + out_size_global_context, planes, BatchNorm)

    def forward(self, x):
        x = self.merge_context(self.local_context(x), self.global_context(x))
        return x


class PyConvSegNet(nn.Module):
    def __init__(self, layers=50, dropout=0.1, classes=2, zoom_factor=8,
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=True,
                 backbone_output_stride=16, backbone_net='resnet'):
        super(PyConvSegNet, self).__init__()
        assert layers in [50, 101, 152, 200]
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = build_backbone_layers(backbone_net,
                                                                                                 layers,
                                                                                                 pretrained,
                                                                                                 backbone_output_stride=backbone_output_stride,
                                                                                                 convert_bn=BatchNorm)
        backbone_output_maps = 2048
        out_merge_all = 256
        self.pyconvhead = PyConvHead(backbone_output_maps, out_merge_all, BatchNorm)

        self.aux = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(256, classes, kernel_size=1)
        )

        self.cls = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_merge_all, classes, kernel_size=1)
        )

    def forward(self, x, y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out_stage3 = self.layer3(x)
        x = self.layer4(out_stage3)

        x = self.pyconvhead(x)

        x = self.cls(x)

        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(x, y)

            aux = self.aux(out_stage3)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss = self.criterion(aux, y)

            return x.max(1)[1], main_loss, aux_loss

        else:
            return x


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    size_input = 473 #817 #577 #473 #713
    input = torch.rand(1, 3, size_input, size_input)#.cuda()
    model = PyConvSegNet(layers=50, dropout=0.1, classes=150, zoom_factor=8,
                      pretrained=False, backbone_output_stride=8, backbone_net='resnet')#.cuda()
    model.eval()
    print(model)
    output = model(input)
    print('PyConvSegNet', output.size())
