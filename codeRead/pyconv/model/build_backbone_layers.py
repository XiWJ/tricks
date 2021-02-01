from collections import OrderedDict
import torch
from torch import nn

from model.backbones import resnet, pyconvresnet, pyconvhgresnet
from util.div.convert_BN import convert_BN


def build_backbone_layers(backbone_net, layers, pretrained, backbone_output_stride=8, convert_bn=None):

    if backbone_net == "pyconvhgresnet":
        if layers == 50:
            backbone = pyconvhgresnet.pyconvhgresnet50()
        elif layers == 101:
            backbone = pyconvhgresnet.pyconvhgresnet101()
        elif layers == 152:
            backbone = pyconvhgresnet.pyconvhgresnet152()

        if pretrained:
            print("Load pretrained model:  ", pretrained)
            backbone.load_state_dict(torch.load(pretrained), strict=True)

        if convert_bn and not isinstance(convert_bn, torch.nn.BatchNorm2d):#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print("Converting Batch Norm to: ", convert_bn)
            backbone = convert_BN(backbone, convert_bn)

        layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        layer1, layer2, layer3, layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4

        if backbone_output_stride == 8:
            for n, m in layer3.named_modules():
                if 'conv2_1' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif "conv2_2" in n:
                    m.dilation, m.padding, m.stride = (2, 2), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        if backbone_output_stride == 16:
            for n, m in layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        return layer0, layer1, layer2, layer3, layer4

    if backbone_net == 'pyconvresnet':
        if layers == 50:
            backbone = pyconvresnet.pyconvresnet50()
        elif layers == 101:
            backbone = pyconvresnet.pyconvresnet101()
        elif layers == 152:
            backbone = pyconvresnet.pyconvresnet152()

        if pretrained:
            print("Load pretrained model:  ", pretrained)
            backbone.load_state_dict(torch.load(pretrained), strict=True)

        if convert_bn and not isinstance(convert_bn, torch.nn.BatchNorm2d):#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print("Converting Batch Norm to: ", convert_bn)
            backbone = convert_BN(backbone, convert_bn)

        layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        layer1, layer2, layer3, layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4

        if backbone_output_stride == 8:
            for n, m in layer3.named_modules():
                if 'conv2_1' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif "conv2_2" in n:
                    m.dilation, m.padding, m.stride = (2, 2), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        if backbone_output_stride == 16:
            for n, m in layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        return layer0, layer1, layer2, layer3, layer4

    if backbone_net == 'resnet':
        if layers == 50:
            backbone = resnet.resnet50()
        elif layers == 101:
            backbone = resnet.resnet101()
        elif layers == 152:
            backbone = resnet.resnet152()

        if pretrained:
            print("Load pretrained model:  ", pretrained)
            backbone.load_state_dict(torch.load(pretrained), strict=True)

        if convert_bn and not isinstance(convert_bn, torch.nn.BatchNorm2d):#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print("Converting Batch Norm to: ", convert_bn)
            backbone = convert_BN(backbone, convert_bn)

        layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        layer1, layer2, layer3, layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4

        if backbone_output_stride == 8:
            for n, m in layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        if backbone_output_stride == 16:
            for n, m in layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        return layer0, layer1, layer2, layer3, layer4




