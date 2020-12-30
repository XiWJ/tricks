import torch
import torch.nn as nn

## step.2 position attention
class PositionAttentionModule(nn.Module):
    def __init__(self, in_channels=64):
        super(PositionAttentionModule, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)   # conv B in dual attention
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)     # conv C
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)        # conv D

        self.gamma = nn.Parameter(torch.zeros(1))   # learnable parameter initialize from zero

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        N, C, H, W = x.size()
        query = self.query_conv(x).view(N, -1, H*W).permute(0, 2, 1) # (N, C, H, W) -> (N, C', H, W) -> (N, H*W, C')
        key = self.key_conv(x).view(N, -1, H*W) # (N, C, H, W) -> (N, C', H, W) -> (N, C', H*W)

        # calculate correlation
        enery = torch.bmm(query, key)   # (N, H*W, C') bmm (N, C', H*W) -> (N, H*W, H*W)
        # spatial normalize
        attention = self.softmax(enery) # (N, H*W, H*W) softmax along spatial dimension (dim=-1, H*W)

        value = self.value_conv(x).view(N, -1, H*W) # (N, C, H, W) -> (N, C, H*W)

        # NOTE!! transpose the attention from (N, H*W, H*W) to (N, H*W, H*W), swap dim=1 <-> dim=2
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (N, C, H*W) bmm (N, H*W, H*W) -> (N, C, H*W)
        out = out.view(N, C, H, W)  # (N, C, H, W)
        out = self.gamma * out + x  # gamma * out + x

        return out

## step.2 channel attention
class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super(ChannelAttentionModule, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        N, C, H, W = x.size()
        query = x.view(N, C, -1)    # (N, C, H, W) -> (N, C, H*W)
        key = x.view(N, C, -1).permute(0, 2, 1) # (N, C, H, W) -> (N, H*W, C)

        # calculate correlation
        enery = torch.bmm(query, key)   # (N, C, C)
        enery = torch.max(enery, -1, keepdim=True)[0].expand_as(enery) - enery  # trick?
        attention = self.softmax(enery) # (N, C, C) softmax along dim=-1 C channel

        value = x.view(N, C, -1)    # (N, C, H*W)

        out = torch.bmm(attention, value)   # bmm del C(already softmax dimension)
        out = out.view(N, C, H, W)
        out = self.gamma * out + x

        return out

class BasicModule(nn.Module):
    def __init__(self, in_channels=64):
        super(BasicModule, self).__init__()
        ## step.1 convolution
        # convolution before attention modules
        self.conv2pam = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.conv2cam = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        ## step.2 attention
        # position attention
        self.PA = PositionAttentionModule(in_channels)
        # channel attention
        self.CA = ChannelAttentionModule()

        ## step.3 convolution & output
        # convolution after attention modules
        self.pam2conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.cam2conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        # output layer
        self.conv_out = nn.Sequential(
            # nn.Dropout2d(0.1, False),
            nn.Conv2d(in_channels, in_channels, 1)
        )

    def forward(self, x):
        pam_out = self.conv2pam(x)
        pam_out = self.PA(pam_out)
        pam_out = self.pam2conv(pam_out)

        cam_out = self.conv2cam(x)
        cam_out = self.CA(cam_out)
        cam_out = self.cam2conv(cam_out)

        feat_sum = pam_out + cam_out
        output = self.conv_out(feat_sum)

        return output


if __name__ == '__main__':
    basicModule = BasicModule()
    basicModule.eval()
    input = torch.ones((4, 64, 32, 24))
    output = basicModule(input)
    print('s')