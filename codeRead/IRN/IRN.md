# IRN code read
## ResNet50
[url](https://github.com/jiwoon-ahn/irn/blob/master/net/resnet50.py)
```
def resnet50(pretrained=True, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict)
    return model
```
## GronpNorm & Upsample in nn.Sequential
```
self.fc_dp6 = nn.Sequential(
            nn.Conv2d(768, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )

edge_out = self.fc_edge6(x)
```
## mean shift in nn.Sequential
[url](https://github.com/jiwoon-ahn/irn/blob/a2ece0034d0a5a9635e4e131cd4fa6e2a779d316/net/resnet50_irn.py#L87)
```
self.fc_dp7 = nn.Sequential(
            nn.Conv2d(448, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1, bias=False),
            self.mean_shift
        )
        
class MeanShift(nn.Module):

    def __init__(self, num_features):
        super(Net.MeanShift, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))

    def forward(self, input):
        if self.training:
            return input
        return input - self.running_mean.view(1, 2, 1, 1)

dp_out = self.fc_dp7(torch.cat([dp1, dp2, dp_up3], dim=1))
```