# Code Read Logs
## 1. 自定义组合卷积层
来源: PointMVSNet 

conv + bn + relu

[conv.py](PointMVSNet/model/nn/conv.py)
```python
class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        """default initialization"""
        init_uniform(self.conv)
        if self.bn is not None:
            init_bn(self.bn)
```

## 2. 使用yaml + parser输入网络配置
来源: pyconv

[yaml文件](pyconv/config/ade20k/ade20k_pyconvresnet50_pyconvsegnet.yaml)
```python
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pyconvresnet50_pyconvsegnet.yaml', help='config file') # 指定yaml文件
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def main():
    global args, logger
    args = get_parser() # 使用parser
    
    gray_folder = os.path.join(args.save_folder, 'gray')
```

## 3. 使用logger输出info
来源: pyconv get_logger

[使用解释](https://www.cnblogs.com/xianyulouie/p/11041777.html)
```python
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def main():
    ···
    logger = get_logger()
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    ···
```

## 4. 魔改ResNet
详见: pyconv

## 5. 网络中单独参数设置
详见: vit-pytorch/nn.Parameter()

[link](https://github.com/lucidrains/vit-pytorch/blob/85314cf0b6c4ab254fed4257d2ed069cf4f8f377/vit_pytorch/vit_pytorch.py#L97)
```python
self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
x += self.pos_embedding[:, :(n + 1)]
```

## 6. 记录输出到本地txt
- log日志记录并返回到console
- 参考[FrameNet](https://github.com/hjwdzh/FrameNet/blob/master/src/train_affine_dorn.py)
```python
args = parser.parse_args()
if args.save != "":
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    fp = open(args.save + '/logs.txt', 'w')
def log(str):
    if args.save != "":
        fp.write("%s\n" % (str))
        fp.flush()
    print(str)
    
def main():
    log('=> will save everything to {}'.format(args.save_path))
```

## 6. 学习率warmup和调整
详见: CascadeStereo

args.lrepochs = "10,12,14,16:2"

在10,12,14,16 epoch学习率除以2

WarmupMultiStepLR函数在[util](./utils.py)中. lr_scheduler.step()放在loss.backward() 和 optimizer.step() 后面
```python
def train():
    milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0/3, warmup_iters=500, last_epoch=len(TrainImgLoader) * start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        ...
        for batch_idx, sample in enumerate(TrainImgLoader): 
            ...
            output = model(...)
            ...
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
```