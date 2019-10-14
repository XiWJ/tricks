# PointMVSNet code tricks
PointMVSNet 的code也是宝藏代码, 是目前读过的代码中格式最为完备的, 今后将此作为基础框架应用起来.
## config设置
config设置采用yaml+config.py格式，具体位置在[yaml](https://github.com/callmeray/PointMVSNet/blob/master/configs/dtu_wde3.yaml)+[config.py](https://github.com/callmeray/PointMVSNet/blob/master/pointmvsnet/config.py)
```python
def load_cfg_from_file(cfg_filename):
    """Load config from a file

    Args:
        cfg_filename (str):

    Returns:
        CfgNode: loaded configuration

    """
    with open(cfg_filename, "r") as f:
        cfg = load_cfg(f)

    cfg_template = _C
    cfg_template.merge_from_other_cfg(cfg)
    return cfg_template

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Point-MVSNet Training")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="/home/xwj/github/PointMVSNet/configs/dtu_wde3.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args
    
args = parse_args()
cfg = load_cfg_from_file(args.config_file) # 加载yaml配置文件
cfg.merge_from_list(args.opts) # merge命令行输入opts
cfg.freeze()
```

## 训练/测试的logger
设置logger，在train或者test阶段可以实时记录日志在本地.txt文件，同时输出在console上。代码位置[logger](https://github.com/callmeray/PointMVSNet/blob/master/pointmvsnet/utils/logger.py)，先setup_logger，使用时候就logger.info("..."). 在train函数中配合logger = logging.getLogger("logger.name"）使用
```python
def setup_logger(name, save_dir, prefix="", timestamp=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        timestamp = time.strftime(".%m_%d_%H_%M_%S") if timestamp else ""
        prefix = "." + prefix if prefix else ""
        log_file = os.path.join(save_dir, "log{}.txt".format(prefix + timestamp))
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
    
logger = setup_logger("pointmvsnet", output_dir, prefix="train")
logger.info("Using {} GPUs".format(num_gpus))

def train(cfg, output_dir=""):
    logger = logging.getLogger("pointmvsnet.trainer")
```

## build optimizer
构建优化器，并将bn层的参数单独拿出来不进行weight decay, 该代码没有freeze bn层, 原因是它没有预训练. 代码位置在[solver.py](https://github.com/callmeray/PointMVSNet/blob/master/pointmvsnet/solver.py)
```python
# 对model参数进行group，并将bn层的参数单独拿出来
def group_weight(module, weight_decay):
    group_decay = []
    group_no_decay = []
    keywords = [".bn."]

    for m in list(module.named_parameters()):
        exclude = False
        for k in keywords:
            if k in m[0]:
                print("Weight decay exclude: "+m[0])
                group_no_decay.append(m[1])
                exclude = True
                break
        if not exclude:
            print("Weight decay include: " + m[0])
            group_decay.append(m[1])

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay, weight_decay=weight_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

# 构建优化器
def build_optimizer(cfg, model):
    name = cfg.SOLVER.TYPE # name=RMSprop
    if hasattr(torch.optim, name):
        def builder(cfg, model):
            return getattr(torch.optim, name)(
                group_weight(model, cfg.SOLVER.WEIGHT_DECAY),
                lr=cfg.SOLVER.BASE_LR,
                **cfg.SOLVER[name],
            )
    elif name in _OPTIMIZER_BUILDERS:
        builder = _OPTIMIZER_BUILDERS[name]
    else:
        raise ValueError("Unsupported type of optimizer.")

    return builder(cfg, model)
    
optimizer = build_optimizer(cfg, model)
```

## build lr scheduler
构建学习率的调度器，代码位置在[solver.py](https://github.com/callmeray/PointMVSNet/blob/master/pointmvsnet/solver.py). 每2 epoch降0.9.
```yaml
SCHEDULER:
  TYPE: "StepLR"
  INIT_EPOCH: 4
  MAX_EPOCH: 16
  StepLR:
    gamma: 0.9
    step_size: 2
```
```python
def build_scheduler(cfg, optimizer):
    name = cfg.SCHEDULER.TYPE # name=StepLR
    if hasattr(torch.optim.lr_scheduler, name):
        def builder(cfg, optimizer):
            return getattr(torch.optim.lr_scheduler, name)(
                optimizer,
                **cfg.SCHEDULER[name],
            )
    elif name in _OPTIMIZER_BUILDERS:
        builder = _OPTIMIZER_BUILDERS[name]
    else:
        raise ValueError("Unsupported type of optimizer.")

    return builder(cfg, optimizer)
    
# build lr scheduler
scheduler = build_scheduler(cfg, optimizer)

# main loop
for epoch in range(start_epoch, max_epoch):
    cur_epoch = epoch + 1
    scheduler.step()
```

## build checkpointer
构建checkpointer, 输入model, optimizer, scheduler, save_dir, logger. save_dir是模型训练文件保存位置. 代码位置在[checkpoint.py](https://github.com/callmeray/PointMVSNet/blob/master/pointmvsnet/utils/checkpoint.py). 直接拿来用,别管为什么, 知道怎么用就好.
```python
# Checkpointer 类
class Checkpointer(object):
    def __init__(
            self,
            model,
            optimizer=None,
            scheduler=None,
            save_dir="",
            logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, resume=True):
        if resume and self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self.model.load_state_dict(checkpoint.pop("model"), False)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

def train():
	# build checkpointer
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                logger=logger)

    # 加载checkpoint文件
    checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, resume=cfg.AUTO_RESUME)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD
    # 初始训练epoch
    start_epoch = checkpoint_data.get("epoch", 4)
    
    for epoch in range(start_epoch, max_epoch):
    	cur_epoch = epoch + 1
        train_meters = train_model(model,
                                   loss_fn,
                                   metric_fn,
                                   image_scales=cfg.MODEL.TRAIN.IMG_SCALES,
                                   inter_scales=cfg.MODEL.TRAIN.INTER_SCALES,
                                   isFlow=(cur_epoch > cfg.SCHEDULER.INIT_EPOCH),
                                   data_loader=train_data_loader,
                                   optimizer=optimizer,
                                   curr_epoch=epoch,
                                   tensorboard_logger=tensorboard_logger,
                                   log_period=cfg.TRAIN.LOG_PERIOD,
                                   output_dir=output_dir,
                                   )

        # 保存checkpoint文件
        if cur_epoch % ckpt_period == 0 or cur_epoch == max_epoch:
            checkpoint_data["epoch"] = cur_epoch
            checkpoint_data[best_metric_name] = best_metric
            checkpointer.save("model_{:03d}".format(cur_epoch), **checkpoint_data)
```

## build data loader
构建dataloader, 在训练时分别构建train和val两种dataloader.
```python
def build_data_loader(cfg, mode="train"):
    if mode == "train":
        dataset = DTU_Train_Val_Set(
            root_dir=cfg.DATA.TRAIN.ROOT_DIR,
            dataset_name="train",
            num_view=cfg.DATA.TRAIN.NUM_VIEW,
            interval_scale=cfg.DATA.TRAIN.INTER_SCALE,
            num_virtual_plane=cfg.DATA.TRAIN.NUM_VIRTUAL_PLANE,
        )
    elif mode == "val":
        dataset = DTU_Train_Val_Set(
            root_dir=cfg.DATA.VAL.ROOT_DIR,
            dataset_name="val",
            num_view=cfg.DATA.VAL.NUM_VIEW,
            interval_scale=cfg.DATA.TRAIN.INTER_SCALE,
            num_virtual_plane=cfg.DATA.TRAIN.NUM_VIRTUAL_PLANE,
        )
    elif mode == "test":
        dataset = DTU_Test_Set(
            root_dir=cfg.DATA.TEST.ROOT_DIR,
            dataset_name="test",
            num_view=cfg.DATA.TEST.NUM_VIEW,
            height=cfg.DATA.TEST.IMG_HEIGHT,
            width=cfg.DATA.TEST.IMG_WIDTH,
            interval_scale=cfg.DATA.TEST.INTER_SCALE,
            num_virtual_plane=cfg.DATA.TEST.NUM_VIRTUAL_PLANE,
        )
    else:
        raise ValueError("Unknown mode: {}.".format(mode))

    if mode == "train":
        batch_size = cfg.TRAIN.BATCH_SIZE
    else:
        batch_size = cfg.TEST.BATCH_SIZE

    data_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=(mode == "train"),
        num_workers=cfg.DATA.NUM_WORKERS,
    )

    return data_loader

def train():
	# build data loader
    train_data_loader = build_data_loader(cfg, mode="train")
    val_period = cfg.TRAIN.VAL_PERIOD
    val_data_loader = build_data_loader(cfg, mode="val") if val_period > 0 else None
```

## build tensorboard logger
构建并使用tensorboard, 记录训练变量. 代码位置在[tensorboard_logger.py](https://github.com/callmeray/PointMVSNet/blob/master/pointmvsnet/utils/tensorboard_logger.py)
```python
class TensorboardLogger(object):
    def __init__(self, log_dir, keywords=_KEYWORDS):
        self.log_dir = osp.join(log_dir, "events.{}".format(time.strftime("%m_%d_%H_%M_%S")))
        mkdir(self.log_dir)
        self.keywords = keywords
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def add_scalars(self, meters, step, prefix=""):
        for k, meter in meters.items():
            for keyword in _KEYWORDS:
                if keyword in k:
                    if isinstance(meter, AverageMeter):
                        v = meter.global_avg
                    elif isinstance(meter, (int, float)):
                        v = meter
                    elif isinstance(meter, torch.Tensor):
                        v = meter.cpu().item()
                    else:
                        raise TypeError()

                    self.writer.add_scalar(osp.join(prefix, k), v, global_step=step)

    def add_image(self, img, step, prefix=""):
        assert len(img.size()) == 3
        self.writer.add_image(osp.join(prefix, "_img"), img, global_step=step)
        
def train():
	# build tensorboard logger (optionally by comment)
    tensorboard_logger = TensorboardLogger(output_dir)
    
    # 写scalar
    tensorboard_logger.add_scalars(loss_dict, curr_epoch * total_iteration + iteration, prefix="train")
    tensorboard_logger.add_scalars(metric_dict, curr_epoch * total_iteration + iteration, prefix="train")
```

## build MetricLogger
build MetricLogger for training. 使用时候就update. 位置[metric_logger.py](https://github.com/callmeray/PointMVSNet/blob/master/pointmvsnet/utils/metric_logger.py)
```python
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            count = 1
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = v.item()
                else:
                    count = v.numel()
                    v = v.sum().item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, count)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattr__(self, attr)

    def __str__(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.avg, meter.global_avg)
            )
        return self.delimiter.join(metric_str)

    @property
    def summary_str(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(metric_str)
        
def train():
	# 定义MetricLogger
	meters = MetricLogger(delimiter="  ")
    # 使用MetricLogger
    losses = sum(loss_dict.values())
    meters.update(loss=losses, **loss_dict, **metric_dict)
    # 带MetricLogger输出log
    logger.info(
                meters.delimiter.join(
                    [
                        "EPOCH: {epoch:2d}",
                        "iter: {iter:4d}",
                        "{meters}",
                        "lr: {lr:.2e}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    epoch=curr_epoch,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )
```
输出
```bash
INFO: EPOCH:  4  iter:    0  loss: 9.0616 (9.0616)  coarse_loss: 1.8164 (1.8164)  flow1_loss: 2.4203 (2.4203)  flow2_loss: 4.8249 (4.8249)  <1_pct_cor: 0.0647 (0.0647)  <3_pct_cor: 0.2270 (0.2270)  <1_pct_flow1: 0.0756 (0.0756)  <3_pct_flow1: 0.2459 (0.2459)  <1_pct_flow2: 0.1038 (0.1038)  <3_pct_flow2: 0.3318 (0.3318)  time: 14961.6561 (14961.6561)  data: 0.6848 (0.6848)  lr: 5.00e-04  max mem: 6735
```

## norm_image
归一化输入的image，位置在[preprocess.py](https://github.com/callmeray/PointMVSNet/blob/master/pointmvsnet/utils/preprocess.py)
```python
def norm_image(img):
    """ normalize image input """
    img = img.astype(np.float32)
    var = np.var(img, axis=(0, 1), keepdims=True)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 1e-7)
    
image = cv2.imread(paths["view_image_paths"][view])
image = norm_image(image)
```

## load_cam_dtu
加载camera.txt文件，load_cam_dtu函数不重要， **重要的是file.read().split()** 的运用，位置在[io.py](https://github.com/callmeray/PointMVSNet/blob/master/pointmvsnet/utils/io.py)
```python
def load_cam_dtu(file, num_depth=0, interval_scale=1.0):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    words = file.read().split() # 直接读取.txt所有数据
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = num_depth
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (num_depth - 1)
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (num_depth - 1)
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam
    
cam = io.load_cam_dtu(open(paths["view_cam_paths"][view]),
                      num_depth=self.num_virtual_plane,
                      interval_scale=self.interval_scale)
```

## grid_sample
```python
def feature_fetcher(pts): 
    # pts reference view 点云的世界坐标
    with torch.no_grad():
        num_pts = pts.size(2) # point数目
        pts_expand = pts.unsqueeze(1).contiguous().expand(batch_size, num_view, 3, num_pts) \
            .contiguous().view(curr_batch_size, 3, num_pts) # 维度变换
        if cam_extrinsics is None:
            transformed_pts = pts_expand.type(torch.float).transpose(1, 2)
        else:
            cam_extrinsics = cam_extrinsics.view(curr_batch_size, 3, 4)	# 相机外参
            R = torch.narrow(cam_extrinsics, 2, 0, 3)					# 旋转矩阵
            t = torch.narrow(cam_extrinsics, 2, 3, 1).expand(curr_batch_size, 3, num_pts)	# 平移向量
            transformed_pts = torch.bmm(R, pts_expand) + t				# Q^c = RQ^w + t 相机坐标 reference view下
            transformed_pts = transformed_pts.type(torch.float).transpose(1, 2)
        x = transformed_pts[..., 0]		# Q^c = [X^c, Y^c, Z^c]
        y = transformed_pts[..., 1]		# Y^c
        z = transformed_pts[..., 2]		# Z^c

        normal_uv = torch.cat(
            [torch.div(x, z).unsqueeze(-1), torch.div(y, z).unsqueeze(-1), torch.ones_like(x).unsqueeze(-1)],
            dim=-1)		# uv = 1/Z^c * Q^c = [u, v, 1] 归一化
        uv = torch.bmm(normal_uv, cam_intrinsics.transpose(1, 2)) # q = K dot Q^c / Z^c = K dot uv, 像素坐标
        uv = uv[:, :, :2]

        grid = (uv - 0.5).view(curr_batch_size, num_pts, 1, 2)	# 像素坐标 [u, v]
        grid[..., 0] = (grid[..., 0] / float(width - 1)) * 2 - 1.0 # u
        grid[..., 1] = (grid[..., 1] / float(height - 1)) * 2 - 1.0 # v

        # grid_sample实际上是一种采样, 根据gird存储的点云坐标信息, 去feature_maps对应位置找特征向量, 保存在最后的pts_feature中.
        # feature_maps (B, C, H, W)
        # grid 		(B, point_num, 1, 2) 
        # grid[3] -> [2] -> [u, v] 坐标, 归一化(-1, 1), [u] -1代表最左边, [v] -1代表最上边.
        # grid中坐标超过 (-1, 1)的, 在feature_maps找不到, 直接填充0.
        pts_feature = F.grid_sample(feature_maps, grid, mode=self.mode) 
```

## 自定义Conv
自定义conv组合, conv+bn+relu结构, 并进行初始化<br>
[conv.py](https://github.com/callmeray/PointMVSNet/blob/master/pointmvsnet/nn/conv.py)
```python
from torch import nn
import torch.nn.functional as F

from .init import init_uniform, init_bn


class Conv1d(nn.Module):
    """Applies a 1D convolution over an input signal composed of several input planes.
    optionally followed by batch normalization and relu activation

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv1d, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, momentum=bn_momentum) if bn else None
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


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
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
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
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


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
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
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        self.init_weights()

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        """default initialization"""
        init_uniform(self.conv)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
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
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        self.init_weights()

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        """default initialization"""
        init_uniform(self.conv)
        if self.bn is not None:
            init_bn(self.bn)
```
使用
```python
from pointmvsnet.nn.conv import *

class ImageConv(nn.Module):
    def __init__(self, base_channels):
        super(ImageConv, self).__init__()
        self.base_channels = base_channels
        self.out_channels = 8 * base_channels
        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1, bias=False)
        )

    def forward(self, imgs):
        out_dict = {}

        conv0 = self.conv0(imgs)
        out_dict["conv0"] = conv0
        conv1 = self.conv1(conv0)
        out_dict["conv1"] = conv1
        conv2 = self.conv2(conv1)
        out_dict["conv2"] = conv2
        conv3 = self.conv3(conv2)
        out_dict["conv3"] = conv3

        return out_dict
```

## get_knn_3d
真的NB, 通过卷积运行来找point KNN紧邻, 而且是矩阵中每个点同时找, 同时计算distance.<br>
输入点云 (B, 3, D, H, W), 3是(x,y,z)世界坐标, $D *H * W$是point点总数.<br>
输出每个点最k紧邻的索引 idx (B, DHW, k)<br>
[torch_utils.py](https://github.com/callmeray/PointMVSNet/blob/master/pointmvsnet/utils/torch_utils.py)
```python
def get_knn_3d(xyz, kernel_size=5, knn=20):
    """ Use 3D Conv to compute neighbour distance and find k nearest neighbour
          xyz: (B, 3, D, H, W)

      Returns:
        idx: (B, D*H*W, k)
    """
    batch_size, _, depth, height, width = list(xyz.size())
    assert (kernel_size % 2 == 1)
    hk = (kernel_size // 2)
    k2 = kernel_size ** 2
    k3 = kernel_size ** 3

    t = np.zeros((kernel_size, kernel_size, kernel_size, 1, kernel_size ** 3))
    ind = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                t[i, j, k, 0, ind] -= 1.0
                t[hk, hk, hk, 0, ind] += 1.0
                ind += 1
    weight = np.zeros((kernel_size, kernel_size, kernel_size, 3, 3 * k3))
    weight[:, :, :, 0:1, :k3] = t
    weight[:, :, :, 1:2, k3:2 * k3] = t
    weight[:, :, :, 2:3, 2 * k3:3 * k3] = t
    weight = torch.tensor(weight).float()

    weights_torch = torch.Tensor(weight.permute((4, 3, 0, 1, 2))).to(xyz.device)
    dist = F.conv3d(xyz, weights_torch, padding=hk)

    dist_flat = dist.contiguous().view(batch_size, 3, k3, -1)
    dist2 = torch.sum(dist_flat ** 2, dim=1)

    _, nn_idx = torch.topk(-dist2, k=knn, dim=1)
    nn_idx = nn_idx.permute(0, 2, 1)
    d_offset = nn_idx // k2 - hk
    h_offset = (nn_idx % k2) // kernel_size - hk
    w_offset = nn_idx % kernel_size - hk

    idx = torch.arange(depth * height * width).to(xyz.device)
    idx = idx.view(1, -1, 1).expand(batch_size, -1, knn)
    idx = idx + (d_offset * height * width) + (h_offset * width) + w_offset

    idx = torch.clamp(idx, 0, depth * height * width - 1)

    return idx
    
nn_idx = get_knn_3d(xyz, len(interval_list), knn=self.k)
```

## gather_knn
PointMVSNet作者代码功力真的强, gather_knn是用C++完成编译生成.so文件, 通过CUDA编译, 形成pythorch接口<br>
[gather_knn.py](https://github.com/callmeray/PointMVSNet/blob/master/pointmvsnet/functions/gather_knn.py)
```python
# import torch before loading cuda extension
import torch

try:
    from pointmvsnet.functions import dgcnn_ext
except ImportError:
    print("Please compile source files before using dgcnn cuda extension.")


class GatherKNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature, index):
        ctx.save_for_backward(index)
        feature_neighbour = dgcnn_ext.gather_knn_forward(feature, index)
        return feature_neighbour

    @staticmethod
    def backward(ctx, grad_output):
        knn_inds = ctx.saved_tensors[0]
        grad_features = dgcnn_ext.gather_knn_backward(grad_output, knn_inds)
        return grad_features, None

gather_knn = GatherKNN.apply

def test_gather_knn():
    torch.manual_seed(1)
    batch_size = 2
    num_inst = 5
    channels = 4
    k = 3

    feature_tensor = torch.rand(batch_size, channels, num_inst).cuda(0)
    # knn_inds = torch.ones([batch_size, num_inst, k], dtype=torch.int64).cuda(0)
    # knn_inds[:, :, 2] = 2
    # knn_inds[:, 0, 2] = 3
    knn_inds = torch.randint(0, num_inst, [batch_size, num_inst, k]).long().cuda(0)

    feature_tensor_gather = torch.zeros_like(feature_tensor).copy_(feature_tensor)
    feature_tensor_gather.requires_grad = True
    feature_tensor_cuda = torch.zeros_like(feature_tensor).copy_(feature_tensor)
    feature_tensor_cuda.requires_grad = True

    feature_expand = feature_tensor_gather.unsqueeze(2).expand(batch_size, channels, num_inst, num_inst)
    knn_inds_expand = knn_inds.unsqueeze(1).expand(batch_size, channels, num_inst, k)
    feature_gather = torch.gather(feature_expand, 3, knn_inds_expand)

    feature_cuda = gather_knn(feature_tensor_cuda, knn_inds)
    print("Forward:", feature_gather.allclose(feature_cuda))

    feature_gather.backward(torch.ones_like(feature_gather))
    feature_cuda.backward(torch.ones_like(feature_cuda))
    grad_gather = feature_tensor_gather.grad
    grad_cuda = feature_tensor_cuda.grad
    print("Backward:", grad_gather.allclose(grad_cuda))


if __name__ == "__main__":
    test_gather_knn()
```
在EdgeConv网络中, gather edge_feature 中每个pixel的k个紧邻的feature
```python
class EdgeConvNoC(nn.Module):
    def __init__(self, in_channels, out_channels):
     	...

    def forward(self, feature, knn_inds):
        ...

        if feature.is_cuda:
            # gather edge_feature 中每个pixel的k个紧邻的feature
            neighbour_feature = gather_knn(edge_feature, knn_inds)  # 32x25600x16
        else:
            # pytorch gather
            knn_inds_expand = knn_inds.unsqueeze(1).expand(batch_size, channels, num_points, k)
            edge_feature_expand = edge_feature.unsqueeze(2).expand(batch_size, -1, num_points, num_points)
            neighbour_feature = torch.gather(edge_feature_expand, 3, knn_inds_expand)

        ...

        return ...
```

