# PointMVSNet code tricks
PointMVSNet 的code也是宝藏代码
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
构建优化器，并将bn层的参数单独拿出来，代码位置在[solver.py](https://github.com/callmeray/PointMVSNet/blob/master/pointmvsnet/solver.py)
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
构建学习率的调度器，代码位置在[solver.py](https://github.com/callmeray/PointMVSNet/blob/master/pointmvsnet/solver.py)
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
```

## build checkpointer

## build data loader