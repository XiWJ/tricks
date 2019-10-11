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