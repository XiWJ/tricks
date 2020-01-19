# training script tricks
- save_path = save_path_formatter(args, parser)
- https://github.com/sunghoonim/DPSNet/blob/c1ddae65dbcf03e4d093f3ee6961512270b66c9d/train.py#L72
- 自动根据训练的参数与默认参数不一样，建立保存的文件夹
```
def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path/timestamp
```
- adjust_learning_rate(args, optimizer, epoch)
- https://github.com/sunghoonim/DPSNet/blob/c1ddae65dbcf03e4d093f3ee6961512270b66c9d/train.py#L154
- 每个epoch进行学习率的调整
```
def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```
- batch_time = AverageMeter()
- https://github.com/sunghoonim/DPSNet/blob/c1ddae65dbcf03e4d093f3ee6961512270b66c9d/train.py#L181
- 一个记录平均值和当前值的类
```
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i):
        self.val = [0]*i
        self.avg = [0]*i
        self.sum = [0]*i
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        assert(len(val) == self.meters)
        self.count += n
        for i,v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)
```
- parameters = chain(pointmvsnet.parameters())
- https://github.com/sunghoonim/DPSNet/blob/c1ddae65dbcf03e4d093f3ee6961512270b66c9d/train.py#L139
```
 parameters = chain(dpsnet.parameters())
    optimizer = torch.optim.Adam(parameters, args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)
```