# Code Read CascadeStereo
github [url](https://github.com/alibaba/cascade-stereo)

## 1. torch.no_grad warpper for functions
不解释, 直接用. 在要no_grad的函数前加一句@make_nograd_func

@make_nograd_func函数在[util](./utils.py)中.
```python
@make_nograd_func
def test_sample_depth(model, model_loss, sample, args):
    if is_distributed:
        model_eval = model.module
    else:
        model_eval = model
    model_eval.eval()
```

## 2. 打印parser
print_args也在[util](./utils.py)中.
```python
# print arguments
def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")

if __name__ == '__main__':
    args = parser.parse_args()
    print_args(args)
```

## 3. 统计model参数
```python
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
```

## 4. 学习率warmup和steps
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

## 5. 保存checkpoint
gc.collect() # 回收内存
```python
# checkpoint
if (not is_distributed) or (dist.get_rank() == 0):
    if (epoch_idx + 1) % args.save_freq == 0:
        torch.save({
            'epoch': epoch_idx,
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict()},
            "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
gc.collect() # 回收内存
```

## 6 多线程
[博客解释１](https://note.qidong.name/2018/11/python-multiprocessing/)

[博客解释２](https://www.cnblogs.com/wxys/p/13756552.html)
```python
def filter_depth(pair_folder, scan_folder, out_folder, plyfilename):
    # the pair file
    pair_file = os.path.join(pair_folder, "pair.txt")
    ...
    
def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def pcd_filter_worker(scan):    # scan就是pcd_filter()的testlist, 是个list
    if args.testlist != "all":
        scan_id = int(scan[4:])
        save_name = 'mvsnet{:0>3}_l3.ply'.format(scan_id)
    else:
        save_name = '{}.ply'.format(scan)
    pair_folder = os.path.join(args.testpath, scan)
    scan_folder = os.path.join(args.outdir, scan)
    out_folder = os.path.join(args.outdir, scan)
    filter_depth(pair_folder, scan_folder, out_folder, os.path.join(args.outdir, save_name))


def pcd_filter(testlist, number_worker):

    partial_func = partial(pcd_filter_worker)

    p = Pool(number_worker, init_worker)
    try:
        p.map(partial_func, testlist)
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()

pcd_filter(testlist, args.num_worker) # args.num_worker==4
```