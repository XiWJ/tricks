# PlaneRCNN trick专题
PlaneRCNN是个宝藏code，里面各种trick
## 设置特定层训练
```python
# in class model
def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
    """Sets model layers as trainable if their names match
    the given regular expression.
    """

    for param in self.named_parameters():
        layer_name = param[0]
        trainable = bool(re.fullmatch(layer_regex, layer_name))
        if not trainable:
            param[1].requires_grad = False
  
# in main.py
if options.trainingMode != '':
    ## Specify which layers to train, default is "all"
    layer_regex = {
        ## all layers but the backbone
        "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        ## From a specific Resnet stage and up
        "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        ## All layers
        "all": ".*",
        "classifier": "(classifier.*)|(mask.*)|(depth.*)",
    }
    assert(options.trainingMode in layer_regex.keys())
    layers = layer_regex[options.trainingMode]
    model.set_trainable(layers)
    pass
# trainable model parameters
trainables_wo_bn = [param for name, param in model.named_parameters() if param.requires_grad and not 'bn' in name]
```
## 固定batch norm层
```python
# in class model.init() or model.predict()
## Fix batch norm layers
def set_bn_fix(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad = False

self.apply(set_bn_fix)
```

## Sequential 搭建网络
```python
# in model.init()
self.deconv4 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
# in model.forward()
x = self.deconv4(torch.cat([self.conv4(feature_maps[3]), x], dim=1))
```

## 更换list次序
```python
## Convert from list of lists of level outputs to list of lists
## of outputs across levels.
## e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
outputs = list(zip(*layer_outputs))
outputs = [torch.cat(list(o), dim=1) for o in outputs]
rpn_class_logits, rpn_class, rpn_bbox = outputs
```