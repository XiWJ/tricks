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
```
## 固定batch norm层
```python
# in class model.init()
## Fix batch norm layers
def set_bn_fix(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad = False

self.apply(set_bn_fix)
```