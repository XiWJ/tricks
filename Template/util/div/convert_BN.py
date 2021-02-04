import torch


def convert_BN(module, output_BN, initial_BN=torch.nn.BatchNorm2d):


    mod = module
    if isinstance(module, initial_BN):
        mod = output_BN(module.num_features, module.eps, module.momentum, module.affine)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()

    for name, child in module.named_children():
        mod.add_module(name, convert_BN(child, output_BN))

    return mod
