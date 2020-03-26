## initialization
# most code from https://zhuanlan.zhihu.com/p/62850258
import torch
import math

def tanh(x):
    return torch.tanh(x)

def relu(x):
    return x.clamp_min(0.)

def kaiming(m, h):
    return torch.randn(m, h)*math.sqrt(2./m)

def xavier(m, h):
    return torch.Tensor(m, h).uniform_(-1, 1)*math.sqrt(6./(m+h))

x = torch.randn(512)

for i in range(100):
    a = kaiming(512, 512) #xavier(512, 512)#torch.randn(512, 512) * math.sqrt(1./512)
    x = relu(a @ x) #a @ x

print(x.mean())
print(x.std())
