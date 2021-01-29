# vit-pytorch code read
## vit-pytorch link
[vit-pytorch](https://github.com/lucidrains/vit-pytorch)

## use_einops
[code](./use_einops/use_einops.py)
### rearrange
```python
rgb_path = "./tmp/rgbd_10_to_20_handheld_train_00000/0001.jpg"
images = [cv2.imread(rgb_path)]
# split into 4 smaller part, (top-left, top-right, bottom-left, bottom-right), (1.png, 2.png, 3.png, 4.png)
x = rearrange(images, 'b (p1 h) (p2 w) c -> b h w (p1 p2 c)', p1=2, p2=2)
cv2.imwrite("./tmp/1.png", x[0, :, :, :3])
cv2.imwrite("./tmp/2.png", x[0, :, :, 3:6])
cv2.imwrite("./tmp/3.png", x[0, :, :, 6:9])
cv2.imwrite("./tmp/4.png", x[0, :, :, 9:])
```

###
```python
self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # (1, 1, 1024)
cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)   # (3, 1, 1024)
```

## einsum
[参考link](https://blog.csdn.net/a2806005024/article/details/96462827)
```python
# (B, 16, 65, 64) dot (B, 16, 65, 64) -> (1, 16, 65, 65)
dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
```

## nn.Parameter
[link](https://github.com/lucidrains/vit-pytorch/blob/85314cf0b6c4ab254fed4257d2ed069cf4f8f377/vit_pytorch/vit_pytorch.py#L97)
```python
self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
x += self.pos_embedding[:, :(n + 1)]
```