import numpy as np
import cv2
# pip install einops
from einops import rearrange

# suppose we have a set of 32 images in "h w c" format (height-width-channel)
images = [np.random.randn(30, 40, 3) for _ in range(32)]
x = rearrange(images, 'b h w c -> b h w c')  # (32, 30, 40, 3)

# concatenate images along height (vertical axis), 960 = 32 * 30
x = rearrange(images, 'b h w c -> (b h) w c')  # (960, 40, 3)

# concatenated images along horizontal axis, 1280 = 32 * 40
x = rearrange(images, 'b h w c -> h (b w) c')  # (30, 1280, 3)

# reordered axes to "b c h w" format for deep learning
x = rearrange(images, 'b h w c -> b c h w')  # (32, 3, 30, 40)

# flattened each image into a vector, 3600 = 30 * 40 * 3
x = rearrange(images, 'b h w c -> b (c h w)')  # (32, 3600)

# split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
x = rearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c', h1=2, w1=2)  # (128, 15, 20, 3)

# space-to-depth operation
x = rearrange(images, 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2, w1=2)  # (32, 15, 20, 12)

rgb_path = "./tmp/rgbd_10_to_20_handheld_train_00000/0001.jpg"
images = [cv2.imread(rgb_path)]
# split into 4 smaller part, (top-left, top-right, bottom-left, bottom-right), (1.png, 2.png, 3.png, 4.png)
x = rearrange(images, 'b (p1 h) (p2 w) c -> b h w (p1 p2 c)', p1=2, p2=2)
cv2.imwrite("./tmp/1.png", x[0, :, :, :3])
cv2.imwrite("./tmp/2.png", x[0, :, :, 3:6])
cv2.imwrite("./tmp/3.png", x[0, :, :, 6:9])
cv2.imwrite("./tmp/4.png", x[0, :, :, 9:])
print('s')