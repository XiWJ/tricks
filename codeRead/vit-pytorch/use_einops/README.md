# use einops
## install
```dockerfile
pip install einops
```

## rearrange
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