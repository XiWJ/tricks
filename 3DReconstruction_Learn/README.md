# 3D reconstruction

**NOTE:**   
[1] Forked from https://github.com/alyssaq/3Dreconstruction   
[2] More base knowledge from pdf of CMU CV class and CS231A class.    
3D reconstruction from 2D images pipeline

Steps:
1. Detect 2D points
2. Match 2D points across 2 images
3. Epipolar geometry   
  3a. If both intrinsic and extrinsic camera parameters are known, reconstruct with projection matrices.   
  3b. If only the intrinsic parameters are known, normalize coordinates and calculate the essential matrix.   
  3c. If neither intrinsic nor extrinsic parameters are known, calculate the fundamental matrix.   
4. With fundamental or essential matrix, assume P1 = [I 0] and calulate parameters of camera 2.
5. Triangulate knowing that x1 = P1 * X and x2 = P2 * X.
6. Bundle adjustment to minimize reprojection errors and refine the 3D coordinates.

Note: Steps and code in this repo is my hobby / learning exercise. Ie, its probably not very efficient. If you wish to use a more production-ready library, check out [OpenCV's SFM module](https://github.com/opencv/opencv_contrib/tree/master/modules/sfm). I have a docker environment for it at: https://github.com/alyssaq/reconstruction

## Prerequisites
* Python 3.5+
* Install [OpenCV](http://opencv.org/): [Mac installation steps](https://gist.github.com/alyssaq/f60393545173379e0f3f)
* pip install -r requirements.txt

## Example 3D cube reconstruction
```sh
$ python3 cube_reconstruction.py
```

## Example Dino 3D reconstruction from 2D images
Download images from <http://www.robots.ox.ac.uk/~vgg/data/data-mview.html> and place into `imgs/dinos`
```sh
$ python3 example.py
```

Detected points and matched across 2 images.
![](testsets/dino_2d_points.png?raw=true)

3D reconstructed dino with essential matrix   
![](testsets/dino_3d_reconstructed.png?raw=true)

## 3D to 2D Projection
```sh
$ python3 camera.py
```

3D points of model house from Oxford University VGG datasets.
![](testsets/house_3d.png?raw=true)

Projected points   
![](testsets/3d_to_2d_projection.png?raw=true)
## Datasets
* Oxford University, Visual Geometry Group: http://www.robots.ox.ac.uk/~vgg/data/data-mview.html
* EPFL computer vision lab: http://cvlabwww.epfl.ch/data/multiview/knownInternalsMVS.html

## References
* [Eight point algorithm](http://ece631web.groups.et.byu.net/Lectures/ECEn631%2013%20-%208%20Point%20Algorithm.pdf)
* [Multiple View Geometry in Computer Vision (Hartley & Zisserman)](http://www.robots.ox.ac.uk/~vgg/hzbook/)

## License
[MIT](https://alyssaq.github.io/mit-license)
