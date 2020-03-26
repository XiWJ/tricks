# Compute surface normal with depth GT
## 算法
```python
input: D Depth map，K Camera intrinsic
output: N Surface normal map
def GenerateSurfaceNormal(D, K)
    P (X, Y, Z) <-- f(D, K)
    for i in D:
        S = [p_i]
        for j in i neighborings:
            S <-- P_j
		# 最小二乘fitting
        N_i <-- least_squares_fitting(S)
	# 双边滤波
    bilateralFilter(N, sigma_s=24, sigma_I=0.2)
    
    return N
```
## 运行
matlab version: 2017a+
```
matlab generate_normal_gt_demon
```