# 距离计算
## 两向量的欧式距离计算
distance.py中函数cal_distance_matrix(a, b)<br>
a(m,k)、b(n,k)，计算结果d(m,n)，表示两两间的欧式距离
# 网络冻结fineturn
## 神经网络冻结fineturn再解冻技巧
freeze_fineturn.py展示神经网络冻结训练技巧，pytorch实现，add_param_group函数调用在requires_grad操作之后。