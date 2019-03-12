#!/usr/bin/env python
# coding=utf-8
""" An example showcasing the logging system of Sacred."""
from sacred import Experiment
import torch
import torch.nn.functional as F
ex = Experiment()

"""
compute a(m,k) between b(n,k) distance, oushi distance
"""
def cal_distance_matrix(point_a, point_b):
    '''
    point_a point_b oushi distance
    :param point_a: tensor of size (m, 2)
    :param point_b: tensor of size (n, 2)
    :return: distance matrix of size (m, n)
    '''
    m, n = point_a.size(0), point_b.size(0)

    a_repeat = point_a.repeat(1, n).view(n * m, 2)  # (n*m, 2)
    b_repeat = point_b.repeat(m, 1)  # (n*m, 2)

    distance = torch.nn.PairwiseDistance(keepdim=True)(a_repeat, b_repeat)  # (n*m, 1)

    return distance.view(m, n)

@ex.main
def my_main(_log):
   a = torch.Tensor([[1.,2.],
                     [2.,3.],
                     [4.,5.],
                     [6.,7.],
                     [9.,8.]])
   b = torch.Tensor([[1.,0.],
                     [4.,1.]])
   d = cal_distance_matrix(a, b)

   print(d)


if __name__ == '__main__':
    ex.run_commandline()
