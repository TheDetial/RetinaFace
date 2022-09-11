# -*- coding: utf-8 -*-
from itertools import product as product
from math import ceil  # 向上取整

import torch
#@File      : anchor.py


# 计算有效特征层对应的anchors尺寸大小
class Anchors(object):
    def __init__(self, cfg, image_size=None):
        super(Anchors, self).__init__()
        self.anchors_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']

        # 网络的预设输入尺寸 （比如：640*640）
        self.image_size = image_size

        # 三个有效特征层的宽高（网络输入尺寸/降采样步长后，向上取整）
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]

    # self.feature_maps: [[80, 80], [40, 40], [20, 20]]

    # 循环计算每个有效特征层对应的anchors尺寸
    def get_anchors(self):
        anchors = []

        # 循环1：循环依次取出三个有效特征层
        # k, f = 0, [80, 80]
        # k, f = 1, [40, 40]
        # k, f = 2, [20, 20]
        for k, f in enumerate(self.feature_maps):
            # k=0, anchors_sizes=[16, 32]
            # k=1, anchors_sizes=[64, 128]
            # k=2, anchors_sizes=[256, 512]
            anchors_sizes = self.anchors_sizes[k]  # 取该有效特征层对应的预设anchor尺寸，大特征层给小anchor尺寸，小特征层给大anchor尺寸

            # 循环2：循环每个有效特征层的宽高尺寸
            # 本质为循环每个网格点：每个网格点对应两个先验框，且每个先验框都为正方形
            for i, j in product(range(f[0]), range(f[1])):
                # 有效特征层维度的网格点坐标表示
                # 例：有效特征层为80*80时，
                # 依次取出[i, j] = [ [0,0],[0,1],[0,2],...,[0,77],[0,78],[0,79]
                # 			 ... ...
                # 		     [79,0],[79,1],[79,2],...,[79,77],[79,78],[79,79] ]

                for anchor_size in anchors_sizes:
                    # （1）每个网格点的anchor尺寸均为正方形（由预设网络输入尺寸为正方形决定的） ---归一化
                    s_kx = anchor_size / self.image_size[1]  # anchor尺寸，宽归一化 ---16/640
                    s_ky = anchor_size / self.image_size[0]  # anchor尺寸，高归一化 ---16/640
                    # （2）网格点坐标  ---归一化
                    # i,j加0.5后变成小数再做归一化,相当于把grid网格点坐标变成格子的中心点的形式
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    # （3）取出网格点坐标cx,cy
                    for cy, cx in product(dense_cy, dense_cx):
                        # anchors：中心点坐标+宽高的形式
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.Tensor(anchors).view(-1, 4)

        if self.clip:
            output.clamp_(max=1, min=0)  # 截断
        return output

# output维度：[16800, 4]
# 总anchor个数为16800：2*(80*80+40*40+20*20)=16800
# 维度为4：中心点坐标+宽高共4个参数
# 每个grid网格点给两个正方形先验框anchor尺寸：
# 返回的anchor形式如下：
# ...
# [[0.875, 0.975, 0.4, 0.4]
# [0.875, 0.975, 0.8, 0.8]
# [0.925, 0.975, 0.4, 0.4]
# [0.925, 0.975, 0.8, 0.8]
# [0.975, 0.975, 0.4, 0.4]
# [0.975, 0.975, 0.8, 0.8]]
# 依次类推...
