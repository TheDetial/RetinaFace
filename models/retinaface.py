# -*- coding: utf-8 -*-
#@File      : retinaface.py
#@Function  : retinaface定义

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from torchvision import models

from backbones import resnet18, resnet34, resnet50, resnet101, resnet152
from backbones import mobilev1025
from models.modules import *

# 每个先验框内是否包含人脸的概率
class ClassHead(nn.Module):
    def __init__(self, inchannels=64, num_anchors=2):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1_1 = nn.Conv2d(inchannels, self.num_anchors*2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x_in):
        out = self.conv1_1(x_in)
        out = out.permute(0, 2, 3, 1).contiguous()  # 调整shape维度： --->(n, h, w, num_anchors*2)

        return out.view(out.shape[0], -1, 2)  # (n, 所有的先验框, 每个先验框包含人脸的概率)

# 人脸框预测: xywh
class BboxHead(nn.Module):
    def __init__(self, inchannels=64, num_anchors=2):
        super(BboxHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1_1 = nn.Conv2d(inchannels, self.num_anchors*4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x_in):
        out = self.conv1_1(x_in)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)

# 关键点预测：5个，每个关键点xy
class LandmarkHead(nn.Module):
    def __init__(self, inchannels=64, num_anchors=2):
        super(LandmarkHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1_1 = nn.Conv2d(inchannels, self.num_anchors*5*2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x_in):
        out = self.conv1_1(x_in)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 5*2)

# retinaface模型定义
class RetinaFace(nn.Module):
    def __init__(self, cfg=None, pretrain=False, mode='train'):
        super(RetinaFace, self).__init__()
        self.pretrained = pretrain

        backbone = mobilev1025(self.pretrained)

        # 2、取有效特征层输出
        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_list = [cfg['in_channel']*2, cfg['in_channel']*4, cfg['in_channel']*8]

        # 3、构建特征金字塔：调用fpn模块
        self.fpn = FPN(in_channels_list, cfg['out_channel'])

        # 4、提升感受野：调用SSH模块 -->每个有效特征层均送给SSH
        self.ssh1 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh2 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh3 = SSH(cfg['out_channel'], cfg['out_channel'])

        # 5、检测头
        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

        # 模式选择：训练或测试
        self.mode = mode

    # 3个有效特征层
    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, x_in):

        # 1、backbone提取特征
        out = self.body.forward(x_in)

        # 2、fpn:特征金字塔
        fpn = self.fpn.forward(out)

        # 3、ssh处理
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])

        features = [feature1, feature2, feature3]

        # 4、结果堆叠
        # 每个SSH的输出均送给类别预测、bbox预测和landmarks预测
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        # 5、return
        # 返回三个有效特征层的堆叠结果：bbox,class,landmarks分开堆叠
        if self.mode == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)

        return output
