# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# conv2d_3*3+bn+leakyrelu
def conv_bn(in_channels, out_channels, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

# conv2d_1*1+bn+leakyrelu
def conv_bn1X1(in_channels, out_channels, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

# conv2d_3*3+bn
def convbn_norelu(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
    )

# retinaface中的SSH多尺度模块
class SSH(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSH, self).__init__()
        assert out_channels % 4 == 0
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1

        # 3*3卷积
        self.convbn33_norelu = convbn_norelu(in_channels, out_channels//2, stride=1)

        # 5*5卷积：使用两个3*3串联
        self.convbn55_1 = conv_bn(in_channels, out_channels//4, stride=1, leaky=leaky)
        self.convbn55_2 = convbn_norelu(out_channels//4, out_channels//4, stride=1)

        # 7*7卷积：使用三个3*3串联
        self.convbn77_2 = conv_bn(out_channels//4, out_channels//4, stride=1, leaky=leaky)  # 接convbn55_1的输出
        self.convbn77_3 = convbn_norelu(out_channels//4, out_channels//4, stride=1)

    def forward(self, x_in):
        x_out = self.convbn33_norelu(x_in)  # feat1

        x_out_55_1 = self.convbn55_1(x_in)
        x_out_55 = self.convbn55_2(x_out_55_1)  # feat2

        x_out_77_2 = self.convbn77_2(x_out_55_1)  # feat3
        x_out_77 = self.convbn77_3(x_out_77_2)

        out = torch.cat([x_out, x_out_55, x_out_77], dim=1)  # out_channels： 32+16+16
        out = F.relu(out)

        return out

# fpn结构
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()

        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1

        # backbone后三个维度特征层横向输出先做1*1卷积调整至相同通道，用于后续addition
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, x_in):

        # 1、获取三个维度的有效特征层
        #  C3 64*80*80
        #  C4 128*40*40
        #  C5 256*20*20
        x_in = list(x_in.values())
        # 2、有效特征层用1*1拉维度
        #  C3 64*80*80
        #  C4 64*40*40
        #  C5 64*20*20
        output1 = self.output1(x_in[0])
        output2 = self.output2(x_in[1])
        output3 = self.output3(x_in[2])

        # 3、output3上采样后和output2做add
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode='nearest')
        output2 = output2 + up3
        output2 = self.merge2(output2)

        # 4、output2上采样后和output1做add
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode='nearest')
        output1 = output1 + up2
        output1 = self.merge1(output1)

        # 5、return
        out = [output1, output2, output3]  # [80*80, 40*40, 20*20]

        return out

