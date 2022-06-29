# -*- coding: utf-8 -*-
import os
import argparse
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils import letterbox_image, preprocess_input
from utils.decoding import *
from backbones import __backbones__
from models import __models__

# 测试模式1：	只对图像进行检测可视化并存到本地，不计算评价指标；
# 测试模式2：	只计算评价指标不进行检测可视化；
#
# @function:    测试脚本：对测试模式1的实现;
# @author:
# @date:
# @last_edit:

#  ---  start  ---
parser = argparse.ArgumentParser(description='RetinaFace face/landmarks detection!')
parser.add_argument('--backbone', default='fd', help='select a backbone structure', choices=__backbones__.keys())  # "resnet/mobilenet"
parser.add_argument('--model', default='fd', help='select a model structure', choices=__models__.keys())  # "retinaface"
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')  # 载入训练好的模型 (路径+模型名字)
parser.add_argument('--input_height', type=int, default=640, help='cnn input height')
parser.add_argument('--input_width', type=int, default=360, help='cnn input width')
parser.add_argument('--conf_thres', type=float, default=0.5, help='nms-confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.5, help='nms-iou threshold')
parser.add_argument('--letterbox_image', action='store_true', help='limitation image size')  # 是否开启预处理：图像的不失真resize
# parser.add_argument('--mode', default='dir_predict', help='select a test mode')  # 测试模式设置
parser.add_argument('--img_dir', type=str, required=True, help='directory of test images')  # 测试图像所在的文件夹目录
parser.add_argument('--save_dir', type=str, required=True, help='save directory of test images results')  # 测试结果保存文件夹目录
args = parser.parse_args()

# face_detection 类定义
class face_detection(object):

    # 初始化
    def __init__(self, **kwargs):

        # 0、确定backbone
        if args.backbone == 'mobilev1025':
            self.cfg = cfg_mnet
        elif args.backbone == 'resnet50':
            self.cfg = cfg_re50
        else:
            raise ValueError('Unsupported backbone - `{}`, Please use mobilenet, resnet50.'.format(args.backbone))


        # 1、 模型设置，默认使用cuda
        self.pretrain = False
        self.model = __models__[args.model](self.cfg, self.pretrain, mode='eval')
        #  前
        checkpoint = torch.load(args.loadckpt)
        self.model.load_state_dict(checkpoint)
        print("loading the model in logdir: {}".format(args.loadckpt))
        # 后
        self.model = nn.DataParallel(self.model)
        self.model.cuda()
        self.model.eval()

        self.letterbox_image = args.letterbox_image

        # 默认开启图像不失真resize
        # if self.letterbox_image:
        # self.anchors = Anchors(self.cfg, image_size=[self.input_shape[0], self.input_shape[1]]).get_anchors()

        # 预设网络输入尺寸--->和训练保持一致
        self.input_height = args.input_height
        self.input_width = args.input_width

        # 阈值
        self.conf_thres = args.conf_thres
        self.nms_thres = args.nms_thres

    # # 图片检测+解码+nms+可视化绘制：返回可视化绘制后的图像
    def detect_img(self, image):

        # 0、copy用于后续绘制
        old_image = image.copy()
        image = np.array(image, np.float32)
        ori_height, ori_width, _ = np.shape(image)

        # 1、计算尺度scale，用于后续将输出的预测框转换到原始图像的高宽尺寸
        scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]  # 图像原始尺寸
        scale_lands = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        # 2、图像不失真resize预处理
        if self.letterbox_image:
            # 按照预设网络输入尺寸：图像resize和设置anchor尺寸
            image = letterbox_image(image, [self.input_width, self.input_height])  # 从原始尺寸resize到预设输入尺寸
            self.anchors = Anchors(self.cfg, image_size=[self.input_width, self.input_height]).get_anchors()  # 获取预设输入尺寸的anchor尺寸先验框
        else:
            # 否则直接按照图像原始尺寸设置anchor尺寸
            self.anchors = Anchors(self.cfg, image_size=(ori_height, ori_width)).get_anchors()  # 获取图像原始尺寸的anchor尺寸先验框

        # 3、进行检测
        with torch.no_grad():
            # 图像预处理
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)
            self.anchors = self.anchors.cuda()
            image = image.cuda()
            # 模型输出
            # bbox, conf, landmarks
            location, conf, landms = self.model(image)

            # 解码
            # (1) bbox预测框解码
            boxes = decode(location.data.squeeze(0), self.anchors, self.cfg['variance'])  # 得到[x1, y1, x2, y2]
            # (2) 人脸置信度
            #  序号为0是背景概率，序号为1是人脸概率
            conf = conf.data.squeeze(0)[:, 1:2]
            # (3) landmarks人脸关键点解码
            landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])  # 得到5组关键点xy坐标

            # 对所有的预测结果进行堆叠
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)  # 堆叠
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.conf_thres, self.nms_thres)  # nms处理

            # 无人脸，不用可视化，返回原图
            if len(boxes_conf_landms) <= 0:
                return old_image

            # 若使用了letterbox_image需要将预处理时添加的灰条去掉
            if self.letterbox_image:
                # 坐标被拉到图像原始尺寸的[0, 1]之间
                boxes_conf_landms = correct_boxes(boxes_conf_landms, np.array([self.input_height, self.input_width]), np.array([ori_height, ori_width]))

        # 4、人脸框和关键点坐标回到图像原始尺寸
        # 直接乘上图像原始尺寸即可，因为不管是否按照预设尺寸输入网络，坐标大小已经全部回到原始尺寸图像的尺度，且在[0, 1]之间
        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_lands

        # 5、检测结果可视化绘制
        for b in boxes_conf_landms:
            text = "{:.4f}".format(b[4])  # 人脸置信度得分
            b = list(map(int, b))

            # b[0]-b[3]为人脸框的坐标, b[4]为得分, b[5]-b[14]为人脸关键点
            # (1)绘制人脸框bbox
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            # (2)绘制人脸得分
            cv2.putText(old_image, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            print(b[0], b[1], b[2], b[3], text)
            # (3)绘制关键点
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
        return old_image

# 对一个文件夹中的图片进行人脸检测，并进行可视化绘制后保存到本地
def main():
    retinaface = face_detection()
    # 对文件夹内的图像进行检测并可视化
    img_names = os.listdir(args.img_dir)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(args.img_dir, img_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_res = retinaface.detect_img(image)  # 调用检测函数
            img_res = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            cv2.imwrite(os.path.join(args.save_dir, img_name), img_res)

if __name__ == '__main__':
    main()
