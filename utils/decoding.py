# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torchvision.ops import nms

# @function:    预测输出后解码函数
# @author:
# @date:
# @last_edit:

# 注意：网络预测输出的解码操作：是编码操作的逆过程，需要和编码保持一致！！！

# 网络预测bbox中心和宽高解码得到bbox   ---use
def decode(locations, anchors, variances):
    # 得到bbox[x_center, y_center, w, h]
    bboxes = torch.cat((anchors[:, :2] + locations[:, :2] * variances[0] * anchors[:, 2:],
                        anchors[:, 2:] * torch.exp(locations[:, 2:] * variances[1])), 1)
    # 转换形式：转成起点终点坐标表示
    bboxes[:, :2] -= bboxes[:, 2:] / 2  # 人脸框起点坐标[x1, y1]
    bboxes[:, 2:] += bboxes[:, :2]  # 人脸框终点坐标[x2, y2]

    return bboxes


# 网络预测关键点解码   ---use
def decode_landm(pre_lands, anchors, variances):

    # 得到5组关键点坐标[landsx0,landsy0,landsx1,landsy1,landsx2,landsy2,landsx3,landsy3,landsx4,landsy4]
    landms = torch.cat((anchors[:, :2] + pre_lands[:, :2] * variances[0] * anchors[:, 2:],
                        anchors[:, :2] + pre_lands[:, 2:4] * variances[0] * anchors[:, 2:],
                        anchors[:, :2] + pre_lands[:, 4:6] * variances[0] * anchors[:, 2:],
                        anchors[:, :2] + pre_lands[:, 6:8] * variances[0] * anchors[:, 2:],
                        anchors[:, :2] + pre_lands[:, 8:10] * variances[0] * anchors[:, 2:],
                        ), dim=1)

    return landms


# 解码后的结果进行nms处理：调用API   ---use
def non_max_suppression(detect_results, conf_thres, nms_thres):
    '''

    input:
    detect_results: 解码后的预测结果
    conf_thres: 预设置信度阈值（分类）
    nms_thres: 预设nms阈值(iou)

    output:
    best_box: 输出nms处理后的结果

    '''

    # 1、判断若小于预设置信度阈值conf_thres，则直接剔除，不参与后续计算
    mask = detect_results[:, 4] >= conf_thres
    detection = detect_results[mask]

    if len(detection) <= 0:
        return []

    # 2、调用官方API中的nms函数
    keep = nms(detection[:, :4], detection[:, 4], nms_thres)
    best_box = detection[keep]

    return best_box.cpu().numpy()


# nms自定义: 自定义iou计算函数
def cal_iou(b1, b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * np.maximum(inter_rect_y2 - inter_rect_y1, 0)

    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)

    return iou


# 解码后的结果进行nms处理：nms自定义实现
def nms_self(detect_results, conf_thres, nms_thres):
    '''
    detect_results: 解码后的预测结果
    conf_thres: 预设置信度阈值（分类）
    nms_thres: 预设nms阈值(iou)

    '''

    # 1、判断若小于预设置信度阈值conf_thres，则直接剔除，不参与后续计算
    mask = detect_results[:, 4] >= conf_thres
    detection = detect_results[mask]

    if len(detection) <= 0:
        return []

    # 2、根据得分对框进行从大到小排序
    best_box = []
    scores = detection[:, 4]  # 此处的得分score是指网络预测的人脸概率
    arg_sort = np.argsort(scores)[::-1]
    detection = detection[arg_sort]

    # 3、每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除
    while np.shape(detection)[0] > 0:
        best_box.append(detection[0])
        if len(detection) == 1:
            break
        ious = cal_iou(best_box[-1], detection[1:])
        detection = detection[1:][ious < nms_thres]

    return best_box.cou().numpy()

# if letterbox_image：如果使用了加灰条resize
# nms之后的预测结果调整到原图尺寸大小
def correct_boxes(result, input_shape, image_shape):
    '''

    input:
    result: nms之后的结果
    input_shape: 网络输入尺寸 np.array([self.height, self.width])
    image_shape: 图像原始尺寸 np.array([ori_height, ori_width])

    output:
    result: 回到图像原始尺寸大小的结果

    '''
    new_shape = image_shape * np.min(input_shape / image_shape)
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    # 将bbox和landmarks拉回原始尺寸，置信度conf不变
    # 尺度scale
    scale_for_boxs = [scale[1], scale[0], scale[1], scale[0]]  # [x, y, w, h]尺度也要对应scale[1]是height， scale[0]是width
    scale_for_landmarks = [scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1],
                           scale[0]]  # 5组关键点
    # 偏移量offset
    offset_for_boxs = [offset[1], offset[0], offset[1], offset[0]]
    offset_for_landmarks = [offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0],
                            offset[1], offset[0]]

    # 补偿回原始尺寸
    result[:, :4] = (result[:, :4] - np.array(offset_for_boxs)) * np.array(scale_for_boxs)
    result[:, 5:] = (result[:, 5:] - np.array(offset_for_landmarks)) * np.array(scale_for_landmarks)

    return result
