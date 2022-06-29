# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


#@Function  : 编码和损失函数

# ---------------------

# 1、真实框处理
# （1）计算所有真实框和所有先验框的重合程度，和真实框iou大于0.35的先验框被认为可以用于预测获得该真实框；
# （2）对这些和真实框重合程度较大的先验框进行编码操作，编码就是当我们要获得这样的真实框时候，网络的预测结果应该是怎样的；
# （3）编码操作三部分：分类预测结果、框的回归预测结果和关键点回归预测结果的编码；

# 2、损失函数
# 计算处理完的真实框与对应图图片的预测结果计算loss
# bbox:smooth l1,所有正标签的框的预测结果的回归loss
# class:cross entropy,所有种类的预测结果的交叉熵loss
# landmarks: smooth l1,所有正标签的人脸关键点的预测结果的回归loss

# ---------------------

# 1、获取框的左上角和右下角  --use
# 用于计算预设先验框的左上角和右下角点，预设anchors信息[x_center, y_center, w, h]正方形anchor
def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2, boxes[:, :2] + boxes[:, 2:]/2), 1)


# 2、获得框的中心和宽高  --no use
def center_size(boxes):
    return torch.cat((boxes[:, 2:]+boxes[:, :2])/2, boxes[:, 2:] - boxes[:, :2], 1)


# 3、计算所有真实框和先验框的交面积(两个框交集部分的面积)  计算iou子函数  --use
def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)

    # 获得交矩形左上角
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))

    # 获得交矩形右下角
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp((max_xy-min_xy), min=0)

    # 计算先验框和所有真实框的重合面积
    return inter[:, :, 0] * inter[:, :, 1]


# 4、计算两个框的交并比IOU  -- use
def jaccard(box_a, box_b):

    # 1、计算两个框交集部分的面积
    inter = intersect(box_a, box_b)

    # 2、计算两个框各自的面积
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    # 3、计算交并比
    iou = inter / union

    return iou

# 编码操作5和6
# 5、人脸框bbox编码(x,y,w,h)  -- use
def encode(matched, priors, variances):
    '''

    variances:为预设的尺度

    '''
    # 1、网格点中心坐标编码  ---(量化误差)
    # 真实框中心点坐标减去先验框中心坐标
    # (真实框起点和终点坐标相加之后)/2-先验框中心坐标：([x1, y1]+[x2, y2])/2-priors[center_x, center_y]
    g_cxcy = (matched[:, :2]+matched[:, 2:])/2 - priors[:, :2]
    # retinaface论文中的做法：量化误差拉一下尺度
    g_cxcy /= (variances[0] * priors[:, 2:])

    # 2、宽高编码(w, h)
    # (真实框终点减去起点坐标之后)/先验框宽高： ([x2, y2]-[x1, y1])/priors[anchor_w, anchor_h]
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    # retinaface论文中的做法：宽高拉一下尺度
    g_wh = torch.log(g_wh) / variances[1]

    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors 4]

# 6、人脸关键点编码5*(x, y)  -- use
# 用到了尺度variances，编码之后为真实关键点xy和先验框中心坐标之间的误差，作为网络需要去预测的东西
def encode_landm(matched, priors, variances):

    matched = torch.reshape(matched, (matched.size(0), 5, 2))

    # (0)取先验框的预设参数[center_x, center_y, anchor_w, anchor_h]
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)

    # 减去中心坐标后除上宽高
    # step1: 真实关键点坐标减去先验框中心坐标: [x,y]-priors[center_x, center_y]
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    # step2: # retinaface论文中的做法：量化误差拉一下尺度(和人脸框中心点坐标操作流程一样)
    g_cxcy /= (variances[0] * priors[:, :, 2:])

    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    return g_cxcy

# -- use 
def log_sum_exp(x):
    x_max = x.data.max()
    # torch.log(): return y=log(x)  (以e为底)
    # torch.exp(): return y=e^x
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# 真实框和先验框匹配过程   -- use
def match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    
    # 1、真实框和先验框的交并比IOU
    # overlaps:保存真实框和先验框的iou,shape:[truths.shape[0],priors.shape[0]]，即每一行保存一个truth和所有priors的iou。
    # 注：每一个gt框和所有的先验框计算iou
    overlaps = jaccard(truths, point_form(priors))

    # 2、求所有真实框和先验框的最大IOU
    # 第一个max：为每个truth真实框匹配最好的先验框
    # best_prior：对每个gt框匹配的最好的先验框prior
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_overlap.squeeze_(1)
    best_prior_idx.squeeze_(1)

    # 3、求所有先验框和真实框的最大IOU
    # 第二个max:为每个先验框匹配最好的truth真实框
    # best_truth：对每个先验框prior匹配的最好的gt框
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_overlap.squeeze_(0)
    best_truth_idx.squeeze_(0)

    # 4、首先要保证每个真实框至少有一个对应先验框（重要）
    # index_fill_：防止匹配的框因为阈值太低被过滤掉
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    # 更新best_truth_idx：存入与真实框最大IOU的先验框的序号
    # 确保每一个真实框都能匹配到一个先验框
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    # 最终的目标是给每个先验框匹配一个标签,因此在每个先验框匹配最好gt框的基础上进行修改，将其中gt框匹配到最好的先验框修改为该gt框
    # 5、获取每个先验框对应的真实框（重要）[num_priors, 4]
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx]
    matches_landm = landms[best_truth_idx]
    # 若重合程度小于threshold则认为是背景
    conf[best_truth_overlap < threshold] = 0

    # 6、利用真实框和先验框进行编码
    # 人脸框编码：中心坐标量化误差+宽高
    loc = encode(matches, priors, variances)
    # 人脸关键点编码
    landm = encode_landm(matches_landm, priors, variances)

    # loc_t,conf_t,landm_t为当前图片的最终标签
    loc_t[idx] = loc
    conf_t[idx] = conf
    landm_t[idx] = landm


# 继承自 SSD MultiBoxLoss   -- use
class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, neg_pos, variance, cuda=True):
        super(MultiBoxLoss, self).__init__()
        '''
        num_class:      2
        overlap_thresh: 0.35 计算正负样本时的iou阈值
        neg_pos:        7 

        '''

        # 类别是否人脸,retinaface是2类
        self.num_classes = num_classes

        # IOU大于该阈值认为该先验框可以用来预测
        self.threshold = overlap_thresh

        # 正负样本的比例 
        self.negpos_ratio = neg_pos
        self.variance = variance
        self.cuda = cuda

    def forward(self, predictions, priors, targets):
        '''
        input:
        predictions:网络预测结果
        priors:先验框anchor
        targets:真实框信息
        
        output:
        loss_l: 人脸框bbox损失
        loss_c: 人脸置信度（概率）损失
        loss_landm: 关键点损失

        '''
        # predictions网络预测结果：预测框, 预测置信度, 预测关键点 
        loc_data, conf_data, landm_data = predictions

        num = loc_data.size(0)  # 预测框信息，网络预测了多少个人脸框bbox？？？
        num_priors = (priors.size(0))  # 预设anchors尺寸的个数，尺度[16800, 4]
        
        # 默认：torch.Tensor==torch.FloatTensor
        # 初始化用于存储：和先验框匹配后的真实框信息
        loc_t = torch.Tensor(num, num_priors, 4)  # 
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)

        # 取gt标签
        for idx in range(num):
            # 获得真实框与标签
            truths = targets[idx][:, :4].data  # gt_bbox人脸框
            labels = targets[idx][:, -1].data  # gt_pos标志位
            landms = targets[idx][:, 4:14].data  # gt_landmarks关键点

            defaults = priors.data  # 所有所有所有预设anchors先验框信息
            
            # 计算先验框和真实框的匹配关系
            # match: 这些idx的真实框应该由哪些先验框负责预测
            # 注：子函数中更新tensor，全局中的tensor会同步更新
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx) 

        zeros = torch.tensor(0)
        if self.cuda:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()
            zeros = zeros.cuda()

        # 情况1：没有人脸框的标签为0
        # 情况2：有人脸真实框且有人脸关键点的的标签为1
        # 情况3：有人脸真实框但没有人脸关键点的标签为-1
        # 因为有些图像有人脸框，但是框内没有关键点标注（人头角度过大）
        # 计算人脸关键点Loss时，pos1 = conf_t > zeros
        # 计算人脸框Loss时，pos = conf_t != zeros
        
        #（1）人脸关键点 loss
        # pos1： 取出置信度大于0的用于计算landm的损失值
        poslands = conf_t > zeros
        pos_idx1 = poslands.unsqueeze(poslands.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
        
        #（2）人脸框 loss
        pos = conf_t != zeros  # 此处解释有问题：pos标志位不等于0表示有框，不一定有关键点
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)  # 正样本
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        conf_t[pos] = 1
        batch_conf = conf_data.view(-1, self.num_classes)
        
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c[pos.view(-1, 1)] = 0  # 将正样本置零
        loss_c = loss_c.view(num, -1)
        # Hard Negative Mining --正负样本均衡策略
        # 本质为：将每个负样本的loss按照从大到小的顺序进行排序后选择前n个
        # (1)两次sort,得到大小排序，越大的序号越小(降序)，然后取前n=self.negpos_ratio*num_pos个负样本用于计算分类损失
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_pos = pos.long().sum(1, keepdim=True)

        # (2)限制负样本数量
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)  # neg是一个01元素组成的mask

        # 求和：计算每一张图片内部有多少正样本
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)  # 正样本
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)  # 负样本
        
        # (3)人脸置信度 分类loss
        # 取出用于训练的正样本和负样本， 计算置信度loss
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        N = max(num_pos.data.sum().float(), 1)  # 有人脸有框
        loss_l /= N
        loss_c /= N

        num_pos_landm = poslands.long().sum(1, keepdim=True)  # 有人脸有框有关键点
        N1 = max(num_pos_landm.data.sum().float(), 1)
        loss_landm /= N1
        
        # output:
        # loss_l: 人脸框bbox损失
        # loss_c: 人脸置信度（概率）confidence损失
        # loss_landm: 关键点landmarks损失

        return loss_l, loss_c, loss_landm
