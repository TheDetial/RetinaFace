# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.config import cfg_mnet, cfg_re50
from datasets.dataloader import detection_collate
from utils.anchors import Anchors
from utils.utils import *
from utils.log import LossHistory

from datasets import __datasets__
from backbones import __backbones__
from models import __models__

# @function:    RetinaFace训练脚本
# @author:      TheDetial
# @date:        2022/06
# @last_edit:   2022/06

#  ---  start  ---
parser = argparse.ArgumentParser(description='RetinaFace face/landmarks detection!')
parser.add_argument('--backbone', default='fd', help='select a backbone structure', choices=__backbones__.keys())  # "resnet/mobilenet"
parser.add_argument('--model', default='fd', help='select a model structure', choices=__models__.keys())  # "retinaface"
parser.add_argument('--dataloader', required=True, help='dataset loader name', choices=__datasets__.keys())  # "face detection dataload"
parser.add_argument('--trainlist', required=True, help='training list')  # train list.txt
parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size', help='Size of batch)')  # 默认给解冻训练时的batch_size
parser.add_argument('--freeze_lr', type=float, default=0.0001, help='base learning rate')
parser.add_argument('--unfreeze_lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--trainloss', default='fd', help='select a model structure', choices=__models__.keys())  # "loss function"
parser.add_argument('--bbpretrain', action='store_true', help='continue training the model from pretrained weights')  # backbone是否使用预训练模型
parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epoch to train')
parser.add_argument('--freeze_epoches', type=int, required=True, help='number of epochs to train')
parser.add_argument('--max_epoches', type=int, required=True, help='number of epochs to train')
parser.add_argument('--freeze_train', action='store_true', help='split net training for freeze and unfreeze stage')  # 是否使用冻结和解冻训练
# if resume
parser.add_argument('--resume', action='store_true', help='continue training the model')  # 继续训练
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')  # resume 整体网络

# start
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.benchmark = True
Cuda = True

imgload = __datasets__[args.dataloader]
criterionLoss = __models__[args.trainloss]

# train one
def train_one_epoch(net_train, net, optimizer, criterion, epoch, epoch_step, train_loader, Epoches, anchors, cfg, loss_history):
    # net.train()
    total_location_loss = 0
    total_cls_loss = 0
    total_landmark_loss = 0

    print("Start train: ")
    with tqdm(total=epoch_step, desc=f'Epoch {epoch+1}/{Epoches}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            if iteration >= epoch_step:
                break
            images, targets = batch[0], batch[1]
            if len(images) == 0:
                continue

            # 被包裹起来的部分不参与track梯度
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor).cuda()  # 图像转tensor
                targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]  # 标签转tensor

            optimizer.zero_grad()
            predictions = net_train(images)

            # 计算损失
            # bbox,classification,landmarks
            location_loss, class_loss, landmarks_loss = criterion(predictions, anchors, targets)


            loss = cfg['loc_weight'] * location_loss + class_loss + landmarks_loss

            loss.backward()
            optimizer.step()

            total_location_loss += cfg['loc_weight'] * location_loss.item()  # 额外权重？？？
            total_cls_loss += class_loss.item()
            total_landmark_loss += landmarks_loss.item()

            # classification loss, bbox loss, landmarks loss
            pbar.set_postfix(**{'Confidence Loss': total_cls_loss / (iteration + 1),
                                'Regression Loss': total_location_loss / (iteration + 1),
                                'Landmarks Loss': total_landmark_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    print('Saving state, iter:', str(epoch + 1))
    torch.save(net.state_dict(), args.logdir + 'Epoch%d-Total_Loss_%.4f.pth' % (
    (epoch + 1), (total_cls_loss + total_location_loss + total_landmark_loss) / (epoch_step + 1)))
    loss_history.append_loss((total_location_loss + total_cls_loss + total_landmark_loss) / (epoch_step + 1))

# train all
def train(net_train, net, learning_rate, start_epoch, end_epoches, batch_size, criterion, anchors, cfg, loss_history):
    # 优化器
    optimizer = optim.Adam(net_train.parameters(), learning_rate, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
    train_dataset = imgload(args.trainlist, cfg['train_image_size'])
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                              drop_last=False, collate_fn=detection_collate)

    epoch_step = train_dataset.get_len() // batch_size

    for epoch in range(start_epoch, end_epoches):
        train_one_epoch(net_train, net, optimizer, criterion, epoch, epoch_step, train_loader, end_epoches, anchors, cfg, loss_history)
        lr_scheduler.step()

# 主函数
def main():
    # 1、确定backbone
    if args.backbone == 'mobilev1025':
        cfg = cfg_mnet
    elif args.backbone == 'resnet50':
        cfg = cfg_re50
    else:
        raise ValueError('Unsupported backbone - `{}`, Please use mobilenet, resnet50.'.format(args.backbone))

    # 2、设置模型
    model = __models__[args.model](cfg=cfg, pretrain=args.bbpretrain)  # backbone pre-train，非all model pre-train
    # 若不使用预训练模型进行权重初始化  --从零开始训练
    # if not args.bbpretrain:
    #     weights_init(model)

    # 3、训练模式选择  --resume 继续训练
    if args.resume and not args.bbpretrain:
        print("loading the lastest model in logdir: {}".format(args.checkpoint_path))
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint)
        print('resume from a model ... ... ')


    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)  # 多卡
        model_train = model_train.cuda()
    print("Number of model parameters: {}".format(sum([p.data.nelement() for p in model.parameters()])))

    # 3、获取先验框anchors
    anchors = Anchors(cfg, image_size=(cfg['train_image_size'], cfg['train_image_size'])).get_anchors()  # 得到的anchor尺寸为：[16800, 4]
    anchors = anchors.cuda()

    # 4、损失函数、log和pth保存
    # 有无人脸:num_class=2, 区分正负样本阈值: 0.35, 正负样本比例: 7
    criterion = criterionLoss(2, 0.35, 7, cfg['variance'], Cuda)  #
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    loss_history = LossHistory(args.logdir)

    # 5、开始训练
    if args.freeze_train:  # 5.1 在先冻结-后解冻的训练模式下
        # （1）先冻结训练
        for param in model.body.parameters():  # 此处看yolov4中网络定义body=mobilev1025 return 中间层
            param.requires_grad = False
        train(model_train, model, args.freeze_lr, args.start_epoch, args.freeze_epoches, args.batch_size * 2, criterion, anchors, cfg, loss_history)  # 冻结训练时，batch_size大一点
        # （2）后解冻训练
        for param in model.body.parameters():
            param.requires_grad = True
        train(model_train, model, args.unfreeze_lr, args.freeze_epoches, args.max_epoches, args.batch_size, criterion, anchors, cfg, loss_history)  # 解冻训练时，batch_size小一点
    else:  # 5.2 在直接更新所有参数的训练模式下(默认状态：参数全部更新)
        train(model_train, model, args.unfreeze_lr, args.start_epoch, args.max_epoches, args.batch_size, criterion, anchors, cfg, loss_history)  # 训练更新所有参数==解冻训练，batch_size小一点

if __name__ == '__main__':
    main()
