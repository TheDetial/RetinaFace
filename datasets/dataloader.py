# -*- coding: utf-8 -*-
#@File      :
#@Author    : TheDetial
#@Email     :
#@Date      : 2022/
#@Last_edit : 2022/
#@Function  : 用于人脸+关键点检测数据载入代码

import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image

# dataloader: face-path+bbox-4点+landmarks-5点
class DataGenerator(data.Dataset):

    '''
    input: 
            img.list:数据list.txt，原始标签形式为：
            # img
            x1,y1,w,h,landsx0,landsy0,landsx1,landsy1,landsx2,landsy2,landsx3,landsy3,landsx4,landsy4
            x1,y1,w,h,landsx0,landsy0,landsx1,landsy1,landsx2,landsy2,landsx3,landsy3,landsx4,landsy4
            x1,y1,w,h,landsx0,landsy0,landsx1,landsy1,landsx2,landsy2,landsx3,landsy3,landsx4,landsy4
            ...
            img_size:送给网络的预设尺寸（默认640*640）
    return: 
            返回非tensor形式的图像image+GT标签
            GT标签形式为：
            img,x1,y1,x2,y2,landsx0,landsy0,landsx1,landsy1,landsx2,landsy2,landsx3,landsy3,landsx4,landsy4,pos
    '''

    def __init__(self, img_list, img_size):  # img.list, img_size送给网络的预设尺寸（默认640*640）
        
        self.img_list = img_list
        self.img_size = img_size
        self.imgs_path, self.words = self.process_label()

    def __len__(self):
        return len(self.imgs_path)

    def get_len(self):
        return len(self.imgs_path)


    def __getitem__(self, index):
        
        img = Image.open(self.imgs_path[index])  # 取该index图像
        labels = self.words[index]
        annotations = np.zeros((0, 15))

        # 1、若该张index的图像无标签 
        if len(labels) == 0:
            return img, annotations

        # 2、若该张index的图像有标签 
        for idx, label in enumerate(labels):
            
            #（1）裁剪起始点+裁剪终点+5组关键点+标签=15个坐标值,[[0,0,...,0]]：1行15列
            annotation = np.zeros((1, 15))  
             
            #（2）原始label中bbox坐标为：起始点(x1, y1)+宽高(w, h)的形式 
            # 取人脸真实框bbox位置：[x1, y1, x1+w, y1+h]
            annotation[0, 0] = label[0]  # 人脸框起点x1
            annotation[0, 1] = label[1]  # 人脸框起点y1
            annotation[0, 2] = label[0] + label[2]  # 人脸框终点x2
            annotation[0, 3] = label[1] + label[3]  # 人脸框终点y2

            # （3）取人脸关键点坐标(x,y)共5组、
            annotation[0, 4] = label[4]  # l0_x
            annotation[0, 5] = label[5]  # l0_y
            annotation[0, 6] = label[6]  # l1_x
            annotation[0, 7] = label[7]  # l1_y
            annotation[0, 8] = label[8]  # l2_x
            annotation[0, 9] = label[9]  # l2_y
            annotation[0, 10] = label[10]  # l3_x
            annotation[0, 11] = label[11]  # l3_y
            annotation[0, 12] = label[12]  # l4_x
            annotation[0, 13] = label[13]  # l4_y
            
            #（6）判断关键点情况并打上对应标签:此处只给-1或1，默认list.txt文件中的图像全部有bbox
            if annotation[0, 4] < 0:
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1
            
            #（7）某张图像可能有多个人脸框+关键点的标签
            annotations = np.append(annotations, annotation, axis=0)
        
        #（8）target为该index图像的所有标签信息组成的数组 
        target = np.array(annotations)
        #（9）对该张index图像和其对应的所有标签进行预处理
        img, target = self.data_preprocess(img, target, [self.img_size, self.img_size])
        img = np.array(np.transpose(self.preprocess_input(img), (2, 0, 1)), dtype=np.float32)

        return img, target

    def preprocess_input(self, input_image):
        input_image -= np.array((104, 117, 123), np.float32)
        return input_image

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a)+a

    # 注意：一张图像中有多个人脸框，每个框有其对应的裁减尺寸和关键点坐标
    # 返回所有的图像数组，每张图像对应的标签数组
    def process_label(self):
        imgs_path = []
        landmarks = []
        f = open(self.img_list, 'r')  # 打开标签.txt
        lines = f.readlines()

        isFirst = True  # 处理完一张图像的标志
        labels = []
        for line in lines:
            line = line.rstrip()
            # 1、'#' 开头：为图像img.path所在的行
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()  # list.copy()直接复制列表
                    landmarks.append(labels_copy)
                    labels.clear()
                img_path = line[2:]  # 取图像path  --right
                imgs_path.append(img_path)
            # 2、非'#'开头的为该img的标签信息：包括裁剪尺寸和关键点坐标
            else:
                linenew = line.split(' ')
                label = [float(x) for x in linenew]  # 每一行坐标单独存
                labels.append(label)  # 该张img的所有关键点坐标全部存完，[[],[],...,[]]
        landmarks.append(labels)
        return imgs_path, landmarks

    def data_preprocess(self, image, targes, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        iw, ih = image.size
        h, w = input_shape
        box = targes

        # 1、图像缩放以及长宽的扭曲
        new_ar = w/h * self.rand(1-jitter, 1+jitter)/self.rand(1-jitter, 1+jitter)
        scale = self.rand(0.25, 3.25)

        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 2、图像多余部分加上灰条
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 3、翻转
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 4、色域扭曲
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32)/255, cv2.COLOR_RGB2HSV)

        x[..., 0] += hue*360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        # 5、真实框（坐标）调整
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2, 4, 6, 8, 10, 12]] = box[:, [0, 2, 4, 6, 8, 10, 12]]*nw/iw + dx
            box[:, [1, 3, 5, 7, 9, 11, 13]] = box[:, [1, 3, 5, 7, 9, 11, 13]]*nh/ih + dy

            if flip:
                box[:, [0, 2, 4, 6, 8, 10, 12]] = w - box[:, [2, 0, 6, 4, 8, 12, 10]]
                box[:, [5, 7, 9, 11, 13]] = box[:, [7, 5, 9, 13, 11]]

            # 人脸框的中心坐标点（注意）
            center_x = (box[:, 0] + box[:, 2])/2  # x_center = (起点+终点)/2
            center_y = (box[:, 1] + box[:, 3])/2  # y_center = (起点+终点)/2

            # np.logical_and:逻辑与
            box = box[np.logical_and(np.logical_and(center_x>0, center_y>0), np.logical_and(center_x<w, center_y<h))]

            box[:, 0:14][box[:, 0:14]<0] = 0
            box[:, [0,2,4,6,8,10,12]][box[:, [0,2,4,6,8,10,12]]>w] = w
            box[:, [1,3,5,7,9,11,13]][box[:, [1,3,5,7,9,11,13]]>h] = h

            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)]
        

        box[:, 4:-1][box[:, -1]==-1] = 0

        box[:, [0,2,4,6,8,10,12]] /= w  # 所有x/640，直接用网络预设输入尺寸做归一化
        box[:, [1,3,5,7,9,11,13]] /= h  # 所有y/640，直接用网络预设输入尺寸做归一化
        box_data = box
        return image_data, box_data

def detection_collate(batch):
    images = []
    targets = []
    for img, box in batch:
        if len(box)==0:
            continue
        images.append(img)
        targets.append(box)
    images = np.array(images)
    return images, targets
