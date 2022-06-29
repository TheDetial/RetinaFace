#-*-coding:utf-8-*-
from backbones.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from backbones.mobileNetV1_025 import mobilev1025
__backbones__ = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "mobilev1025": mobilev1025,
}
