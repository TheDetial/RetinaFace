# -*- coding: utf-8 -*-

from models.retinaface import RetinaFace
from models.loss import MultiBoxLoss

__models__ = {
    "retinaface": RetinaFace,
    "multiboxloss": MultiBoxLoss,
}
