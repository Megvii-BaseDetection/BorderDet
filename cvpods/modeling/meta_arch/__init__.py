# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# import all the meta_arch, so they will be registered

from .centernet import CenterNet
from .fcos import FCOS
from .borderdet import BorderDet
from .panoptic_fpn import PanopticFPN
from .rcnn import GeneralizedRCNN, ProposalNetwork
from .reppoints import RepPoints
from .retinanet import RetinaNet
from .semantic_seg import SemanticSegmentor, SemSegFPNHead
from .ssd import SSD
from .tensormask import TensorMask
from .yolov3 import YOLOv3
from .efficientdet import EfficientDet
from .pointrend import (
    PointRendROIHeads,
    CoarseMaskHead,
    StandardPointHead,
    PointRendSemSegHead,
)
from .dynamic4seg import DynamicNet4Seg
from .fcn import FCNHead
