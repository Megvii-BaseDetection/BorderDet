#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File               :   fcos_config.py
@Time               :   2020/05/07 23:56:09
@Author             :   Benjin Zhu
@Contact            :   poodarchu@gmail.com
@Last Modified by   :   Benjin Zhu
@Last Modified time :   2020/05/07 23:56:09
'''

from .base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        # META_ARCHITECTURE="RetinaNet",
        RESNETS=dict(OUT_FEATURES=["res3", "res4", "res5"]),
        FPN=dict(IN_FEATURES=["res3", "res4", "res5"]),
        FCOS=dict(
            NUM_CLASSES=80,
            IN_FEATURES=["p3", "p4", "p5", "p6", "p7"],
            NUM_CONVS=4,
            FPN_STRIDES=[8, 16, 32, 64, 128],
            PRIOR_PROB=0.01,
            CENTERNESS_ON_REG=False,
            NORM_REG_TARGETS=False,
            SCORE_THRESH_TEST=0.05,
            TOPK_CANDIDATES_TEST=1000,
            NMS_THRESH_TEST=0.6,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            IOU_LOSS_TYPE="iou",
            CENTER_SAMPLING_RADIUS=0.0,
            OBJECT_SIZES_OF_INTEREST=[
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, float("inf")],
            ],
        ),
    ),
)


class FCOSConfig(BaseDetectionConfig):
    def __init__(self):
        super(FCOSConfig, self).__init__()
        self._register_configuration(_config_dict)


config = FCOSConfig()
