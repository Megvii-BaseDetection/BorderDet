#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File               :   yolo_config.py
@Time               :   2020/05/07 23:55:49
@Author             :   Benjin Zhu
@Contact            :   poodarchu@gmail.com
@Last Modified by   :   Benjin Zhu
@Last Modified time :   2020/05/07 23:55:49
'''

from .base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        PIXEL_MEAN=(0.485, 0.456, 0.406),
        PIXEL_STD=(0.229, 0.224, 0.225),
        DARKNET=dict(
            DEPTH=53,
            STEM_OUT_CHANNELS=32,
            WEIGHTS="s3://generalDetection/cvpods/ImageNetPretrained/custom/darknet53.mix.pth",
            OUT_FEATURES=["dark3", "dark4", "dark5"]
        ),
        YOLO=dict(
            CLASSES=80,
            IN_FEATURES=["dark3", "dark4", "dark5"],
            ANCHORS=[
                [[116, 90], [156, 198], [373, 326]],
                [[30, 61], [62, 45], [42, 119]],
                [[10, 13], [16, 30], [33, 23]],
            ],
            CONF_THRESHOLD=0.01,  # TEST
            NMS_THRESHOLD=0.5,
            IGNORE_THRESHOLD=0.7,
        ),
    ),
)


class YOLO3Config(BaseDetectionConfig):
    def __init__(self):
        super(YOLO3Config, self).__init__()
        self._register_configuration(_config_dict)


config = YOLO3Config()
