#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File               :   base_classification_config.py
@Time               :   2020/05/07 23:56:17
@Author             :   Benjin Zhu
@Contact            :   poodarchu@gmail.com
@Last Modified by   :   Benjin Zhu
@Last Modified time :   2020/05/07 23:56:17
'''

from cvpods.configs.base_config import BaseConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        PIXEL_MEAN=[0.406, 0.456, 0.485],  # BGR
        PIXEL_STD=[0.225, 0.224, 0.229],
        BACKBONE=dict(FREEZE_AT=-1, ),  # do not freeze
        RESNETS=dict(
            NUM_CLASSES=None,
            DEPTH=None,
            OUT_FEATURES=["linear"],
            NUM_GROUPS=1,
            # Options: FrozenBN, GN, "SyncBN", "BN"
            NORM="BN",
            ACTIVATION=dict(
                NAME="ReLU",
                INPLACE=True,
            ),
            # Whether init last bn weight of each BasicBlock or BottleneckBlock to 0
            ZERO_INIT_RESIDUAL=True,
            WIDTH_PER_GROUP=64,
            # Use True only for the original MSRA ResNet; use False for C2 and Torch models
            STRIDE_IN_1X1=False,
            RES5_DILATION=1,
            RES2_OUT_CHANNELS=256,
            STEM_OUT_CHANNELS=64,
            DEFORM_ON_PER_STAGE=[False, False, False, False],
            DEFORM_MODULATED=False,
            DEFORM_NUM_GROUPS=1,

            # Deep Stem
            DEEP_STEM=False,
            # Apply avg after conv2 in the BottleBlock
            # When AVD=True, the STRIDE_IN_1X1 should be Falss
            AVD=False,
            # Apply avg_down to the downsampling layer for residual path
            AVG_DOWN=False,
            # Radix in ResNeSt
            RADIX=1,
            # Bottleneck_width in ResNeSt
            BOTTLENECK_WIDTH=64,
        ),
    ),
    SOLVER=dict(
        IMS_PER_DEVICE=32,  # defalut: 8 gpus x 32 = 256
    ),
)


class BaseClassificationConfig(BaseConfig):
    def __init__(self):
        super(BaseClassificationConfig, self).__init__()
        self._register_configuration(_config_dict)


config = BaseClassificationConfig()
