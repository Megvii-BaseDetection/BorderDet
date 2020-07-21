# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved

from .cityscapes import CityScapesDataset
from .coco import COCODataset
from .imagenet import ImageNetDataset
from .voc import VOCDataset
from .widerface import WiderFaceDataset
from .lvis import LVISDataset
from .citypersons import CityPersonsDataset
from .crowdhuman import CrowdHumanDataset

__all__ = [
    "COCODataset",
    "VOCDataset",
    "CityScapesDataset",
    "ImageNetDataset",
    "WiderFaceDataset",
    "LVISDataset",
    "CityPersonsDataset",
    "CrowdHumanDataset",
]
