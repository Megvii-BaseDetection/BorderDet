import os.path as osp

from cvpods.configs.rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        MASK_ON=False,
        BACKBONE=dict(
            FREEZE_AT=0,
        ),
        RESNETS=dict(
            DEPTH=50,
            NORM="SyncBN",
        ),
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(60000, 80000),
            MAX_ITER=90000,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.02,
        ),
        IMS_PER_BATCH=16,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(640, 672, 704, 736, 768, 800),
                      max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        )
    ),
    TEST=dict(
        EVAL_PERIOD=10000,
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]),
)


class FasterRCNNConfig(RCNNFPNConfig):
    def __init__(self):
        super(FasterRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = FasterRCNNConfig()
