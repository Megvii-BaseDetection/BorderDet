#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File               :   base_config.py
@Time               :   2020/05/07 23:55:17
@Author             :   Benjin Zhu
@Contact            :   poodarchu@gmail.com
@Last Modified by   :   Feng Wang
@Last Modified time :   2020/06/19 19:04:32
'''

import os
import pprint
import re
from ast import literal_eval

from colorama import Back, Fore

from easydict import EasyDict as edict

from .config_helper import (_assert_with_logging,
                            _check_and_coerce_cfg_value_type, diff_dict,
                            find_key, highlight, update)

_config_dict = dict(
    MODEL=dict(
        DEVICE="cuda",
        # Path (possibly with schema like catalog://, detectron2://, s3://) to a checkpoint file
        # to be loaded to the model. You can find available models in the model zoo.
        WEIGHTS="",
        # Indicate whether convert final checkpoint to use as pretrain weights of preceeding model
        AS_PRETRAIN=False,
        # Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR)
        # To train on images of different number of channels, just set different mean & std.
        PIXEL_MEAN=[103.530, 116.280, 123.675],
        # When using pre-trained models in Detectron1 or any MSRA models,
        # std has been absorbed into its conv1 weights, so the std needs to be
        # set 1. Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
        PIXEL_STD=[1.0, 1.0, 1.0],
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=(800,), max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        ),
        # Whether the model needs RGB, YUV, HSV etc.
        FORMAT="BGR",
        # Mask R-CNN supports either "polygon" or "bitmask" as ground truth.
        MASK_FORMAT="polygon",
    ),
    DATASETS=dict(
        CUSTOM_TYPE=("ConcatDataset", dict()),
        # List of the dataset names for training.
        # Must be registered in cvpods/data/datasets/paths_route
        TRAIN=(),
        # List of the pre-computed proposal files for training, which must be consistent
        # with datasets listed in DATASETS.TRAIN.
        PROPOSAL_FILES_TRAIN=(),
        # Number of top scoring precomputed proposals to keep for training
        PRECOMPUTED_PROPOSAL_TOPK_TRAIN=2000,
        # List of the dataset names for testing.
        # Must be registered in cvpods/data/datasets/paths_route
        TEST=(),
        # List of the pre-computed proposal files for test, which must be consistent
        # with datasets listed in DATASETS.TEST.
        PROPOSAL_FILES_TEST=(),
        # Number of top scoring precomputed proposals to keep for test
        PRECOMPUTED_PROPOSAL_TOPK_TEST=1000,
    ),
    DATALOADER=dict(
        # Number of data loading threads
        NUM_WORKERS=2,
        # Default sampler for dataloader
        SAMPLER_TRAIN="DistributedGroupSampler",
        # Repeat threshold for RepeatFactorTrainingSampler
        REPEAT_THRESHOLD=0.0,
        # If True, the dataloader will filter out images that have no associated
        # annotations at train time.
        FILTER_EMPTY_ANNOTATIONS=True,
    ),
    SOLVER=dict(
        # Configs of lr scheduler
        LR_SCHEDULER=dict(
            NAME="WarmupMultiStepLR",
            MAX_ITER=40000,
            MAX_EPOCH=None,
            # STEPS supports both iterations and epochs.
            # If MAX_EPOCH are specified, STEPS will be calculated automatically
            STEPS=(30000,),
            WARMUP_FACTOR=1.0 / 1000,
            WARMUP_ITERS=1000,
            # WARMUP_METHOD in "linear", "constant", "brunin"
            WARMUP_METHOD="linear",
            # Decrease learning rate by GAMMA.
            GAMMA=0.1,
        ),
        OPTIMIZER=dict(
            NAME="SGD",
            BASE_LR=0.001,
            # Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for biases.
            # This is not useful (at least for recent models). You should avoid
            # changing these and they exist only to reproduce Detectron v1 training if desired.
            BIAS_LR_FACTOR=1.0,
            WEIGHT_DECAY=0.0001,
            # The weight decay that's applied to parameters of normalization layers
            WEIGHT_DECAY_NORM=0.0,
            MOMENTUM=0.9,
        ),
        # Gradient clipping
        CLIP_GRADIENTS=dict(
            ENABLED=False,
            # - "value": the absolute values of elements of each gradients are clipped
            # - "norm": the norm of the gradient for each parameter is clipped thus
            #   affecting all elements in the parameter
            CLIP_TYPE="value",
            # Maximum absolute value used for clipping gradients
            CLIP_VALUE=1.0,
            # Floating point number p for L-p norm to be used with the "norm"
            # gradient clipping type; for L-inf, please specify .inf
            NORM_TYPE=2.0,
        ),
        # Save a checkpoint after every this number of iterations
        CHECKPOINT_PERIOD=5000,
        # Number of images per batch across all machines.
        # If we have 16 GPUs and IMS_PER_BATCH = 32,
        # each GPU will see 2 images per batch.
        IMS_PER_BATCH=16,
        IMS_PER_DEVICE=2,
        BATCH_SUBDIVISIONS=1,
    ),
    TEST=dict(
        # For end-to-end tests to verify the expected accuracy.
        # Each item is [task, metric, value, tolerance]
        # e.g.: [['bbox', 'AP', 38.5, 0.2]]
        EXPECTED_RESULTS=[],
        # The period (in terms of steps) to evaluate the model during training.
        # If using positive EVAL_PERIOD, every #EVAL_PERIOD iter will evaluate automaticly.
        # If EVAL_PERIOD = 0, model will be evaluated after training.
        # If using negative EVAL_PERIOD, no evaluation will be applied.
        EVAL_PERIOD=0,
        # The sigmas used to calculate keypoint OKS. See http://cocodataset.org/#keypoints-eval
        # When empty it will use the defaults in COCO.
        # Otherwise it should have the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
        KEYPOINT_OKS_SIGMAS=[],
        # Maximum number of detections to return per image during inference (100 is
        # based on the limit established for the COCO dataset).
        DETECTIONS_PER_IMAGE=100,
        AUG=dict(
            ENABLED=False,
            MIN_SIZES=(400, 500, 600, 700, 800, 900, 1000, 1100, 1200),
            MAX_SIZE=4000,
            FLIP=True,
        ),
        PRECISE_BN=dict(ENABLED=False, NUM_ITER=200),
    ),
    # Directory where output files are written
    OUTPUT_DIR="./output",
    # Set seed to negative to fully randomize everything.
    # Set seed to positive to use a fixed seed. Note that a fixed seed does not
    # guarantee fully deterministic behavior.
    SEED=-1,
    # Benchmark different cudnn algorithms.
    # If input images have very different sizes, this option will have large overhead
    # for about 10k iterations. It usually hurts total time, but can benefit for certain models.
    # If input images have the same or similar sizes, benchmark is often helpful.
    CUDNN_BENCHMARK=False,
    # The period (in terms of steps) for minibatch visualization at train time.
    # Set to 0 to disable.
    VIS_PERIOD=0,
    # global config is for quick hack purposes.
    # You can set them in command line or config files,
    # Do not commit any configs into it.
    GLOBAL=dict(
        HACK=1.0,
        DUMP_TRAIN=True,
        DUMP_TEST=False,
    ),
)


class BaseConfig(object):

    def __init__(self):
        self._config_dict = {}
        self._register_configuration(_config_dict)

    def _register_configuration(self, config):
        """
        Register all key and values of config as BaseConfig's attributes.

        Args:
            config (dict): custom config dict
        """
        self._config_dict = update(self._config_dict, config)
        for k, v in self._config_dict.items():
            if hasattr(self, k):
                if isinstance(v, dict):
                    setattr(self, k, update(getattr(self, k), v))
                else:
                    setattr(self, k, v)
            elif isinstance(v, dict):
                setattr(self, k, edict(v))
            else:
                setattr(self, k, v)

    def merge_from_list(self, cfg_list):
        """
        Merge config (keys, values) in a list (e.g., from command line) into
        this CfgNode.

        Args:
            cfg_list (list): cfg_list must be divided exactly.
            For example, `cfg_list = ['FOO.BAR', 0.5]`.
        """
        _assert_with_logging(
            len(cfg_list) % 2 == 0,
            "Override list has odd length: {}; it must be a list of pairs".format(
                cfg_list
            ),
        )
        # root = self
        for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
            key_list = full_key.split(".")
            d = self
            for subkey in key_list[:-1]:
                _assert_with_logging(
                    hasattr(d, subkey), "Non-existent key: {}".format(full_key)
                )
                d = getattr(d, subkey)
            subkey = key_list[-1]
            _assert_with_logging(
                hasattr(d, subkey), "Non-existent key: {}".format(full_key)
            )
            value = self._decode_cfg_value(v)
            value = _check_and_coerce_cfg_value_type(
                value, getattr(d, subkey), subkey, full_key
            )
            setattr(d, subkey, value)

    def link_log(self, link_name="log"):
        """
        create a softlink to output dir.

        Args:
            link_name(str): name of softlink
        """
        if os.path.islink(link_name) and os.readlink(link_name) != self.OUTPUT_DIR:
            os.system("rm " + link_name)
        if not os.path.exists(link_name):
            cmd = "ln -s {} {}".format(self.OUTPUT_DIR, link_name)
            os.system(cmd)

    @classmethod
    def _decode_cfg_value(cls, value):
        """
        Decodes a raw config value (e.g., from a yaml config files or command
        line argument) into a Python object.
        If the value is a dict, it will be interpreted as a new CfgNode.
        If the value is a str, it will be evaluated as literals.
        Otherwise it is returned as-is.

        Args:
            value (dict or str): value to be decoded
        """
        # Configs parsed from raw yaml will contain dictionary keys that need to be
        # converted to CfgNode objects
        if isinstance(value, dict):
            return cls(value)
        # All remaining processing is only applied to strings
        if not isinstance(value, str):
            return value
        # Try to interpret `value` as a:
        #   string, number, tuple, list, dict, boolean, or None
        try:
            value = literal_eval(value)
        # The following two excepts allow v to pass through when it represents a
        # string.
        #
        # Longer explanation:
        # The type of v is always a string (before calling literal_eval), but
        # sometimes it *represents* a string and other times a data structure, like
        # a list. In the case that v represents a string, what we got back from the
        # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
        # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
        # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
        # will raise a SyntaxError.
        except ValueError:
            pass
        except SyntaxError:
            pass
        return value

    def _get_param_list(self) -> list:
        """
        get parameter(attribute) list of current config object

        Returns:
            list: attribute list
        """
        param_list = [
            name
            for name in self.__dir__()
            if not name.startswith("_") and not callable(getattr(self, name))
        ]
        return param_list

    def diff(self, cfg) -> dict:
        """
        diff given config with current config object

        Args:
            cfg(BaseConfig): given config, could be any subclass of BaseConfig

        Returns:
            dict: contains all diff pair
        """
        assert isinstance(cfg, BaseConfig), "config is not a subclass of BaseConfig"
        diff_result = {}
        self_param_list = self._get_param_list()
        conf_param_list = cfg._get_param_list()
        for param in self_param_list:
            if param not in conf_param_list:
                diff_result[param] = getattr(self, param)
            else:
                self_val, conf_val = getattr(self, param), getattr(cfg, param)
                if self_val != conf_val:
                    if isinstance(self_val, dict):
                        diff_result[param] = diff_dict(self_val, conf_val)
                    else:
                        diff_result[param] = self_val
        return diff_result

    def show_diff(self, cfg):
        """
        print diff between current config object and given config object
        """
        return pprint.pformat(edict(self.diff(cfg)))

    def find(self, key: str, show=True, color=Fore.BLACK + Back.YELLOW) -> dict:
        """
        find a given key and its value in config

        Args:
            key (str): the string you want to find
            show (bool): if show is True, print find result; or return the find result
            color (str): color of `key`, default color is black(foreground) yellow(background)

        Returns:
            dict: if  show is False, return dict that contains all find result

        Example::

            >>> from config import config        # suppose you are in your training dir
            >>> config.find("weights")
        """
        key = key.upper()
        find_result = {}
        param_list = self._get_param_list()
        for param in param_list:
            param_value = getattr(self, param)
            if re.search(key, param):
                find_result[param] = param_value
            elif isinstance(param_value, dict):
                find_res = find_key(param_value, key)
                if find_res:
                    find_result[param] = find_res
        if not show:
            return find_result
        else:
            pformat_str = pprint.pformat(edict(find_result))
            print(highlight(key, pformat_str, color))

    def __repr__(self):
        param_dict = edict(
            {param: getattr(self, param) for param in self._get_param_list()}
        )
        return pprint.pformat(param_dict)


config = BaseConfig()
