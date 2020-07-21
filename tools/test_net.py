# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in cvpods.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use cvpods as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import glob
import logging
import os
import re
import sys
sys.path.insert(0, '.')

from collections import OrderedDict
import torch

from cvpods.checkpoint import DetectionCheckpointer
from cvpods.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from cvpods.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    CityPersonsEvaluator,
    CrowdHumanEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    WiderFaceEvaluator,
    ClassificationEvaluator,
    verify_results,
)
from cvpods.modeling import GeneralizedRCNNWithTTA
from cvpods.utils import comm, PathManager

from config import config
from net import build_model


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, dataset, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        dump = config.GLOBAL.DUMP_TEST

        evaluator_list = []
        meta = dataset.meta
        evaluator_type = meta.evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    dataset,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                    dump=dump,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg", "citypersons"]:
            evaluator_list.append(
                COCOEvaluator(dataset_name, meta, cfg, True, output_folder, dump))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(
                COCOPanopticEvaluator(dataset_name, meta, output_folder, dump))
        if evaluator_type == "cityscapes":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesEvaluator(dataset_name, meta, dump)
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name, meta, dump)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, meta, cfg, True, output_folder, dump)
        if evaluator_type == "citypersons":
            evaluator_list.append(
                CityPersonsEvaluator(dataset_name, meta, cfg, True, output_folder, dump))
        if evaluator_type == "crowdhuman":
            return CrowdHumanEvaluator(dataset_name, meta, cfg, True, output_folder, dump=True)
        if evaluator_type == "widerface":
            return WiderFaceEvaluator(dataset_name, meta, cfg, True, output_folder, dump)
        if evaluator_type == "classification":
            return ClassificationEvaluator(dataset_name, meta, cfg, True, output_folder, dump)
        if hasattr(cfg, "EVALUATORS"):
            for evaluator in cfg.EVALUATORS:
                evaluator_list.append(
                    evaluator(dataset_name, meta, True, output_folder, dump=False))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("cvpods.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        res = cls.test(cfg, model, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def test_argument_parser():
    parser = default_argument_parser()
    parser.add_argument("--start-iter", type=int, default=0, help="start iter used to test")
    parser.add_argument("--end-iter", type=int, default=None,
                        help="end iter used to test")
    parser.add_argument("--debug", action="store_true", help="use debug mode or not")
    return parser


def main(args):
    config.merge_from_list(args.opts)
    cfg, logger = default_setup(config, args)
    if args.debug:
        batches = int(cfg.SOLVER.IMS_PER_BATCH / 8 * args.num_gpus)
        if cfg.SOLVER.IMS_PER_BATCH != batches:
            cfg.SOLVER.IMS_PER_BATCH = batches
            logger.warning("SOLVER.IMS_PER_BATCH is changed to {}".format(batches))

    if "MODEL.WEIGHTS" in args.opts:
        if cfg.MODEL.WEIGHTS.endswith(".pth") and not PathManager.exists(cfg.MODEL.WEIGHTS):
            ckpt_name = cfg.MODEL.WEIGHTS.split("/")[-1]
            model_prefix = cfg.OUTPUT_DIR.split("cvpods_playground")[1][1:]
            remote_file_path = os.path.join(cfg.OSS.DUMP_PREFIX, model_prefix, ckpt_name)
            logger.warning(f"The specified ckpt file ({cfg.MODEL.WEIGHTS}) was not found locally,"
                           f" try to load the corresponding dump file on OSS ({remote_file_path}).")
            cfg.MODEL.WEIGHTS = remote_file_path
        valid_files = [cfg.MODEL.WEIGHTS]
    else:
        list_of_files = glob.glob(os.path.join(cfg.OUTPUT_DIR, '*.pth'))

        assert list_of_files, "No checkpoint file found in {}.".format(cfg.OUTPUT_DIR)
        list_of_files.sort(key=os.path.getctime)
        latest_file = list_of_files[-1]
        if not args.end_iter:
            valid_files = [latest_file]
        else:
            files = [f for f in list_of_files if str(f) <= str(latest_file)]
            valid_files = []
            for f in files:
                try:
                    model_iter = int(re.split(r'(model_|\.pth)', f)[-3])
                except Exception:
                    logger.warning("remove {}".format(f))
                    continue
                if args.start_iter <= model_iter <= args.end_iter:
                    valid_files.append(f)
            assert valid_files, "No .pth files satisfy your requirement"

    # * means all if need specific format then *.csv
    for current_file in valid_files:
        cfg.MODEL.WEIGHTS = current_file
        model = build_model(cfg)

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))


if __name__ == "__main__":
    args = test_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
