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
import logging
import os
import pickle as pkl
import sys
sys.path.insert(0, '.')  # noqa: E402
from collections import OrderedDict
import torch
from colorama import Fore, Style

from cvpods.checkpoint import DetectionCheckpointer
from cvpods.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from cvpods.evaluation import (  # isort:skip
    CityPersonsEvaluator, CityscapesEvaluator,
    ClassificationEvaluator, COCOEvaluator,
    COCOPanopticEvaluator, DatasetEvaluators,
    LVISEvaluator, PascalVOCDetectionEvaluator,
    SemSegEvaluator, WiderFaceEvaluator, verify_results
)
from cvpods.modeling import GeneralizedRCNNWithTTA
from cvpods.utils import comm

from config import config
from net import build_model


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
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
        dump = config.GLOBAL.DUMP_TRAIN

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
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, meta, output_folder, dump))
        elif evaluator_type == "cityscapes":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesEvaluator(dataset_name, meta, dump)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name, meta, dump)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, meta, cfg, True, output_folder, dump)
        elif evaluator_type == "citypersons":
            evaluator_list.append(
                CityPersonsEvaluator(dataset_name, meta, cfg, True, output_folder, dump))
        elif evaluator_type == "widerface":
            return WiderFaceEvaluator(dataset_name, meta, cfg, True, output_folder, dump)
        if evaluator_type == "classification":
            return ClassificationEvaluator(dataset_name, meta, cfg, True, output_folder, dump)
        if hasattr(cfg, "EVALUATORS"):
            for evaluator in cfg.EVALUATORS:
                evaluator_list.append(evaluator(dataset_name, meta, True, output_folder, dump=True))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("pm.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        res = cls.test(cfg, model, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def stage_main(args, cfg, build):
    cfg.merge_from_list(args.opts)
    cfg, logger = default_setup(cfg, args)
    model_build_func = build

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg, model_build_func)
    trainer.resume_or_load(resume=args.resume)

    if args.eval_only:
        DetectionCheckpointer(
            trainer.model, save_dir=cfg.OUTPUT_DIR, resume=args.resume).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, trainer.model)
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, trainer.model))
        return res

    # check wheather worksapce has enough storeage space
    # assume that a single dumped model is 700Mb
    file_sys = os.statvfs(cfg.OUTPUT_DIR)
    free_space_Gb = (file_sys.f_bfree * file_sys.f_frsize) / 2**30
    eval_space_Gb = (cfg.SOLVER.LR_SCHEDULER.MAX_ITER // cfg.SOLVER.CHECKPOINT_PERIOD) * 700 / 2**10
    if eval_space_Gb > free_space_Gb:
        logger.warning(f"{Fore.RED}Remaining space({free_space_Gb}GB) "
                       f"is less than ({eval_space_Gb}GB){Style.RESET_ALL}")

    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )

    trainer.train()

    if comm.is_main_process() and cfg.MODEL.AS_PRETRAIN:
        # convert last ckpt to pretrain format
        convert_to_pretrained_model(
            input=os.path.join(cfg.OUTPUT_DIR, "model_final.pth"),
            save_path=os.path.join(cfg.OUTPUT_DIR, "model_final_pretrain_weight.pkl")
        )


def convert_to_pretrained_model(input, save_path):
    obj = torch.load(input, map_location="cpu")
    obj = obj["model"]

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith("encoder_q.") and not k.startswith("network"):
            continue
        old_k = k
        if k.startswith("encoder_q."):
            k = k.replace("encoder_q.", "")
        elif k.startswith("network"):
            k = k.replace("network.", "")
        print(old_k, "->", k)
        newmodel[k] = v.numpy()

    res = {
        "model": newmodel,
        "__author__": "MOCO" if k.startswith("encoder_q.") else "CLS",
        "matching_heuristics": True
    }

    with open(save_path, "wb") as f:
        pkl.dump(res, f)


def main(args):
    if isinstance(config, list):
        assert isinstance(build_model, list) and len(config) == len(build_model)
        for cfg, build in zip(config, build_model):
            stage_main(args, cfg, build)
    else:
        stage_main(args, config, build_model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    if isinstance(config, list):
        assert len(config) > 0
        print("soft link first config in list to {}".format(config[0].OUTPUT_DIR))
        config[0].link_log()
    else:
        print("soft link to {}".format(config.OUTPUT_DIR))
        config.link_log()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
