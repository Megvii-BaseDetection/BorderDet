# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Set, Type, Union
import torch

from .registry import OPTIMIZERS, LR_SCHEDULERS

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"


def _create_gradient_clipper(cfg) -> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """
    cfg = cfg.clone()

    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, cfg.CLIP_VALUE, cfg.NORM_TYPE)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, cfg.CLIP_VALUE)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg.CLIP_TYPE)]


def _generate_optimizer_class_with_gradient_clipping(
    optimizer_type: Type[torch.optim.Optimizer], gradient_clipper: _GradientClipper
) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """

    def optimizer_wgc_step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                gradient_clipper(p)
        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer_type.__name__ + "WithGradientClip",
        (optimizer_type,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(cfg, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer instance of some type OptimizerType to become an instance
    of the new dynamically created class OptimizerTypeWithGradientClip
    that inherits OptimizerType and overrides the `step` method to
    include gradient clipping.
    Args:
        cfg: CfgNode
            configuration options
        optimizer: torch.optim.Optimizer
            existing optimizer instance
    Return:
        optimizer: torch.optim.Optimizer
            either the unmodified optimizer instance (if gradient clipping is
            disabled), or the same instance with adjusted __class__ to override
            the `step` method and include gradient clipping
    """
    if not cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
        return optimizer
    grad_clipper = _create_gradient_clipper(cfg.SOLVER.CLIP_GRADIENTS)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        type(optimizer), grad_clipper
    )
    optimizer.__class__ = OptimizerWithGradientClip
    return optimizer


def build_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config
    """
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    if cfg.SOLVER.OPTIMIZER.NAME == "SGD":
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module in model.modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.OPTIMIZER.BASE_LR
                weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
                if isinstance(module, norm_module_types):
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY_NORM
                elif key == "bias":
                    # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                    # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                    # hyperparameters are by default exactly the same as for regular
                    # weights.
                    lr = cfg.SOLVER.OPTIMIZER.BASE_LR * cfg.SOLVER.OPTIMIZER.BIAS_LR_FACTOR
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        optimizer = OPTIMIZERS.get("SGD")(
            params,
            cfg.SOLVER.OPTIMIZER.BASE_LR,
            momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER.NAME == "AdamW":
        lr = cfg.SOLVER.OPTIMIZER.BASE_LR
        optimizer = OPTIMIZERS.get("AdamW")(
            model.parameters(),
            lr=lr,
            betas=cfg.SOLVER.OPTIMIZER.BETAS,
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY,
            amsgrad=cfg.SOLVER.OPTIMIZER.AMSGRAD
        )
    elif cfg.SOLVER.OPTIMIZER.NAME == "Adam":
        lr = cfg.SOLVER.OPTIMIZER.BASE_LR
        optimizer = OPTIMIZERS.get("Adam")(
            model.parameters(),
            lr=lr,
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY,
            amsgrad=cfg.SOLVER.OPTIMIZER.AMSGRAD
        )
    elif cfg.SOLVER.OPTIMIZER.NAME == "SGD_GATE_LR_MULTI":
        # For DynamicRouting
        # multiply lr for gating function
        gate_lr_multi = cfg.SOLVER.OPTIMIZER.GATE_LR_MULTI
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for name, module in model.named_modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.OPTIMIZER.BASE_LR
                weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
                if isinstance(module, norm_module_types):
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY_NORM
                elif key == "bias":
                    # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                    # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                    # hyperparameters are by default exactly the same as for regular
                    # weights.
                    lr = cfg.SOLVER.OPTIMIZER.BASE_LR * cfg.SOLVER.OPTIMIZER.BIAS_LR_FACTOR
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY

                if gate_lr_multi > 0.0 and "gate_conv" in name:
                    lr *= gate_lr_multi

                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        optimizer = torch.optim.SGD(
            params,
            cfg.SOLVER.OPTIMIZER.BASE_LR,
            momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM)

    if hasattr(cfg.SOLVER.OPTIMIZER, "LARS"):
        if cfg.SOLVER.OPTIMIZER.LARS.ENABLED:
            eps = cfg.SOLVER.OPTIMIZER.LARS.EPS
            trust_coef = cfg.SOLVER.OPTIMIZER.LARS.TRUST_COEF
            optimizer = OPTIMIZERS.get("LARS")(optimizer, eps, trust_coef)

    return optimizer


def build_lr_scheduler(
    cfg, optimizer: torch.optim.Optimizer, epoch_iters: int = -1
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER.NAME
    if name == "WarmupMultiStepLR":
        return LR_SCHEDULERS.get(name)(
            optimizer,
            cfg.SOLVER.LR_SCHEDULER.STEPS,
            cfg.SOLVER.LR_SCHEDULER.GAMMA,
            warmup_factor=cfg.SOLVER.LR_SCHEDULER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.LR_SCHEDULER.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return LR_SCHEDULERS.get(name)(
            optimizer,
            cfg.SOLVER.LR_SCHEDULER.MAX_ITER,
            warmup_factor=cfg.SOLVER.LR_SCHEDULER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.LR_SCHEDULER.WARMUP_METHOD,
            epoch_iters=epoch_iters,
        )
    elif name == "LambdaLR":
        return LR_SCHEDULERS.get(name)(
            optimizer,
            cfg.SOLVER.LR_SCHEDULER.LAMBDA_SCHEDULE)
    elif name == "OneCycleLR":
        return LR_SCHEDULERS.get(name)(
            optimizer,
            cfg.SOLVER.LR_SCHEDULER.MAX_LR,
            total_steps=cfg.SOLVER.LR_SCHEDULER.MAX_ITER,
            pct_start=cfg.SOLVER.LR_SCHEDULER.PCT_START,
            base_momentum=cfg.SOLVER.LR_SCHEDULER.BASE_MOM,
            max_momentum=cfg.SOLVER.LR_SCHEDULER.MAX_MOM,
            div_factor=cfg.SOLVER.LR_SCHEDULER.DIV_FACTOR
        )
    elif name == "PolyLR":
        return LR_SCHEDULERS.get(name)(
            optimizer,
            cfg.SOLVER.LR_SCHEDULER.MAX_ITER,
            cfg.SOLVER.LR_SCHEDULER.POLY_POWER,
            warmup_factor=cfg.SOLVER.LR_SCHEDULER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.LR_SCHEDULER.WARMUP_METHOD,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
