import logging
import torch

from cvpods.layers import ShapeSpec
from cvpods.modeling.meta_arch import RetinaNet
from cvpods.modeling.meta_arch.retinanet import permute_all_cls_and_box_to_N_HWA_K_and_concat
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone.fpn import build_retinanet_resnet_fpn_backbone
from cvpods.modeling.anchor_generator import DefaultAnchorGenerator
from cvpods.modeling.losses import smooth_l1_loss

from sigmoid_dr_loss import SigmoidDRLoss


class DRRetinaNet(RetinaNet):
    def __init__(self, cfg):
        super(DRRetinaNet, self).__init__(cfg)
        self.regress_norm = cfg.MODEL.RETINANET.REG_NORM
        self.box_cls_loss_func = SigmoidDRLoss()

    def losses(self, gt_classes, gt_anchors_deltas, pred_class_logits,
               pred_anchor_deltas):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`RetinaNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_anchor_deltas, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        pos_inds = torch.nonzero((gt_classes >= 0)
                                 & (gt_classes != self.num_classes)).squeeze(1)

        retinanet_regression_loss = smooth_l1_loss(
            pred_anchor_deltas[pos_inds],
            gt_anchors_deltas[pos_inds],
            beta=self.smooth_l1_loss_beta,
            # size_average=False,
            reduction="sum",
        ) / max(1, pos_inds.numel() * self.regress_norm)

        labels = torch.ones_like(gt_classes)
        # convert labels from 0~79 to 1~80
        labels[pos_inds] += gt_classes[pos_inds]
        labels[gt_classes == -1] = gt_classes[gt_classes == -1]
        labels[gt_classes == self.num_classes] = 0
        labels = labels.int()

        retinanet_cls_loss = self.box_cls_loss_func(pred_class_logits, labels)

        return {
            "loss_cls": retinanet_cls_loss,
            "loss_box_reg": retinanet_regression_loss
        }


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = build_retinanet_resnet_fpn_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_anchor_generator(cfg, input_shape):

    return DefaultAnchorGenerator(cfg, input_shape)


def build_model(cfg):

    cfg.build_backbone = build_backbone
    cfg.build_anchor_generator = build_anchor_generator

    model = DRRetinaNet(cfg)
    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model
