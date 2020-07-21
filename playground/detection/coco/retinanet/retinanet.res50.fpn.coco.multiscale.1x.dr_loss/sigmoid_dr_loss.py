import torch
from torch import nn
import torch.nn.functional as F
import math
"""
    PyTorch Implementation for DR Loss
    Reference
    CVPR'20: "DR Loss: Improving Object Detection by Distributional Ranking"
    Copyright@Alibaba Group Holding Limited
"""


class SigmoidDRLoss(nn.Module):
    def __init__(self,
                 pos_lambda=1,
                 neg_lambda=0.1 / math.log(3.5),
                 L=6.,
                 tau=4.):
        super(SigmoidDRLoss, self).__init__()
        self.margin = 0.5
        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda
        self.L = L
        self.tau = tau

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        dtype = targets.dtype
        device = targets.device
        class_range = torch.arange(1,
                                   num_classes + 1,
                                   dtype=dtype,
                                   device=device).unsqueeze(0)
        t = targets.unsqueeze(1)
        pos_ind = (t == class_range)
        neg_ind = (t != class_range) * (t >= 0)
        pos_prob = logits[pos_ind].sigmoid()
        neg_prob = logits[neg_ind].sigmoid()
        neg_q = F.softmax(neg_prob / self.neg_lambda, dim=0)
        neg_dist = torch.sum(neg_q * neg_prob)
        if pos_prob.numel() > 0:
            pos_q = F.softmax(-pos_prob / self.pos_lambda, dim=0)
            pos_dist = torch.sum(pos_q * pos_prob)
            loss = self.tau * torch.log(
                1. + torch.exp(self.L *
                               (neg_dist - pos_dist + self.margin))) / self.L
        else:
            loss = self.tau * torch.log(
                1. + torch.exp(self.L *
                               (neg_dist - 1. + self.margin))) / self.L
        return loss
