from contextlib import contextmanager

import torch
from torch.optim.optimizer import Optimizer

from cvpods._C import compute_adaptive_lr

from cvpods.solver.registry import OPTIMIZERS

__all__ = ['LARS']


@OPTIMIZERS.register()
class LARS(Optimizer):
    """Implements 'LARS (Layer-wise Adaptive Rate Scaling)'__ as Optimizer a
    :class:`~torch.optim.Optimizer` wrapper.
    __ : https://arxiv.org/abs/1708.03888
    Wraps an arbitrary optimizer like :class:`torch.optim.SGD` to use LARS. If
    you want to the same performance obtained with small-batch training when
    you use large-batch training, LARS will be helpful::

    Args:
        optimizer (Optimizer):
            optimizer to wrap
        eps (float, optional):
            epsilon to help with numerical stability while calculating the
            adaptive learning rate
        trust_coef (float, optional):
            trust coefficient for calculating the adaptive learning rate

    Example::
        base_optimizer = optim.SGD(model.parameters(), lr=0.1)
        optimizer = LARS(optimizer=base_optimizer)
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    """

    def __init__(self, optimizer, eps=1e-8, trust_coef=0.001):
        if eps < 0.0:
            raise ValueError('invalid epsilon value: , %f' % eps)
        if trust_coef < 0.0:
            raise ValueError("invalid trust coefficient: %f" % trust_coef)

        self.optim = optimizer
        self.eps = eps
        self.trust_coef = trust_coef
        self.adaptive_lr = torch.ones([])

    def __getstate__(self):
        lars_dict = {}
        lars_dict['eps'] = self.eps
        lars_dict['trust_coef'] = self.trust_coef
        lars_dict['adaptive_lr'] = self.adaptive_lr
        return (self.optim, lars_dict)

    def __setstate__(self, state):
        self.optim, lars_dict = state

        self.eps = lars_dict['eps']
        self.trust_coef = lars_dict['trust_coef']
        self.adaptive_lr = lars_dict['adaptive_lr']

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.optim)

    @property
    def param_groups(self):
        return self.optim.param_groups

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    @contextmanager
    def hide_weight_decays(self):
        weight_decays = []

        for group in self.optim.param_groups:
            if 'weight_decay' in group:
                weight_decays.append(group['weight_decay'])
                group['weight_decay'] = 0
            else:
                weight_decays.append(None)

        try:
            yield weight_decays
        finally:
            for group, weight_decay in zip(self.optim.param_groups, weight_decays):
                if weight_decay is None:
                    continue
                group['weight_decay'] = weight_decay

    def apply_adaptive_lrs(self, weight_decays):
        with torch.no_grad():
            for group, weight_decay in zip(self.optim.param_groups, weight_decays):
                if weight_decay is None:
                    weight_decay = 0.0
                for p in group['params']:
                    if p.grad is None:
                        continue

                    param_norm = p.norm()
                    grad_norm = p.grad.norm()

                    # The optimizer class has no method to change `dtype` of
                    # its inner tensors (like `adaptive_lr`) and to select to
                    # use CPU or GPU with Tensor. LARS's interface follows the
                    # optimizer class's interface, so LARS cannot change
                    # `dtype` of inner tensors explicitly also. In that
                    # context, we have constructed LARS can modify its member
                    # variable's spec implicitly by comparing with given spec
                    # by the original optimizer's element.
                    param_norm_spec = (param_norm.is_cuda, param_norm.type())
                    adaptive_lr_spec = (self.adaptive_lr.is_cuda, self.adaptive_lr.type())

                    if param_norm_spec != adaptive_lr_spec:
                        self.adaptive_lr = torch.ones_like(param_norm)

                    # calculate adaptive lr & weight decay
                    adaptive_lr = compute_adaptive_lr(
                        param_norm,
                        grad_norm,
                        weight_decay,
                        self.eps,
                        self.trust_coef,
                        self.adaptive_lr)

                    p.grad.add_(weight_decay, p.data)
                    p.grad.mul_(adaptive_lr)

    def step(self, *args, **kwargs):
        with self.hide_weight_decays() as weight_decays:
            self.apply_adaptive_lrs(weight_decays)
            return self.optim.step(*args, **kwargs)
