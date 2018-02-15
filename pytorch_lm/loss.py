#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Implements loss functions not in torch.nn."""

import torch
import torch.nn.functional as F
import torch.nn.modules.loss as L


class SequenceLoss(L._WeightedLoss):
    """
    Implements TensorFlow's sequence_loss. It is basically the same as
    `CrossEntropyLoss`, but expects a three-dimensional input and (by default)
    returns a two-dimensional output. It also makes it possible to reduce this
    matrix along both (batch and time) dimensions.

    This class is necessary to properly able to implement BPTT, since
    `CrossEntropyLoss` would simply average the loss over both (batch and time)
    dimensions, which is not how BPTT works.

    In a way, this version is a bit more advanced that TF's, because it allows
    specifying different aggregation functions along the minebatch and time
    dimensions.
    It does not use the usual `size_average` argument.

    If provided, the optional argument `weight` should be a 1D `Tensor`
    assigning weight to each of the classes. This is particularly useful when
    you have an unbalanced training set.

    `input` has to be a 3D `Tensor` of size `(minibatch, time steps, C)`.

    This criterion expects a class index (0 to C-1) as the
    `target` for each value of a 2D tensor of size `minibatch x time steps`.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
           If given, has to be a Tensor of size `C`
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When size_average is
            ``True``, the loss is averaged over non-ignored targets.
        reduce_across_batch (str, optional): Whether to average (``'mean'``) or
            ``'sum'`` the losses over the batch dimension, or keep the dimension
            (``None``). The default is ``'mean'``.
        reduce_across_timesteps (str, optional): Whether to average (``'mean'``) or
            ``'sum'`` the losses over the time dimension, or keep the dimension
            (``None``). The default is ``'mean'``.
    """
    def __init__(self, weight=None, ignore_index=-100,
                 reduce_across_batch='mean', reduce_across_timesteps='mean'):
        super(SequenceLoss, self).__init__(weight, False)
        self.ignore_index = ignore_index
        self.reduce_across_batch = self.__aggr_func(reduce_across_batch)
        self.reduce_across_timesteps = self.__aggr_func(reduce_across_timesteps)

    @staticmethod
    def __aggr_func(func_str):
        if not func_str:
            return None
        if func_str == 'sum':
            return torch.sum
        if func_str == 'mean':
            return torch.mean
        else:
            raise ValueError('Invalid aggregation "{}"'.format(func_str))

    def forward(self, input, target):
        L._assert_no_grad(target)
        flat_input = input.view(-1, input.size(2))
        flat_targets = target.view(-1)
        flat_losses = F.cross_entropy(flat_input, flat_targets, self.weight,
                                      True, self.ignore_index, False)
        losses = flat_losses.view(target.size(0), target.size(1))
        if self.reduce_across_timesteps:
            losses = self.reduce_across_timesteps(losses, dim=1, keepdim=True)
        if self.reduce_across_batch:
            losses = self.reduce_across_batch(losses, dim=0, keepdim=True)
        return torch.squeeze(losses)
