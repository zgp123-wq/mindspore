# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import mindspore.ops as ops
import mindspore.numpy as np
# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = ops.Abs(input - target)
    cond = n < beta
    loss = np.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
