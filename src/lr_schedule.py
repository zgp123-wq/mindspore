"""Learning rate schedule"""

import math
import numpy as np
from src.model_utils.config import config


def get_lr(global_step, lr_init, lr_end, lr_max, warmup_epochs1, warmup_epochs2,
           warmup_epochs3, warmup_epochs4, warmup_epochs5, total_epochs, steps_per_epoch):
    """
    generate learning rate array

    Args:
       global_step(int): total steps of the training
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_epochs(float): number of warmup epochs
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps1 = steps_per_epoch * warmup_epochs1
    warmup_steps2 = warmup_steps1 + steps_per_epoch * warmup_epochs2
    warmup_steps3 = warmup_steps2 + steps_per_epoch * warmup_epochs3
    warmup_steps4 = warmup_steps3 + steps_per_epoch * warmup_epochs4
    warmup_steps5 = warmup_steps4 + steps_per_epoch * warmup_epochs5
    step_radio = [1e-4, 1e-3, 1e-2, 0.1]
    for i in range(total_steps):
        if i < warmup_steps1:
            lr = lr_init * (warmup_steps1 - i) / (warmup_steps1) + \
                 (lr_max * step_radio[0]) * i / (warmup_steps1 * 3)
        elif warmup_steps1 <= i < warmup_steps2:
            lr = 1e-5 * (warmup_steps2 - i) / (warmup_steps2 - warmup_steps1) + \
                 (lr_max * step_radio[1]) * (i - warmup_steps1) / (warmup_steps2 - warmup_steps1)
        elif warmup_steps2 <= i < warmup_steps3:
            lr = 1e-4 * (warmup_steps3 - i) / (warmup_steps3 - warmup_steps2) + \
                 (lr_max * step_radio[2]) * (i - warmup_steps2) / (warmup_steps3 - warmup_steps2)
        elif warmup_steps3 <= i < warmup_steps4:
            lr = 1e-3 * (warmup_steps4 - i) / (warmup_steps4 - warmup_steps3) + \
                (lr_max * step_radio[3]) * (i - warmup_steps3) / (warmup_steps4 - warmup_steps3)
        elif warmup_steps4 <= i < warmup_steps5:
            lr = 1e-2 * (warmup_steps5 - i) / (warmup_steps5 - warmup_steps4) + \
                lr_max * (i - warmup_steps4) / (warmup_steps5 - warmup_steps4)
        else:
            lr = lr_end + \
                (lr_max - lr_end) * \
                (1. + math.cos(math.pi * (i - warmup_steps5) / (total_steps - warmup_steps5))) / 2.
        if lr < 0.0:
            lr = 0.0
        lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate
