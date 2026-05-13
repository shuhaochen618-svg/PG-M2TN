"""
Learning Rate Scheduler with Linear Warmup + Cosine Annealing
===============================================================
Implements a warmup phase followed by cosine decay, designed
for multi-task training stability.
"""

import numpy as np


class WarmupCosineScheduler:
    """
    Linear warmup followed by cosine annealing learning rate schedule.

    Args:
        optimizer      : PyTorch optimizer.
        warmup_epochs  : Number of linear warmup epochs.
        total_epochs   : Total training epochs.
        base_lr        : Base learning rate (reached after warmup).
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr

    def step(self, epoch):
        """
        Update learning rate for the given epoch.

        Returns:
            lr : Current learning rate.
        """
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / \
                       (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr
