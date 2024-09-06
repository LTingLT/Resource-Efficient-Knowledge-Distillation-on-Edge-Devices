import numpy as np
import torch
import warnings
import time
import torch.distributed as dist

class LinearWarmupAndCosineAnneal(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warm_up, T_max, last_epoch=-1, smooth=1e-9):
        self.warm_up = int(warm_up * T_max)
        self.T_max = T_max - self.warm_up
        self.smooth = smooth
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("If use the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        if self.last_epoch == 0:
            return [lr / (self.warm_up + 1) for lr in self.base_lrs]
        elif self.last_epoch <= self.warm_up:
            c = (self.last_epoch + 1) / self.last_epoch
            return [group['lr'] * c for group in self.optimizer.param_groups]
        else:
            le = self.last_epoch - self.warm_up

            if le > self.T_max:
                warnings.warn(f"Epoch {self.last_epoch}: reached maximum number of iterations {self.T_max + self.warm_up}. This is unexpected behavior, and this SimCLR implementation was not tested in this regime!")

            return [(1 + np.cos(np.pi * le / self.T_max)) /
                    (1 + np.cos(np.pi * (le - 1) / self.T_max) + self.smooth) *
                    group['lr']
                    for group in self.optimizer.param_groups]
