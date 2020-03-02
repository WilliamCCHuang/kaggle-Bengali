import math
import warnings
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR


class FlatCosineAnnealing(LambdaLR):

    def __init__(self, optimizer, epochs, flat_duration, last_epoch=-1):
        self.epochs = epochs
        self.flat_duration = flat_duration

        super(FlatCosineAnnealing, self).__init__(optimizer, lr_lambda=self.compute_lr, last_epoch=last_epoch)

    def compute_lr(self, epoch):
        if epoch < self.flat_duration:
            return 1.0
        else:
            ratio = float(epoch + 1 - self.flat_duration) / (self.epochs - self.flat_duration)
            ratio = (1 + math.cos(math.pi * ratio)) / 2.0
            
            return ratio