from typing import Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss, _WeightedLoss, CrossEntropyLoss


__all__ = [
    'TopkCrossEntropyLoss',
    'MultiTaskLoss',
    'MultiTaskCrossEntropyLoss'
] # TODO: add FocalLoss after FocalLoss is done


class TopkCrossEntropyLoss(_WeightedLoss):
    """
    The implementation of Ohem loss.
    Paper:
        https://www.zpascal.net/cvpr2016/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf
    
    Arguments:
        topk {float} -- ratio of top k samples. It would choose the largest `int(topk*batch_size)` samples from data.
    
    Returns:
        [torch.tensor] -- [description]
    """

    def __init__(self, topk=0.7, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        assert isinstance(topk, float), 'The argument `topk` must have the type of `float`.'
        assert 0 < topk <= 1, 'The value of the argument `topk` must be between 0 and 1.'

        super(TopkCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.topk = topk
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        batch_size = input.size(0)

        loss = F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction='none')
        loss, index = torch.topk(loss, k=int(self.topk * batch_size), dim=0)

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()


class LabelSmoothingCrossEntropyLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        assert 0 <= smoothing <=1
        super(LabelSmoothingCrossEntropyLoss, self).__init__()

        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)

        with torch.no_grad():
            smooth_target = torch.zeros_like(input)
            smooth_target.fill_(self.smoothing / (target.size(-1) - 1))
            smooth_target.scatter_(1, target.data.unsqueeze(-1), 1.0 - self.smoothing)
        loss = (-smooth_target * log_prob).sum(-1).mean()

        return loss


class FocalLoss(nn.Module):
    # TODO: check whether it is correct
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1) # (N, C, H, W) => (N, C, H*W)
            input = input.transpose(1, 2) # (N, C, H*W) => (N, H*W, C)
            input = input.continguous().view(-1, input.size(2)) # (N*H*W, C)
        target = target.view(-1, 1) # (N, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:

            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class MultiTaskLoss(nn.Module):
    """
    total_loss = task_weights[0] * criterions[0](inputs, targets[0]) + ... +
                 task_weights[-1] * criterions[-1](inputs, targets[-1])
    
    Arguments:
        criterions {list} -- list of criterions
        task_weights {list} -- weights for weighted summation of all losses
    
    Returns:
        total_loss {torch.Tensor} -- sum of all task losses
        losses {list} -- list of task losses
    """

    def __init__(self, criterions: list, task_weights: list=None):
        super(MultiTaskLoss, self).__init__()

        if task_weights:
            assert len(criterions) == len(task_weights), \
                'The number of `criterions` is not consistent of that of `task_weights`.'

        task_weights = task_weights or [1.0 / len(criterions)] * len(criterions)
        assert np.sum(task_weights) == 1.0, \
            f'The sum of `task_weights` should be equal to one, but got {np.sum(task_weights)}.'
            
        self.n_task = len(criterions)
        self.criterions = criterions
        self.task_weights = task_weights # torch.register_buffer(...)

    def forward(self, inputs, targets):
        assert isinstance(inputs, (tuple, list)), 'The inputs of multiplt tasks should be a tuple or a list.'
        assert isinstance(targets, (tuple, list)), 'The targets of multiplt tasks should be a tuple or a list.'
        assert len(inputs) == len(self.criterions), 'The number of `inputs` is not consistent of that of `criterions`.'
        assert len(targets) == len(self.criterions), 'The number of `targets` is not consistent of that of `criterions`.'

        losses = [criterion(input, target) for criterion, input, target in zip(self.criterions, inputs, targets)]
        total_loss = sum(weight * loss for weight, loss in zip(self.task_weights, losses))

        return total_loss, losses


class MultiTaskCrossEntropyLoss(MultiTaskLoss):
    """
    total_loss = task_weights[0] * CrossEntropy(inputs, targets[0]) + ... +
                 task_weights[-1] * CrossEntropy(inputs, targets[-1])

    Returns:
        total_loss {torch.Tensor} -- sum of all losses of each task
        losses {list} -- loss of each task
    """

    def __init__(self, n_task: int, task_weights: list=None, weight: torch.Tensor=None,
                 size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        criterions = [CrossEntropyLoss(weight, size_average, ignore_index, reduce, reduction) for _ in range(n_task)]
        super(MultiTaskCrossEntropyLoss, self).__init__(criterions, task_weights)


class MultiTaskLabelSmoothingCrossEntropyLoss(MultiTaskLoss):
    """
    total_loss = task_weights[0] * LabelSmoothingLoss(inputs, targets[0]) + ... +
                 task_weights[-1] * LabelSmoothingLoss(inputs, targets[-1])

    Returns:
        total_loss {torch.Tensor} -- sum of all losses of each task
        losses {list} -- loss of each task
    """

    def __init__(self, n_task: int, tast_weights: list=None, smoothings: Union[float, list]=0.1):
        if isinstance(smoothings, list):
            assert len(smoothings) == n_task, 'The number of `smoothings` should equal to `n_task`.'
        elif isinstance(smoothings, float):
            assert 0 <= smoothings <= 1.0, 'The elements of `smoothings` should be between 0.0 and 1.0.'
            smoothings = [smoothings] * n_task

        self.smoothings = smoothings
        criterions = [LabelSmoothingCrossEntropyLoss(smoothing) for smoothing in smoothings]
        super(MultiTaskLabelSmoothingCrossEntropyLoss, self).__init__(criterions, tast_weights)


if __name__ == "__main__":
    n_task = 2

    # test MultiTaskCrossEntropy
    cross_entropy_loss = CrossEntropyLoss()
    multi_task_criterion = MultiTaskCrossEntropyLoss(n_task, task_weights=None)

    print(multi_task_criterion.task_weights)

    inputs = [torch.randn(3, 5, requires_grad=True) for _ in range(n_task)]
    targets = [torch.empty(3, dtype=torch.long).random_(5) for _ in range(n_task)]
    
    print([cross_entropy_loss(input, target).item() for input, target in zip(inputs, targets)])

    total_loss, losses = multi_task_criterion(inputs, targets)
    print(total_loss.item())
    print([loss.item() for loss in losses])

    # test MultiTaskLabelSmoothingLoss
    label_smoothing_loss = LabelSmoothingCrossEntropyLoss(smoothing=0.1)
    multi_task_criterion = MultiTaskLabelSmoothingCrossEntropyLoss(n_task, tast_weights=None, smoothings=0.1)

    print([label_smoothing_loss(input, target).item() for input, target in zip(inputs, targets)])

    total_loss, losses = multi_task_criterion(inputs, targets)
    print(total_loss.item())
    print([loss.item() for loss in losses])