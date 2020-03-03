import argparse
import numpy as np
import pandas as pd
from typing import Union

import torch
import torch.nn as nn


def str2bool(v):
    if isinstance(v, bool):
        return bool
    if v.lower() in ('yes', 'true', 't', 'y', '1'):  
        return True  
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):  
        return False  
    else:  
        raise argparse.ArgumentTypeError('Boolean value expected.') 


def check_args_to_run(args):
    print('\nThe arguments:')
    for arg in vars(args):
        print(f'* {arg} = {getattr(args, arg)}')

    result = input('\nDo you wanna run? (y/n):')

    if result in ['y', 'Y']:
        print('Run!')
    elif result in ['n', 'N']:
        print('Exit!')
        exit()
    else:
        raise ValueError('You can only type in \'y\', \'Y\', \'n\', or \'N\'.')


def change_to_tensor(x: Union[np.array, pd.DataFrame], requires_grad: bool=True) -> torch.tensor:
    if isinstance(x, pd.DataFrame):
        x = x.values

    return torch.tensor(x, dtype=torch.float, requires_grad=requires_grad)


def train_on_batch(model, X, y, criterion, optimizer):
    model.train()

    optimizer.zero_grad()
    outputs = model(X)

    # `loss` is a torch.tensor for single task, and
    # `loss` is a tuple consisting of torch.tensor for multiple tasks
    loss = criterion(outputs, y)

    if isinstance(loss, tuple):
        # multiple tasks
        total_loss, losses = loss
    else:
        # single task
        total_loss = loss

    total_loss.backward()
    optimizer.step()

    return loss


def evaluate_on_batch(model, X, y, criterion=None, eval_funcs=[]):
    if not criterion and not eval_funcs:
        raise ValueError('One of `criterion` or `eval_func` must be assigned.')

    loss = None
    evaluations = []

    model.eval()
    with torch.no_grad():
        output = model(X)

        if criterion:
            loss = criterion(output, y)

            if isinstance(loss, tuple):
                # for multiple tasks
                total_loss, losses = loss
                loss = total_loss.item(), [loss.item() for loss in losses]
            else:
                loss = loss.item()

    evaluations = [eval_func(y.detach().numpy(), output.detach().numpy()) for eval_func in eval_funcs]

    return loss, evaluations


if __name__ == "__main__":
    import inspect
    test = nn.Module()
    print(str(test))
    
