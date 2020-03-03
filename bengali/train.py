import os
import sys
sys.path.append('/home/jarvis1121/AI/Kaggle/Bengali/kaggle-Bengali')

import argparse
import torch.optim as optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from base_models import BaseCNNModel
from utils import load_labels, load_images
from lightning_models import BengaliLightningModel
from datasets import BengaliTrainDataset, BengaliTestDataset

from pytorch_utils.losses import FocalLoss, MultiTaskLoss, MultiTaskCrossEntropyLoss, MultiTaskLabelSmoothingCrossEntropyLoss
from pytorch_utils.optimizers import RAdam, Ranger, LookaheadAdam
from pytorch_utils.schedulers import FlatCosineAnnealing
from pytorch_utils.utils import str2bool, check_args_to_run


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', help='The index of this experiment', default=None)

    parser.add_argument('--model_name', default='se_resnext101_32x4d')
    parser.add_argument('--activation', default='ReLU')

    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--test_size', type=float, default=0.3)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--loss', default='crossentropyloss')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--focal_alpha', type=float, default=1)
    parser.add_argument('--focal_gamma', type=float, default=2)

    parser.add_argument('--optimizer', default='Ranger')
    parser.add_argument('--lookahead', type=str2bool, default='False')
    parser.add_argument('--lookahead_alpha', type=float, default=0.5)
    parser.add_argument('--lookahead_k', type=int, default=6)

    parser.add_argument('--scheduler', default='ReduceLROnPlateau')
    parser.add_argument('--reduce_factor', type=float, default=0.1)
    parser.add_argument('--flat_duration', type=float, default=0.7)

    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--task_weights', type=float, nargs='+', default=[1./3, 1./3, 1./3])
    
    return parser


def check_args(args):
    try:
        int(args.exp)
    except:
        raise ValueError(f'Expected integer, but got {args.exp}')

    assert args.model_name in [
        'resnet18', 'resnet152',
        'densenet121', 'densenet161',
        'se_resnet50', 'se_resnet152',
        'se_resnext50_32x4d', 'se_resnext101_32x4d',
        'efficientnet-b0',
        'efficientnet-b7'
    ]

    assert args.activation.lower() in [
        'relu',
        'mish'
    ]
    
    assert args.loss in [
        'crossentropyloss',
        'topkcrossentropyloss',
        'labelsmoothingcrossentropyloss',
        'focalloss',
    ]

    assert args.optimizer.lower() in [
        'adam', 'radam', 'ranger'
    ]

    assert args.scheduler.lower() in [
        'reducelronplateau',
        'flatcosineannealing'
    ]

    if args.scheduler.lower() == 'reducelronplateau':
        assert 0.0 < args.reduce_factor < 1.0
    # print(args.lookahead, type(args.lookahead))
    # exit()

    assert isinstance(args.task_weights, list), args.task_weights
    assert sum(args.task_weights) == 1.0, args.task_weights


def build_loss(args):
    if args.loss.lower() == 'crossentropyloss':
        return MultiTaskCrossEntropyLoss(n_task=3, 
                                task_weights=args.task_weights)

    if args.loss.lower() == 'labelsmoothingcrossentropyloss':
        return MultiTaskLabelSmoothingCrossEntropyLoss(n_task=3, 
                                tast_weights=args.tast_weights, 
                                smoothings=args.smoothing)

    if args.loss.lower() == 'focalloss':
        return MultiTaskLoss(criterions=[
            FocalLoss(alpha=args.alpha, gamma=args.gamma),
            FocalLoss(alpha=args.alpha, gamma=args.gamma),
            FocalLoss(alpha=args.alpha, gamma=args.gamma)
        ], task_weights=args.task_weights)


def build_optimizer(args, model):
    if args.optimizer.lower() == 'adam':
        if str2bool(args.lookahead):
            optimizer = LookaheadAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'radam':
        if args.lookahead:
            optimizer = Ranger(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = Ranger(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return optimizer


def build_scheduler(args, optimizer):
    if args.scheduler.lower() == 'reducelronplateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.reduce_factor)
    else:
        # 'flatcosineannealing'
        return FlatCosineAnnealing(optimizer, epochs=args.epochs, flat_duration=args.flat_duration)


def main():
    # args
    parser = build_parser()
    args = parser.parse_args()
    check_args(args)
    check_args_to_run(args)

    # data
    print('\n===== Starting preparing data =====\n')

    labels = load_labels()
    images = load_images(mode='train')

    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=args.test_size) # TODO: discuss

    train_dataset = BengaliTrainDataset(images=train_images, labels=train_labels, size=args.image_size)
    val_dataset = BengaliTrainDataset(images=val_images, labels=val_labels, size=args.image_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count())
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=os.cpu_count())

    print('\n===== Completed preparing data =====')

    # model
    criterions = build_loss(args)
    base_cnn_model = BaseCNNModel(model_name=args.model_name, 
                                  hidden_dim=args.hidden_dim, 
                                  dropout=args.dropout,
                                  activation=args.activation)

    optimizer = build_optimizer(args, base_cnn_model)
    scheduler = build_scheduler(args, optimizer)

    model = BengaliLightningModel(base_model=base_cnn_model,
                                  train_dataloader=train_dataloader,
                                  val_dataloader=val_dataloader,
                                  criterions=criterions,
                                  optimizer=optimizer,
                                  scheduler=scheduler)
    
    # callbacks
    filepath = f'/home/jarvis1121/AI/Kaggle/Bengali/kaggle-Bengali/models/trial_{int(args.exp)}'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss',
                                                       verbose=1, mode='min')
    
    print('\n===== Starting training =====')
    
    # train
    trainer = pl.Trainer(max_epochs=args.epochs,
                         gpus=args.gpus,
                         early_stop_callback=False,
                         checkpoint_callback=checkpoint_callback)

    trainer.fit(model)

    print('\n===== End training =====')


if __name__ == "__main__":
    main()
