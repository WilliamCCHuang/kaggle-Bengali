import os
os.sys.path.append('/Users/william/Documents/kaggle/kaggle-Bengali')

import argparse
import torch.optim as optim

import pytorch_lightning as pl

from base_models import BaseCNNModel
from lightning_models import BengaliLightningModel
from datasets import BengaliTrainDataset, BengaliTestDataset
from utils import load_images

from pytorch_utils.losses import *
from pytorch_utils.optimizers import RAdam


TRAIN_DIR = 'data/' # TODO:
TEST_DIR = 'data/' #TODO:


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', default='crossentropyloss')
    parser.add_argument('--hidden_dim', default=128)
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--weight_decay', default=1e-4)
    parser.add_argument('--smoothing', default=0.1)
    parser.add_argument('--alpha', default=1)
    parser.add_argument('--gamma', default=2)
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--task_weights', nargs='+', default=[1./3, 1./3, 1./3])
    
    return parser
    

def check_args(args):
    assert args.loss in [
        'crossentropyloss',
        'topkcrossentropyloss',
        'labelsmoothingcrossentropyloss',
        'focalloss',
    ]

    assert isinstance(args.task_weights, list), args.task_weights
    assert sum(args.task_weights) == 1.0, args.task_weights


def build_loss(args):
    if args.loss == 'crossentropyloss':
        return MultiTaskCrossEntropyLoss(n_task=3, task_weights=args.task_weights)
    if args.loss == 'labelsmoothingcrossentropyloss':
        return MultiTaskLabelSmoothingCrossEntropyLoss(n_task=3, tast_weights=args.tast_weights, smoothings=args.smoothing)
    if args.loss == 'focalloss':
        return MultiTaskLoss(criterions=[
            FocalLoss(alpha=args.alpha, gamma=args.gamma),
            FocalLoss(alpha=args.alpha, gamma=args.gamma),
            FocalLoss(alpha=args.alpha, gamma=args.gamma)
        ], task_weights=args.task_weights)


def main():
    # args
    parser = build_parser()
    args = parser.parse_args()

    check_args(args)

    # data
    df_train = load_images(mode='test') # TODO:
    train_dataloader = None # TODO:
    val_dataloader = None # TODO:

    # model
    criterions = build_loss(args)
    base_cnn_model = BaseCNNModel(model_name='se_resnext50_32x4d', hidden_dim=args.hidden_dim, dropout=args.dropout)
    optimizer = RAdam(base_cnn_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, )

    model = BengaliLightningModel(base_model=base_cnn_model,
                                  train_dataloader=train_dataloader,
                                  val_dataloader=val_dataloader,
                                  criterions=criterions,
                                  optimizer=optimizer)
    
    # train
    trainer = pl.Trainer(max_epochs=args.epochs,
                         gpus=args.gpus,
                         early_stop_callback=False,
                         checkpoint_callback=None) # TODO: checkpoint
    trainer.fit(model)


if __name__ == "__main__":
    main()