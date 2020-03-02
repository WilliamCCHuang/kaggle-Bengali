import os
# os.sys.path.append('/Users/william/Documents/kaggle/kaggle-Bengali')

import argparse

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from base_models import BaseCNNModel
from utils import load_labels, load_images
from lightning_models import BengaliLightningModel
from datasets import BengaliTrainDataset, BengaliTestDataset

from pytorch_utils.losses import *
from pytorch_utils.optimizers import RAdam


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='se_resnext101_32x4d')
    parser.add_argument('--image_size', default=128)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--hidden_dim', default=128)
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--loss', default='crossentropyloss')
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
    assert args.model_name in [
        'resnet18', 'resnet152',
        'densenet121', 'densenet161',
        'se_resnet50', 'se_resnet152',
        'se_resnext50_32x4d', 'se_resnext101_32x4d',
        'efficientnet-b0',
        'efficientnet-b7'
    ]

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
    labels = load_labels()
    images = load_images(mode='train')

    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.3) # TODO: test_size

    train_dataset = BengaliTrainDataset(images=train_images, labels=train_labels, size=args.size)
    val_dataset = BengaliTrainDataset(images=val_images, labels=val_labels, size=args.size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count())
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=os.cpu_count())

    # model
    criterions = build_loss(args)
    base_cnn_model = BaseCNNModel(model_name=args.model_name, hidden_dim=args.hidden_dim, dropout=args.dropout)
    optimizer = RAdam(base_cnn_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model = BengaliLightningModel(base_model=base_cnn_model,
                                  train_dataloader=train_dataloader,
                                  val_dataloader=val_dataloader,
                                  criterions=criterions,
                                  optimizer=optimizer)
    
    # callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath='models/trial_1', monitor='val_total_loss',
                                                       verbose=1, mode='min')
    # train
    trainer = pl.Trainer(max_epochs=args.epochs,
                         gpus=args.gpus,
                         early_stop_callback=False,
                         checkpoint_callback=checkpoint_callback)
    trainer.fit(model)


if __name__ == "__main__":
    main()