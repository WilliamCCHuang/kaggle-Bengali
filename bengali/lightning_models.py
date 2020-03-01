import os
os.sys.path.append('~/AI/Kaggle/Bengalai/kaggle-Bengali/bengali')

import numpy as np

import torch
import pytorch_lightning as pl

from metrics import multi_task_macro_recall
from pytorch_lightning_utils.multi_task_models import MultiTaskLightningModel


class BengaliLightningModel(MultiTaskLightningModel):

    def __init__(self, base_model, train_dataloader, val_dataloader,
                 criterions, optimizer, scheduler=None):
        super(BengaliLightningModel, self).__init__(base_model, criterions)
        self._train_dataloader = train_dataloader # add _ in the head in order not to conflict
        self._val_dataloader = val_dataloader # add _ in the head in order not to conflict
        self.optimizer = optimizer
        self.scheduler = scheduler # modify `MultiTaskLightningModel.get_lr()` if change name of self.scheduler

    def validation_step(self, batch, batch_idx):
        x_train, y_trains = batch

        y_preds = self.forward(x_train)

        total_loss, losses = self.criterions(y_preds, y_trains)

        total_loss = total_loss.item()
        losses = [loss.item() for loss in losses]

        y_preds = [torch.argmax(y_pred, axis=1).cpu().numpy() for y_pred in y_preds]
        y_trains = [y_train.cpu().numpy() for y_train in y_trains]

        output = {
            'batch_size': x_train.size(0),
            'y_trains': y_trains,
            'y_preds': y_preds,
            'total_val_loss': total_loss,
            **{
                f'val_loss_{i+1}': loss for i, loss in enumerate(losses)
            }
        }

        return output

    def validation_end(self, outputs):
        total_data = sum(output['batch_size'] for output in outputs)
        
        total_val_loss = sum(output['total_val_loss'] * output['batch_size'] for output in outputs) / total_data
        val_loss_1 = sum(output['val_loss_1'] * output['batch_size'] for output in outputs) / total_data
        val_loss_2 = sum(output['val_loss_2'] * output['batch_size'] for output in outputs) / total_data
        val_loss_3 = sum(output['val_loss_3'] * output['batch_size'] for output in outputs) / total_data

        y_preds_1 = np.hstack([output['y_preds'][0] for output in outputs])
        y_preds_2 = np.hstack([output['y_preds'][1] for output in outputs])
        y_preds_3 = np.hstack([output['y_preds'][2] for output in outputs])
        
        y_trains_1 = np.hstack([output['y_trains'][0] for output in outputs])
        y_trains_2 = np.hstack([output['y_trains'][1] for output in outputs])
        y_trains_3 = np.hstack([output['y_trains'][2] for output in outputs])

        total_recall, recalls = multi_task_macro_recall(true_graphemes=y_trains_1, pred_graphemes=y_preds_1,
                                                        true_vowels=y_trains_2, pred_vowels=y_preds_2,
                                                        true_consonants=y_trains_3, pred_consonants=y_preds_3,
                                                        n_grapheme=10, n_vowel=10, n_consonant=10)

        tensorboard_logs = {
            'val_total_loss': total_val_loss,
            **{
                f'val_loss_{i+1}': loss for i, loss in enumerate([val_loss_1, val_loss_2, val_loss_3])
            },
            'val_total_recall': total_recall,
            'val_graphemes_recall': recalls[0],
            'val_vowels_recall': recalls[1],
            'val_consonants_recall': recalls[2]
        } # data for tensorboard can be a python number, np.array, or torch.tensor

        return {'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        optimizers = [self.optimizer]
        schedulers = [self.scheduler] if self.scheduler else []

        return optimizers, schedulers

    @pl.data_loader
    def train_dataloader(self):
        return self._train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        return self._val_dataloader