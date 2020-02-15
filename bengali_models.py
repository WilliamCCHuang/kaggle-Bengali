import numpy as np

import torch
import pytorch_lightning as pl

from kaggle_utils import multi_task_macro_recall
from pytorch_lightning_utils.multi_task_models import MultiTaskBaseModel


class BengaliModel(MultiTaskBaseModel):

    def __init__(self, model, train_dataloader, val_dataloader,
                 criterions, optimizer, scheduler=None):
        super(BengaliModel, self).__init__(model, criterions)
        self._train_dataloader = train_dataloader # add _ in the head in order not to conflict
        self._val_dataloader = val_dataloader # add _ in the head in order not to conflict
        self.optimizer = optimizer
        self.scheduler = scheduler

    def validation_step(self, batch, batch_idx):
        # print(f'batch index: {batch_idx}')
        x_train, y_trains = batch

        # print('valid:', batch_idx)
        # print('data:', x_train.device)
        # print('model:', next(self.model.parameters()).device)

        y_preds = self.forward(x_train)

        total_loss, losses = self.criterions(y_preds, y_trains)

        total_loss = total_loss.item()
        losses = [loss.item() for loss in losses]

        # print(f'total_loss: {total_loss}, loss_1: {losses[0]}, loss_2: {losses[1]}, loss_3 {losses[2]}')

        y_preds = [torch.argmax(y_pred, axis=1).cpu().numpy() for y_pred in y_preds]
        y_trains = [y_train.cpu().numpy() for y_train in y_trains]

        # print('y_preds:')
        # print(y_preds)
        # print(f'y_trains:')
        # print(y_trains)

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
        # print('end')
        total_data = sum(output['batch_size'] for output in outputs)
        
        total_val_loss = sum(output['total_val_loss'] * output['batch_size'] for output in outputs) / total_data
        val_loss_1 = sum(output['val_loss_1'] * output['batch_size'] for output in outputs) / total_data
        val_loss_2 = sum(output['val_loss_2'] * output['batch_size'] for output in outputs) / total_data
        val_loss_3 = sum(output['val_loss_3'] * output['batch_size'] for output in outputs) / total_data

        # print(f'total_val_loss: {total_val_loss}, val_loss_1: {val_loss_1}, val_loss_2: {val_loss_2}, val_loss_3: {val_loss_3}')

        y_preds_1 = np.hstack([output['y_preds'][0] for output in outputs])
        y_preds_2 = np.hstack([output['y_preds'][1] for output in outputs])
        y_preds_3 = np.hstack([output['y_preds'][2] for output in outputs])

        # print('y_preds_1')
        # print(y_preds_1)
        
        y_trains_1 = np.hstack([output['y_trains'][0] for output in outputs])
        y_trains_2 = np.hstack([output['y_trains'][1] for output in outputs])
        y_trains_3 = np.hstack([output['y_trains'][2] for output in outputs])

        # print('y_trains_1')
        # print(y_trains_1)

        total_recall, recalls = multi_task_macro_recall(true_graphemes=y_trains_1, pred_graphemes=y_preds_1,
                                                        true_vowels=y_trains_2, pred_vowels=y_preds_2,
                                                        true_consonants=y_trains_3, pred_consonants=y_preds_3,
                                                        n_grapheme=10, n_vowel=10, n_consonant=10)

        # print(f'total_recall: {total_recall}, recall_1: {recalls[0]}, recall_2: {recalls[1]}, recall_3: {recalls[2]}')

        tensorboard_logs = {
            'total_val_loss': total_val_loss,
            **{
                f'val_loss_{i+1}': loss for i, loss in enumerate([val_loss_1, val_loss_2, val_loss_3])
            },
            'total_recall': total_recall,
            'graphemes_recall': recalls[0],
            'vowels_recall': recalls[1],
            'consonants_recall': recalls[2]
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