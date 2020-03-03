import numpy as np

import pytorch_lightning as pl


class MultiTaskLightningModel(pl.LightningModule):
    """
    Need to be inherited and implement the following methods
    * validation_step (optional)
    * validation_end (optional)
    * test_step (optional)
    * test_end (optional)
    * configure_optimizers (required)
    * train_dataloader (required)
    * val_dataloader (optional)
    * test_dataloader (optional)
    
    Arguments:
        model {nn.Module} -- base model
        hparams {} -- hyper-parameters
        criterions {list} -- list of criterions for each task
    """
    def __init__(self, base_model, criterions):
        super(MultiTaskLightningModel, self).__init__()
        self.base_model = base_model
        self.criterions = criterions

    def forward(self, inputs):
        outputs = self.base_model(inputs) # outputs: list

        return outputs

    def training_step(self, batch, batch_idx):
        x_train, y_trains = batch  # x_train: torch.tensor, y_trains: list of torch.tensors

        y_preds = self.forward(x_train) # y_preds: list of torch.tensors

        total_loss, losses = self.criterions(y_preds, y_trains) # train loss should be a torch.tensor

        tensorboard_logs = {
            **{
                f'loss_{i+1}': loss for i, loss in enumerate(losses)
            }
        } # data for tensorboard can be a python number, np.array, or torch.tensor

        lr_dict = self.get_lr()
        tensorboard_logs.update(lr_dict)

        return {'loss': total_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def get_lr(self):
        lr_dict = {}

        if hasattr(self, 'optimizer'):
            lrs = [group['lr'] for group in self.optimizer.param_groups]

            if len(lrs) == 1:
                lr_dict['lr'] = lrs[0]
            else:
                for i, lr in enumerate(lrs):
                    lr_dict[f'lr_{i}'] = lr

        return lr_dict
