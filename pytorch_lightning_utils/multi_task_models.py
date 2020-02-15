import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class MultiTaskBaseModel(pl.LightningModule):
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
    def __init__(self, model, criterions):
        super(MultiTaskBaseModel, self).__init__()
        self.model = model
        self.criterions = criterions

    def forward(self, inputs):
        outputs = self.model(inputs) # outputs: list

        return outputs

    def training_step(self, batch, batch_idx):
        x_train, y_trains = batch  # x_train: torch.tensor, y_trains: list of torch.tensors

        # print('train:', batch_idx)
        # print('data:', x_train.device)
        # print('model:', next(self.model.parameters()).device)

        y_preds = self.forward(x_train) # y_preds: list of torch.tensors

        total_loss, losses = self.criterions(y_preds, y_trains) # train loss should be a torch.tensor

        tensorboard_logs = {
            'total_loss': total_loss,
            **{
                f'loss_{i+1}': loss for i, loss in enumerate(losses)
            }
        } # data for tensorboard can be a python number, np.array, or torch.tensor

        return {'loss': total_loss, 'log': tensorboard_logs}