import argparse

from bengali_models import BengaliModel
from bengali_metrics import multi_task_macro_recall
from bengali_datasets import GraphemeTrainDataset, GraphemeTestDataset
from bengali_utils import load_images

from pytorch_utils.models import BaseCNNModel
from pytorch_utils.losses import MultiTaskCrossEntropyLoss
from pytorch_utils.optimizers import RAdam

import pytorch_lightning as pl


TRAIN_DIR = 'data/origin' # TODO:
TEST_DIR = 'data/origin' #TODO:


def build_args():
    parser = argparse.ArgumentParser()
    

def main():
    # data
    df_train = load_images(mode='test') # TODO:
    train_dataloader = None # TODO:
    val_dataloader = None # TODO:

    # model
    criterions = MultiTaskCrossEntropyLoss(n_task=3)
    base_cnn_model = BaseCNNModel(model_name='se_resnext50_32x4d', hidden_dim=128, dropout=0.5)
    optimizer = RAdam(base_cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    model = BengaliModel(model=base_cnn_model,
                         train_dataloader=train_dataloader,
                         val_dataloader=val_dataloader,
                         criterions=criterions,
                         optimizer=optimizer)
    trainer = pl.Trainer(max_epochs=100, early_stop_callback=False, gpus=1, checkpoint_callback=None) # TODO: checkpoint
    trainer.fit(model)


if __name__ == "__main__":
    main()