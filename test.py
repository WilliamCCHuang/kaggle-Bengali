import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from bengali_models import BengaliModel
from pytorch_utils.losses import MultiTaskCrossEntropyLoss

import pytorch_lightning as pl


class TestDataset(Dataset):

    def __init__(self):
        super(TestDataset, self).__init__()
        self.ref_dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())

    def __len__(self):
        return 10

    def __getitem__(self, index):
        x, y = self.ref_dataset[index % 10]

        return x, [y, y, y]


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        
        self.l1 = nn.Linear(28 * 28, 64)
        self.l2 = nn.Linear(64, 10)
        # self.l3 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        y = self.l1(x)
        y = F.relu(y)
        y = self.l2(y)

        return y, y, y


def main():
    train_dataloader = DataLoader(TestDataset(), batch_size=5, shuffle=False)
    val_dataloader = DataLoader(TestDataset(), batch_size=5, shuffle=False)

    base_model = BaseModel()
    model = BengaliModel(base_model, train_dataloader, val_dataloader,
                         MultiTaskCrossEntropyLoss(n_task=3), optim.Adam(base_model.parameters()))

    trainer = pl.Trainer(max_epochs=20, early_stop_callback=False) # for cpu
    # trainer = pl.Trainer(max_epochs=20, early_stop_callback=False, gpus=1) # for gpu
    trainer.fit(model)
    

if __name__ == "__main__":
    main()