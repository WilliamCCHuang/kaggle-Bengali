import math
import numpy as np

from torch.utils.data import Dataset

from utils import load_labels, load_images, crop_resize


class BengaliTrainDataset(Dataset):

    def __init__(self, images, labels, size=128, transform=None):
        super(BengaliTrainDataset, self).__init__()

        self.size = size
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        img = crop_resize(img, size=self.size)
        img = img.astype(np.float32).reshape(1, self.size, self.size)
        img /= 255.

        y1, y2, y3 = self.labels[idx]
        
        if self.transform:
            img = self.transform(image=img)['image']

        return img, y1, y2, y3


class BengaliTestDataset(Dataset):
    
    def __init__(self, images, size=128, transform=None, TTA=False):
        super(BengaliTestDataset, self).__init__()

        self.TTA = TTA
        self.size = size
        self.images = images
        self.transform = transform

        if TTA:
            if transform is None:
                raise ValueError('`transform` cannot be `None` when `TTA` is `True`')
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        img = crop_resize(img, size=self.size)
        img = img.astype(np.float32).reshape(1, self.size, self.size)
        img /= 255.

        if not self.TTA:
            return img
        if self.TTA and self.transform:
            images = [img]
            for _ in range(self.TTA - 1):
                aug_img = self.transform(image=img)['image']
                images.append(aug_img)
        
            return images


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    from torch.utils.data import DataLoader

    labels = load_labels()
    images = load_images(mode='train', indices=[0])
    dataset = BengaliTrainDataset(images, labels, size=128)
    dataloader = DataLoader(dataset, batch_size=5)
    
    for images, labels1, labels2, labels3 in dataloader:
        print('images:', images.size())
        print(labels1)
        print(labels2)
        print(labels3)
        break