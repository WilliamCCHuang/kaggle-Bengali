import math
import numpy as np

from torch.utils.data import Dataset


class GraphemeTrainDataset(Dataset):

    def __init__(self, df_images, df_labels=None, transform=None):
        super(GraphemeTrainDataset, self).__init__()

        self.df_image = df_images
        self.df_label = df_labels
        self.transform = transform
        self.size = (137, 236)
        
    def __len__(self):
        return len(self.df_image)
    
    def __getitem__(self, idx):
        image = self.df_image.iloc[idx][1:].values.reshape(*self.size).astype(np.uint8)

        label1 = self.df_label.grapheme_root.values[idx]
        label2 = self.df_label.vowel_diacritic.values[idx]
        label3 = self.df_label.consonant_diacritic.values[idx]
        
        if self.transform:
            augment = self.transform(image=image)
            image = augment['image']

        return image, label1, label2, label3


class GraphemeTestDataset(Dataset):
    
    def __init__(self, df_images, transform=None, TTA=False):
        super(GraphemeTestDataset, self).__init__()

        self.TTA = TTA
        self.df_images = df_images
        self.transform = transform
        self.size = (137, 236)
        
    def __len__(self):
        return len(self.df_images)
    
    def __getitem__(self, idx):
        image = self.df_images.iloc[idx][1:].values.reshape(*self.size).astype(np.uint8)

        if not self.TTA:
            return image
        if self.TTA and self.transform:
            images = [image]
            for _ in range(self.TTA - 1):
                augment = self.transform(image=image)
                image = augment['image']
                images.append(image)
        
            return images


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from bengali_utils import load_images

    df_images = load_images(mode='test')
    dataset = GraphemeTestDataset(df_images, TTA=False)
    img = dataset[0]
    plt.imshow(img)
    plt.show()