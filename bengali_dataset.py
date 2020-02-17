import math
import numpy as np

from torch.utils.data import Dataset


# TODO: test
class GraphemeDataset(Dataset):
    def __init__(self, df_image, df_label=None, mode='train', transform=None, TTA=True):
        assert mode in ['train', 'test']

        self.df_image = df_image
        self.df_label = df_label
        self.mode = mode
        self.transform = transform
        self.TTA = TTA

        size = math.sqrt(df_image.iloc[0][1:].values.size())
        assert int(size) == size
        self.size = size
        
    def __len__(self):
        return len(self.df_image)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            image = self.df_image.iloc[idx][1:].values.reshape(self.size, self.size).astype(np.uint8)

            label1 = self.df_label.grapheme_root.values[idx]
            label2 = self.df_label.vowel_diacritic.values[idx]
            label3 = self.df_label.consonant_diacritic.values[idx]
            
            if self.transform:
                augment = self.transform(image=image)
                image = augment['image']

            return image, label1, label2, label3
        else:
            image = self.df_image.iloc[idx][1:].values.reshape(self.size, self.size).astype(np.uint8)

            if self.TTA and self.transform:
                augment = self.transform(image=image)
                image = augment['image']
            
            return image