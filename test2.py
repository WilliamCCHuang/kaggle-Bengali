import numpy as np
import matplotlib.pyplot as plt
from albumentations.augmentations.transforms import Cutout

from bengali_datasets import GraphemeTestDataset
from bengali_utils import load_images


def main():
    df_images = load_images(mode='test')
    transform = Cutout(num_holes=1, max_h_size=50, max_w_size=50, fill_value=0, p=1.0)
    dataset = GraphemeTestDataset(df_images, transform, TTA=2)
    _, img = dataset[0]
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()