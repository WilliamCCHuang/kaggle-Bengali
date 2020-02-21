import os
import cv2
import numpy as np
import pandas as pd


def create_dirs(file_path):
    dir_path = os.path.dirname(file_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_images(mode):
    # TODO: concatenate all feather files
    
    assert mode in ['train', 'test']

    df_list = []

    if mode == 'train':
        for i in range(4):
            df = pd.read_feather(f'data/origin/train_image_data_{i}.feather')
            df_list.append(df)
    else:
        for i in range(4):
            df = pd.read_feather(f'data/origin/test_image_data_{i}.feather')
            df_list.append(df)
    
    df = pd.concat(df_list)

    return df


def crop_char_image(image, threshold=5./255.):
    assert image.ndim == 2
    is_black = image > threshold

    is_black_vertical = np.sum(is_black, axis=0) > 0
    is_black_horizontal = np.sum(is_black, axis=1) > 0
    left = np.argmax(is_black_horizontal)
    right = np.argmax(is_black_horizontal[::-1])
    top = np.argmax(is_black_vertical)
    bottom = np.argmax(is_black_vertical[::-1])
    height, width = image.shape
    cropped_image = image[left:height - right, top:width - bottom]
    return cropped_image


def resize(image, size=128):
    if isinstance(size, int):
        size = (size, size)

    return cv2.resize(image, size)


def submit():
    pass # TODO:


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df_images = load_images(mode='test')
    img = df_images.iloc[0][1:].values.reshape(137, 236).astype(np.uint8) / 255.0
    img = 1 - img
    img = crop_char_image(img, threshold=20.0/255.0)
    img = resize(img, size=128)
    plt.imshow(img)
    plt.show()