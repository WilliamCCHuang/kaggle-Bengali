import os
import cv2
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager


@contextmanager
def timer(name):
    """
    utility timer function to check how long a piece of code might take to run.
    :param name: name of the code fragment to be timed
    :yield: time taken for the code to run
    """
    t0 = time.time()
    print('[%s] in progress' % name)
    yield
    print('[%s] done in %.0f s' %(name, time.time() - t0))


def create_dirs(file_path):
    dir_path = os.path.dirname(file_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def parquet_to_feather(filepath):
    filename, extension = os.path.splitext(filepath)
    
    assert extension == '.parquet', f'The file {filepath} is not a parquet file'

    feather_filepath = filename + '.feather'

    df = pd.read_parquet(filepath)
    df.to_feather(feather_filepath)


def load_labels():
    df_labels = pd.read_csv('~/AI/Kaggle/Bengalai/Data/bengaliai-cv19/train.csv')

    return df_labels


def load_images(mode):
    # TODO: concatenate all feather files
    
    assert mode in ['train', 'test']

    df_list = []

    if mode == 'train':
        for i in range(4):
            df = pd.read_feather(f'~/AI/Kaggle/Bengalai/Data/bengaliai-cv19/train_image_data_{i}.feather')
            df_list.append(df)
    else:
        for i in range(4):
            df = pd.read_parquet(f'~/AI/Kaggle/Bengalai/Data/bengaliai-cv19/test_image_data_{i}.parquet')
            df_list.append(df)
    
    df = pd.concat(df_list)

    return df


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def crop_resize(img, size=128, pad=16):
    # crop a box around pixels larger than the threshold

    width = 236
    height = 137

    # some images contain lines at the sides
    ymin, ymax, xmin, xmax = bbox(img[5:-5, 5:-5] > 80)

    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if xmin > 13 else 0
    ymin = ymin - 10 if ymin > 10 else 0
    xmax = xmax + 13 if xmax < width - 13 else width
    ymax = ymax + 10 if ymax < height + 10 else height
    
    img = img[ymin:ymax, xmin:xmax]

    # remove low intensity pixels as noise
    img[img < 28] = 0
    
    lx, ly = xmax - xmin, ymax - ymin
    l = max(lx, ly) + pad

    # make sure that the aspect ration is kept in rescaling
    img = np.pad(img, [((l - ly) // 2,), ((l - lx) // 2,)], mode='constant')

    return resize(img, size)

    # assert image.ndim == 2
    # is_black = image > threshold

    # is_black_vertical = np.sum(is_black, axis=0) > 0
    # is_black_horizontal = np.sum(is_black, axis=1) > 0
    # left = np.argmax(is_black_horizontal)
    # right = np.argmax(is_black_horizontal[::-1])
    # top = np.argmax(is_black_vertical)
    # bottom = np.argmax(is_black_vertical[::-1])
    # height, width = image.shape
    # cropped_image = image[left:height - right, top:width - bottom]
    # return cropped_image


def resize(img, size=128):
    if isinstance(size, int):
        size = (size, size)

    return cv2.resize(img, size)


def submit():
    pass # TODO:


if __name__ == "__main__":
    # transform parquet to feather
    # train_files = [f'~/AI/Kaggle/Bengalai/Data/bengaliai-cv19/train_image_data_{i}.parquet' for i in range(4)]
    # for file in train_files:
    #     parquet_to_feather(file)

    # exit()

    # compare the speed between opening parquet files and opening feather files
    # import time
    # start = time.time()
    # for i in range(4):
    #     pd.read_parquet(f'~/AI/Kaggle/Bengalai/Data/bengaliai-cv19/train_image_data_{i}.parquet')
    # print('parquet:', time.time() - start)

    # start = time.time()
    # for i in range(4):
    #     pd.read_feather(f'~/AI/Kaggle/Bengalai/Data/bengaliai-cv19/train_image_data_{i}.feather')
    # print('feather:', time.time() - start)
    # exit()

    import matplotlib.pyplot as plt

    df_images = load_images(mode='test')

    for i in range(5):
        img = df_images.iloc[i][1:].values.reshape(137, 236).astype(np.uint8)
        img = 255 - img
        img = crop_resize(img, size=128)
        plt.imshow(img)
        plt.show()