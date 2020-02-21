import numpy as np
import pandas as pd

from bengali_utils import (
    create_dirs,
    load_images,
    crop_char_image,
    resize
)


def main(mode, size=128, file_path=None):
    df_images = load_images(mode)
    df_new = pd.DataFrame(columns=['image_id'] + [str(i) for i in range(size**2)])
    df_new.image_id = df_images.image_id
    
    for i in range(len(df_images)):
        img = df_images.iloc[i, 1:].values.reshape(137, 236).astype(np.uint8) / 255.0
        img = 1 - img
        img = crop_char_image(img, threshold=20.0/255.0)
        img = resize(img, size)
        df_new.iloc[i, 1:] = img.reshape(-1)

    df_new = df_new.reset_index().drop('index', axis=1)
    
    if file_path:
        assert file_path.endswith('feather')
        create_dirs(file_path)
        df_new.to_feather(file_path)


if __name__ == "__main__":
    main(mode='test', size=128,
         file_path='data/128x128/test_images.feather')