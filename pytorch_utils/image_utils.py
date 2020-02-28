from typing import Union

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms


def load_image(img_path: str, mode: str='RGB', size: Union[int, tuple, list]=None) -> Image.Image:
    img = Image.open(img_path)
    img = img.convert(mode=mode)
    
    if size is not None:
        img = img.resize(size)
    
    return img


def image_to_tensor(img: Image.Image) -> torch.tensor:
    tensor = transforms.ToTensor()(img)
    tensor = torch.unsqueeze(tensor, dim=0)
    
    return tensor

def bbox(h, w, length):
    y = np.random.randint(h)
    x = np.random.randint(w)
    
    y1 = int(np.clip(y - length // 2, 0, h))
    y2 = int(np.clip(y + length // 2, 0, h))
    x1 = int(np.clip(x - length // 2, 0, w))
    x2 = int(np.clip(x + length // 2, 0, w))

    return x1, y1, x2, y2

def cutout(img: torch.Tensor, n_holes: int, length: Union[int, float]) -> torch.Tensor:
    img.requires_grad_(False)
    
    h, w = list(img.size())[-2:]
    
    if 2 <= img.dim() <= 3:
        # 1 image
        for _ in range(n_holes):
            x1, y1, x2, y2 = bbox(h, w, length)
            img[..., y1:y2, x1:x2] = 0.0
    elif img.dim() == 4:
        # batch images
        for i in range(img.size(0)):
            for _ in range(n_holes):
                x1, y1, x2, y2 = bbox(h, w, length)
                img[i, :, y1:y2, x1:x2] = 0.0
    else:
        raise ValueError('Image has wrong shape')

    return img
    