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

def mixup(img: torch.Tensor, lab, alpha = 1.0) -> torch.Tensor:
    """
    Input x = (batchsize, channel, H, W)
    Output order:
    mixed images, mixed labels
    """
    img_copy   = img.clone().detach()
    lab_copy   = lab.clone().detach()
    lam        = np.random.beta(alpha, alpha)
    batch_size = img.size(0)
    
    if img.dim() == 4:
        for i in range(batch_size):
            rand_index = int(np.random.randint(batch_size, size = 1))
            if  i == rand_index:
                mix_img_i   = lam * img[i, ...] + (1 - lam)* img[(rand_index+1)%batch_size, ...] 
                mix_lab_i   = lam * lab[i, ...] + (1 - lam)* lab[(rand_index+1)%batch_size, ...]
                img_copy[i, ...] =  mix_img_i
                lab_copy[i, ...] =  mix_lab_i
                
            else:
                mix_img_i   = lam * img[i, ...] + (1 - lam)* img[(rand_index+1), ...] 
                mix_lab_i   = lam * lab[i, ...] + (1 - lam)* lab[(rand_index+1), ...]
                img_copy[i, ...] =  mix_img_i
                lab_copy[i, ...] =  mix_lab_i
    else:
        raise RuntimeError('Image has wrong shape, must with (B, C, H, W)')

    return img_copy, lab_copy

def cutmix(img: torch.Tensor, lab: torch.Tensor, n_holes, length) -> torch.Tensor:
    img_copy   = img.clone().detach()
    lab_copy   = lab.clone().detach()
    batch_size = img.size(0)
    h, w       = list(img.size())[-2:]
    lab_ratio  = length**2.0/(h*w)

    if img.dim() == 4:
        for i in range(batch_size):
            rand_index = int(np.random.randint(batch_size, size = 1))

            for n in range(n_holes):
                x1, y1, x2, y2 = bbox(h, w, length)

                if  i == rand_index:
                    img_copy[i,:, y1:y2, x1:x2] = img[(rand_index+1)%batch_size, :, y1:y2, x1:x2]
                    lab_copy[i, :] = lab[i, :] + lab_ratio * lab[(rand_index+1)%batch_size, :]
                    
                else:
                    img_copy[i,:, y1:y2, x1:x2] = img[rand_index, :, y1:y2, x1:x2]
                    lab_copy[i, :] = lab[i, :] + lab_ratio * lab[rand_index, :]
    else:
        raise RuntimeError('Input has wrong a shape, must with (B, C, H, W)')   
    return  img_copy, lab_copy


if __name__ == "__main__":
    img = load_image('transformer.jpeg')
    print(type(img))
    