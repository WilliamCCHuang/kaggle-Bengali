from typing import Union

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms


def load_image(image_path: str, mode: str='RGB', size: Union[int, tuple, list]=None) -> np.array:
    img = Image.open(image_path)
    img = image.convert(mode=mode)
    
    if size is not None:
        img = img.resize(size, Image.ANTIALIAS)
    
    img = np.array(img)

    return img


def resize(img: Union[np.array, Image.Image], size: Union[int, list, tuple]=128) -> np.array:
    if isinstance(size, int):
        size = (size, size)

    if isinstance(img, np.array):
        return cv2.resize(img, size)
    elif isinstance(img, Image.Image):
        img = img.resize(size, Image.ANTIALIAS)
        img = np.array(img)

        reutnr img
    else:
        raise ValueError(f'Unrecognize image type {type(img)}')


def image_to_tensor(image: Union[np.array, Image.Image]) -> torch.tensor:
    """
    Change an image to a tensor with the addition first dimension indicating batch size
    
    Arguments:
        image {Union[np.array, Image.Image]} -- [description]
    
    Returns:
        torch.tensor -- [description]
    """

    tensor = transforms.ToTensor()(image)
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


def cutout(image: torch.Tensor, n_holes: int, length: Union[int, float]) -> torch.Tensor:
    image.requires_grad_(False)
    
    h, w = list(image.size())[-2:]
    
    if 2 <= image.dim() <= 3:
        # 1 image
        for _ in range(n_holes):
            x1, y1, x2, y2 = bbox(h, w, length)
            image[..., y1:y2, x1:x2] = 0.0
    elif image.dim() == 4:
        # batch images
        for i in range(image.size(0)):
            for _ in range(n_holes):
                x1, y1, x2, y2 = bbox(h, w, length)
                image[i, :, y1:y2, x1:x2] = 0.0
    else:
        raise ValueError('Image has wrong shape')

    return image


def mixup(image: torch.Tensor, label, alpha=1.0) -> torch.Tensor:
    """
    Input x = (batchsize, channel, H, W)
    Output order:
    mixed images, mixed labels
    """

    if image.dim() != 4:
        raise ValueError('Image has wrong shape, must with (B, C, H, W)')

    bz = image.size(0)
    lam = np.random.beta(alpha, alpha)
    image_copy = image.clone().detach()
    label_copy = label.clone().detach()
    
    for i in range(bz):
        rand_index = int(np.random.randint(bz, size=1))
        if  i == rand_index:
            mix_image_i = lam * image[i, ...] + (1 - lam)* image[(rand_index + 1) % bz, ...] 
            mix_label_i = lam * label[i, ...] + (1 - lam)* label[(rand_index + 1) % bz, ...]
            image_copy[i, ...] = mix_image_i
            label_copy[i, ...] = mix_label_i
        else:
            mix_image_i = lam * image[i, ...] + (1 - lam)* image[(rand_index + 1), ...] 
            mix_label_i = lam * label[i, ...] + (1 - lam)* label[(rand_index + 1), ...]
            image_copy[i, ...] = mix_image_i
            label_copy[i, ...] = mix_label_i

    return image_copy, label_copy


def cutmix(image: torch.Tensor, label: torch.Tensor, n_holes, length) -> torch.Tensor:
    if image.dim() != 4:
        raise ValueError('Input has wrong a shape, must with (B, C, H, W)')

    bz, _, h, w = list(img.size())
    image_copy = image.clone().detach()
    label_copy = label.clone().detach()
    label_ratio = length**2.0 / (h * w)

    for i in range(bz):
        rand_index = int(np.random.randint(bz, size=1))
        for _ in range(n_holes):
            x1, y1, x2, y2 = bbox(h, w, length)

            if  i == rand_index:
                image_copy[i, :, y1:y2, x1:x2] = image[(rand_index + 1) % bz, :, y1:y2, x1:x2]
                label_copy[i, :] = label[i, :] + label_ratio * label[(rand_index + 1) % bz, :]
            else:
                image_copy[i,:, y1:y2, x1:x2] = img[rand_index, :, y1:y2, x1:x2]
                label_copy[i, :] = label[i, :] + label_ratio * label[rand_index, :]
                    
    return  image_copy, label_copy


if __name__ == "__main__":
    img = load_image('transformer.jpeg')
    print(type(img))
    