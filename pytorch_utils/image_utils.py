import PIL
from typing import Union

import torch
import torchvision.transforms as transforms


def load_image(img_path: str, mode='RGB', size: Union[int, tuple, list]=None) -> PIL.Image.Image:
    img = PIL.Image.open(img_path)
    img = img.convert(mode=mode)
    
    if size is not None:
        img = img.resize(size)
    
    return img


def image_to_tensor(img: PIL.Image.Image) -> torch.tensor:
    tensor = transforms.ToTensor()(img)
    tensor = torch.unsqueeze(tensor, dim=0)
    
    return tensor

    
if __name__ == "__main__":
    img = load_image('transformer.jpeg')
    print(type(img))
    