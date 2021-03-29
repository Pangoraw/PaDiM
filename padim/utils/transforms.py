from typing import Tuple, Union
from pathlib import Path

from torchvision import transforms
from torchvision.transforms import Compose
from torch import Tensor
from PIL import Image


def process_image(path: Union[Path, str], img_transforms=None) -> Tensor:
    if img_transforms is None:
        img_transforms = default_transforms()

    img = Image.open(path)
    img = img_transforms(img)
    return img


def default_transforms(size: Tuple[int, int] = (128, 128)) -> Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            inplace=True,
        )
    ])
