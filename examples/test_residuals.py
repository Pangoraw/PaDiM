import os
from typing import Union, List, Literal
import sys
from pathlib import Path
import pickle

import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
import torch
from torch import Tensor
import cv2

sys.path.append('./')

from padim import PaDiM
from padim.datasets import (
    LimitedDataset,
    SemmacapeTestDataset,
)
from padim.utils import propose_regions, floating_IoU


class MVTecDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[Path, str],
        query_list: List[str],
        mode: Literal["train", "test"],
        transforms: Compose,
        debug: bool,
    ) -> None:

        self.data_dir = Path(data_dir) / mode / "good"
        self.transforms = transforms
        self.debug = debug

        self.stem_list = os.listdir(self.data_dir)

    def __getitem__(self, index: int) -> Tensor:

        stem = self.stem_list[index]

        img_path = str(self.data_dir / f"{stem}")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transforms(img)

        return img

    def _save_transformed_images(self, index: int, img: Tensor, mask: Tensor) -> None:

        img = img.permute(1, 2, 0).detach().numpy()
        mask = mask.detach().numpy()
        plt.figure(figsize=(9, 3))

        plt.subplot(131)
        plt.title("Input Image")
        plt.imshow(img)
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.subplot(132)
        plt.title("Ground Truth")
        plt.imshow(mask)
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.subplot(133)
        plt.title("Supervision")
        plt.imshow(img)
        plt.imshow(mask, alpha=0.5)
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.tight_layout()
        plt.savefig(f"{self.stem_list[index]}.png")

    def __len__(self) -> int:

        return len(self.stem_list)

class MVTecTestDataset(Dataset):
    def __init__(self, good_data_dir, defective_data_dir, transforms):
        self.defective_data_dir = defective_data_dir
        self.good_data_dir = good_data_dir
        self.transforms = transforms

        self.defective_samples = os.listdir(self.defective_data_dir)
        self.good_samples = os.listdir(self.good_data_dir)

    def _read_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img)
        return img

    def __getitem__(self, index):
        if index < len(self.good_samples):
            img_path = self.good_data_dir + self.good_samples[index]
            return 0, self._read_image(img_path)

        img_path = self.defective_data_dir + self.defective_samples[index - len(self.good_samples)]
        return 1, self._read_image(img_path)

    def __len__(self):
        return len(self.good_samples) + len(self.defective_samples)

img_transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Resize((128, 128)),
  transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        inplace=True,
    )
])
dataloader = DataLoader(
    batch_size=30,
    dataset=MVTecDataset(
        query_list="toothbrush",
        data_dir="./data/toothbrush",
        mode="train",
        transforms=img_transforms,
        debug=False
    )
)


model = PaDiM()
for imgs in tqdm(dataloader):
    model.train_one_batch(imgs)

test_dataloader = DataLoader(dataset=MVTecTestDataset(
  good_data_dir="./data/toothbrush/test/good/",
  defective_data_dir="./data/toothbrush/test/defective/",
  transforms=img_transforms,
))
distances = []
y_trues = []
for labels, new_imgs in test_dataloader:
    distances.extend(model.predict(new_imgs))
    y_trues.extend(labels)
amaps = torch.tensor(np.array(distances), dtype=torch.float32)

params = model.get_residuals()
with open('params.pckl', 'wb') as f:
    pickle.dump(params, f)

with open('params.pckl', 'rb') as f:
    params = pickle.load(f)

new_model = PaDiM.from_residuals(*params, device="cpu")

distances = []
for _, new_imgs in test_dataloader:
    distances.extend(model.predict(new_imgs))

new_amaps = torch.tensor(np.array(distances), dtype=torch.float32)
print("diff: ", (amaps - new_amaps).mean())
