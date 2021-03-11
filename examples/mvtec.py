import os
import sys
from pathlib import Path
from typing_extensions import Literal
from typing import Union, List, Tuple

import cv2
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize

sys.path.append("./")

from padim import PaDiM
from padim.utils import mean_smoothing, compute_roc_score, compute_pro_score


model = PaDiM(
    num_embeddings=120,
    device="cpu",
    backbone="resnet18",
)


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

dataloader = DataLoader(
  batch_size=30,
  dataset=MVTecDataset(
    query_list="toothbrush",
    data_dir="./data/toothbrush",
    mode="train",
    transforms=Compose([
      ToTensor(),
      Resize((128, 128)),
    ]),
    debug=False
))
for imgs in tqdm(dataloader):
    model.train_one_batch(imgs)

transforms = Compose([
  ToTensor(),
  Resize((128, 128)),
])
test_dataloader = DataLoader(dataset=MVTecTestDataset(
  good_data_dir="./data/toothbrush/test/good/",
  defective_data_dir="./data/toothbrush/test/defective/",
  transforms=transforms,
))
distances = []
y_trues = []
for labels, new_imgs in test_dataloader:
    distances.extend(model.predict(new_imgs))
    y_trues.extend(labels)
amaps = torch.tensor(np.array(distances), dtype=torch.float32)

"""
mask_files = [
  "./data/toothbrush/ground_truth/defective/" + img_file.replace(".png", "_mask.png")
  for img_file in test_dataloader.dataset.stem_list
]

def get_mask(mask_file):
    mask = Image.open(mask_file)
    return np.array(mask, dtype=int)

masks = torch.squeeze(torch.tensor(np.array([
  transforms(get_mask(mask_file)).numpy()
  for mask_file in mask_files
])), 1)
print(">> mask shape", masks.shape)
"""

img_w = 128
img_h = 128
amaps = amaps.permute(1, 0).view(len(test_dataloader), 32, 32).unsqueeze(dim=1)
amaps = F.interpolate(amaps, size=(img_h, img_w), mode="bilinear", align_corners=False)
amaps = mean_smoothing(amaps)
amaps = (amaps - amaps.min()) / (amaps.max() - amaps.min())
amaps = amaps.squeeze().numpy()

roc_score = compute_roc_score(amaps, y_trues, test_dataloader.dataset.good_samples + test_dataloader.dataset.defective_samples)
# pro_score = compute_pro_score(amaps, y_trues)

print("roc score: ", roc_score)
print("pro score: ", pro_score)
