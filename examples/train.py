import os
from pathlib import Path
from typing_extensions import Literal
from typing import Union, List, Tuple

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize

from padim import PaDiM


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
        print(img_path)
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


dataloader = DataLoader(dataset=MVTecDataset(
  query_list="toothbrush",
  data_dir="./data/toothbrush",
  mode="train",
  transforms=Compose([
    ToTensor(),
  ]),
  debug=False
))
for imgs in tqdm(dataloader):
    model.train_one_batch(imgs)

test_dataloader = DataLoader(dataset=MVTecDataset(
  query_list="toothbrush",
  data_dir="./data/toothbrush",
  mode="test",
  transforms=Compose([
    ToTensor(),
  ]),
  debug=False
))
distances = model.test(test_dataloader)
amaps = torch.tensor(np.array(distances), dtype=torch.float32)

img_w = new_imgs.shape(2)
img_h = new_imgs.shape(3)
amaps = amaps.permute(1, 0).view(b, h, w).unsqueeze(dim=1)
amaps = F.interpolate(amaps, size=(img_h, img_w), mode="bilinear", align_corners=False)
amaps = mean_smoothing(amaps)
amaps = (amaps - amaps.min()) / (amaps.max() - amaps.min())
amaps = amaps.squeeze().numpy()

roc_score = compute_roc_score(amaps, np.array(masks))
pro_score = compute_pro_score(amaps, np.array(masks))

print("roc score: ", roc_score)
print("pro score: ", pro_score)
