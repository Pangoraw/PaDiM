import os
import pickle

import torch
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

from padim import PaDiM, PaDiMShared
from padim.datasets import LimitedDataset


class TrainingDataset(Dataset):
    def __init__(self, data_dir, img_transforms):
        self.data_dir = data_dir
        self.data_frame = pd.read_csv("./empty_ranking.csv", index_col=0)
        self.transforms = img_transforms

    def __getitem__(self, index):
        if index % 2 == 0:
            direction = -1
        else:
            direction = 1
        index = direction * index // 2

        img_path = self.data_dir + self.data_frame.iloc[index][0]
        img = Image.open(img_path)
        img = self.transforms(img)

        return img, 1

    def __len__(self):
        length, _ = self.data_frame.shape
        return length


def train(cfg):
    LIMIT = cfg.train_limit
    PARAMS_PATH = cfg.params_path
    SHARED = cfg.shared
    size = tuple(map(int, cfg.size.split("x")))

    if SHARED:
        Model = PaDiMShared
    else:
        Model = PaDiM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    padim = Model(device=device, backbone="wide_resnet50", size=size)
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    if "semmacape" in cfg.train_folder:
        training_dataset = TrainingDataset(
            data_dir=cfg.train_folder,
            img_transforms=img_transforms,
        )
    else:
        training_dataset = ImageFolder(root=cfg.train_folder, transform=img_transforms)

    #n_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", 12))
    dataloader = DataLoader(
        batch_size=32,
        num_workers=8,
        dataset=LimitedDataset(limit=LIMIT, dataset=training_dataset),
    )

    for batch in tqdm(dataloader):
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = batch[0]
        padim.train_one_batch(batch)

    print(">> Saving params")
    params = padim.get_residuals()
    with open(PARAMS_PATH, 'wb') as f:
        pickle.dump(params, f)
    print(f">> Params saved at {PARAMS_PATH}")

    return padim
