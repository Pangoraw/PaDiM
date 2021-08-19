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
from padim.datasets import (
    LimitedDataset,
    IfremerTrainingDataset,
    SemmacapeDataset,
    KelomniaTrainingDataset,
)


class IfremerDataset(Dataset):
    def __init__(self, data_dir, img_transforms):
        self.data_dir = data_dir
        self.transforms = img_transforms
        self.files = os.listdir(self.data_dir)

    def __getitem__(self, index):
        location = self.data_dir + self.files[index]
        img = Image.open(location)
        img = self.transforms(img)
        return img, 1

    def __len__(self):
        return len(self.files)


class TrainingDataset(Dataset):
    def __init__(self, data_dir, img_transforms, ranking_file="./empty_ranking.csv"):
        self.data_dir = data_dir
        self.data_frame = pd.read_csv(ranking_file, index_col=0)
        self.transforms = img_transforms

    def __getitem__(self, index):
        if index % 2 == 0:
            direction = -1
        else:
            direction = 1
        index = direction * index // 2

        file_name = self.data_frame.iloc[index][0]
        img_path = file_name if file_name.startswith("/") else self.data_dir + file_name
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
    padim = Model(
            device=device,
            backbone=cfg.backbone,
            size=size, load_path=cfg.load_path,
            mode="random" if not cfg.semi_ortho else "semi_orthogonal")
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    if "seals" in cfg.train_folder:
        # training_dataset = IfremerTrainingDataset(
        #     transform=img_transforms,
        #     homo_dir=cfg.train_folder + "416_homogeneous/",
        #     hetero_dir=cfg.train_folder + "416_non_homogeneous/",
        #     homo_frequency=cfg.homo_frequency,
        # )
        training_dataset = TrainingDataset(
            data_dir=cfg.train_folder,
            img_transforms=img_transforms,
            ranking_file="./seal_empty_ranking.csv",
        )
        print(f"Dataset with {len(training_dataset)} samples")
    elif "turtles" in cfg.train_folder.lower():
        # training_dataset = SemmacapeDataset(
        #     transforms=img_transforms,
        #     data_dir=cfg.train_folder,
        # )
        training_dataset = TrainingDataset(
            data_dir=cfg.train_folder,
            img_transforms=img_transforms,
            ranking_file="./turtles_empty_ranking.csv",
        )
    elif "ifremer" in cfg.train_folder.lower():
        training_dataset = TrainingDataset(
            data_dir=cfg.train_folder,
            img_transforms=img_transforms,
            ranking_file="./ifremer_empty_ranking.csv",
        )
        print(f"Dataset with {len(training_dataset)} samples")
    elif "semmacape" in cfg.train_folder:
        training_dataset = TrainingDataset(
            data_dir=cfg.train_folder,
            img_transforms=img_transforms,
        )
    else:
        training_dataset = ImageFolder(root=cfg.train_folder, transform=img_transforms)

    n_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", 12))
    dataloader = DataLoader(
        batch_size=32,
        num_workers=n_cpus,
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
