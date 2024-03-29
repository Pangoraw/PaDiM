"""
Dataset files specific to the Semmacape project
"""
import os
from os import path

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class OutlierExposureDataset(Dataset):
    def __init__(self, normal_dataset, outlier_dataset, frequency=2):
        self.frequency = frequency
        self.normal_dataset = normal_dataset
        self.normal_dataset_idx = 0
        self.outlier_dataset = outlier_dataset
        self.outlier_dataset_idx = 0

    def __getitem__(self, index):
        cls = 1 - int((index % self.frequency) == 1)
        if cls == 1:
            dataset = self.normal_dataset
            idx = self.normal_dataset_idx
            self.normal_dataset_idx += 1
        else:
            dataset = self.outlier_dataset
            idx = self.outlier_dataset_idx
            self.outlier_dataset_idx += 1
        img, _ = dataset[idx]
        return img, cls

    def __len__(self):
        return self.frequency * len(self.normal_dataset) // (self.frequency - 1)


class LimitedDataset(Dataset):
    """
    Limits the underlying Dataset to only N samples
    """

    def __init__(self, dataset, limit=-1):
        """
        Params
        ======
            dataset: Dataset - the underlying dataset
            limit: int - the number of sample to limit to
        """
        super().__init__()
        self.dataset = dataset

        if limit == -1:  # limit of -1 is no limit
            limit = len(self.dataset)
        self.length = min(len(self.dataset), limit)

    def __getitem__(self, index):
        if index >= self.length:
            return None

        return self.dataset[index]

    def __len__(self):
        return self.length


class SemmacapeTestDataset(Dataset):
    """
    Loads anomalous images from a folder of images and box txt files
    """

    def __init__(self, transforms, data_dir):
        super().__init__()

        self.data_dir = data_dir
        self.transforms = transforms
        self.image_files = [
            f for f in os.listdir(data_dir) if f.endswith(".jpg")
        ]

    def __getitem__(self, index):
        img_location = path.join(self.data_dir, self.image_files[index])

        img = Image.open(img_location)
        w, h = img.size
        img = self.transforms(img)

        _, w, h = img.shape
        mask = np.zeros((w, h))
        is_image_normal = "normal" in img_location

        if not is_image_normal:
            with open(img_location.replace(".jpg", ".txt")) as f:
                boxes = [
                    [float(x) for x in l.split(" ")[1:-1]]
                    for l in f.readlines()
                ]
            for cx, cy, bw, bh in boxes:
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                mask[y1:y2, x1:x2] = 1.0

        return (img_location, img, mask, int(is_image_normal))

    def __len__(self):
        return len(self.image_files)


class SemmacapeDataset(Dataset):
    """
    Loads normal images from a folder of images
    """

    def __init__(self, transforms, data_dir):
        super().__init__()

        self.data_dir = data_dir
        self.transforms = transforms

        self.image_files = [
            f for f in os.listdir(data_dir) if f.endswith(".jpg")
        ]

    def __getitem__(self, index: int) -> Tensor:
        file_path = self.image_files[index]
        img = Image.open(path.join(self.data_dir, file_path))
        img = self.transforms(img)

        return img

    def __len__(self) -> int:
        return len(self.image_files)
