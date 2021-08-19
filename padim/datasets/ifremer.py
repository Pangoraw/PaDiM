import os

from torch.utils.data import Dataset
from PIL import Image

DEFAULT_HOMO_DIR="/share/projects/semmacape/Data_Tristan_07_09_20/416_homogeneous/"
DEFAULT_HETERO_DIR="/share/projects/semmacape/Data_Tristan_07_09_20/416_non_homogeneous/"


class IfremerTrainingDataset(Dataset):
    def __init__(self,
        transform,
        homo_dir=DEFAULT_HOMO_DIR,
        hetero_dir=DEFAULT_HETERO_DIR,
        homo_frequency=2,
    ):
        self.homo_dir = homo_dir
        self.homo_files = os.listdir(homo_dir)
        self.homo_idx = 0
        self.hetero_dir = hetero_dir
        self.hetero_files = os.listdir(hetero_dir)
        self.hetero_idx = 0
        self.transform = transform
        self.homo_frequency = homo_frequency

    def __getitem__(self, index):
        homo_img = (index % self.homo_frequency) == 0
        if not homo_img:
            location = self.hetero_dir + self.hetero_files[self.hetero_idx]
            self.hetero_idx += 1
        else:
            location = self.homo_dir + self.homo_files[self.homo_idx]
            self.homo_idx += 1

        img = Image.open(location)
        img = self.transform(img)
        return img, 1

    def __len__(self):
        return self.homo_frequency * min(len(self.homo_files), len(self.hetero_files))


