import os

import torch
from torch.utils.data import Dataset
from PIL import Image

DEFAULT_TRAIN_FILE_LIST_LOC="/share/home/berg/turtles_empty.txt"
DEFAULT_TEST_FILE_LIST_LOC="/share/home/berg/turtles_non_empty_new_annotations.txt"


class AbstractKelomniaDataset(Dataset):
    def __init__(self,
        transform,
        file_list_loc,
    ):
        with open(file_list_loc, "r") as f:
            self.files = [f.strip() for f in f.readlines()]

        self.transform = transform


    def __getitem__(self, index):
        img = Image.open(self.files[index])
        return self.transform(img), 1


    def __len__(self):
        return len(self.files)


class KelomniaTestingDataset(AbstractKelomniaDataset):
    def __init__(
        self,
        transform,
        file_list_loc=DEFAULT_TEST_FILE_LIST_LOC
    ):
        super(KelomniaTestingDataset, self).__init__(transform, file_list_loc)

    def __getitem__(self, index):
        location = self.files[index]

        img = Image.open(location)

        is_image_normal = "normal" in location

        w, h = img.size
        mask = torch.zeros(w, h)

        return location, self.transform(img), mask, int(is_image_normal)


class KelomniaTrainingDataset(AbstractKelomniaDataset):
    def __init__(
        self,
        transform,
        file_list_loc=DEFAULT_TRAIN_FILE_LIST_LOC
    ):
        super(KelomniaTrainingDataset, self).__init__(transform, file_list_loc)
