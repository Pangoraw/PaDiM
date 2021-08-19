import os
from os import path

import torch
from torch.utils.data import Dataset
from PIL import Image


class MVTecADTestDataset(Dataset):
    def __init__(self, root, transform, mask_transform):
        self.test_dir = path.join(root, "test")
        self.ground_truth_dir = path.join(root, "ground_truth")
        self.classes = os.listdir(self.test_dir)
        self.transform = transform
        self.mask_transform = mask_transform

        self.current_class = 0
        self.current_class_idx = 0
        self.classes_files = {
            i: os.listdir(path.join(self.test_dir, cls)) 
            for i, cls in enumerate(self.classes)
        }
    
    def __getitem__(self, index):
        if self.current_class_idx == len(self.classes_files[self.current_class]):
            self.current_class_idx = 0
            self.current_class += 1

        item_file = path.join(self.classes[self.current_class], self.classes_files[self.current_class][self.current_class_idx])
        img_file = path.join(self.test_dir, item_file)

        mask_file = item_file.replace(".png", "_mask.png")
        mask_file = path.join(self.ground_truth_dir, mask_file)
        img = Image.open(img_file)
        img = img.convert("RGB")
        img = self.transform(img)

        is_good_img = self.classes[self.current_class] == "good"
        if not is_good_img:
            mask = Image.open(mask_file)
            mask = self.mask_transform(mask)
            mask[mask != 0] = 1.
        else:
            mask = torch.zeros((1,) + img.shape[1:])

        self.current_class_idx += 1
        return img, mask, int(not is_good_img)

    def __len__(self):
        return sum(len(files) for files in self.classes_files.values())
