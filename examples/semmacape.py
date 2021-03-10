import argparse
import sys

from tqdm import tqdm
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

sys.path.append('./')

from padim import PaDiM
from padim.datasets import (
    LimitedDataset,
    SemmacapeTestDataset,
)
from padim.utils import propose_regions, floating_IoU


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

        return img

    def __len__(self):
        length, _ = self.data_frame.shape
        return length


def get_args():
    parser = argparse.ArgumentParser(prog="PaDiM trainer")
    parser.add_argument("--train_limit", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    return parser.parse_args()


cfg = get_args()
LIMIT = cfg.train_limit
THRESHOLD = cfg.threshold
IOU_THRESHOLD = cfg.iou_threshold

padim = PaDiM(device="cuda:0", backbone="wide_resnet50")
img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
])
dataloader = DataLoader(
    batch_size=32,
    num_workers=2,
    dataset=LimitedDataset(limit=LIMIT, dataset=TrainingDataset(
        data_dir="/share/home/berg/scripts/416_empty/",
        img_transforms=img_transforms,
    )),
)

for batch in tqdm(dataloader):
    padim.train_one_batch(batch)

test_dataset = SemmacapeTestDataset(
    data_dir="/share/projects/semmacape/Data_Semmacape_2/416_non_empty/",
    transforms=img_transforms,
)
test_dataloader = DataLoader(
    batch_size=1,
    dataset=LimitedDataset(dataset=test_dataset, limit=200)
)

classes = {}

n_proposals = 0
positive_proposals = 0
for loc, img, mask in tqdm(test_dataloader):
    # 1. Prediction
    res = padim.predict(img)
    res = (res - res.min()) / (res.max() - res.min())
    res = res.reshape((32, 32))

    def normalize_box(box):
        x1, y1, x2, y2 = box
        return x1 / 32, y1 / 32, abs(x1 - x2) / 32, abs(y1 - y2) / 32
    preds = propose_regions(res, threshold=THRESHOLD)
    preds = [normalize_box(box) for box in preds]  # map from 0,32 to 0,1
    n_proposals += len(preds)

    # 2. Collect GT boxes
    # PATH = '/share/projects/semmacape/Data_Semmacape_2/416_non_empty/'
    with open(
        loc[0].replace('.jpg', '_with_name_label.txt'),
        'r'
    ) as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        cls, cx, cy, bw, bh, _ = line.split(' ')
        if cls not in classes:
            # classes[cls] = (detected, total number of GT)
            classes[cls] = (0, 1)
        else:
            classes[cls] = (classes[cls][0], classes[cls][1] + 1)
        x1, y1 = float(cx) - float(bw) / 2, float(cy) - float(bh) / 2
        w, h = float(bw), float(bh)
        box = (x1, y1, w, h)

        for pred in preds:
            iou = floating_IoU(box, pred)
            if iou >= IOU_THRESHOLD:
                positive_proposals += 1
                classes[cls] = (classes[cls][0] + 1, classes[cls][1])
                break  # Dont count GT box more than once

print(f"positive proposals: {positive_proposals}")
print(f"total proposals: {n_proposals}")
print(f"PPR: {positive_proposals / n_proposals}")
for cls, (detected, n_gt) in classes.items():
    print(f">> {cls}: recall: {detected / n_gt}")
