import argparse
import sys
import pickle

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append('./')

from padim import PaDiM
from padim.datasets import (
    LimitedDataset,
    SemmacapeTestDataset,
)
from padim.utils import propose_regions, floating_IoU


def get_args():
    parser = argparse.ArgumentParser(prog="PaDiM tester")
    parser.add_argument("--test_limit", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--params_path", type=str)
    return parser.parse_args()


cfg = get_args()
LIMIT = cfg.test_limit
THRESHOLD = cfg.threshold
IOU_THRESHOLD = cfg.iou_threshold
PARAMS_PATH = cfg.params_path
LATTICE = 104

with open(PARAMS_PATH, 'rb') as f:
    params = pickle.load(f)

padim = PaDiM.from_residuals(*params, device="cuda:0")

img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((416, 416)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
test_dataset = SemmacapeTestDataset(
    data_dir="/share/projects/semmacape/Data_Semmacape_2/416_non_empty/",
    transforms=img_transforms,
)
test_dataloader = DataLoader(
    batch_size=1,
    dataset=LimitedDataset(dataset=test_dataset, limit=LIMIT)
)

classes = {}

n_proposals = 0
n_included = 0
positive_proposals = 0

means, covs, _ = padim.get_params()
means, covs = means.cpu().numpy(), covs.cpu().numpy()
inv_cvars = padim._get_inv_cvars(covs)
for loc, img, mask in tqdm(test_dataloader):
    # 1. Prediction
    res = padim.predict(img, params=(means, inv_cvars))
    res = (res - res.min()) / (res.max() - res.min())
    res = res.reshape((LATTICE, LATTICE))

    def normalize_box(box):
        x1, y1, x2, y2 = box
        return (
            x1 / LATTICE,
            y1 / LATTICE,
            abs(x1 - x2) / LATTICE,
            abs(y1 - y2) / LATTICE
        )
    preds = propose_regions(res, threshold=THRESHOLD)
    preds = [normalize_box(box) for box in preds]  # map from 0,LATTICE to 0,1
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
            x2, y2, w2, h2 = pred
            # check inclusion
            if (x2 >= x1
                    and x2 + w2 <= x1 + w
                    and y2 >= y1
                    and y2 + h2 <= y1 + h):
                n_included += 1

print(f"positive proposals: {positive_proposals}")
print(f"total proposals: {n_proposals}")
print(f"included: {n_included}")
print(f"PPR: {positive_proposals / n_proposals}")
for cls, (detected, n_gt) in classes.items():
    print(f">> {cls}: recall: {detected / n_gt}")
