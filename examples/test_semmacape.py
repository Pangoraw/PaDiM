import argparse
import sys
import pickle

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append('./')

from padim import PaDiM, PaDiMShared, PaDiMSVDD
from padim.datasets import (
    LimitedDataset,
    SemmacapeTestDataset,
)
from padim.utils import propose_regions, floating_IoU


def get_args():
    parser = argparse.ArgumentParser(prog="PaDiM tester")
    parser.add_argument("--params_path", required=True, type=str)
    parser.add_argument("--test_limit", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--min_area", type=int, default=2)
    parser.add_argument("--use_nms", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--deep", action="store_true")
    return parser.parse_args()


cfg = get_args()
LIMIT = cfg.test_limit
THRESHOLD = cfg.threshold
IOU_THRESHOLD = cfg.iou_threshold
PARAMS_PATH = cfg.params_path
MIN_AREA = cfg.min_area
USE_NMS = cfg.use_nms
LATTICE = 104
SHARED = cfg.shared
DEEP = cfg.deep

if DEEP:
    Model = PaDiMSVDD
elif SHARED:
    Model = PaDiMShared
else:
    Model = PaDiM

with open(PARAMS_PATH, 'rb') as f:
    params = pickle.load(f)

padim = Model.from_residuals(*params, device="cuda:0")

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
n_gt = 0
sum_iou = 0
total_positive_proposals = 0
positive_proposals = 0

means, covs, _ = padim.get_params()
inv_cvars = padim._get_inv_cvars(covs)
for loc, img, mask in tqdm(test_dataloader):
    # 1. Prediction
    res = padim.predict(img, params=(means, inv_cvars))
    res = (res - res.min()) / (res.max() - res.min())
    res = res.reshape((LATTICE, LATTICE)).cpu()

    def normalize_box(box):
        x1, y1, x2, y2, s = box
        return (
            x1 / LATTICE,
            y1 / LATTICE,
            abs(x1 - x2) / LATTICE,
            abs(y1 - y2) / LATTICE,
            s
        )
    preds = propose_regions(
        res,
        threshold=THRESHOLD,
        min_area=MIN_AREA,
        use_nms=USE_NMS,
    )
    preds = [normalize_box(box) for box in preds]  # map from 0,LATTICE to 0,1
    n_proposals += len(preds)

    img_proposals_counted = False

    # 2. Collect GT boxes
    with open(
        loc[0].replace('.jpg', '_with_name_label.txt'),
        'r'
    ) as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        n_gt += 1
        cls, cx, cy, bw, bh, _ = line.split(' ')
        if cls not in classes:
            # classes[cls] =
            # (# detected, # proposals, sum(iou), total number of GT)
            classes[cls] = (0, len(preds), 0, 1)
            # Hypothesis: only one class per image
            img_proposals_counted = True
        else:
            n_proposals_offset = 0 if img_proposals_counted else len(preds)
            classes[cls] = (
                classes[cls][0],
                classes[cls][1] + n_proposals_offset,
                classes[cls][2],
                classes[cls][3] + 1
            )
        x1, y1 = float(cx) - float(bw) / 2, float(cy) - float(bh) / 2
        w, h = float(bw), float(bh)
        box = (x1, y1, w, h)

        box_detected = False
        for pred in preds:
            iou = floating_IoU(box, pred)
            sum_iou += iou
            classes[cls] = (
                classes[cls][0],
                classes[cls][1],
                classes[cls][2] + iou,
                classes[cls][3],
            )
            total_positive_proposals += 1
            if not box_detected and iou >= IOU_THRESHOLD:
                # Positive detection
                # Count detected box only once
                positive_proposals += 1
                classes[cls] = (
                    classes[cls][0] + 1,
                    classes[cls][1],
                    classes[cls][2],
                    classes[cls][3]
                )
                box_detected = True
            x2, y2, w2, h2, = pred[:4]
            # check inclusion
            if (x2 >= x1
                    and x2 + w2 <= x1 + w
                    and y2 >= y1
                    and y2 + h2 <= y1 + h):
                n_included += 1

print(f"positive proposals: {positive_proposals}")
print(f"total positive proposals: {total_positive_proposals}")
print(f"total proposals: {n_proposals}")
print(f"included: {n_included}")
if n_proposals == 0:
    print("PPR: 0")
    print("MEAN_IOU: 0")
else:
    print(f"PPR: {positive_proposals / n_proposals}")
    print(f"MEAN_IOU: {sum_iou / n_proposals}")
print(f"RECALL: {positive_proposals / n_gt}")
for cls, (detected, n_cls_proposals, cls_sum_iou, n_cls_gt) in classes.items():
    print(f">> {cls}: recall: {detected / n_cls_gt}")
    if n_cls_proposals == 0:
        print(f">> {cls}: precision: 0")
        print(f">> {cls}: mean_iou: 0")
    else:
        print(f">> {cls}: precision: {detected / n_cls_proposals}")
        print(f">> {cls}: mean_iou: {cls_sum_iou / n_cls_proposals}")
