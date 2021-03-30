import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder, CIFAR10
from torchvision import transforms, utils

sys.path.append("../padim")
sys.path.append("../deep_svdd/src")

from padim.datasets import LimitedDataset, OutlierExposureDataset
from padim import PaDiMSVDD

logging.basicConfig(filename="logs/padeep.log", level=logging.INFO)
handler = logging.StreamHandler(sys.stdout)

root = logging.getLogger()
root.addHandler(handler)

parser = argparse.ArgumentParser(prog="PaDeep test")
parser.add_argument("--train_folder", required=True)
parser.add_argument("--test_folder", required=True)
parser.add_argument("--oe_folder")
parser.add_argument("--oe_frequency", type=int)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--ae_n_epochs", type=int, default=1)
parser.add_argument("--train_limit", type=int, default=-1)
parser.add_argument("--pretrain", action="store_true")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

padeep = PaDiMSVDD(backbone="wide_resnet50", device=device)

img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((416, 416)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        inplace=True,
    ),
])

normal_dataset = LimitedDataset(
    ImageFolder(root=args.train_folder,
                target_transform=lambda _: 1,  # images are always normal
                transform=img_transforms),
    limit=args.train_limit,
)
if args.oe_folder is not None:
    train_dataset = OutlierExposureDataset(
        normal_dataset=normal_dataset,
        outlier_dataset=ImageFolder(root=arg.oe_folder,
                                    transform=img_transforms),
        frequency=args.oe_frequency,
    )
else:
    train_dataset = normal_dataset

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    num_workers=16,
    shuffle=True,
)

train_normal_dataloader = DataLoader(
    dataset=normal_dataset,
    batch_size=16,
    num_workers=16,
    shuffle=True,
)

test_dataloader = DataLoader(
    dataset=ImageFolder(root=args.test_folder, transform=img_transforms),
    batch_size=4,
    shuffle=True,
)
test_iter = iter(test_dataloader)
test_batch, _ = next(test_iter)

if args.pretrain:
    root.info("Starting pretraining")
    padeep.pretrain(train_dataloader, n_epochs=args.ae_n_epochs)
    root.info("Pretraining done")

root.info("Starting training")
padeep.train(
    train_dataloader,
    n_epochs=args.n_epochs,
    test_images=test_batch,
    outlier_exposure=True,
)

results = padeep.predict(test_batch)

utils.save_image(test_batch, "inputs.png")
utils.save_image(results, "outputs.png")
