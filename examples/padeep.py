import argparse
import logging
import pickle
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

sys.path.append("../padim")

from padim.datasets import LimitedDataset, OutlierExposureDataset
from padim import PaDiMSVDD

logging.basicConfig(filename="logs/padeep.log", level=logging.INFO)
handler = logging.StreamHandler(sys.stdout)

root = logging.getLogger()
root.addHandler(handler)

torch.autograd.set_detect_anomaly(True)

def train(args):
    PARAMS_PATH = args.params_path
    USE_SELF_SUPERVISION = args.use_self_supervision

    device = "cuda" if torch.cuda.is_available() else "cpu"

    padeep = PaDiMSVDD(backbone="wide_resnet50",
                       device=device,
                       n_svdds=args.n_svdds)

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
        ImageFolder(
            root=args.train_folder,
            target_transform=lambda _: 1,  # images are always normal
            transform=img_transforms),
        limit=args.train_limit,
    )
    if args.oe_folder is not None:
        train_dataset = OutlierExposureDataset(
            normal_dataset=normal_dataset,
            outlier_dataset=ImageFolder(root=args.oe_folder,
                                        transform=img_transforms),
            frequency=args.oe_frequency,
        )
    else:
        train_dataset = normal_dataset

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        num_workers=16,
        shuffle=False,
    )

    train_normal_dataloader = DataLoader(
        dataset=normal_dataset,
        batch_size=16,
        num_workers=16,
        shuffle=True,
    )

    if args.pretrain:
        root.info("Starting pretraining")
        padeep.pretrain(train_normal_dataloader, n_epochs=args.ae_n_epochs)
        root.info("Pretraining done")

    root.info("Starting training")
    padeep.train(
        train_dataloader,
        n_epochs=args.n_epochs,
        outlier_exposure=True,
        self_supervision=USE_SELF_SUPERVISION,
    )

    print(">> Saving params")
    params = padeep.get_residuals()
    with open(PARAMS_PATH, 'wb') as f:
        pickle.dump(params, f)
    print(f">> Params saved at {PARAMS_PATH}")

    return padeep
