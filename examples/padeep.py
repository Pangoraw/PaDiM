import argparse
import logging
import os
import pickle
import sys
from functools import partial

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

sys.path.append("../padim")

from padim.datasets import (
    LimitedDataset,
    IfremerTrainingDataset,
    OutlierExposureDataset,
    SemmacapeDataset,
    KelomniaTrainingDataset,
)
from padim import PaDiMSVDD
from padim.panf import PaDiMNVP
from semmacape import TrainingDataset, IfremerDataset
from test_semmacape import test as test_semmacape

logging.basicConfig(filename="logs/padeep.log", level=logging.INFO)
handler = logging.StreamHandler(sys.stdout)

root = logging.getLogger()
root.addHandler(handler)

def train(args, experiment=None):
    PARAMS_PATH = args.params_path
    USE_SELF_SUPERVISION = args.use_self_supervision
    SIZE = tuple(map(int, args.size.split("x")))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.nf:
        padeep = PaDiMNVP(n_svdds=args.n_svdds,
                          num_embeddings=args.num_embeddings,
                          backbone=args.backbone,
                          device=device,
                          size=SIZE,
                          mode="random" if not args.semi_ortho else "semi_orthogonal",
                          load_path=args.load_path,
                          nf="maf")
    else:
        padeep = PaDiMSVDD(num_embeddings=args.num_embeddings,
                           backbone=args.backbone,
                           device=device,
                           n_svdds=args.n_svdds,
                           size=SIZE,
                           mode="random" if not args.semi_ortho else "semi_orthogonal",
                           load_path=args.load_path)

    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(SIZE),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            inplace=True,
        ),
    ])

    if "seals" in args.train_folder:
        # training_dataset = IfremerTrainingDataset(
        #     transform=img_transforms,
        #     homo_dir=args.train_folder + "416_homogeneous/",
        #     hetero_dir=args.train_folder + "416_non_homogeneous/",
        #     homo_frequency=args.homo_frequency,
        # )
        training_dataset = TrainingDataset(
            data_dir=args.train_folder,
            img_transforms=img_transforms,
            ranking_file="./seal_empty_ranking.csv",
        )
        print(f"Dataset with {len(training_dataset)} samples")
    elif "turtles" in args.train_folder.lower():
        # training_dataset = SemmacapeDataset(
        #     transforms=img_transforms,
        #     data_dir=args.train_folder,
        # )
        # training_dataset = KelomniaTrainingDataset(
        #     transform=img_transforms,
        # )
        training_dataset = TrainingDataset(
            data_dir=args.train_folder,
            img_transforms=img_transforms,
            ranking_file="./turtles_empty_ranking.csv",
        )
    elif "ifremer" in args.train_folder.lower():
        # training_dataset = IfremerTrainingDataset(
        #     transform=img_transforms,
        # )
        training_dataset = TrainingDataset(
            data_dir=args.train_folder,
            img_transforms=img_transforms,
            ranking_file="./ifremer_empty_ranking.csv",
        )
        print(f"Dataset with {len(training_dataset)} samples")
    elif "semmacape" in args.train_folder:
        training_dataset = TrainingDataset(
            data_dir=args.train_folder,
            img_transforms=img_transforms,
        )
    else:
        training_dataset = ImageFolder(
            root=args.train_folder,
            target_transform=lambda _: 1,  # images are always normal
            transform=img_transforms,
        )

    normal_dataset = LimitedDataset(
        dataset=training_dataset,
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

    n_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", 12))

    # Since realnvp makes use of batch normalization
    batch_size = 1 if args.nf else 16

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=n_cpus,
        shuffle=False,
    )

    from sklearn.metrics import roc_auc_score

    if False and "ifremer" in args.train_folder:
        test_dataset = ImageFolder(root="/share/projects/semmacape/Unsupervised/image_dissimilarity/data/val/",
            transform=img_transforms, target_transform=lambda x: int(test_dataset.class_to_idx["wave_and_glitters"] != x))

        def test_cb(epoch):
            if epoch % 5 != 0: return

            y_trues = []
            preds = []
            for img, cls in test_dataset: 
                img = img.to(device)
                pred = padeep.predict(img.unsqueeze(0)).max().item()
                y_trues.append(cls)
                preds.append(pred)
            score = roc_auc_score(y_trues, preds)
            print(f"roc_auc_score: {score:.3f}")
    # elif "ifremer" in args.train_folder.lower() or "turtles" in args.train_folder.lower():
    #     def test_cb(epoch):
    #         if epoch != 0 and ((epoch+1) % 5 != 0):
    #             return
    #         for t in range(1, 7):
    #             threshold = t / 10
    #             args_copy = argparse.Namespace(**vars(args))
    #             args_copy.test_limit = 10
    #             results = test_semmacape(args_copy, padeep, threshold)
    #             experiment.checkpoint(
    #                 step=10*epoch+t,
    #                 metrics=results,
    #                 primary_metric=("f1_score", "maximize"),
    #             )

    train_normal_dataloader = DataLoader(
        dataset=normal_dataset,
        batch_size=16,
        num_workers=n_cpus,
        shuffle=True,
    )

    if args.pretrain:
        root.info("Starting pretraining")
        padeep.pretrain(train_normal_dataloader, n_epochs=args.ae_n_epochs, input_size=args.num_embeddings)
        root.info("Pretraining done")

    root.info("Starting training")
    padeep.train(
        train_dataloader,
        n_epochs=args.n_epochs,
        outlier_exposure=args.oe_folder is not None,
        test_cb=None, # test_cb if "ifremer" in args.train_folder.lower() or "turtles" in args.train_folder.lower() else None,
        self_supervision=USE_SELF_SUPERVISION,
    )

    print(">> Saving params")
    params = padeep.get_residuals()
    with open(PARAMS_PATH, 'wb') as f:
        pickle.dump(params, f)
    print(f">> Params saved at {PARAMS_PATH}")

    return padeep
