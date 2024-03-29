import argparse
import os
import pickle
import sys

import keepsake
import torch

sys.path.append("./")

from padim import PaDiM, PaDiMShared, PaDiMSVDD

from padeep import train as train_padeep
from semmacape import train as train_padim
from test_semmacape import test as test_semmacape
from test_mvtec import test as test_mvtec


def parse_args():
    parser = argparse.ArgumentParser(prog="PaDiM")
    parser.add_argument("--train_folder", required=True, type=str)
    parser.add_argument("--test_folder", required=True, type=str)
    parser.add_argument("--params_path", required=True, type=str)
    parser.add_argument("--train_limit", type=int, default=-1)

    # Testing params
    parser.add_argument("--test_limit", type=int, default=-1)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--min_area", type=int, default=2)
    parser.add_argument("--use_nms", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--compare_all",
                        action="store_true",
                        help="For original PaDiM only")
    parser.add_argument("--size",
                        default="416x416",
                        help="image size [default=416x416]")

    # Params for PaDeep
    parser.add_argument("--oe_folder")
    parser.add_argument("--oe_frequency", type=int)

    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--ae_n_epochs", type=int, default=1)
    parser.add_argument("--n_svdds", type=int, default=1)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--use_self_supervision", action="store_true")

    return parser.parse_args()


def main():
    cfg = parse_args()
    params_dict = cfg.__dict__

    if cfg.deep:
        method = "padeep"
    elif cfg.shared:
        method = "shared"
    elif cfg.compare_all:
        method = "invariant"
    else:
        method = "padim"
    params_dict["method"] = method

    print("Starting experiment")
    experiment = keepsake.init(
        params=params_dict,
    )
    print("Experiment started")

    if os.path.exists(cfg.params_path):
        with open(cfg.params_path, "rb") as f:
            params = pickle.load(f)
        if cfg.deep:
            Model = PaDiMSVDD
        elif cfg.shared:
            Model = PaDiMShared
        else:
            Model = PaDiM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Model.from_residuals(*params, device=device)
    else:
        if cfg.deep:
            model = train_padeep(cfg)
        else:
            model = train_padim(cfg)

    if "semmacape" in cfg.test_folder:
        for t in range(1, 8):
            threshold = t / 10
            results = test_semmacape(cfg, model, threshold)
            experiment.checkpoint(
                step=t,
                metrics=results,
                primary_metric=("f1_score", "maximize"),
            )
    else:
        results = test_mvtec(cfg, model)
        experiment.checkpoint(
            metrics=results,
            primary_metric=("roc_auc_score", "maximize"),
        )

    experiment.stop()

if __name__ == "__main__":
    main()
