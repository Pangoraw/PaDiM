import argparse
import os
import pickle
import sys
from functools import partial
from itertools import product

import keepsake
import torch

sys.path.append("./")

from padim import PaDiM, PaDiMShared, PaDiMSVDD
from padim.panf import PaDiMNVP

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
    parser.add_argument("--load_path")
    parser.add_argument("--threshold", type=int)

    # Testing params
    parser.add_argument("--test_limit", type=int, default=-1)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--min_area", type=int, default=2)
    parser.add_argument("--use_nms", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--semi_ortho", action="store_true")
    parser.add_argument("--compare_all",
                        action="store_true",
                        help="For original PaDiM only")
    parser.add_argument("--size",
                        default="416x416",
                        help="image size [default=416x416]")

    # Params for PaDeep
    parser.add_argument("--oe_folder")
    parser.add_argument("--oe_frequency", type=int)
    parser.add_argument("--homo_frequency", type=int, default=3)

    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--ae_n_epochs", type=int, default=1)
    parser.add_argument("--n_svdds", type=int, default=1)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--use_self_supervision", action="store_true")
    parser.add_argument("--num_embeddings", type=int, default=100)
    parser.add_argument("--backbone", default="wide_resnet50", choices=["resnet18", "resnet50", "wide_resnet50"])
    parser.add_argument("--nf", action="store_true")
    parser.add_argument("--test_ious", action="store_true")

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
    elif cfg.semi_ortho:
        method = "semi_ortho"
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
        if cfg.deep and cfg.nf:
            Model = PaDiMNVP
        elif cfg.deep:
            Model = PaDiMSVDD
        elif cfg.shared:
            Model = PaDiMShared
        elif cfg.semi_ortho:
            Model = PaDiM
        else:
            Model = PaDiM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Model.from_residuals(*params, device=device)
    else:
        if cfg.deep:
            model = train_padeep(cfg, experiment)
        else:
            model = train_padim(cfg)

    if "semmacape" in cfg.test_folder.lower():
        if cfg.test_ious:
            i = 0
            print("testing iou/thresholds matrix")
            for (thresh, iou) in product([.1, .2, .3, .4, .5, .6, .7, .8, .9], [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]):
                print(f"threshold = {thresh:.2f}, iou = {iou:.2f}")
                cfg.iou_threshold = iou
                results = test_semmacape(cfg, model, thresh, True)
                experiment.checkpoint(
                    step=i,
                    metrics=results,
                    primary_metric=("f1_score", "maximize"),
                )
                i += 1
        elif cfg.threshold is not None:
            threshold = cfg.threshold / 10
            results = test_semmacape(cfg, model, threshold)
            experiment.checkpoint(
                step=cfg.threshold,
                metrics=results,
                primary_metric=("f1_score", "maximize"),
            )
        else:
            computed_roc_auc = False
            auroc = -1
            for t in range(1, 10):
                threshold = t / 10
                results = test_semmacape(cfg, model, threshold, computed_roc_auc)

                if "roc_auc_score" in results:
                    computed_roc_auc = True
                    auroc = results["roc_auc_score"]
                elif computed_roc_auc and "roc_auc_score" not in results:
                    results["roc_auc_score"] = auroc

                experiment.checkpoint(
                    step=t,
                    metrics=results,
                    primary_metric=("f1_score", "maximize"),
                )
        # t = 1 if "ifremer" in cfg.test_folder.lower() else 6
        # threshold = t / 10
        # results = test_semmacape(cfg, model, threshold)
        # experiment.checkpoint(
        #     step=t,
        #     metrics=results,
        #     primary_metric=("f1_score", "maximize"),
        # )
    else:
        results = test_mvtec(cfg, model)
        experiment.checkpoint(
            metrics=results,
            primary_metric=("roc_auc_score", "maximize"),
        )

    experiment.stop()

if __name__ == "__main__":
    main()
