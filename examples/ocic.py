import argparse
import pickle
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm

sys.path.append("../padim/")

from padim.ocic import OCIC, OCICSVDD


def parse_args():
  parser = argparse.ArgumentParser("OCIC Trainer")
  parser.add_argument("--train_folder", required=True)
  parser.add_argument("--params_path", required=True)
  parser.add_argument("--method", default="gaussian", choices=["gaussian", "svdd"])
  parser.add_argument("--mode", default="deep", choices=["deep", "shallow"])
  return parser.parse_args()


def main(args):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  img_transforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((32, 32)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
  ])
  dataset = ImageFolder(root=args.train_folder, transform=img_transforms)
  dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,
    num_workers=32,
  )

  if args.method == "gaussian":
    Model = OCIC
    opt_args = []
  elif args.method == "svdd":
    Model = OCICSVDD
    opt_args = (50,)
  else:
    raise NotImplementedError(args.method)

  ocic = Model(device, args.mode)
  ocic.train(dataloader, *opt_args)

  params = ocic.get_residuals()
  with open(args.params_path, "wb") as f:
    pickle.dump((args.method,) + params, f)


if __name__ == "__main__":
  args = parse_args()
  main(args)
