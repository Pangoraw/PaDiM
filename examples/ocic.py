import argparse
import pickle
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.models import ImageFolder
from torchvision import transforms
from tqdm import tqdm

sys.path.append("../padim/")

from padim.ocic import OCIC, OCICSVDD


def parse_args():
  parser = argparse.ArgumentParser("OCIC Trainer")
  parser.add_argument("--train_folder", required=True)
  parser.add_argument("--params_path", required=True)
  parser.add_argument("--method", default="gaussian", choices=["gaussian", "svdd"])
  return parser.parse_args()


def main(args):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  dataset = ImageFolder(root=args.train_folder, transform=transforms.ToTensor())
  dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,
    num_workers=16,
  )

  if args.method == "gaussian":
    Model = OCIC
  elif args.method == "svdd":
    Model = OCICSVDD
  else:
    raise NotImplementedError(args.method)

  ocic = Model(device)
  ocic.train(dataloader)

  params = ocic.get_residuals()
  with open(args.params_path, "wb") as f:
    pickle.dump((args.method,) + params, f)


if __name__ == "__main__":
  args = parse_args()
  main(args)
