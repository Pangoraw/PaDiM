import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils

sys.path.append('../padim')
sys.path.append('../deep_svdd/src')

from padim import PaDiMSVDD


logging.basicConfig(filename='logs/padeep.log', level=logging.INFO)
handler = logging.StreamHandler(sys.stdout)

root = logging.getLogger()
root.addHandler(handler)
root.info('Starting training')

parser = argparse.ArgumentParser(prog="PaDeep test")
parser.add_argument("--train_folder", required=True)
parser.add_argument("--test_folder", required=True)
parser.add_argument("--n_epochs", type=int, default=1)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

padeep = PaDiMSVDD(backbone='wide_resnet50', 
                   device=device)

img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((416, 416)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        inplace=True,
    ),
])

train_dataloader = DataLoader(
    dataset=ImageFolder(root=args.train_folder, transform=img_transforms),
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

padeep.train_home_made(train_dataloader, n_epochs=args.n_epochs, test_images=test_batch)

results = padeep.predict(next_batch)

utils.save_image(test_batch, 'inputs.png')
utils.save_image(results, 'outputs.png')
