import sys
import argparse

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

sys.path.append('../padim')
sys.path.append('../deep_svdd/src')

from padim import PaDiMSVDD


parser = argparse.ArgumentParser(prog="PaDeep test")
parser.add_argument("--train_folder", required=True)
parser.add_argument("--test_folder", required=True)

args = parser.parse_args()

padeep = PaDiMSVDD(args.train_folder,
                   args.test_folder,
                   backbone='wide_resnet50')

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

padeep.train_home_made(train_dataloader, n_epochs=1)
# padeep.test()
