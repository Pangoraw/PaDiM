import sys
import argparse

sys.path.append('../padim')
sys.path.append('../deep_svdd/src')

from padim import PaDiMSVDD


parser = argparse.ArgumentParser(prog="PaDeep test")
parser.add_argument("--train_folder", required=True)
parser.add_argument("--test_folder", required=True)

args = parser.parse_args()

padeep = PaDiMSVDD(args.train_folder, args.test_folder,
                   backbone='wide_resnet50')
padeep.train()

padeep.test()
