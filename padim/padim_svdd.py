from typing import Union
import logging

import numpy as np
import torch
from torch import Tensor, device as Device, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm

from deep_svdd.src.deepSVDD import DeepSVDD
from deep_svdd.src.base import BaseADDataset

from padim.base import PaDiMBase


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


class TransformingDataset(Dataset):
    def __init__(self, root, img_transforms, target_transform):
        self.img_dataset = ImageFolder(root=root, transform=img_transforms)
        self.target_transform = target_transform
        self.current_img_index = 0
        img, _ = self.img_dataset[0]
        self.current_img_embeddings = self.target_transform(img)

    def __getitem__(self, index):
        img_index = index // (104 * 104)
        if self.current_img_index != img_index:
            self.current_img_index = img_index
            img, _ = self.img_dataset[img_index]
            self.current_img_embeddings = self.target_transform(img)

        embedding_idx = index % (104 * 104)
        return (self.current_img_embeddings[embedding_idx, :].reshape(
            (1, 10, 10)), 0, 0)

    def __len__(self):
        return len(self.img_dataset) * 104 * 104


class ImageFolderADDataset(BaseADDataset):
    def __init__(self, train_image_folder, test_image_folder, img_transforms,
                 target_transform):
        super(ImageFolderADDataset, self).__init__(train_image_folder)
        self.train_set = TransformingDataset(train_image_folder,
                                             img_transforms, target_transform)
        self.test_set = TransformingDataset(test_image_folder,
                                            img_transforms,
                                            target_transform=target_transform)

    def loaders(self,
                batch_size: int,
                shuffle_train=True,
                shuffle_test=False,
                num_workers: int = 0):
        train_dataloader = DataLoader(num_workers=num_workers,
                                      batch_size=batch_size,
                                      shuffle=shuffle_train,
                                      dataset=self.train_set)
        test_dataloader = DataLoader(num_workers=num_workers,
                                     batch_size=batch_size,
                                     shuffle=shuffle_test,
                                     dataset=self.test_set)

        return train_dataloader, test_dataloader


class PaDiMSVDD(PaDiMBase):
    """A variant of the PaDiM architecture using Deep-SVDD as
    the normal distribution instead of a multi-variate gaussian
    """

    def __init__(
        self,
        num_embeddings: int = 100,
        device: Union[str, Device] = "cpu",
        backbone: str = "resnet18",
        **kwargs,
    ):
        super(PaDiMSVDD, self).__init__(num_embeddings, device, backbone)
        self.svdd = DeepSVDD()
        self.svdd.set_network("mnist_LeNet")

        self._init_params(**kwargs)

    def _init_params(self,
                     objective='one-class',
                     R=0.0,
                     nu=0.1,
                     lr: float = 0.001,
                     weight_decay=1e-6,
                     lr_milestones=(),
                     optimizer_name='adam'):
        assert objective in (
            'one-class', 'soft-boundary'
        ), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        self.R = torch.tensor(R, device=self.device)
        self.c = None

        self.lr = lr
        self.nu = nu
        # number of training epochs for soft-boundary
        # Deep SVDD before radius R gets updated
        self.warm_up_n_epochs = 10

        self.lr_milestones = lr_milestones
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, **kwargs):
        self.svdd.train(self.image_ad_dataset, device=self.device, **kwargs)

    def _embed_batch_flatten(self, imgs):
        embeddings = self._embed_batch(imgs)
        _, C, _, _ = embeddings.shape
        return embeddings.view((-1, 1, 10, 10))

    def train_home_made(self, dataloader, n_epochs=10):
        logger = logging.getLogger()

        self.svdd.net = self.svdd.net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(self.svdd.net.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestones, gamma=0.1)

        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self._init_center_c(dataloader)
            logger.info('Center c initialized.')

        self.svdd.net.train()
        for epoch in tqdm(range(n_epochs)):
            loss_epoch = 0.0
            n_batches = 0
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)

                optimizer.zero_grad()

                embeddings = self._embed_batch_flatten(imgs)
                outputs = self.svdd.net(embeddings)
                dist = torch.sum((outputs - self.c)**2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R**2
                    loss = self.R**2 + (1 / self.nu) * torch.mean(
                        torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)

                loss.backward()
                optimizer.step()

                if (self.objective == 'soft-boundary') and (
                        epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu),
                                               device=self.device)
                loss_epoch += loss.item()
                n_batches += 1
            logger.info('\tEpoch {}/{}\tLoss: {:.8f}'.format(
                epoch + 1, n_epochs, loss_epoch / n_batches))

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('\tLR Scheduler: new learning rate is %g' %
                            float(scheduler.get_lr()[0]))

        logger.info('Finished training.')

        return self.svdd.net

    def _init_center_c(self, dataloader, eps=0.1):
        n_samples = 0
        c = torch.zeros(self.svdd.net.rep_dim, device=self.device)
        self.svdd.net.eval()
        with torch.no_grad():
            for inputs, _ in tqdm(dataloader):
                inputs = inputs.to(self.device)
                inputs = self._embed_batch_flatten(inputs)

                outputs = self.svdd.net(inputs)
                n_samples += outputs.size(0)
                c += torch.sum(outputs, dim=0)
        c /= n_samples

        # If c_i is too close to 0, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def predict(self, batch: Tensor):
        self.svdd.net.eval()

        embeddings = self._embed_batch_flatten(batch)
        outputs = self.svdd.net(embeddings)
        dists = torch.sum((outputs - self.c) ** 2, dim=1)
        if self.objective == 'soft-boundary':
            scores = dists - self.R ** 2
        else:
            scores = dists

        # Return anomaly maps
        return scores.reshape((-1, 1, 104, 104))
