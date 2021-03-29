from typing import Union

import numpy as np
import torch
from torch import Tensor, device as Device
from torch.utils.data import DataLoader
from scipy.spatial.distance import mahalanobis

from padim.base import PaDiMBase


class PaDiMShared(PaDiMBase):
    def __init__(
        self,
        num_embeddings: int = 100,
        device: Union[str, Device] = "cpu",
        backbone: str = "resnet18",
    ):
        super(PaDiMShared, self).__init__(num_embeddings, device, backbone)
        self.N = 0
        self.mean = torch.zeros((self.num_embeddings, ), device=self.device)
        self.cov = torch.zeros((self.num_embeddings, self.num_embeddings),
                               device=self.device)

    def train_one_batch(self, imgs: Tensor):
        """
        Handle only one batch, updating the internal state
        Params
        ======
            imgs: Tensor - batch tensor of size (b * c * w * h)
        """
        with torch.no_grad():
            patches = self._embed_batch_flatten(imgs.to(self.device))  # n * c
            n_patches = patches.size(0)
            for i in range(n_patches):
                patch = patches[i]
                self.cov += torch.outer(patch, patch)  # c * c
            self.mean += patches.sum(dim=0)
            self.N += n_patches

    def train(self, dataloader: DataLoader):
        """
        End-to-end training of the model
        Params
        ======
            dataloader: DataLoader - a dataset dataloader feeding images
        Returns
        =======
            mean: Tensor - the computed mean vector
            cov: Tensor - the computed covariance matrice
        """
        for imgs in dataloader:
            self.train_one_batch(imgs)
        return self.get_params()

    def predict(self, new_imgs: Tensor) -> Tensor:
        mean, cov, _ = self.get_params()
        mean, cov = mean.cpu().numpy(), cov.cpu().numpy()
        inv_cov = np.linalg.inv(cov)
        embeddings = self._embed_batch(new_imgs)
        b, c, w, h = embeddings.shape
        embeddings = embeddings.reshape(b, c, w * h).cpu().numpy()

        distances = []
        for i in range(h * w):
            distance = [
                mahalanobis(e[:, i], mean, inv_cov) for e in embeddings
            ]
            distances.append(distance)

        return np.array(distances)

    def get_params(self, epsilon: float = 0.01):
        """
        Computes the mean vector and covariance matrice from the
        indermediary state
        Params
        ======
            epsilon: float - coefficient for the identity matrix
        Returns
        =======
            means: Tensor - the computed mean vectors
            covs: Tensor - the computed covariance matrices
            embedding_ids: Tensor - the embedding indices
        """
        mean = self.mean.detach().clone()
        cov = self.cov.detach().clone()

        identity = torch.eye(self.num_embeddings).to(self.device)
        mean /= self.N
        cov -= self.N * torch.outer(mean, mean)
        cov /= self.N - 1  # corrected covariance
        cov += epsilon * identity

        return mean, cov, self.embedding_ids
