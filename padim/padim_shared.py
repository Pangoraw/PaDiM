from typing import Union, Tuple

import torch
from torch import Tensor, device as Device
from torch.utils.data import DataLoader

from padim.base import PaDiMBase
from padim.utils.distance import mahalanobis_sq


class PaDiMShared(PaDiMBase):
    """
    Like PaDiM, but the multi-variate gaussian representation is shared
    between all patches
    """

    def __init__(
        self,
        num_embeddings: int = 100,
        device: Union[str, Device] = "cpu",
        backbone: str = "resnet18",
        size: Union[None, Tuple[int, int]] = None,
    ):
        super(PaDiMShared, self).__init__(num_embeddings, device, backbone, size)
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

    def predict(self, new_imgs: Tensor, params=None) -> Tensor:
        if params is None:
            mean, cov, _ = self.get_params()
            inv_cov = torch.inverse(cov)
        else:
            mean, inv_cov = params
        embeddings = self._embed_batch(new_imgs)
        b, c, w, h = embeddings.shape
        assert b == 1, f"batch size should be 1, got {b}"
        embeddings = embeddings.reshape(c, w * h).permute(1, 0)
        distances = mahalanobis_sq(embeddings, mean, inv_cov)
        return torch.sqrt(distances)

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

    def _get_inv_cvars(self, cov):
        return torch.inverse(cov)

    def get_residuals(self):
        def detach_numpy(t: Tensor):
            return t.detach().cpu().numpy()

        backbone = self._get_backbone()
        return (self.N, detach_numpy(self.mean), detach_numpy(self.cov),
                detach_numpy(self.embedding_ids), backbone)

    @staticmethod
    def from_residuals(N: int, mean, cov, embedding_ids, backbone, device):
        num_embeddings, = embedding_ids.shape
        padim = PaDiMShared(num_embeddings=num_embeddings,
                            backbone=backbone,
                            device=device)
        padim.embedding_ids = torch.tensor(embedding_ids, device=device)
        padim.N = N
        padim.mean = torch.tensor(mean, device=device)
        padim.cov = torch.tensor(cov, device=device)
        return padim
