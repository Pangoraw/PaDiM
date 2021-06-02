from typing import Tuple, Union, List

import numpy as np
from numpy import ndarray as NDArray

import torch
from torch import Tensor, device as Device
from torch.utils.data import DataLoader

from padim.base import PaDiMBase
from padim.utils.distance import mahalanobis_multi, mahalanobis_sq


class PaDiM(PaDiMBase):
    """
    The PaDiM model
    """

    def __init__(
        self,
        num_embeddings: int = 100,
        device: Union[str, Device] = "cpu",
        backbone: str = "resnet18",
        size: Union[None, Tuple[int, int]] = None,
    ):
        super(PaDiM, self).__init__(num_embeddings, device, backbone, size)
        self.N = 0
        self.means = torch.zeros(
            (self.num_patches, self.num_embeddings)).to(self.device)
        self.covs = torch.zeros((self.num_patches, self.num_embeddings,
                                 self.num_embeddings)).to(self.device)

    def train_one_batch(self, imgs: Tensor) -> None:
        """
        Handle only one batch, updating the internal state
        Params
        ======
            imgs: Tensor - batch tensor of size (b * c * w * h)
        """
        with torch.no_grad():
            # b * c * w * h
            embeddings = self._embed_batch(imgs.to(self.device))
            b = embeddings.size(0)
            embeddings = embeddings.reshape(
                (-1, self.num_embeddings, self.num_patches))  # b * c * (w * h)
            for i in range(self.num_patches):
                patch_embeddings = embeddings[:, :, i]  # b * c
                for j in range(b):
                    self.covs[i, :, :] += torch.outer(
                        patch_embeddings[j, :],
                        patch_embeddings[j, :])  # c * c
                self.means[i, :] += patch_embeddings.sum(dim=0)  # c
            self.N += b  # number of images

    def train(self, dataloader: DataLoader) -> Tuple[Tensor, Tensor, Tensor]:
        """
        End-to-end training of the model
        Params
        ======
            dataloader: DataLoader - a dataset dataloader feeding images
        Returns
        =======
            means: Tensor - the computed mean vectors
            covs: Tensor - the computed covariance matrices
        """
        for imgs in dataloader:
            self.train_one_batch(imgs)

        means, covs, embedding_ids = self.get_params()
        return means, covs, embedding_ids

    def get_params(self,
                   epsilon: float = 0.01) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes the mean vectors and covariance matrices from the
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
        means = self.means.detach().clone()
        covs = self.covs.detach().clone()

        identity = torch.eye(self.num_embeddings).to(self.device)
        means /= self.N
        for i in range(self.num_patches):
            covs[i, :, :] -= self.N * torch.outer(means[i, :], means[i, :])
            covs[i, :, :] /= self.N - 1  # corrected covariance
            covs[i, :, :] += epsilon * identity  # constant term

        return means, covs, self.embedding_ids

    def test(self, dataloader: DataLoader) -> List[NDArray]:
        """
        Consumes the given dataloader and outputs the corresponding
        distance matrices
        Params
        ======
            dataloader: DataLoader - a dataloader of image tensors
        Returns
        =======
            distances: ndarray - the (N * (w * h)) distance matrix
        """
        distances = []
        means, covs, _ = self.get_params()
        means, covs = means.cpu().numpy(), covs.cpu().numpy()
        inv_cvars = self._get_inv_cvars(covs)
        for new_imgs in dataloader:
            new_distances = self.predict(new_imgs, params=(means, inv_cvars))
            distances.extend(new_distances)
        return np.array(distances)

    def _get_inv_cvars(self, covs: Tensor) -> NDArray:
        inv_cvars = torch.inverse(covs)
        return inv_cvars

    def predict(self,
                new_imgs: Tensor,
                params: Tuple[Tensor, Tensor] = None,
                compare_all: bool = False) -> Tensor:
        """
        Computes the distance matrix for each image * patch
        Params
        ======
            imgs: Tensor - (b * W * H) tensor of images
            params: [(Tensor, Tensor)] - optional precomputed parameters
        Returns
        =======
            distances: Tensor - (c * b) array of distances
        """
        if params is None:
            means, covs, _ = self.get_params()
            inv_cvars = self._get_inv_cvars(covs)
        else:
            means, inv_cvars = params
        embeddings = self._embed_batch(new_imgs)
        b, c, w, h = embeddings.shape
        # not required, but need changing of testing code
        assert b == 1, f"The batch should be of size 1, got b={b}"
        embeddings = embeddings.reshape(c, w * h).permute(1, 0)

        if compare_all:
            distances = mahalanobis_multi(embeddings, means, inv_cvars)
            distances, _ = distances.min(dim=0)
        else:
            distances = mahalanobis_sq(embeddings, means, inv_cvars)
        return torch.sqrt(distances)

    def get_residuals(self) -> Tuple[int, NDArray, NDArray, NDArray, str]:
        """
        Get the intermediary data needed to stop the training and resume later
        Returns
        =======
            N: int - the number of images
            means: Tensor - the sums of embedding vectors
            covs: Tensor - the sums of the outer product of embedding vectors
            embedding_ids: Tensor - random dimensions used for size reduction
        """
        backbone = self._get_backbone()

        def detach_numpy(t: Tensor) -> NDArray:
            return t.detach().cpu().numpy()

        return (self.N, detach_numpy(self.means), detach_numpy(self.covs),
                detach_numpy(self.embedding_ids), backbone)

    @staticmethod
    def from_residuals(N: int, means: NDArray, covs: NDArray,
                       embedding_ids: NDArray, backbone: str,
                       device: Union[Device, str]):
        num_embeddings, = embedding_ids.shape
        padim = PaDiM(num_embeddings=num_embeddings,
                      device=device,
                      backbone=backbone)
        padim.embedding_ids = torch.tensor(embedding_ids).to(device)
        padim.N = N
        padim.means = torch.tensor(means).to(device)
        padim.covs = torch.tensor(covs).to(device)

        return padim
