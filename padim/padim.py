from typing import Tuple, Union, List

import numpy as np
from numpy import ndarray as NDArray

import torch
from torch import Tensor, device as Device
from torch.nn import Module
from torch.utils.data import DataLoader
from scipy.spatial.distance import mahalanobis

from padim.utils import embeddings_concat
from padim.backbones import ResNet18, WideResNet50


class PaDiM:
    def __init__(
        self,
        num_embeddings: int = 100,
        device: Union[str, Device] = "cpu",
        backbone: str = "resnet18",
    ):
        self.device = device
        self.num_embeddings = num_embeddings

        self._init_backbone(backbone)

        self.N = 0
        self.embedding_ids = torch.randperm(self.max_embeddings_size)[
            : self.num_embeddings
        ].to(self.device)
        self.means = torch.zeros((self.num_embeddings, self.num_patches)).to(
            self.device
        )
        self.covs = torch.zeros(
            (self.num_embeddings, self.num_embeddings, self.num_patches)
        ).to(self.device)

    def _init_backbone(self, backbone: str) -> None:
        if backbone == "resnet18":
            self.model = ResNet18().to(self.device)
        elif backbone == "wide_resnet50":
            self.model = WideResNet50().to(self.device)
        else:
            raise Exception(
                f"unknown backbone {backbone}, choose one of ['resnet18', 'wide_resnet50']"
            )

        self.num_patches = self.model.num_patches
        self.max_embeddings_size = self.model.embeddings_size

    def _embed_batch(self, imgs: Tensor) -> Tensor:
        with torch.no_grad():
            feature_1, feature_2, feature_3 = self.model(imgs.to(self.device))
        embeddings = embeddings_concat(feature_1, feature_2)
        embeddings = embeddings_concat(embeddings, feature_3)
        embeddings = torch.index_select(embeddings, dim=1, index=self.embedding_ids)
        return embeddings

    def train_one_batch(self, imgs: Tensor) -> None:
        """
        Handle only one batch, updating the internal state
        Params
        ======
            imgs: Tensor - batch tensor of size (b * c * w * h)
        """
        with torch.no_grad():
            embeddings = self._embed_batch(imgs.to(self.device))  # b * c * w * h
            b = embeddings.size(0)
            embeddings = embeddings.reshape(
                (-1, self.num_embeddings, self.num_patches)
            )  # b * c * (w * h)
            for i in range(self.num_patches):
                patch_embeddings = embeddings[:, :, i]  # b * c
                for j in range(b):
                    self.covs[:, :, i] += torch.outer(
                        patch_embeddings[j, :], patch_embeddings[j, :]
                    )  # c * c
                self.means[:, i] += patch_embeddings.sum(dim=0)  # c
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

    def get_params(self, epsilon: float = 0.01) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes the mean vectors and covariance matrices from the indermediary state
        Params
        ======
            epsilon: float - coefficient for the identity matrix
        Returns
        =======
            means: Tensor - the computed mean vectors
            covs: Tensor - the computed covariance matrices
        """
        means = self.means.detach().clone()
        covs = self.covs.detach().clone()

        identity = torch.eye(self.num_embeddings).to(self.device)
        means /= self.N
        for i in range(self.num_patches):
            covs[:, :, i] -= self.N * torch.outer(means[:, i], means[:, i])
            covs[:, :, i] /= self.N - 1  # corrected covariance
            covs[:, :, i] += epsilon * identity  # constant term

        return means, covs, self.embedding_ids

    def test(self, dataloader: DataLoader) -> List[NDArray]:
        """
        Consumes the given dataloader and outputs the corresponding distance matrices
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
        for new_imgs in dataloader:
            new_distances = self.predict(new_imgs, params=(means, covs))
            distances.extend(new_distances)
        return np.array(distances)

    def predict(
        self, new_imgs: Tensor, params: Tuple[NDArray, NDArray] = None
    ) -> NDArray:
        """
        Computes the distance matrix for each image * patch
        Params
        ======
            imgs: Tensor - (b * W * H) tensor of images
            params: [(ndarray, ndarray)] - optional precomputed distribution parameters
        Returns
        =======
            distances: ndarray - (c * b) array of distances
        """
        if params is None:
            means, covs, _ = self.get_params()
            means, covs = means.cpu().numpy(), covs.cpu().numpy()
        else:
            means, covs = params
        embeddings = self._embed_batch(new_imgs)
        b, c, w, h = embeddings.shape
        embeddings = embeddings.reshape(b, c, w * h).cpu().numpy()

        distances = []
        for i in range(h * w):
            mean = means[:, i]
            cvar_inv = np.linalg.inv(covs[:, :, i])
            distance = [mahalanobis(e[:, i], mean, cvar_inv) for e in embeddings]
            distances.append(distance)

        return np.array(distances)

    def get_residuals(self) -> Tuple[int, Tensor, Tensor, Tensor]:
        """
        Get the intermediary data needed to stop the training and resume later
        Returns
        =======
            N: int - the number of images
            means: Tensor - the sums of embedding vectors
            covs: Tensor - the sums of the outer product of embedding vectors
            embedding_ids: Tensor - the random dimensions used for size reduction
        """
        return self.N, self.means, self.covs, self.embedding_ids

    @staticmethod
    def from_resisuals(N: int, means: NDArray, covs: NDArray, embedding_ids: NDArray):
        pass
