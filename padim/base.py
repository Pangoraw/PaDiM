from typing import Union, Tuple

import torch
from torch import Tensor, device as Device

from padim.utils import embeddings_concat
from padim.backbones import ResNet18, WideResNet50


class PaDiMBase:
    """The embedding backbone shared by PaDiM and PaDiMSVDD
    """

    def __init__(self, num_embeddings: int, device: Union[str, Device],
                 backbone: str, size=None):
        self.device = device
        self.num_embeddings = num_embeddings

        if size is not None:
            self._init_backbone_with_size(backbone, size)
        else:
            self._init_backbone(backbone)

        self.embedding_ids = torch.randperm(
            self.max_embeddings_size)[:self.num_embeddings].to(self.device)

    def _get_backbone(self):
        if isinstance(self.model, ResNet18):
            backbone = "resnet18"
        elif isinstance(self.model, WideResNet50):
            backbone = "wide_resnet50"
        else:
            raise NotImplementedError()

        return backbone

    def _init_backbone_with_size(self, backbone: str, size: Tuple[int, int]) -> None:
        self._init_backbone(backbone)
        empty_batch = torch.zeros((1, 3) + size, device=self.device)
        feature_1, _, _ = self.model(empty_batch)
        _, _, w, h = feature_1.shape
        self.num_patches = w * h
        self.model.num_patches = w * h

    def _init_backbone(self, backbone: str) -> None:
        if backbone == "resnet18":
            self.model = ResNet18().to(self.device)
        elif backbone == "wide_resnet50":
            self.model = WideResNet50().to(self.device)
        else:
            raise Exception(f"unknown backbone {backbone}, "
                            "choose one of ['resnet18', 'wide_resnet50']")

        self.num_patches = self.model.num_patches
        self.max_embeddings_size = self.model.embeddings_size

    def _embed_batch(self, imgs: Tensor, with_grad: bool = False) -> Tensor:
        with torch.set_grad_enabled(with_grad):
            feature_1, feature_2, feature_3 = self.model(imgs.to(self.device))
        embeddings = embeddings_concat(feature_1, feature_2)
        embeddings = embeddings_concat(embeddings, feature_3)
        embeddings = torch.index_select(
            embeddings,
            dim=1,
            index=self.embedding_ids,
        )
        return embeddings

    def _embed_batch_flatten(self, imgs, *args):
        embeddings = self._embed_batch(imgs, *args)
        _, C, _, _ = embeddings.shape
        return embeddings.permute(0, 2, 3, 1).reshape((-1, C))
