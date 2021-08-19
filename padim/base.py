from typing import Union, Tuple

import torch
from torch import Tensor, device as Device

from padim.utils import embeddings_concat
from padim.backbones import ResNet18, ResNet50, WideResNet50


def fast_embeddings_concat(f1: Tensor, f2: Tensor, f3: Tensor):
    n, c1, w, h, = f1.shape
    _, c2, w2, h2 = f2.shape
    _, c3, w3, h3 = f3.shape

    Z = torch.zeros(n, c1+c2+c3, w, h, device=f1.device)
    Z[:,0:c1,:,:] = f1
    Z[:,c1:c1+c2,:,:] = f2.repeat_interleave(repeats=h//h2, dim=3).repeat_interleave(repeats=w//w2, dim=2)
    Z[:,c1+c2:,:,:] = f3.repeat_interleave(repeats=h//h3, dim=3).repeat_interleave(repeats=w//w3, dim=2)

    return Z


def _generate_W(F: int, k: int, device: str):
    omega = torch.randn((F, k), device=device)
    q, r = torch.linalg.qr(omega)
    W = q @ torch.sign(torch.diag(torch.diag(r)))
    return W


class PaDiMBase:
    """The embedding backbone shared by PaDiM and PaDiMSVDD
    """

    def __init__(self, num_embeddings: int, device: Union[str, Device],
                 backbone: str, size=None, load_path: str = None, mode="random"):
        self.device = device
        self.num_embeddings = num_embeddings

        if size is not None:
            self._init_backbone_with_size(backbone, size, load_path)
        else:
            self._init_backbone(backbone, load_path)

        self.embedding_mode = mode if mode is not None else "random"
        if mode is None or mode == "random":
            self.embedding_ids = torch.randperm(
                self.max_embeddings_size)[:self.num_embeddings].to(self.device)
        elif mode == "semi_orthogonal":
            self.W = _generate_W(self.max_embeddings_size, self.num_embeddings, device)

    def _get_backbone(self):
        if isinstance(self.model, ResNet18):
            backbone = "resnet18"
        elif isinstance(self.model, ResNet50):
            backbone = "resnet50"
        elif isinstance(self.model, WideResNet50):
            backbone = "wide_resnet50"
        else:
            raise NotImplementedError()

        return backbone

    def _init_backbone_with_size(self, backbone: str, size: Tuple[int, int], load_path: str = None) -> None:
        self._init_backbone(backbone, load_path)
        empty_batch = torch.zeros((1, 3) + size, device=self.device)
        feature_1, _, _ = self.model(empty_batch)
        _, _, w, h = feature_1.shape
        self.num_patches = w * h
        self.model.num_patches = w * h

    def _init_backbone(self, backbone: str, load_path: str = None) -> None:
        if backbone == "resnet18":
            self.model = ResNet18(load_path).to(self.device)
        elif backbone == "resnet50":
            self.model = ResNet50(load_path).to(self.device)
        elif backbone == "wide_resnet50":
            self.model = WideResNet50(load_path).to(self.device)
        else:
            raise Exception(f"unknown backbone {backbone}, "
                            "choose one of ['resnet18', 'resnet50', 'wide_resnet50']")

        self.num_patches = self.model.num_patches
        self.max_embeddings_size = self.model.embeddings_size

    def _embed_batch(self, imgs: Tensor, with_grad: bool = False) -> Tensor:
        self.model.eval()
        with torch.set_grad_enabled(with_grad):
            feature_1, feature_2, feature_3 = self.model(imgs.to(self.device))
        # embeddings = fast_embeddings_concat(feature_1, feature_2, feature_3)
        embeddings = embeddings_concat(feature_1, feature_2)
        embeddings = embeddings_concat(embeddings, feature_3)
        if self.embedding_mode == "random":
            embeddings = torch.index_select(
                embeddings,
                dim=1,
                index=self.embedding_ids,
            )
        elif self.embedding_mode == "semi_orthogonal":
            embeddings = embeddings.permute(2, 3, 1, 0)
            embeddings = torch.matmul(self.W.T, embeddings)
            embeddings = embeddings.permute(3, 2, 0, 1)
        return embeddings

    def _embed_batch_flatten(self, imgs, *args):
        embeddings = self._embed_batch(imgs, *args)
        _, C, _, _ = embeddings.shape
        return embeddings.permute(0, 2, 3, 1).reshape((-1, C))
