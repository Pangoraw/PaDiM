import torch
from torch import Tensor


def mahalanobis_sq(x: Tensor, mu: Tensor, sigma_inv: Tensor) -> Tensor:
    """
    The squared mahalanobis distance using PyTorch
    """
    delta = x - mu
    dist = torch.dot(delta, torch.mv(sigma_inv, delta))
    return dist
