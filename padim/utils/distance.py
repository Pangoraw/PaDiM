import torch
from torch import Tensor


def mahalanobis_sq(x: Tensor, mu: Tensor, sigma_inv: Tensor) -> Tensor:
    """
    The squared mahalanobis distance using PyTorch
    """
    print(x.shape)
    print(mu.shape)
    delta = (x - mu).view((104 * 104, 100, 1))
    print(delta.shape)
    temp = torch.matmul(sigma_inv, delta).view((104 * 104, 100, 1))
    dist = torch.matmul(delta.view((104 * 104, 1, 100)), temp)
    return dist
