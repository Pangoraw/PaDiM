import torch
from torch import Tensor


def batch_mahalanobis_sq(x: Tensor, mu: Tensor, sigma_inv: Tensor) -> Tensor:
    """
    Allows calling `mahalanobis_sq` on a batch of tensors
    Params
    ======
        x: The input tensors of size (b, h * w, c)
        mu: The mean tensors of size (h * w, c)
        sigma_inv: The inverted covariance matrices of size (h * w, c, c)
    Returns
    =======
        dist: Distance tensor of size (b, h * w, 1)
    """
    result = mahalanobis_sq(x, mu, sigma_inv)
    return torch.diagonal(result, dim1=1,
                          dim2=2).squeeze(1).unsqueeze(-1)  # (b, h * w, 1)


def mahalanobis_sq(x: Tensor, mu: Tensor, sigma_inv: Tensor) -> Tensor:
    """
    The squared mahalanobis distance using PyTorch
    Params
    ======
        x: The input tensors of size (h * w, c)
        mu: The mean tensors of size (h * w, c)
        sigma_inv: The inverted covariance matrices of size (h * w, c, c)
    Returns
    =======
        dist: Distance tensor of size (h * w, 1)
    """
    delta = x - mu  # (h * w, c)
    temp = torch.matmul(sigma_inv, delta.unsqueeze(-1))  # (h * w, c, 1)
    dist = torch.matmul(delta.unsqueeze(1), temp)  # (h * w, 1, 1)
    return dist.squeeze(1)  # (h * w, 1)
