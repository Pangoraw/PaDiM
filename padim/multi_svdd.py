import torch
from torch import nn, Tensor

from padim.deep_svdd import build_network


def _build_svdd(*args, **kwargs):
    return build_network(*args, **kwargs)


class MultiDeepSVDD(nn.Module):
    def __init__(self, n_svdds=1, *args, **kwargs):
        super(MultiDeepSVDD, self).__init__()
        self.n_svdds = n_svdds

        self.svdds = [
            _build_svdd(*args, **kwargs)
            for _ in range(self.n_svdds)
        ]

    def forward(self, x: Tensor) -> Tensor:
        results = torch.stack(
            (svdd(x) for svdd in self.svdds),
            dim=0
        )
        return results.min(dim=0)
