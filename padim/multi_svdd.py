import torch
from torch import optim, nn, Tensor
from tqdm import tqdm

from padim.deep_svdd import build_network, build_autoencoder


class MultiDeepSVDD(nn.Module):
    def __init__(self, n_svdds=1, *args, **kwargs):
        super(MultiDeepSVDD, self).__init__()
        self.n_svdds = n_svdds

        self.svdds = nn.ModuleList([
            build_network(*args, **kwargs)
            for _ in range(self.n_svdds)
        ])

    def forward(self, x: Tensor) -> Tensor:
        results = torch.stack(
            [svdd(x) for svdd in self.svdds],
            dim=1
        )  # (b * h * w) * n * rep_dim
        return results


class MultiAutoEncoder(nn.Module):
    def __init__(self, n, *args, **kwargs):
        super(MultiAutoEncoder, self).__init__()
        self.n = n
        self.auto_encoders = nn.ModuleList([
            build_autoencoder(*args, **kwargs)
            for _ in range(self.n)
        ])
        self.current_auto_encoder = 0

    def forward(self, x):
        self.current_auto_encoder = (self.current_auto_encoder + 1) % self.n
        res = self.auto_encoders[self.current_auto_encoder](x)
        return res
