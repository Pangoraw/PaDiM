from typing import Union, Tuple
from math import sqrt

import torch
from torch import Tensor, device as Device, optim, nn
from tqdm import tqdm

from padim.base import PaDiMBase
from padim.normalizing_flows.maf import RealNVP


class MultiHeadedRealNVP(nn.Module):
    def __init__(self, n_svdds: int = 1):
        self.nets = nn.ModuleList([])

    def forward(self, x):
        return


class PaDiMNVP(PaDiMBase):

    def __init__(
        self,
        num_embeddings: int = 100,
        device: Union[str, Device] = "cpu",
        backbone: str = "resnet18",
        size: Union[None, Tuple[int, int]] = None,
    ):
        super(PaDiMNVP, self).__init__(num_embeddings, device, backbone, size)

        self.net = RealNVP(n_blocks=3,
                input_size=num_embeddings, hidden_size=20, n_hidden=2)

    def train(self, dataloader, n_epochs=10, *args, **kwargs):
        self.net = self.net.to(self.device)

        optimizer = optim.Adam(self.net.parameters(),
                               lr=0.001,
                               weight_decay=1e-6,
                               amsgrad=False)

        for epoch in tqdm(range(n_epochs)):
            total_loss = 0
            i = 0
            for imgs, _ in dataloader:
                embeddings = self._embed_batch_flatten(imgs)
                loss = - self.net.log_prob(embeddings, None).mean(0)

                i+=1
                print(f"batch {i}: loss: {loss.item()}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"epoch {epoch+1} loss: {total_loss:.3f}")

    def predict(self, batch, params=None):
        """
        Computes the negative log-probability for each path of the batch image
        """
        b = batch.size(0)
        self.net.eval()
        embeddings = self._embed_batch_flatten(batch)
        w = int(sqrt(embeddings.size(0)))
        loss = -self.net.log_prob(embeddings, None)
        return loss.view(b, w, w)

    def _get_inv_cvars(self, a):
        """
        Noop, like `get_params()`
        """
        return a

    def get_params(self):
        return torch.zeros((1, )), torch.zeros((1, 1)), self.embedding_ids

    def get_residuals(self):
        def detach_numpy(t: Tensor):
            return t.detach().cpu().numpy()

        backbone = self._get_backbone()
        net_dict = self.net.state_dict()
        return net_dict, detach_numpy(self.embedding_ids), backbone

    @staticmethod
    def from_residuals(state_dict, embedding_ids, device="cuda"):
        padim = PaDiMNVP(device)
        padim.net.load_state_dict(state_dict)
        padim.embedding_ids = torch.tensor(embedding_ids, device=device)

        return padim
