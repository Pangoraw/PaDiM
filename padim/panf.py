from typing import Union, Tuple
from math import sqrt

import torch
from torch import Tensor, device as Device, optim, nn
from tqdm import tqdm

from padim.base import PaDiMBase
from padim.normalizing_flows.maf import RealNVP, MAF


class MultiHeadedNF(nn.Module):

    def __init__(self, nf, n_nfs: int = 1, *args, **kwargs):
        super(MultiHeadedNF, self).__init__()
        self.nets = nn.ModuleList([
            nf(*args, **kwargs)
            for _ in range(n_nfs)
        ])

    def forward(self, x):
        return torch.stack(
            [nf(x)[0] for nf in self.nets],
            dim=1,
        )

    def log_prob(self, x, y=None):
        return torch.stack(
            [nf.log_prob(x, y) for nf in self.nets],
            dim=1,
        )


class PaDiMNVP(PaDiMBase):

    def __init__(
        self,
        n_svdds=1,
        num_embeddings: int = 100,
        device: Union[str, Device] = "cpu",
        backbone: str = "resnet18",
        size: Union[None, Tuple[int, int]] = None,
        load_path: str = None,
        mode = "random",
        nf = "realnvp",
    ):
        super(PaDiMNVP, self).__init__(num_embeddings, device, backbone, size, load_path=load_path, mode=mode)
    
        nf = RealNVP if nf == "realnvp" else MAF

        # (n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu', input_order='sequential', batch_norm=True):
        self.net = MultiHeadedNF(
            nf,
            n_svdds,
            n_blocks=7,
            input_size=num_embeddings,
            hidden_size=130,
            n_hidden=3
        )

    def train(self, dataloader, n_epochs=10, *args, **kwargs):
        self.net = self.net.to(self.device)

        optimizer = optim.Adam(self.net.parameters(),
                               lr=0.001,
                               weight_decay=1e-6,
                               amsgrad=False)

        for epoch in tqdm(range(n_epochs)):
            total_loss = 0
            for imgs, _ in dataloader:
                embeddings = self._embed_batch_flatten(imgs)
                logprobs, _ = self.net.log_prob(embeddings, None).max(dim=1)
                loss = -logprobs.mean()

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
        with torch.no_grad():
            logprobs, _ = self.net.log_prob(embeddings, None).max(dim=1)
            loss = -logprobs

        assert b == 1
        loss = loss.view(b, w, w)
        loss -= loss.min() # map to [0, max]

        return loss

    def _get_inv_cvars(self, a):
        """
        Noop, like `get_params()`
        """
        return a

    def get_params(self):
        return torch.zeros((1, )), torch.eye(2), self.embedding_ids if self.embedding_mode == "random" else self.W

    def get_residuals(self):
        def detach_numpy(t: Tensor):
            return t.detach().cpu().numpy()

        backbone = self._get_backbone()
        net_dict = self.net.state_dict()
        return net_dict, detach_numpy(self.embedding_ids if self.embedding_mode == "random" else self.W), backbone

    @staticmethod
    def from_residuals(state_dict, embedding_ids, backbone, device="cuda"):
        mode = "random" if len(embedding_ids.shape) == 1 else "semi_orthogonal"

        if mode == "semi_orthogonal":
            _, num_embeddings = embedding_ids.shape
        else:
            num_embeddings, = embedding_ids.shape

        n_svdds = 0
        for key in state_dict.keys():
            if key.startswith("nets."):
                n_svdds = max(n_svdds, int(key[5:].split(".")[0]))
        n_svdds += 1

        print("loading with", n_svdds, "nets")

        padim = PaDiMNVP(
                n_svdds=n_svdds,
                backbone=backbone,
                device=device,
                mode=mode,
                nf="maf")

        padim.net.load_state_dict(state_dict)
        padim.net.eval()
        padim.net = padim.net.to(device)

        if mode == "random":
            padim.embedding_ids = torch.tensor(embedding_ids, device=device)
        else:
            padim.W = torch.tensor(embedding_ids, device=device)

        return padim
