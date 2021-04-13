import torch
from torch import nn


cel = nn.CrossEntropyLoss()


def self_supervised_loss(patches, classifier, device="cpu"):
    """
    Params
    ======
        patches: Tensor - size (b, c)
        classifier: PositionClassifier
    Returns
    =======
        cross_entropy_loss: Tensor - the self-supervised cross-entropy loss
    """
    n_patches = patches.size(0)
    indices = torch.arange(0, n_patches, dtype=torch.long).to(device)  # b
    neighbors_patches = torch.randint_like(indices, -4, 4)  # b

    indices += neighbors_patches
    indices[neighbors_patches <= -2] -= 102
    indices[neighbors_patches >= 2] += 102
    indices = torch.clamp(indices, min=0, max=104 * 104)

    shifted_patches = torch.index_select(patches, dim=0, index=indices)
    logits = classifier(patches, shifted_patches)

    return cel(logits, neighbors_patches + 4)


class PositionClassifier(nn.Module):
    """Fully-connected classifier used for the self-supervision loss
    """

    def __init__(self, classifier_d, class_num=8):
        super(PositionClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(classifier_d, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, class_num),
        )

    def forward(self, x1, x2):
        x = x1 - x2
        return self.model(x)
