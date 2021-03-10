from typing import Tuple

from torch import Tensor
from torch.nn import Module
from torchvision.models import resnet18


class ResNet18(Module):
    embeddings_size = 448
    num_patches = 32 * 32

    def __init__(self) -> None:
        super().__init__()
        self.resnet18 = resnet18(pretrained=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return the three intermediary layers from the ResNet18
        pre-trained model.
        Params
        ======
            x: Tensor - the input tensor of size (b * c * w * h)
        Returns
        =======
            feature_1: Tensor - the residual from layer 1
            feature_2: Tensor - the residual from layer 2
            feature_3: Tensor - the residual from layer 3
        """
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        feature_1 = self.resnet18.layer1(x)
        feature_2 = self.resnet18.layer2(feature_1)
        feature_3 = self.resnet18.layer3(feature_2)

        return feature_1, feature_2, feature_3
