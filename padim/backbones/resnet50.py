from typing import Tuple

from torch import Tensor
from torch.nn import Module
from torchvision.models import resnet50


class ResNet50(Module):
    embeddings_size = 1792
    num_patches = 32 * 32

    def __init__(self) -> None:
        super().__init__()
        self.resnet50 = resnet50(pretrained=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return the three intermediary layers from the ResNet50
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
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        feature_1 = self.resnet50.layer1(x)
        feature_2 = self.resnet50.layer2(feature_1)
        feature_3 = self.resnet50.layer3(feature_2)

        return feature_1, feature_2, feature_3
