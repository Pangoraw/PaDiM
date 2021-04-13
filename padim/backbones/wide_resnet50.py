from typing import Tuple

from torch import Tensor
from torch.nn import Module
from torchvision.models import wide_resnet50_2


class WideResNet50(Module):
    embeddings_size = 1792
    num_patches = 104 * 104

    def __init__(self) -> None:
        super().__init__()
        self.wide_resnet50 = wide_resnet50_2(pretrained=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return the three intermediary layers from the WideResNet50
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
        x = self.wide_resnet50.conv1(x)
        x = self.wide_resnet50.bn1(x)
        x = self.wide_resnet50.relu(x)
        x = self.wide_resnet50.maxpool(x)

        feature_1 = self.wide_resnet50.layer1(x)
        feature_2 = self.wide_resnet50.layer2(feature_1)
        feature_3 = self.wide_resnet50.layer3(feature_2)

        return feature_1, feature_2, feature_3

