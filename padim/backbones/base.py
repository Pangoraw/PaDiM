from typing import Tuple

import torch
from torch import nn, Tensor
from torchvision.models import resnet18, resnet50, wide_resnet50_2


class EncoderBase(nn.Module):
    def __init__(self, resnet_builder, load_path=None) -> None:
        super().__init__()
        if load_path is None:
            self.resnet = resnet_builder(pretrained=True)
        else:
            state_dict = torch.load(load_path)
            convert = lambda k: k.replace("encoder.resnet.", "")
            self.resnet = resnet_builder()
            self.resnet.load_state_dict(state_dict, strict=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return the three intermediary layers from the ResNet
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
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        feature_1 = self.resnet.layer1(x)
        feature_2 = self.resnet.layer2(feature_1)
        feature_3 = self.resnet.layer3(feature_2)

        return feature_1, feature_2, feature_3


class ResNet50(EncoderBase):
    embeddings_size = 1792
    num_patches = 32 * 32

    def __init__(self, load_path=None) -> None:
        super(ResNet50, self).__init__(resnet50, load_path)


class ResNet18(EncoderBase):
    embeddings_size = 448
    num_patches = 32 * 32

    def __init__(self, load_path=None) -> None:
        super(ResNet18, self).__init__(resnet18, load_path)


class WideResNet50(EncoderBase):
    embeddings_size = 1792
    num_patches = 104 * 104

    def __init__(self, load_path=None) -> None:
        super(WideResNet50, self).__init__(wide_resnet50_2, load_path)