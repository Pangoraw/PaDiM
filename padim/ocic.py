import torch
from torch import Tensor, nn
from torchvision.models import resnet18

from padim.utils.distance import mahalanobis_sq


class OCIC:
  """
  One Class Image Classifier based on a pre-trained ResNet model
  """

  def __init__(self, device):
    self.device = device
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.backbone = resnet18(pretrained=True).to(device).eval()
    self.feature_size = 448
    self.mean = torch.zeros((self.feature_size,), device=device)
    self.cov_sum = torch.zeros((self.feature_size, self.feature_size), device=device)
    self.N = 0

  def _embed(self, x: Tensor) -> Tensor:
    with torch.no_grad():
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        feature_1 = self.backbone.layer1(x)
        feature_2 = self.backbone.layer2(feature_1)
        feature_3 = self.backbone.layer3(feature_2)

        feature_1 = self.avgpool(feature_1).squeeze(2).squeeze(2)
        feature_2 = self.avgpool(feature_2).squeeze(2).squeeze(2)
        feature_3 = self.avgpool(feature_3).squeeze(2).squeeze(2)

        return torch.cat((feature_1, feature_2, feature_3), dim=1)

  def train_one_batch(self, batch: Tensor):
    embeddings = self._embed(batch)  # n * 512
    b = embeddings.size(0)
    for i in range(b):
      self.cov_sum += torch.outer(embeddings[i, :], embeddings[i, :])
    self.mean += embeddings.sum(dim=0)
    self.N += b

  def train(self, dataloader):
    for batch in dataloader:
      self.train_one_batch(batch)

  def get_params(self, epsilon: float = .01):
    means = self.mean.detach().clone()
    covs = self.cov_sum.detach().clone()

    identity = torch.eye(self.feature_size).to(self.device)
    means /= self.N
    covs -= self.N * torch.outer(means, means)
    covs /= self.N - 1
    covs += epsilon * identity

    return means, covs

  @staticmethod
  def _get_inv_cvar(cov):
    return torch.inverse(cov)

  def get_residuals(self):
    def numpy_detach(t):
      return t.detach().cpu().numpy()

    return numpy_detach(self.mean), numpy_detach(self.cov_sum), self.N

  @staticmethod
  def from_residuals(mean, cov_sum, N, device):
    ocic = OCIC(device)
    ocic.mean = torch.tensor(mean, device=device)
    ocic.cov_sum = torch.tensor(cov_sum, device=device)
    ocic.N = N
    return ocic

  def predict(self, imgs, params=None):
    if params is None:
      mean, cov = self.get_params()
      inv_cvar = self._get_inv_cvar(cov)
    else:
      mean, inv_cvar = params

    embeddings = self._embed(imgs)
    distances = mahalanobis_sq(embeddings, mean, inv_cvar)
    return torch.sqrt(distances)
