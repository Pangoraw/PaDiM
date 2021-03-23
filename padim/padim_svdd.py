from typing import Union

from torch import Tensor, device as Device
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from deep_svdd.src.deepSVDD import DeepSVDD
from deep_svdd.src.base import BaseADDataset

from padim.base import PaDiMBase


class TransformingDataset(Dataset):
    def __init__(self, root, img_transforms, target_transform):
        self.img_dataset = ImageFolder(root=root, transform=img_transforms)
        self.target_transform = target_transform
        self.current_img_index = 0
        img, _ = self.img_dataset[0]
        self.current_img_embeddings = self.target_transform(img)

    def __getitem__(self, index):
        img_index = index // (104*104)
        if self.current_img_index != img_index:
            self.current_img_index = img_index
            img, _ = self.img_dataset[img_index]
            self.current_img_embeddings = self.target_transform(img)

        embedding_idx = index % (104 * 104)
        return self.current_img_embeddings[embedding_idx, :].reshape((1, 10, 10)), 0, 0

    def __len__(self):
        return len(self.img_dataset) * 104 * 104


class ImageFolderADDataset(BaseADDataset):
    def __init__(self, train_image_folder, test_image_folder, img_transforms,
                 target_transform):
        super(ImageFolderADDataset, self).__init__(train_image_folder)
        self.train_set = TransformingDataset(train_image_folder,
                                             img_transforms, target_transform)
        self.test_set = TransformingDataset(test_image_folder,
                                            img_transforms,
                                            target_transform=target_transform)

    def loaders(self,
                batch_size: int,
                shuffle_train=True,
                shuffle_test=False,
                num_workers: int = 0):
        train_dataloader = DataLoader(num_workers=num_workers,
                                      batch_size=batch_size,
                                      shuffle=shuffle_train,
                                      dataset=self.train_set)
        test_dataloader = DataLoader(num_workers=num_workers,
                                     batch_size=batch_size,
                                     shuffle=shuffle_test,
                                     dataset=self.test_set)

        return train_dataloader, test_dataloader


class PaDiMSVDD(PaDiMBase):
    """A variant of the PaDiM architecture using Deep-SVDD as
    the normal distribution instead of a multi-variate gaussian
    """

    def __init__(
        self,
        train_image_folder,
        test_image_folder,
        num_embeddings: int = 100,
        device: Union[str, Device] = "cpu",
        backbone: str = "resnet18",
    ):
        super(PaDiMSVDD, self).__init__(num_embeddings, device, backbone)
        self.svdd = DeepSVDD()
        self.svdd.set_network("mnist_LeNet")

        def encoder(img: Tensor) -> Tensor:
            """Transforms images to embedding vectors"""
            embeddings = self._embed_batch(img.unsqueeze(0))  # N * C * H * W
            _, C, H, W = embeddings.shape
            embeddings = embeddings.view((H * W, C))
            return embeddings

        img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((416, 416)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                inplace=True,
            ),
        ])
        self.image_ad_dataset = ImageFolderADDataset(
            train_image_folder,
            test_image_folder,
            img_transforms=img_transforms,
            target_transform=encoder)

    def train(self, **kwargs):
        self.svdd.train(self.image_ad_dataset, device=self.device, **kwargs)

    def test(self, **kwargs):
        self.svdd.test(self.image_ad_dataset, device=self.device)
