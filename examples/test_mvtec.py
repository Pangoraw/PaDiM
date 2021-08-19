import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from padim.datasets import LimitedDataset, MVTecADTestDataset
from padim.utils import compute_pro_score, propose_regions_cv2 as propose_regions, floating_IoU


def test(cfg, padim):
    LIMIT = cfg.test_limit
    TEST_FOLDER = cfg.test_folder
    size = tuple(map(int, cfg.size.split("x")))

    predict_args = {}
    if cfg.compare_all:
        predict_args["compare_all"] = cfg.compare_all

    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = MVTecADTestDataset(root=TEST_FOLDER,
       transform=img_transforms,
       mask_transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(size)]),
       #target_transform=lambda x: int(test_dataset.class_to_idx["good"] == x))
    )

    test_dataloader = DataLoader(batch_size=1,
                                 dataset=LimitedDataset(dataset=test_dataset,
                                                        limit=LIMIT))

    y_trues = []
    y_preds = []

    masks = []
    amaps = []

    means, covs, _ = padim.get_params()
    inv_cvars = padim._get_inv_cvars(covs)

    pbar = tqdm(test_dataloader)
    for img, mask, y_true in pbar:
        res = padim.predict(img, params=(means, inv_cvars), **predict_args)
        preds = [res.max().item()]

        y_trues.extend(y_true.numpy())
        y_preds.extend(preds)

        masks.append(mask.unsqueeze(0).cpu())
        amaps.append(res.unsqueeze(0).cpu())

    gaussian_smoothing = transforms.GaussianBlur(9)

    amaps = torch.cat(amaps)
    amaps = F.interpolate(amaps, size, mode="bilinear", align_corners=True)
    amaps = gaussian_smoothing(amaps)
    amaps -= amaps.min()
    amaps /= amaps.max()
    masks = torch.cat(masks)

    amaps = amaps.squeeze().cpu().numpy()
    masks = masks.squeeze().cpu().numpy()

    roc_score = roc_auc_score(y_trues, y_preds)
    print(f"roc_auc_score: {roc_score}")

    pro_score = compute_pro_score(amaps, masks)
    print(f"pro_score: {pro_score}")

    return {
        "roc_auc_score": roc_score,
        "pro_score": pro_score,
    }
