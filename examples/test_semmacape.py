from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from padim.datasets import (
    LimitedDataset,
    SemmacapeTestDataset,
)
from padim.utils import propose_regions_cv2 as propose_regions, floating_IoU

grouped_classes_labels = ["Dauphins", "Oiseaux"]
accepted_classes = {
    # Dauphins: 0
    "Dauphin_BleuBlanc": 0,
    "Dauphin_Commun": 0,
    "Grand_Dauphin": 0,
    "Delphinid_Ind.": 0,

    # Oiseaux: 1
    "Sterne_Pose": 1,
    "Petit_Puffin_Pose": 1,
    "Goeland_Pose": 1,
    "Fou_Bassan_Pose": 1,

    "Sterne_Vol": 1,
    "Petit_Puffin_Vol": 1,
    "Goeland_Vol": 1,
    "Fou_Bassan_Vol": 1,
}

def test(cfg, padim, t):
    LIMIT = cfg.test_limit
    THRESHOLD = t
    IOU_THRESHOLD = cfg.iou_threshold
    MIN_AREA = cfg.min_area
    USE_NMS = cfg.use_nms
    LATTICE = 104
    TEST_FOLDER = cfg.test_folder

    predict_args = {}
    if cfg.compare_all:
        predict_args["compare_all"] = cfg.compare_all

    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416, 416)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = SemmacapeTestDataset(
        data_dir=TEST_FOLDER,
        transforms=img_transforms,
    )
    test_dataloader = DataLoader(batch_size=1,
                                 dataset=LimitedDataset(dataset=test_dataset,
                                                        limit=LIMIT))

    classes = {}

    n_proposals = 0
    n_included = 0
    n_gt = 0
    sum_iou = 0
    total_positive_proposals = 0
    positive_proposals = 0

    y_trues = []
    y_preds = []

    means, covs, _ = padim.get_params()
    inv_cvars = padim._get_inv_cvars(covs)
    pbar = tqdm(test_dataloader)
    for loc, img, _, y_true in pbar:
        # 1. Prediction
        res = padim.predict(img, params=(means, inv_cvars), **predict_args)
        preds = [res.max().item()]
        res = (res - res.min()) / (res.max() - res.min())
        res = res.reshape((LATTICE, LATTICE)).cpu()

        y_trues.extend(y_true.numpy())
        y_preds.extend(preds)

        # Don't do proposals and counts for normal images
        if y_true[0] == 1:
            continue

        def normalize_box(box):
            x1, y1, x2, y2, s = box
            return (x1 / LATTICE, y1 / LATTICE, abs(x1 - x2) / LATTICE,
                    abs(y1 - y2) / LATTICE, s)

        preds = propose_regions(
            res,
            threshold=THRESHOLD,
            min_area=MIN_AREA,
            use_nms=USE_NMS,
        )
        preds = [normalize_box(box)
                 for box in preds]  # map from 0,LATTICE to 0,1
        n_proposals += len(preds)

        img_proposals_counted = False

        # 2. Collect GT boxes
        with open(loc[0].replace('.jpg', '_with_name_label.txt'), 'r') as f:
            lines = f.readlines()

        n_gt += len(lines)
        for line in lines:
            cls, cx, cy, bw, bh, _ = line.split(' ')
            if cls not in classes:
                # classes[cls] =
                # (# detected, # proposals, sum(iou), total number of GT)
                classes[cls] = (0, len(preds), 0, len(lines))
                # Hypothesis: only one class per image
            elif not img_proposals_counted:
                classes[cls] = (
                    classes[cls][0],
                    classes[cls][1] + len(preds),
                    classes[cls][2],
                    classes[cls][3] + len(lines),
                )
            img_proposals_counted = True

            x1, y1 = float(cx) - float(bw) / 2, float(cy) - float(bh) / 2
            w, h = float(bw), float(bh)
            box = (x1, y1, w, h)

            box_detected = False
            for pred in preds:
                iou = floating_IoU(box, pred)
                sum_iou += iou
                classes[cls] = (
                    classes[cls][0],
                    classes[cls][1],
                    classes[cls][2] + iou,
                    classes[cls][3],
                )
                total_positive_proposals += 1
                if not box_detected and iou >= IOU_THRESHOLD:
                    # Positive detection
                    # Count detected box only once
                    positive_proposals += 1
                    classes[cls] = (classes[cls][0] + 1, classes[cls][1],
                                    classes[cls][2], classes[cls][3])
                    box_detected = True
                x2, y2, w2, h2, = pred[:4]
                # check inclusion
                if (x2 >= x1 and x2 + w2 <= x1 + w and y2 >= y1
                        and y2 + h2 <= y1 + h):
                    n_included += 1

        if n_proposals == 0:
            PPR = 0
        else:
            PPR = positive_proposals / n_proposals
        recall = positive_proposals / n_gt
        pbar.set_description(f"PPR: {PPR:.3f} RECALL: {recall:.3f}")

    results = {}

    # from 1 normal to 1 anomalous
    y_trues = list(map(lambda x: 1.0 - x, y_trues))
    roc_score = roc_auc_score(y_trues, y_preds)
    print(f"roc auc score: {roc_score}")
    results["roc_auc_score"] = roc_score

    print(f"positive proposals: {positive_proposals}")
    print(f"total positive proposals: {total_positive_proposals}")
    print(f"total proposals: {n_proposals}")
    print(f"included: {n_included}")

    if n_proposals == 0:
        ppr = 0
        mean_iou = 0
    else:
        ppr = positive_proposals / n_proposals
        mean_iou = sum_iou / n_proposals
    print(f"PPR: {ppr}")
    print(f"MEAN_IOU: {mean_iou}")
    recall = positive_proposals / n_gt
    print(f"RECALL: {recall}")

    results['mean_iou'] = mean_iou
    results['ppr'] = ppr
    results['recall'] = recall

    if ppr == 0:
        f1 = 0
    else:
        f1 = 2 * (ppr * recall) / (ppr + recall)
    results['f1_score'] = f1
    print(f"F1_SCORE: {f1}")

    grouped_classes = {cls: (0, 0, 0, 0) for cls in grouped_classes_labels}
    for cls, data in classes.items():
        acc = grouped_classes[grouped_classes_labels[accepted_classes[cls]]]
        grouped_classes[grouped_classes_labels[accepted_classes[cls]]] = tuple(map(sum, zip(data, acc)))

    classes.update(grouped_classes)

    for cls, (detected, n_cls_proposals, cls_sum_iou,
              n_cls_gt) in classes.items():
        cls_recall = detected / n_cls_gt
        print(f">> {cls}: recall: {cls_recall}")
        if n_cls_proposals == 0:
            cls_ppr = 0
            cls_mean_iou = 0
        else:
            cls_ppr = detected / n_cls_proposals
            cls_mean_iou = cls_sum_iou / n_cls_proposals

        print(f">> {cls}: precision: {cls_ppr}")
        print(f">> {cls}: mean_iou: {cls_mean_iou}")
        results[cls + "_mean_iou"] = cls_mean_iou
        results[cls + "_ppr"] = cls_ppr
        results[cls + "_recall"] = cls_recall

        if cls_ppr == 0:
            cls_f1 = 0
        else:
            cls_f1 = 2 * cls_ppr * cls_recall / (cls_ppr + cls_recall)
        print(f">> {cls}: f1_score: {cls_f1}")
        results[cls + "_f1_score"] = cls_f1

    return results
