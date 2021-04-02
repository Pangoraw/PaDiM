from typing import Tuple, Union, List
from functools import reduce
from itertools import product

import cv2
import numpy as np
from numpy import ndarray as NDArray
import torch


# A box defined as (x1, y1, x2, y2)
Region = Tuple[int, int, int, int]


def propose_regions_cv2(patches, threshold: float = 0.75, **kwargs):
    """
    Faster region proposal

    >>> propose_regions_cv2(np.array([[1, 0, 1], [1, 0, 1], [0, 1, 0]]))
    [(0, 0, 1, 2, 1), (2, 0, 3, 2, 1), (1, 2, 2, 3, 1)]

    >>> propose_regions_cv2(np.array([[0, 0], [0, 0]]))
    []

    """
    if isinstance(patches, torch.Tensor):
        patches = patches.cpu().numpy()
    mask = (patches >= threshold).astype(np.int8)

    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    def stat_to_box(stat):
        x1, y1, w, h, _ = stat
        return (x1, y1, x1 + w, y1 + h, 1)
    boxes = map(stat_to_box, stats[1:])

    return filter_regions(boxes, **kwargs)


def propose_region(
    patches: NDArray,
    threshold: float = 0.75,
    explore_diagonals: bool = False,
) -> Union[None, Region]:
    """Proposes regions from a set of patch activations

    >>> propose_region(np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]]))
    (0, 0, 2, 2, 1)

    >>> propose_region(np.array([[1, 0], [0, 0]]))
    (0, 0, 1, 1, 1)

    >>> propose_region(np.array([[1, 0], [0, 1]]), explore_diagonals=True)
    (0, 0, 2, 2, 1)

    >>> propose_region(np.array([[.1, 0], [0, 0]]), threshold=0.2)

    Params
    ======
        patches: ndarray - (h * w) patch activations
        threshold: float - the activation threshold
    Returns
    =======
        box - the bounding box
    """
    h, w = patches.shape

    first_patch_idx = np.argmax(patches, axis=None)
    px, py = first_patch_idx % w, first_patch_idx // w
    patch_value = patches[py, px]

    if patch_value < threshold:
        return None
    patches[py, px] = 0.0

    directions = [
        (0,  1),  # right
        (0, -1),  # left
        (1,  0),  # down
        (-1, 0),  # up
    ]
    if explore_diagonals:
        directions.extend([
            (-1, -1),  # upper left
            (-1, 1),  # bottom left
            (1, -1),  # upper right
            (1, 1),  # bottom right
        ])

    def explore_neighbors(x, y):
        neighbors = []
        for dx, dy in directions:
            nx, ny = dx + x, dy + y
            # invalid directions
            if nx not in range(w) or ny not in range(h):
                continue
            # patch is activated
            if patches[ny, nx] >= threshold:
                neighbors.append((nx, ny))
                patches[ny, nx] = 0.0

        new_neighbors = []
        for neighbor in neighbors:
            new_neighbors.extend(explore_neighbors(*neighbor))

        neighbors.extend(new_neighbors)
        return neighbors

    neighbors = explore_neighbors(px, py)
    neighbors.append((px, py))
    min_x = reduce(min, map(lambda p: p[0], neighbors), w)
    min_y = reduce(min, map(lambda p: p[1], neighbors), h)
    max_x = reduce(max, map(lambda p: p[0], neighbors), 0) + 1
    max_y = reduce(max, map(lambda p: p[1], neighbors), 0) + 1

    return (min_x, min_y, max_x, max_y, patch_value)


def propose_regions(
    patches: NDArray,
    threshold: float = 0.75,
    **kwargs
) -> List[Region]:
    """Proposes many regions

    >>> propose_regions(np.array([[1, 0, 1], [1, 0, 1], [0, 1, 0]]))
    [(0, 0, 1, 2, 1), (2, 0, 3, 2, 1), (1, 2, 2, 3, 1)]

    >>> propose_regions(np.array([[0, 0], [0, 0]]))
    []

    Params
    ======
        patches: ndarray - (h * w) patch activations
        threshold: float - the activation threshold
    Returns
    =======
        boxes: List[Region] - (x1, y1, x2, y2) bounding boxes
    """
    regions = []
    proposal = propose_region(patches, threshold)
    while proposal is not None:
        regions.append(proposal)
        proposal = propose_region(patches, threshold)

    return filter_regions(regions, **kwargs)


def filter_regions(
    regions: List[Region],
    min_area: int = 1,
    use_nms: bool = False,
    **kwargs,
) -> List[Region]:
    """Filters out regions that are not relevant

    >>> filter_regions([(0, 0, 1, 1)], min_area = 2)
    []

    >>> filter_regions([(0, 0, 1, 1), (0, 0, 2, 1)], min_area = 2)
    [(0, 0, 2, 1)]

    Params
    ======
        regions: List[Tuple[float, float, float, float]] - regions
        min_area: int - the minimum area for a region
    Returns
    =======
        regions: List[Tuple[float, float, float, float]] - regions
    """
    def get_area(region) -> float:
        x1, y1, x2, y2 = region[:4]
        return abs(x1 - x2) * abs(y1 - y2)
    if use_nms:
        regions = non_maximum_suppression(regions, **kwargs)
    return list(filter(lambda r: get_area(r) >= min_area, regions))


def _serialize_region(
    region: Region
) -> List[Tuple[int, int]]:
    """Lists the patches that are part of a region

    >>> _serialize_region((0, 0, 1, 2))
    [(0, 0), (0, 1)]

    Params
    ======
        region - Region the region to serialize
    Returns
    =======
        patches - List[Tuple[int, int]] patches part of the region
    """
    x1, y1, x2, y2 = region
    return list(product(range(x1, x2), range(y1, y2)))


def IoU(r1: Region, r2: Region) -> float:
    """Intersection over Union

    >>> IoU((0, 0, 1, 2), (0, 0, 1, 1))
    0.5

    Params
    ======
        r1 - Region
        r2 - Region
    Returns
    =======
        IoU - the insersection over union
    """
    s1 = _serialize_region(r1)
    s2 = _serialize_region(r2)

    intersection = set(s1).intersection(s2)
    union = set(s1).union(s2)

    return len(intersection) / len(union)


def floating_IoU(
        r1: Tuple[float, float, float, float],
        r2: Tuple[float, float, float, float]
) -> float:
    """IoU for non-grid based boxes

    >>> floating_IoU((0, 0, 1, 1), (0, 0, 1, 2))
    0.5

    Params
    ======
        r1 - (x, y, w, h,...)
        r2 - (x, y, w, h,...)
    Returns
    =======
        IoU - the insersection over union
    """
    x1, y1, w1, h1 = r1[:4]
    x2, y2, w2, h2 = r2[:4]

    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    top = max(y1, y2)
    bottom = min(y1 + h1, y2 + h2)

    overlap_height = bottom - top
    overlap_width = right - left

    if overlap_height <= 0 or overlap_width <= 0:
        overlap_area = 0
    else:
        overlap_area = overlap_height * overlap_width

    area1 = w1 * h1
    area2 = w2 * h2

    union_area = area1 + area2 - overlap_area
    return overlap_area / union_area


def cvt_xyxys_xywhs(box):
    x1, y1, x2, y2, s = box
    return (x1, y1, x2 - x1, y2 - y1, s)


def non_maximum_suppression(
    boxes: List,
    iou_threshold: float = 0.5
) -> List:
    """Filter boxes using non-maximum suppression

    >>> non_maximum_suppression([(0, 0, 1, 2, .2), (0, 0, 1, 1, .4)])
    [(0, 0, 1, 1, 0.4)]

    Params
    ======
        boxes: List[Regions] - regions with score
        iou_threshold: float - the threshold at which to remove boxes
    Returns
    =======
        boxes - List[Regions] - the filtered regions
    """
    # TODO: Fast version using numpy
    # Sort boxes by confidence scores
    boxes = sorted(boxes, key=lambda b: b[4])
    new_boxes = []
    while len(boxes) > 0:
        box = boxes.pop()
        new_boxes.append(box)
        # remove boxes
        boxes = list(filter(
            lambda other_box: floating_IoU(
                cvt_xyxys_xywhs(box),
                cvt_xyxys_xywhs(other_box)) < iou_threshold,
            boxes
        ))

    return new_boxes


if __name__ == "__main__":
    import doctest
    doctest.testmod()
