from typing import Tuple, Union
from functools import reduce

import numpy as np
from numpy import ndarray as NDArray


def propose_region(
    patches: NDArray,
    threshold: float = 0.75
) -> Union[None, Tuple[float, float, float, float]]:
    """Proposes regions from a set of patch activations

    >>> propose_region(np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]]))
    (0, 0, 2, 2)

    >>> propose_region(np.array([[1, 0], [0, 0]]))
    (0, 0, 1, 1)

    >>> propose_region(np.array([[.1, 0], [0, 0]]), threshold=0.2)

    Params
    ======
        patches: ndarray - (h * w) patch activations
        threshold: float
    Returns
    =======
        boxes
    """
    h, w = patches.shape

    first_patch_idx = np.argmax(patches, axis=None)
    px, py = first_patch_idx % w, first_patch_idx // w

    if patches[py, px] < threshold:
        return None

    def explore_neighbors(x, y):
        directions = [
            (0,  1),  # right
            (0, -1),  # left
            (1,  0),  # down
            (-1, 0),  # up
        ]
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

    return (min_x, min_y, max_x, max_y)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
