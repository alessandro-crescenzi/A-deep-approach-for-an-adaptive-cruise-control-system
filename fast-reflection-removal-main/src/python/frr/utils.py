import numpy as np
import torch


def min_max_scale(ar: torch.tensor, new_min: float = 0, new_max: float = 1):
    """
    Scales input array to have values in [0, 1] in-place.

    Args:
        ar (np.array): Input array.
        new_min (float): New minimum.
        new_max (float): New maximum.
    """
    ar -= ar.min()
    ar /= (ar.max() - ar.min())
    ar *= (new_max - new_min) + new_min
