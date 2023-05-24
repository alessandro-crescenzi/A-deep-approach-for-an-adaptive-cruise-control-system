import torch

from frr.base import dct2, idct2, laplacian
from frr.utils import min_max_scale
import numpy as np
import matplotlib.pyplot as plt


class FastReflectionRemoval:
    """
    An instance of this class is able to remove reflections from specified image. It implements
    the algorithm to remove reflections from [Fast Single Image Reflection Suppression via Convex Optimization](https://arxiv.org/pdf/1903.03889.pdf) paper.
    """

    def __init__(self, M, N, h: float, lmbd: float = 0, mu: float = 1, epsilon: float = 1e-8):
        """
        Args:
            h (float): h parameter from the paper. Larger h means more reflections removed, but potentially worse quality of the image.
            lmbd (float, optional): . Defaults to 0.
            mu (float, optional): Defaults to 1.
        """
        if not (0 <= h <= 1):
            raise ValueError(
                "Value of 'h' must be between 0 and 1 (included). Recommended values are between 0 and 0.13.")
        if not (0 <= lmbd <= 1):
            raise ValueError("Value of 'lmbd' must be between 0 and 1 (included). Recommended value is 0.")
        if not (0 <= mu <= 1):
            raise ValueError("Value of 'mu' must be between 0 and 1 (included). Recommended value is 1.")

        self.h = h
        self.lmbd = lmbd
        self.epsilon = epsilon
        self.mu = mu

        # create matrix kappa, where kappa_{mn} = 2 * [cos((pi * m) / M) + cos((pi * n) / N) - 2]
        m = np.cos((np.pi * np.arange(M)) / M)
        n = np.cos((np.pi * np.arange(N)) / N)
        kappa = 2 * (np.add.outer(m, n) - 2)

        self.const = torch.from_numpy(self.mu * (kappa ** 2) - self.lmbd * kappa + self.epsilon).to("cuda:0")

    def _compute_rhs(self, image: torch.tensor) -> torch.tensor:
        """
        Computes right-hand side of equation 7 from the paper.

        Args:
            image (np.ndarray): Input image. Must be normalised to [0, 1] values.

        Returns:
            np.ndarray: Right-hand side of the equation.
        """
        channels = image.shape[-1]

        # iteratively compute for each channel individually
        # L(div(\delta_h(\grad Y)))
        # from eq (7)
        laplacians = torch.zeros(image.shape, device='cuda:0')
        for c in range(channels):
            lapl = laplacian(laplacian(image[..., c], h=self.h))
            laplacians[:, :, c] = lapl

        # computes right-hand side of equation (7)
        # L(...) + \epsilon * Y
        rhs = laplacians + self.epsilon * image

        return rhs

    def _compute_T(self, rhs: torch.tensor) -> torch.tensor:
        """
        Computes T matrix (the original matrix with reflection suppressed).

        Args:
            rhs (torch.tensor): Right-hand side of the equation 7.

        Returns:
            Returns T matrix.
        """

        channels = rhs.shape[-1]

        T = torch.zeros(rhs.shape)

        # perform Poisson DCT to solve partial differential eq
        for c in range(channels):
            rhs_slice = rhs[..., c]

            u = dct2(rhs_slice)

            u = u / self.const
            u = idct2(u)

            T[..., c] = u

        return T

    def remove_reflection(self, image: torch.tensor) -> torch.tensor:
        """
        Removes reflection from specified image.

        Args:
            image (np.ndarray): Image represented as numpy array of shape (H, W, C), where H is height, W is width and C is channels.
            The image is expected to have values in the interval [0, 1].

        Returns:
            np.ndarray: Returns image with removed reflections with values between 0 and 1.
        """
        # if not np.all((0 <= image) & (image <= 1)):
        #     raise ValueError("Input image doesn't have all values between 0 and 1.")
        # if len(image.shape) != 3:
        #     raise ValueError("Input image must have 3 dimensions.")

        T = self._compute_T(self._compute_rhs(image))
        min_max_scale(T)

        return T
