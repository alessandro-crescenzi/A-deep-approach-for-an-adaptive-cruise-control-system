import numpy as np
from typing import Tuple

import torch
import torch_dct as dct


# def gradient(A: np.ndarray) -> Tuple[np.ndarray]:
#     """
#     Computes gradients of input numpy array. Returns tuple, where the first
#     part is gradient in direction of axis 0 (rows), then axis 1 (columns),...
#
#     Args:
#         f (np.ndarray): Input numpy array.
#
#     Returns:
#         Returns tuple of numpy arrays denoting gradients in different directions.
#     """
#
#     rows, cols = A.shape
#
#     grad_x = torch.zeros(A.shape)
#     grad_x[:, 0: cols - 1] = np.diff(A, axis=1)
#
#     grad_x = torch.zeros(A.shape)
#     grad_y[0:rows - 1, :] = np.diff(A, axis=0)
#
#     B = np.concatenate((grad_x[..., np.newaxis], grad_y[..., np.newaxis]), axis=-1)
#
#     return B

def divergence(A):
    m, n, _ = A.shape

    grad_x = A[:, :, 0]
    grad_y = A[:, :, 1]

    T1 = torch.zeros(grad_x.shape, device='cuda:0')
    T1[:, 1:n] = grad_x[:, 0:n-1]
    grad_x = grad_x - T1

    T1 = torch.zeros(grad_x.shape, device='cuda:0')
    T1[1:m, :] = grad_y[0:m - 1, :]
    grad_y = grad_y - T1

    return grad_x + grad_y

    # T = A[:, :, 0]
    # T1 = np.zeros(shape=(m, n))
    # T1[:, 1:n] = T[:, 0:n - 1]
    #
    # B = B + T - T1
    #
    # T = A[:, :, 1]
    # T1 = np.zeros(shape=(m, n))
    # T1[1:m, :] = T[0:m - 1, :]
    #
    # B = B + T - T1
    # return B


def laplacian(img, h: float = None):
    dims = 2

    grad = torch.gradient(img)
    grad[0][-1] = 0
    grad[1][:, -1] = 0
    grads = torch.concat((torch.unsqueeze(grad[1], dim=2), torch.unsqueeze(grad[0], dim=2)), dim=-1)

    if h is not None:
        # remove edges (gradients) smaller than 0
        # norm = np.sqrt(np.sum(grads * grads, axis=-1))
        # norm = np.linalg.norm(grads, axis=-1)
        norm = torch.sqrt(torch.sum(grads * grads, dim=-1))

        # mask = (norm < h)[..., np.newaxis].repeat(dims, axis=-1)
        mask = torch.concat((torch.unsqueeze(norm < h, dim=2), torch.unsqueeze(norm < h, dim=2)), dim=-1)
        grads[mask] = 0

    # and compute its divergence by summing the second-order gradients
    lap = divergence(grads)

    return lap


def dct2(block):
    return dct.dct(torch.transpose(dct.dct(torch.transpose(block, 0, 1), norm='ortho'), 0, 1), norm='ortho')


def idct2(block):
    return dct.idct(torch.transpose(dct.idct(torch.transpose(block, 0, 1), norm='ortho'), 0, 1), norm='ortho')
