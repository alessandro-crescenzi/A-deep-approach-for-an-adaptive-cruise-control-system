import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.utils import _log_api_usage_once


__all__ = [
    "Imadjust",
    "Histeq",
    "Adapthisteq",
    "Conorm"
]


class Imadjust(torch.nn.Module):
    def __init__(self):
        super().__init__()
        _log_api_usage_once(self)

    def forward(self, pic):
        return F.adjust_contrast(pic, 2)


class Histeq(torch.nn.Module):
    def __init__(self):
        super().__init__()
        _log_api_usage_once(self)

    def forward(self, pic):
        return F.equalize(pic.to(dtype=torch.uint8)).to(dtype=torch.float)


class Adapthisteq(torch.nn.Module):
    def __init__(self):
        super().__init__()
        _log_api_usage_once(self)

    def forward(self, pic):
        image = pic.to(dtype=torch.uint8)

        assert image.shape[1] == image.shape[2], "Error in image transformation height and weight are not equal"

        iC, iH, iW = image.shape
        kH = kW = 6

        oH = oW = iW // kW

        for i in range(oH):
            for j in range(oW):
                image[:, i * kH:i * kH + kH, j * kW:j * kW + kW] = F.equalize(
                    image[:, i * kH:i * kH + kH, j * kW:j * kW + kW])

        return image.to(dtype=torch.float)


class Conorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        _log_api_usage_once(self)

    def forward(self, pic):
        return F.gaussian_blur(pic, kernel_size=[5, 5], sigma=[1, 1])

