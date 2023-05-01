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
        return F.equalize(pic)


class Adapthisteq(torch.nn.Module):
    def __init__(self):
        super().__init__()
        _log_api_usage_once(self)

    def forward(self, pic):
        np_image = np.array(pic)
        image = torch.from_numpy(np_image)
        image = image.swapaxes(1, 2).swapaxes(0, 1)

        assert image.shape[1] == image.shape[2], "Error in image transformation height and weight are not equal"

        iC, iH, iW = image.shape
        kH = kW = 6

        oH = oW = iW // kW

        for i in range(oH):
            for j in range(oW):
                image[:, i * kH:i * kH + kH, j * kW:j * kW + kW] = F.equalize(
                    image[:, i * kH:i * kH + kH, j * kW:j * kW + kW])

        return torchvision.transforms.ToPILImage()(image)


class Conorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        _log_api_usage_once(self)

    def forward(self, pic):
        return F.gaussian_blur(pic, kernel_size=[5, 5], sigma=[1, 1])

