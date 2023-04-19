from __future__ import print_function
import torchvision.transforms as transforms
from utils.preprocessing import *

# data augmentation for training and test time Resize all images to 32 * 32 and normalize them to mean = 0 and
# standard-deviation = 1 based on statistics collected from the training set

data_transforms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.3332, 0.3019, 0.3060), (0.2827, 0.2710, 0.2739))
])

# Resize, normalize and jitter image brightness
data_jitter_brightness = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ColorJitter(brightness=3),
    transforms.ToTensor(),
    transforms.Normalize((0.3332, 0.3019, 0.3060), (0.2827, 0.2710, 0.2739))
])

# Resize, normalize and jitter image saturation
data_jitter_saturation = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ColorJitter(saturation=3),
    transforms.ToTensor(),
    transforms.Normalize((0.3332, 0.3019, 0.3060), (0.2827, 0.2710, 0.2739))
])

# Resize, normalize and jitter image contrast
data_jitter_contrast = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ColorJitter(contrast=3),
    transforms.ToTensor(),
    transforms.Normalize((0.3332, 0.3019, 0.3060), (0.2827, 0.2710, 0.2739))
])

# Resize, normalize and jitter image hues
data_jitter_hue = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ColorJitter(hue=0.15),
    transforms.ToTensor(),
    transforms.Normalize((0.3332, 0.3019, 0.3060), (0.2827, 0.2710, 0.2739))
])

# Resize, normalize and blur image
data_blur = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.GaussianBlur(kernel_size=3, sigma=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.3332, 0.3019, 0.3060), (0.2827, 0.2710, 0.2739))
])

# Resize, normalize and rotate image
data_rotate = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.3332, 0.3019, 0.3060), (0.2827, 0.2710, 0.2739))
])

# Resize, normalize and apply a perspective transformation to image
data_perspective = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomPerspective(),
    transforms.ToTensor(),
    transforms.Normalize((0.3332, 0.3019, 0.3060), (0.2827, 0.2710, 0.2739))
])

# Resize, normalize, rotate, translate and shear image
data_affine = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomAffine(degrees=15, shear=2, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.3332, 0.3019, 0.3060), (0.2827, 0.2710, 0.2739))
])

# Resize, normalize and crop image
data_center = transforms.Compose([
    transforms.Resize((60, 60)),
    transforms.CenterCrop(48),
    transforms.ToTensor(),
    transforms.Normalize((0.3332, 0.3019, 0.3060), (0.2827, 0.2710, 0.2739))
])

data_erasing = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.RandomErasing(),
    transforms.Normalize((0.3332, 0.3019, 0.3060), (0.2827, 0.2710, 0.2739))
])

