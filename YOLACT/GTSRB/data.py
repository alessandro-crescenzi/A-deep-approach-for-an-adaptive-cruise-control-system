from __future__ import print_function

import torchvision.transforms as transforms

data_transforms = transforms.Compose([
    transforms.Resize((48, 48))
])

# Resize, normalize and jitter image brightness
data_jitter_brightness = transforms.Compose([
    transforms.Resize((48, 48)),
    # transforms.ColorJitter(brightness=-5),
    transforms.ColorJitter(brightness=1)
])

# Resize, normalize and jitter image saturation
data_jitter_saturation = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ColorJitter(saturation=1),
    # transforms.ColorJitter(saturation=-5)
])

# Resize, normalize and jitter image contrast
data_jitter_contrast = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ColorJitter(contrast=1),
    # transforms.ColorJitter(contrast=-5)
])

# Resize, normalize and jitter image hues
data_jitter_hue = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ColorJitter(hue=0.15)
])

# Resize, normalize and blur image
data_blur = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.GaussianBlur(kernel_size=3, sigma=0.5)
])

# Resize, normalize and rotate image
data_rotate = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomRotation(15)
])

# Resize, normalize and apply a perspective transformation to image
data_perspective = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomPerspective()
])

# Resize, normalize, rotate, translate and shear image
data_affine = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomAffine(degrees=15, shear=2, translate=(0.1, 0.1))
])

# Resize, normalize and crop image 
data_center = transforms.Compose([
    transforms.Resize((60, 60)),
    transforms.CenterCrop(48)
])

