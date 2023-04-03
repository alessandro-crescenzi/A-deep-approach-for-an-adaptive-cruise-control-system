from __future__ import print_function
import torchvision.transforms as transforms
from utils.preprocessing import *

# data augmentation for training and test time Resize all images to 32 * 32 and normalize them to mean = 0 and
# standard-deviation = 1 based on statistics collected from the training set

basic_transformation = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.3332, 0.3019, 0.3060), (0.2827, 0.2710, 0.2739))
])

imadjust_transformation = transforms.Compose([
    transforms.Resize((48, 48)),
    Imadjust(),
    transforms.ToTensor(),
    transforms.Normalize((0.3332, 0.3019, 0.3060), (0.2827, 0.2710, 0.2739))
])

histeq_transformation = transforms.Compose([
    transforms.Resize((48, 48)),
    Histeq(),
    transforms.ToTensor(),
    transforms.Normalize((0.3332, 0.3019, 0.3060), (0.2827, 0.2710, 0.2739))
])

adapthisteq_transformation = transforms.Compose([
    transforms.Resize((48, 48)),
    Adapthisteq(),
    transforms.ToTensor(),
    transforms.Normalize((0.3332, 0.3019, 0.3060), (0.2827, 0.2710, 0.2739))
])

conorm_transformation = transforms.Compose([
    transforms.Resize((48, 48)),
    Conorm(),
    transforms.ToTensor(),
    transforms.Normalize((0.3332, 0.3019, 0.3060), (0.2827, 0.2710, 0.2739))
])
