from __future__ import print_function
import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets
from torch.autograd import Variable
import numpy as np

# Neural Network and Optimizer
from model import ResnetGTSRB, StnGTSRB

### Data Initialization and Loading
from utils.augmentation import data_transforms, data_jitter_hue, data_jitter_brightness, data_jitter_saturation, \
    data_jitter_contrast, data_rotate, data_equalize, data_center, data_blur, \
    data_perspective, data_affine  # augmentation.py in the same folder
from utils.train_val_test import initialize_data

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip must be here")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models.')
args = parser.parse_args()

torch.manual_seed(args.seed)

if torch.cuda.is_available():
    use_gpu = True
    print("Using GPU")
else:
    use_gpu = False
    print("Using CPU")

FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
Tensor = FloatTensor


def train():
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    numClasses = 13

    model = StnGTSRB(numClasses)

    if use_gpu:
        model.cuda()

    model.train()

    initialize_data(args.data)  # extracts the zip files, makes a validation set

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    # loss counters
    iteration = 0
    correct = 0
    training_loss = 0

    weights = torch.ones(numClasses)
    # weights[0:12] = 2
    # weights[12] = 0.5

    if use_gpu:
        weights = weights.cuda()

    num_epochs = args.epochs
    epoch = 0

    print("Begin training!")
    try:
        for epoch in range(num_epochs):

            # Apply data transformations on the training images to augment dataset
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset([datasets.ImageFolder(args.data + '/train_images',
                                                                     transform=data_transforms),
                                                datasets.ImageFolder(args.data + '/train_images',
                                                                     transform=data_jitter_brightness),
                                                datasets.ImageFolder(args.data + '/train_images',
                                                                     transform=data_jitter_saturation),
                                                datasets.ImageFolder(args.data + '/train_images',
                                                                     transform=data_jitter_contrast),
                                                datasets.ImageFolder(args.data + '/train_images',
                                                                     transform=data_jitter_hue),
                                                datasets.ImageFolder(args.data + '/train_images',
                                                                     transform=data_blur),
                                                datasets.ImageFolder(args.data + '/train_images',
                                                                     transform=data_rotate),
                                                datasets.ImageFolder(args.data + '/train_images',
                                                                     transform=data_perspective),
                                                datasets.ImageFolder(args.data + '/train_images',
                                                                     transform=data_affine),
                                                datasets.ImageFolder(args.data + '/train_images',
                                                                     transform=data_equalize),
                                                datasets.ImageFolder(args.data + '/train_images',
                                                                     transform=data_center),
                                                ]),
                batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=use_gpu)

            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(args.data + '/val_images',
                                     transform=data_transforms),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=use_gpu)

            for batch_idx, (images, target) in enumerate(train_loader):
                images, target = Variable(images), Variable(target)
                if use_gpu:
                    images = images.cuda()
                    target = target.cuda()
                optimizer.zero_grad()
                output = model(images)
                loss = F.nll_loss(output, target, weight=weights)
                loss.backward()
                optimizer.step()
                max_index = output.max(dim=1)[1]
                correct += (max_index == target).sum()
                training_loss += loss
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss per example: {:.6f}\tLoss: {:.6f}'.
                          format(epoch, batch_idx * len(images), len(train_loader.dataset),
                                 100. * batch_idx / len(train_loader),
                                 loss.data.item() / (args.batch_size * args.log_interval), loss.data.item()))
                iteration += 1
            print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                training_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))
            correct = 0

            validation(model, val_loader, scheduler, weights)

            print("\nSaving weights:..")
            model.save_weights(os.path.join(args.save_folder, f"gtsr_{epoch}_{iteration}.pth"))

    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')
            model.save_weights(os.path.join(args.save_folder, f"{epoch}_{iteration}_interrupt.pth"))
        exit()


def validation(model, val_loader, scheduler, weights):
    model.eval()
    validation_loss = 0
    correct = 0
    for images, target in val_loader:
        with torch.no_grad():
            images, target = Variable(images), Variable(target)
            if use_gpu:
                images = images.cuda()
                target = target.cuda()
            output = model(images)
            validation_loss += F.nll_loss(output, target, size_average=False,
                                          weight=weights).data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    scheduler.step(np.around(validation_loss, 3))
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


if __name__ == '__main__':
    train()