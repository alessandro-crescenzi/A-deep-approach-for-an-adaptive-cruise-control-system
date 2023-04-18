import argparse
import os
import re

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch GTSRB plotting losses')
parser.add_argument('--file', type=str, metavar='M',
                    help="the file that is used to plot")

global args


def plot(file: str):
    if not os.path.exists("../results"):
        os.makedirs("../results")

    patterns = {
        'train': re.compile(r'Training set: Average loss: (?P<train>\S+),'),
        'val': re.compile(r'Validation set: Average loss: (?P<val>\S+),')
    }
    data = {key: [] for key in patterns}

    f = open(file, 'r')

    for line in f:
        for key, pattern in patterns.items():
            res = pattern.match(line)

            if res is not None:
                datum = res.groupdict()
                try:
                    v = float(datum[key])
                    datum[key] = v
                except ValueError:
                    break

                data[key].append(datum)
                break

    plt.title('Training Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    loss_names = ['Train Loss', 'Val Loss']

    x = [x['train'] for x in data['train']]
    plt.plot(x)
    j = [j['val'] for j in data['val']]
    plt.plot(j)

    plt.legend(loss_names)
    plt.show()
    # plt.savefig(f"../results/{file.split('.')[0]}_lossesPlot.png")


if __name__ == '__main__':
    args = parser.parse_args()

    plot(args.file)
