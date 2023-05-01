from __future__ import print_function
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
import pandas as pd
import torch
from torch.autograd import Variable
from model import ResnetGTSRB, StnGTSRB
from torcheval.metrics import MulticlassConfusionMatrix
from plot_evaluation import plot
import torch.nn.functional as F

from utils.eval_transformation import *


# .py in the same folder

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='PyTorch GTSRB evaluation script')
parser.add_argument('--test_dir', type=str, default='data/test_images', metavar='D',
                    help="folder where test data are located")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--max_voting', default=False, type=str2bool,
                    help="the prediction is the max of the transformations applied")

mapping = {
    0: "Speed Limit 20Kph",
    1: "Speed Limit 30Kph",
    2: "Speed Limit 50Kph",
    3: "Speed Limit 60Kph",
    4: "Speed Limit 70Kph",
    5: "Speed Limit 80Kph",
    6: "Speed Limit 100Kph",
    7: "Speed Limit 120Kph",
    8: "Yield",
    9: "Stop",
    10: "End of Speed Limits",
    11: "Unknown"
}

global args


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def evaluate():
    numClasses = 12
    target = []

    model_name = args.model.split('/')[1].split('.')[0]

    state_dict = torch.load(args.model)
    model = StnGTSRB(numClasses=numClasses)
    model.load_state_dict(state_dict)
    model.eval()

    test_dir = args.test_dir

    transformations = [basic_transformation, imadjust_transformation, histeq_transformation, adapthisteq_transformation,
                       conorm_transformation]

    if 'test_images' in args.test_dir:
        data = pd.read_csv("data/test_images/GT-final_test.csv", delimiter=';')
        target = data.ClassId.tolist()
        for i, el in enumerate(target):
            if el in [0, 1, 2, 3, 4, 5]:
                continue
            elif el == 7:
                target[i] = 6
            elif el == 8:
                target[i] = 7
            elif el == 13:
                target[i] = 8
            elif el == 14:
                target[i] = 9
            elif el == 32:
                target[i] = 10
            else:
                target[i] = 11
        target = torch.tensor(target)

    predictions = torch.zeros(len([x for x in os.listdir(test_dir) if x.endswith('.ppm')]), dtype=torch.int64)

    max_voting = args.max_voting

    for idx, file in enumerate(tqdm(os.listdir(test_dir))):
        if 'ppm' in file:
            output = torch.zeros([1, numClasses], dtype=torch.float32)
            if max_voting:
                with torch.no_grad():
                    for t in transformations:
                        data = t(pil_loader(test_dir + '/' + file))
                        data = data.unsqueeze(0)
                        data = Variable(data)
                        output = output.add(model(data))
                    output = F.log_softmax(output, dim=1)
                    pred = output.data.max(1, keepdim=True)[1]
                    predictions[idx] = pred
            else:
                with torch.no_grad():
                    data = basic_transformation(pil_loader(test_dir + '/' + file))
                    data = data.unsqueeze(0)
                    data = Variable(data)
                    output = model(data)
                    output = F.log_softmax(output, dim=1)
                    pred = output.data.max(1, keepdim=True)[1]
                    predictions[idx] = pred

    confusion_mat = MulticlassConfusionMatrix(numClasses)
    confusion_mat.update(predictions, target)

    df_cm = pd.DataFrame(confusion_mat.compute().numpy(), range(numClasses), range(numClasses))
    df_cm.to_csv(f'results/{model_name}_prediction.csv', index=False)

    tensors_dict = {'predictions': predictions, 'target': target}
    torch.save(tensors_dict, f'results/{model_name}_tensors.pt')

    plot(model_name)


if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists('results/'):
        os.makedirs('results')

    evaluate()

    exit()
