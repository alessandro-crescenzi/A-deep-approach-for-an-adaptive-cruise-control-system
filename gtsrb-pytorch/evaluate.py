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

from utils.augmentation import data_transforms, data_jitter_hue, data_jitter_brightness, data_jitter_saturation, \
    data_jitter_contrast, data_rotate, data_equalize, data_center, data_blur, \
    data_perspective, data_affine  # augmentation.py in the same folder

# .py in the same folder

parser = argparse.ArgumentParser(description='PyTorch GTSRB evaluation script')
parser.add_argument('--test_dir', type=str, default='data/test_images', metavar='D',
                    help="folder where test data are located")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--max_voting', default=False,
                    help="the prediction is the max of the transformations applied")

mapping = {
    0: "Speed Limit 20Kph",
    1: "Speed Limit 30Kph",
    2: "Speed Limit 50Kph",
    3: "Speed Limit 60Kph",
    4: "Speed Limit 70Kph",
    5: "Speed Limit 80Kph",
    6: "End Speed Limit 80Kph",
    7: "Speed Limit 100Kph",
    8: "Speed Limit 120Kph",
    9: "Yield",
    10: "Stop",
    11: "End of Speed Limits",
    12: "Unknown"
}

global args


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def evaluate():
    numClasses = 13
    target = []

    model_name = args.model.split('/')[1].split('.')[0]

    state_dict = torch.load(args.model)
    model = StnGTSRB(numClasses=numClasses)
    model.load_state_dict(state_dict)
    model.eval()

    test_dir = args.test_dir

    transforms = [data_transforms, data_jitter_hue, data_jitter_brightness, data_jitter_saturation,
                  data_jitter_contrast, data_blur, data_rotate, data_equalize, data_center, data_affine,
                  data_perspective]

    if 'test_images' in args.test_dir:
        data = pd.read_csv("data/test_images/GT-final_test.csv", delimiter=';')
        target = data.ClassId.tolist()
        for i, el in enumerate(target):
            if el == 13:
                target[i] = 9
            elif el == 14:
                target[i] = 10
            elif el == 32:
                target[i] = 11
            elif el in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                continue
            else:
                target[i] = 12
        target = torch.tensor(target)

    predictions = torch.zeros(len([x for x in os.listdir(test_dir) if x.endswith('.ppm')]), dtype=torch.int64)

    for idx, file in enumerate(tqdm(os.listdir(test_dir))):
        if 'ppm' in file:
            output = torch.zeros([1, numClasses], dtype=torch.float32)
            with torch.no_grad():
                if args.max_voting:
                    for i in range(0, len(transforms)):
                        data = transforms[i](pil_loader(test_dir + '/' + file))
                        data = data.unsqueeze(0)
                        data = Variable(data)
                        output = output.add(model(data))
                else:
                    data = data_transforms(pil_loader(test_dir + '/' + file))
                    data = data.unsqueeze(0)
                    data = Variable(data)
                    output = model(data)
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
