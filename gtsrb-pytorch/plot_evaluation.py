import argparse
import seaborn as sn
import matplotlib.pyplot as plt
import torch
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
import pandas as pd
import time

parser = argparse.ArgumentParser(description='PyTorch GTSRB ploting performance')
parser.add_argument('--model_name', type=str, metavar='M',
                    help="the model name that is used to evaluate")

global args


def plot(model_name):
    numClasses = 12

    mapping = {
        0:  "20Kph",
        1:  "30Kph",
        2:  "50Kph",
        3:  "60Kph",
        4:  "70Kph",
        5:  "80Kph",
        6:  "100Kph",
        7:  "120Kph",
        8:  "Yield",
        9:  "Stop",
        10: "End All",
        11: "Unknown"
    }

    df_cm = pd.read_csv(f'results/{model_name}_prediction.csv')
    df_cm.columns = list(mapping.values())
    df_cm = df_cm.rename(index=mapping)

    tensors_dict = torch.load(f'results/{model_name}_tensors.pt')
    predictions = tensors_dict['predictions']
    target = tensors_dict['target']

    plt.figure(figsize=(10, 10))
    sn.set(font_scale=1)
    hm = sn.heatmap(df_cm, vmin=0, vmax=20, annot=True, fmt='g', linewidths=.3, square=True)  # font size
    hm.set(xlabel="predictions", ylabel="target")  # for label size

    plt.savefig(f'results/{model_name}_cm_plot_{time.strftime("%Y%m%d-%H%M")}.png')
    # plt.show()

    # Accuracy
    glob_acc = MulticlassAccuracy(average='micro', num_classes=numClasses)
    glob_acc.update(predictions, target)
    print(f"\nGloabal accuracy: {round(glob_acc.compute().data.item(), 4)}\n")

    pc_acc = MulticlassAccuracy(average=None, num_classes=numClasses)
    pc_acc.update(predictions, target)
    # print(f"\nPer class accuracy:")
    pc_acc = pc_acc.compute().tolist()
    # for i, e in enumerate(pc_acc):
    #     print(f"\tClass {i+1}: {round(e, 4)}")

    # Precision
    glob_prec = MulticlassPrecision(average='micro', num_classes=numClasses)
    glob_prec.update(predictions, target)
    print(f"\nGloabal precision: {round(glob_prec.compute().data.item(), 4)}\n")

    pc_prec = MulticlassPrecision(average=None, num_classes=numClasses)
    pc_prec.update(predictions, target)
    # print(f"\nPer class precision:")
    pc_prec = pc_prec.compute().tolist()
    # for i, e in enumerate(pc_prec):
    #     print(f"\tClass {i + 1}: {round(e, 4)}")

    # Recall
    glob_rec = MulticlassRecall(average='micro', num_classes=numClasses)
    glob_rec.update(predictions, target)
    print(f"\nGloabal recall: {round(glob_rec.compute().data.item(), 4)}\n")

    pc_rec = MulticlassRecall(average=None, num_classes=numClasses)
    pc_rec.update(predictions, target)
    # print(f"\nPer class accuracy:")
    pc_rec = pc_rec.compute().tolist()
    # for i, e in enumerate(pc_rec):
    #     print(f"\tClass {i + 1}: {round(e, 4)}")

    df_dict = {
        "Accuracy": pc_acc,
        "Precision": pc_prec,
        "Recall": pc_rec
    }
    apr_df = pd.DataFrame(df_dict)

    f = open(f'results/{model_name}_AccPrecRec_{time.strftime("%Y%m%d-%H%M")}.txt', "a")

    f.write(apr_df.to_markdown())


if __name__ == '__main__':
    args = parser.parse_args()

    plot(args.model_name)

    exit()
