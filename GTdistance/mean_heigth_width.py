import argparse
import json
import os
import random

import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Ground Truth distance CS')
parser.add_argument('--input-dir', type=str, default='data/test_images', metavar='path',
                    help="folder where images to be labebel are located")
parser.add_argument('--ann', type=str, metavar='path',
                    help="folder where labels of images are located")

def getGTimages():
    ann_dirs = [
        'data/annotations/train',
        # 'data/annotations/val',
        # 'data/annotations/test'
    ]

    height_sum_car = 0
    height_sum_truck = 0
    width_sum_car = 0
    width_sum_truck = 0
    num_car = 0
    num_truck = 0

    for img_set in ann_dirs:
        print('Starting %s' % img_set)
        for folder in os.listdir(img_set):
            print('Analyzing %s' % folder)

            for filename in tqdm(os.listdir(os.path.join(img_set, folder))):
                f = filename.split('_')

                json_ann = json.load(open(os.path.join(img_set, folder, filename)))

                for element in json_ann['objects']:
                    if element['label'] == 'car':
                        num_car += 1
                        width_sum_car += element['3d']['dimensions'][1]
                        height_sum_car += element['3d']['dimensions'][2]

                    if element['label'] == 'truck':
                        num_truck += 1
                        width_sum_truck += element['3d']['dimensions'][1]
                        height_sum_truck += element['3d']['dimensions'][2]

    print("Mean Height Car: " + str(height_sum_car/num_car))
    print("Mean Width Car: " + str(width_sum_car / num_car))
    print("Mean Height Truck: " + str(height_sum_truck / num_truck))
    print("Mean Width Truck: " + str(width_sum_truck / num_truck))


if __name__ == '__main__':
    if not os.path.isdir('data/results'):
        os.mkdir('data/results')
    getGTimages()
