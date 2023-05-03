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


COLORS = ((244, 67, 54),
          (233, 30, 99),
          (156, 39, 176),
          (103, 58, 183),
          (63, 81, 181),
          (33, 150, 243),
          (3, 169, 244),
          (0, 188, 212),
          (0, 150, 136),
          (76, 175, 80),
          (139, 195, 74),
          (205, 220, 57),
          (255, 235, 59),
          (255, 193, 7),
          (255, 152, 0),
          (255, 87, 34),
          (121, 85, 72),
          (158, 158, 158),
          (96, 125, 139))


def getGTimages():
    ann_dirs = [
        'data/annotations/train',
        # 'data/annotations/val',
        # 'data/annotations/test'
    ]
    imgs_dirs = [
        'data/images/train',
        # 'data/images/val',
        # 'data/images/test',
    ]
    res_dirs = [
        'data/results/train',
        # 'data/results/val',
        # 'data/results/test',
    ]

    for img_set, ann_set, res_set in zip(imgs_dirs, ann_dirs, res_dirs):
        print('Starting %s' % img_set)
        for folder in os.listdir(ann_set):
            print('Analyzing %s' % folder)
            if not os.path.exists(os.path.join(res_set, folder)):
                os.mkdir(os.path.join(res_set, folder))
            for filename in tqdm(os.listdir(os.path.join(ann_set, folder))):
                f = filename.split('_')
                img_path = os.path.join(img_set, folder, f[0]+'_'+f[1]+'_'+f[2]+'_leftImg8bit.png')
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    json_ann = json.load(open(os.path.join(ann_set, folder, filename)))
                    for element in json_ann['objects']:
                        if element['label'] == 'car':
                            x1 = element['2d']['amodal'][0]
                            x2 = element['2d']['amodal'][0] + element['2d']['amodal'][2]
                            y1 = element['2d']['amodal'][1]
                            y2 = element['2d']['amodal'][1] + element['2d']['amodal'][3]
                            color = random.sample(COLORS, 1)[0]
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            text_str = '%.2f m' % (element['3d']['center'][0] - element['3d']['dimensions'][0])
                            font_face = cv2.FONT_HERSHEY_DUPLEX
                            font_scale = 0.6
                            font_thickness = 1

                            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                            text_pt = (x1, y1 - 3)
                            text_color = [255, 255, 255]

                            cv2.rectangle(img, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                            cv2.putText(img, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                                        cv2.LINE_AA)
                    res_filename = os.path.join(res_set, folder, f[0]+'_'+f[1]+'_'+f[2]+'_distanceGT.png')
                    cv2.imwrite(res_filename, img)


if __name__ == '__main__':
    if not os.path.isdir('data/results'):
        os.mkdir('data/results')
    getGTimages()
