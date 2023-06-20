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

REAL_CAR_WIDTH = 1.76
REAL_CAR_HEIGHT = 1.56
REAL_TRUCK_WIDTH = 2.36
REAL_TRUCK_HEIGHT = 3.20
PIXEL_OFFSET = 3
DISTANCE_OFFSET = 1
IMAGE_CENTER = 1024
MAX_OFFSET = 400

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

    error_width = []
    error_height = []
    percentage_error_width = []
    percentage_error_height = []

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

                    x_nearest_frontal_box = 999999
                    is_car = True

                    fx = json_ann['sensor']['fx']
                    fy = json_ann['sensor']['fy']
                    u0 = json_ann['sensor']['u0']

                    for element in json_ann['objects']:
                        if element['label'] == 'car' or element['label'] == 'car':
                            if element['label'] == 'car':
                                is_car = True
                            else:
                                is_car = False
                            temp_x1 = element['2d']['amodal'][0]
                            temp_x2 = element['2d']['amodal'][0] + element['2d']['amodal'][2]
                            x_center = (temp_x1 + temp_x2) / 2

                            if (abs(IMAGE_CENTER-x_center)<abs(IMAGE_CENTER-x_nearest_frontal_box)):
                                x_nearest_frontal_box = x_center

                                x1 = element['2d']['amodal'][0]
                                x2 = element['2d']['amodal'][0] + element['2d']['amodal'][2]
                                y1 = element['2d']['amodal'][1]
                                y2 = element['2d']['amodal'][1] + element['2d']['amodal'][3]
                                real_distance = element['3d']['center'][0] - element['3d']['dimensions'][0]/2

                    #Estimate distance
                    if abs(IMAGE_CENTER-x_center) > MAX_OFFSET:
                        res_filename = os.path.join(res_set, folder, f[0] + '_' + f[1] + '_' + f[2] + '_distanceGT.png')
                        cv2.imwrite(res_filename, img)
                        continue
                    if is_car:
                        estimated_distance_height = fy * REAL_CAR_HEIGHT / (y2 - y1 - PIXEL_OFFSET) - DISTANCE_OFFSET
                        estimated_distance_width = fx * REAL_CAR_WIDTH / (x2 - x1 - PIXEL_OFFSET) - DISTANCE_OFFSET
                    else:
                        estimated_distance_height = fy * REAL_TRUCK_HEIGHT / (y2 - y1 - PIXEL_OFFSET) - DISTANCE_OFFSET
                        estimated_distance_width = fx * REAL_TRUCK_WIDTH / (x2 - x1 - PIXEL_OFFSET) - DISTANCE_OFFSET

                    error_height.append(abs(real_distance - estimated_distance_height))
                    error_width.append(abs(real_distance-estimated_distance_width))

                    percentage_error_width.append(abs(real_distance - estimated_distance_width) / ((real_distance + estimated_distance_width)/2) *100)
                    percentage_error_height.append(abs(real_distance - estimated_distance_height) / ((real_distance + estimated_distance_height)/2) *100)

                    #Draw bounding box
                    color = random.sample(COLORS, 1)[0]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    text_str = '%.2f m -- %.2f m -- %.2f m' % (real_distance, estimated_distance_height, estimated_distance_width)
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

    total_error_width = 0
    total_error_height = 0
    total_percentage_error_width = 0
    total_percentage_error_height = 0
    for i in range (len(error_height)):
        total_error_width += error_width[i]
        total_error_height += error_height[i]
        total_percentage_error_width += percentage_error_width[i]
        total_percentage_error_height += percentage_error_height[i]


    print ("Total Error width: " + str (total_error_width) + " m")
    print ("Total Error height: " + str(total_error_height) + " m")

    print("Mean Error width: " + str(total_error_width / len(percentage_error_width)) + " m")
    print("Mean Error height: " + str(total_error_height / len(percentage_error_height)) + " m")

    print("Percentage Error width: " + str(total_percentage_error_width / len(percentage_error_width)) + " %")
    print("Percentage Error height: " + str(total_percentage_error_height / len(percentage_error_height)) + " %")

if __name__ == '__main__':
    if not os.path.isdir('data/results'):
        os.mkdir('data/results')
    getGTimages()
