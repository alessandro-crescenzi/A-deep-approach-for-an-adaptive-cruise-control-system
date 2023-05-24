#!/usr/bin/env python
import json
import os
import cv2
import time
import torch
import numpy as np
import argparse
from tqdm import tqdm
import glob
from frr.core import FastReflectionRemoval
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Camera Calibration')
parser.add_argument('--resize', default=100, type=int,
                    help='Percentage of the original image')
parser.add_argument('--video', type=str,
                    help="Video filename to calibrate")
parser.add_argument('--video_extension', default="mp4", help="Video filename extension to calibrate")
parser.add_argument('--input_dir', help="Input directory")


def get_params_from_json():
    try:
        f = open("parameters.json", "r")
    except IOError:
        print("Error: Json file with camera parameters does not appear to exist.")
        return -1

    data = json.load(f)

    ret = data['Camera calibrated']
    camera_matrix = np.asarray(data['Camera matrix'])
    dist = np.asarray(data['Distorsion Parameters'])

    return ret, camera_matrix, dist


def video_undistortion(video_name: str, dir: str, resize_perc: int):
    # Start default camera
    video = cv2.VideoCapture(video_name)
    if not video.isOpened():
        print("Error opening video stream or file")
        exit(-2)

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # Get params of video
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frameSize = (int(width * resize_perc / 100), int(height * resize_perc / 100))

    out = cv2.VideoWriter(dir + "/" + video_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frameSize)

    ret, cameraMatrix, dist = get_params_from_json()
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (width, height), 0, (width, height))

    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (width, height), 5)

    N, M = frameSize

    alg = FastReflectionRemoval(h=0.03, M=M, N=N)

    # Start time
    start = time.time()

    # Grab a few frames
    for _ in tqdm(range(length)):
        ret, frame = video.read()
        if not ret:
            break

        dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        if resize_perc != 100:
            dst = cv2.resize(dst, frameSize)

        # FRR algorithm
        dst = dst / 255
        dst = alg.remove_reflection(dst)
        dst = dst * 255
        # Gaussian filter
        dst = cv2.GaussianBlur(dst, (7, 7), 0)
        # # Contrast control (1.0-3.0)
        alpha = 1
        # Brightness control (0-100)
        beta = 0
        dst = cv2.convertScaleAbs(dst, alpha=alpha, beta=beta)

        # dst = cv2.GaussianBlur(dst, (5, 5), 0)

        out.write(dst)

    # End time
    end = time.time()

    # Time elapsed
    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))

    # Calculate frames per second
    # fps = num_frames / seconds
    # print("Estimated frames per second : {0}".format(fps))

    # Release video
    video.release()
    out.release()


if __name__ == '__main__':
    args = parser.parse_args()
    video_ext = args.video_extension
    name = args.video
    resize_percentage = args.resize
    out_dir = "calibrated_videos"

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if name is not None:
        video_undistortion(name, out_dir, resize_percentage)
