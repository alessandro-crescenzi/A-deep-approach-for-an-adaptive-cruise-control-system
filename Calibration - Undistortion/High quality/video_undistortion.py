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


parser = argparse.ArgumentParser(description='Camera Calibration')
parser.add_argument('--resize', default=100, type=int,
                    help='Percentage of the original image')
parser.add_argument('--video', nargs='+', help="Video filename to calibrate")
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

    # Start time
    start = time.time()

    # Grab a few frames
    for _ in tqdm(range(length)):
        ret, frame = video.read()
        if not ret:
            break
        # print(f"Return value:   {ret}"
        # frameSize = (640, 480)
        # frame = cv2.resize(frame, frameSize)
        # dst = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
        dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        # Preprocessing
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        # Create the mask for the inpainting
        mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        # Gaussian filtering
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        adjusted = cv2.inpaint(blur, mask, 15, cv2.INPAINT_TELEA)
        # Contrast control (1.0-3.0)
        alpha = 1.5
        # Brightness control (0-100)
        beta = 0

        dst = cv2.convertScaleAbs(adjusted, alpha=alpha, beta=beta)
        if resize_perc != 100:
            dst = cv2.resize(dst, frameSize)
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
    inp_dir = args.input_dir
    out_dir = "calibrated_videos"
    videos = []

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if name is not None:
        videos = np.asarray(name).tolist()
    else:
        videos += glob.glob('*.' + video_ext)

    for vid in videos:
        video_undistortion(vid, out_dir, resize_percentage)

    videos = []

    if inp_dir is not None:
        videos = glob.glob(inp_dir+'/*.'+video_ext)
    for vid in videos:
        video_undistortion(inp_dir+'/'+vid, out_dir, resize_percentage)
