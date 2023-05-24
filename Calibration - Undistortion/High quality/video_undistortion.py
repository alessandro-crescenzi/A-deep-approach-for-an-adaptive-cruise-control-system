#!/usr/bin/env python
import json
import os
from multiprocessing.pool import ThreadPool
import cv2
import time
import torch
import numpy as np
import argparse
from tqdm import tqdm
import glob
from queue import Queue

parser = argparse.ArgumentParser(description='Camera Calibration')
parser.add_argument('--video', type=str,
                    help="Video filename to calibrate")
parser.add_argument('--video_extension', default="mp4", help="Video filename extension to calibrate")
parser.add_argument('--video_multiframe', default=4, type=int,
                    help='The number of frames to evaluate in parallel to make videos play at higher fps.')


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


def video_undistortion(video_name: str, out_directory: str):
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

    frameSize = (width, height)

    out = cv2.VideoWriter(out_directory + "/" + video_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frameSize)
    running = True
    vid_done = False

    ret, cameraMatrix, dist = get_params_from_json()
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (width, height), 0, (width, height))

    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (width, height), 5)

    def cleanup_and_exit():
        print()
        pool.terminate()
        video.release()
        out.release()
        exit()

    def get_next_frame(vid):
        frames = []
        for idx in range(args.video_multiframe):
            frame = vid.read()[1]
            if frame is None:
                return frames
            frames.append(frame)
        return frames

    def undistortion(frames):
        return [cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR) for frame in frames]

    def preprocessing(frames):
        return [cv2.GaussianBlur(frame, (5, 5), 0) for frame in frames]

    def write_video():
        try:
            nonlocal frame_buffer, running, vid_done

            while running:
                if not frame_buffer.empty():
                    out.write(frame_buffer.get())
        except:
            import traceback
            traceback.print_exc()

    frame_buffer = Queue()

    first_frames = undistortion(get_next_frame(video))

    sequence = [preprocessing, undistortion]
    pool = ThreadPool(processes=len(sequence) + args.video_multiframe + 2)
    pool.apply_async(write_video)
    active_frames = [{'value': f, 'idx': 0} for f in first_frames]

    try:
        while video.isOpened() and running:

            # Start time
            start = time.time()

            if not vid_done:
                next_frames = pool.apply_async(get_next_frame, args=(video,))
            else:
                next_frames = None

            if not (vid_done and len(active_frames) == 0):
                for frame in active_frames:
                    _args = frame['value']
                    frame['value'] = pool.apply_async(sequence[frame['idx']], args=_args)

                # For each frame whose job was the last in the sequence (i.e. for all final outputs)
                for frame in active_frames:
                    if frame['idx'] == 0:
                        frame_buffer.put(frame['value'])

                # Remove the finished frames from the processing queue
                active_frames = [x for x in active_frames if x['idx'] > 0]

                # Finish evaluating every frame in the processing queue and advanced their position in the sequence
                for frame in list(reversed(active_frames)):
                    frame['value'] = frame['value'].get()
                    frame['idx'] -= 1

                    if frame['idx'] == 0:
                        # Split this up into individual threads for prep_frame since it doesn't support batch size
                        active_frames += [{'value': frame['value'][i], 'idx': 0} for i in range(1, len(frame['value']))]
                        frame['value'] = frame['value']

                # Finish loading in the next frames and add them to the processing queue
                if next_frames is not None:
                    frames = next_frames.get()
                    if len(frames) == 0:
                        vid_done = True
                    else:
                        active_frames.append({'value': frames, 'idx': len(sequence) - 1})

    except KeyboardInterrupt:
        print('\nStopping...')

    # End time
    end = time.time()

    # Time elapsed
    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))

    # Calculate frames per second
    # fps = num_frames / seconds
    # print("Estimated frames per second : {0}".format(fps))

    cleanup_and_exit()


if __name__ == '__main__':
    args = parser.parse_args()
    video_ext = args.video_extension
    name = args.video
    out_dir = "calibrated_videos"

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if name is not None:
        video_undistortion(name, out_dir)
