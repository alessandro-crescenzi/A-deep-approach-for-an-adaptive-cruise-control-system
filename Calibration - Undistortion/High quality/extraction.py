import os

import cv2
from tqdm import tqdm

if __name__ == '__main__':
    out_dir = "extracted_frame"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    video = cv2.VideoCapture("NO20220925-103508-000799.MP4")
    if not video.isOpened():
        print("Error opening video stream or file")
        exit(-2)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frameSize = (int(width), int(height))

    for i in tqdm(range(length)):
        ret, frame = video.read()
        if ret:
            cv2.imwrite(out_dir + "/" + str(i) + '.jpg', frame)

    video.release()