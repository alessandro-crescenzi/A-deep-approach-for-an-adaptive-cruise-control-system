import argparse
import os

from PIL import Image
from tqdm import tqdm
import cv2

parser = argparse.ArgumentParser(description='Frame Extraction')
parser.add_argument('--video', default=None, help="Video filename to extract")


def frame_extraction(video_name: str, out_dir: str):
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

    # Grab a few frames
    for i in tqdm(range(length)):
        ret, frame = video.read()
        if not ret:
            break
        if i % (fps*5) == 0:
            cv2.imwrite(os.path.join(out_dir, f"{i}.jpg"), frame)

    # Release video
    video.release()

    for file in os.listdir(out_dir):
        file_name = file.split('.')[0]
        im1 = Image.open(os.path.join(out_dir, file))
        im1.save(os.path.join(out_dir, file_name+'.png'))
        os.remove(os.path.join(out_dir, file))


if __name__ == '__main__':
    args = parser.parse_args()
    name = args.video

    if name is None:
        exit()

    out_dir = name.split('.')[0]

    if out_dir == '':
        exit()

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))


    frame_extraction(name, out_dir)