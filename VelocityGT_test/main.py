import json
import argparse
import re
import time
from tqdm import tqdm
import cv2

parser = argparse.ArgumentParser(description='Camera Calibration')
parser.add_argument('--video', type=str,
                    help='Video to modify')
parser.add_argument('--offset', type=int,
                    help='number of second to wait to print the new velocity info at the first frame. If offset > 0,'
                         'the video in behind the first info velocity, otherwise the video is in adavance')


def get_time(el):
    h = int(el['timestamp'].split(' ')[1].split(':')[0])
    m = int(el['timestamp'].split(' ')[1].split(':')[1])
    s = int(el['timestamp'].split(' ')[1].split(':')[2])

    return h * 10000 + m * 100 + s


if __name__ == '__main__':
    args = parser.parse_args()

    video_name = args.video
    offset = args.offset

    if video_name is None:
        print("Insert video name")
        exit(-1)

    with open('GPSData.json') as json_file:
        data = json.load(json_file)

    name = re.search(r'NO(?P<time>\S+)', video_name).groupdict()['time']
    video_info = data[name]

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

    frameSize = (int(width), int(height))

    out = cv2.VideoWriter('result.MP4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frameSize)

    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 2
    font_thickness = 2
    x1 = 0
    y1 = 0

    # Start time
    start = time.time()

    # Grab a few frames
    for i in tqdm(range(length)):
        ret, frame = video.read()
        if not ret:
            break

        text_str_prv = ''
        el = video_info[max(0, int(i / fps) - offset)]

        try:
            text_str = 'SPEED: %d km/h' % el['velocity']
            text_str_prv = text_str
        except IndexError:
            text_str = text_str_prv

        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

        text_pt = (x1, y1 + text_h + 20)
        text_color = [255, 255, 255]

        cv2.rectangle(frame, (x1, y1), (x1 + text_w, y1 + text_h + 40), [0, 0, 0], -1)
        cv2.putText(frame, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                    cv2.LINE_AA)
        out.write(frame)

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
