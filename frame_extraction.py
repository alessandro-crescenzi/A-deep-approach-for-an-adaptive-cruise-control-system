import cv2
import os

def main():
    newpath = r'C:\Users\Alvaro\Desktop\Computer vision and cognitive systems\frames' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    capture = cv2.VideoCapture(r'C:\Users\Alvaro\Desktop\Computer vision and cognitive systems\examplee.MP4')
    i = 0
    
    while (True):
        success, frame = capture.read()
        if not success:
            break
        cv2.imwrite(os.path.join(newpath, 'frame' + str(i) + '.jpg'), frame)
        i = i + 1
    
    capture.release()

if __name__ == "__main__":
    main()
