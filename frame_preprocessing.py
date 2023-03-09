import cv2
import os

def main():
    newpath = r'PATH' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    capture = cv2.VideoCapture(r'PATH')
    i = 0
    
    while (True):
        success, frame = capture.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        preprocessed = cv2.inpaint(blur, mask, 15, cv2.INPAINT_TELEA)
        cv2.imwrite(os.path.join(newpath, 'frame' + str(i) + '.jpg'), preprocessed)
        cv2.imwrite(os.path.join(newpath, 'mask' + str(i) + '.jpg'), mask)
        i += 1
    
    capture.release()

if __name__ == "__main__":
    main()
