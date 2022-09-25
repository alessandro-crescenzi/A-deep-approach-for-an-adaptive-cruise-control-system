import json
from dataclasses import field
from xmlrpc.client import boolean
import numpy as np
import cv2
import glob
import argparse
import os

parser = argparse.ArgumentParser(description='Camera Calibration')
parser.add_argument('--chess_size', type=int, nargs='+',
                    help='Chessboard size in term of number of corners', required=True)
parser.add_argument('--file_extension', default="jpg", type=str, help='Image file extension')
parser.add_argument('--with_renaming', default=False,
                    help='It renames image files with standard names before correction')
parser.add_argument('--resize', default=100, type=int,
                    help='Percentage of the original image')


def calibration(chessboardSize, fileExtension, resize_percentage):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('*.jpg')
    out_dir = "./calibration_results"
    os.mkdir(out_dir)

    tmp = cv2.imread(images[0])
    width = int(tmp.shape[1] * resize_percentage / 100)
    height = int(tmp.shape[0] * resize_percentage / 100)
    frameSize = (width, height)

    n_images = len(images)

    if n_images == 0:
        print('No image found... (check file extension)')
        exit(-1)

    for image in images:
        print(image)
        img = cv2.imread(image)
        if resize_percentage != 100:
            img = cv2.resize(img, frameSize)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv2.imwrite(out_dir + "/" + image + '_corners.' + fileExtension, img)

    # print("Object points: ", objpoints)
    # print("Image points: ", imgpoints)

    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    print("Camera Calibrated: ", ret)
    print("\nCamera Matrix: ", cameraMatrix)
    print("\nDistorsion Parameters: ", dist)
    print("\nRotation Vectors: ", rvecs)
    print("\nTranslation Vectors: ", tvecs)

    ######################### UNDISTORTION #######################################

    img = cv2.imread(images[0])
    h, w = img.shape[:2]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 0, (w, h))

    # Undistort
    dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
    # crop the image
    x, y, w, h = roi
    # dst = dst[y:y + h, x:x + w]
    cv2.imwrite(out_dir + "/" + 'calibresult.' + fileExtension, dst)

    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
    dst2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    # dst2 = dst2[y:y + h, x:x + w]
    cv2.imwrite(out_dir + "/" + 'calibresult2.' + fileExtension, dst2)

    ###################### UNDISTORTION ERROR ###################

    # Reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: ", mean_error / len(objpoints))

    ###################### WRITE CALIBRATION RESULTS IN A FILE ###########################

    camera_dict = {
        'Camera calibrated': ret,
        'Camera matrix': cameraMatrix.tolist(),
        'Distorsion Parameters': dist.tolist()
    }

    with open("parameters.json", "w") as outfile:
        json.dump(camera_dict, outfile)

    f = open(out_dir + "/" + "calibration_results.txt", "w")
    f.write("----- CALIBRATION RESULTS -----")
    f.write("\n\nCamera Calibrated: " + str(ret))
    f.write("\n\nCamera Matrix:\n\n" + str(cameraMatrix))
    f.write("\n\nDistorsion Parameters:\n\n" + str(dist))
    f.write("\n\nRotation Vectors:\n\n" + str(rvecs))
    f.write("\n\nTranslation Vectors:\n\n" + str(tvecs))
    f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    chessboardSize = tuple(args.chess_size)
    fileExtension = args.file_extension
    resize = args.resize
    calibration(chessboardSize, fileExtension, resize)
