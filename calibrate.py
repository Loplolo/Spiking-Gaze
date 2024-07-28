#!/usr/bin/env python
import cv2
import numpy as np
import os
import sys 
import json
import scipy.io
import argparse

def record_loop(path):
    img_count=0
    video = cv2.VideoCapture(0)
    cv2.namedWindow('display')

    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    ok, frame = video.read()

    if not ok:
        print("Cannot read video")
        sys.exit()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    chessboard_size = (6,9)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []  
    imgpoints = []

    while True:

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.equalizeHist(gray_frame)

        ok, frame = video.read()
        img = frame.copy()

        if not ok:
            break

        k = cv2.waitKey(1) & 0xff

        ret, corners = cv2.findChessboardCorners(gray_frame, chessboard_size, None) 
        cv2.drawChessboardCorners(gray_frame, chessboard_size, corners, ret)

        if ret:

            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray_frame, corners, chessboard_size, (-1, -1), criteria)

            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(gray_frame, chessboard_size, corners, ret)

            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_frame.shape[::-1], None, None)
            
            calibration_data = {
                "cameraMatrix": camera_matrix.tolist(),
                "distCoeffs": dist_coeffs.tolist(),
            }

            os.makedirs(path, exist_ok=True)
            
            scipy.io.savemat(os.path.join(path, 'Camera.mat'), calibration_data)
            print("Calibration data saved")

        cv2.imshow("display", img)

        if k == 27: return -1
        elif k != 255: print(k)

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration_path', type=str, default='./dataset/custom/p00/Calibration', help='Path to save calibration data')
    args = parser.parse_args()
    record_loop(args.calibration_path)
