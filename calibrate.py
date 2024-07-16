#!/usr/bin/env python
import cv2
import numpy as np
import os
import sys 
import json

path = "./calibration"                           

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
                "camera_matrix": camera_matrix.tolist(),
                "dist_coeffs": dist_coeffs.tolist()
            }
                
            with open(os.path.join(path, 'calibration.json'), 'w') as json_file:
                json.dump(calibration_data, json_file, indent=4)

            print("Calibration data saved")

        cv2.imshow("display", img)

        if k == 27: return -1
        elif k != 255: print(k)

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    record_loop(path)
