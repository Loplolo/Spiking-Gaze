#!/usr/bin/env python
import cv2
import numpy as np
import os
import sys 

path = "./calibration/images"                           

def calibrate(path):
    pass
    #TODO

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

    while True:

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok, frame = video.read()
        display = frame.copy()

        if not ok:
            break

        cv2.imshow("display", display)
        k = cv2.waitKey(1) & 0xff
        
        if k == 27: return -1 # ESC pressed, check nel main loop -> break
        elif k == 122:
            # z pressed
            pass

        elif k == 115:
            # s pressed
            img_count += 1
            fname = os.path.join(path, "Frame_{}.jpg".format(img_count))
            cv2.imwrite(fname, gray_frame)

            print(fname + " saved!")

        elif k != 255: print(k)

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    record_loop(path)
    calibrate(path)
