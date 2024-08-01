import sys 
import cv2 
import os
import numpy as np
import re
import matplotlib.pyplot as plt
from utils import preprocess_image, undistort_image, load_calibration

def infer_loop(model, image_size, calib_path):

    video = cv2.VideoCapture(0)
    cv2.namedWindow('display')

    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    ok, frame = video.read()

    calib_data = load_calibration(calib_path)
    frame = undistort_image(frame, calib_data["cameraMatrix"], calib_data["distCoeffs"])       

    if not ok:
        print("Cannot read video")
        sys.exit()

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    face = None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    while True:

        ok, frame = video.read()
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        display = frame.copy()

        if not ok:
            break

        timer = cv2.getTickCount()
        try:
            for (x, y, w, h) in faces:
                margin = 30  
                x -= margin
                y -= margin
                w += 2 * margin
                h += 2 * margin

                x, y, w, h = max(x, 0), max(y, 0), min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
                cv2.rectangle(display, (x, y), (x + w, y + h), (255, 255, 0), 2)
                
                face = frame[y:y+h, x:x+w]

        except Exception as exc:
            print(exc)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(display, "FPS : " + str(int(fps)), (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 170, 50), 2)
        try:
            
            image = preprocess_image(face, image_size[:2])
            prediction = model.predict(image)

            draw_gaze_vector_3d(ax, prediction)

            cv2.imshow("display", display)

        except Exception as e:
            print(e)

        k = cv2.waitKey(1) & 0xff
        
        if k == 27: return -1
        elif k != 255: print(k)

    cv2.destroyAllWindows()
    cv2.waitKey(0)

    video.release()

def draw_gaze_vector_3d(ax, gaze_vector):

    ax.cla()
    ax.quiver(0, 0, 0,
            gaze_vector[0], gaze_vector[1], gaze_vector[2],
            color='red', arrow_length_ratio=0.1, linewidth=1)

    ax.view_init(elev=-90, azim=-90)  

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.draw()
    plt.pause(0.001)