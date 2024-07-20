import sys 
import cv2 
import os
import numpy as np
import re
import matplotlib.pyplot as plt
from utils import preprocess_image, undistort_image, load_camera_calibration

def infer_loop(model, image_size, calib_path):

    if(model.batch_size != 1):
        print("Couldn't predict value, batch_size must be 1 for inference")
        return

    video = cv2.VideoCapture(0)
    cv2.namedWindow('display')

    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    ok, frame = video.read()
    camera_matrix, dist_coeffs, _ , _ = load_camera_calibration(calib_path)
    frame = undistort_image(frame, camera_matrix, dist_coeffs)       

    if not ok:
        print("Cannot read video")
        sys.exit()

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    face = None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    camera_matrix, dist_coeffs = load_camera_calibration(calib_path)

    while True:

        ok, frame = video.read()
        img = undistort_image(img, calib_path, camera_matrix, dist_coeffs)       
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        display = frame.copy()

        if not ok:
            break

        timer = cv2.getTickCount()
        try:
            for (x,y,w,h) in faces:
                cv2.rectangle(display,(x,y),(x+w,y+h),(255,255,0),2)
                center = (x + w // 2, y + h // 2)
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


def record_loop(path):
    
    video = cv2.VideoCapture(0)
    cv2.namedWindow('display')

    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    ok, frame = video.read()

    if not ok:
        print("Cannot read video")
        sys.exit()

    img_count = 0
    for root, _, files in os.walk(path):
        for file in files:
            num = int(re.search('Face_(\d*).jpg', os.path.join(root, file)).group(1)) 
            img_count = num if num > img_count else img_count
        
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        ok, frame = video.read()
        display = frame.copy()

        if not ok:
            break
        timer = cv2.getTickCount()
        try:
            for (x,y,w,h) in faces:
                cv2.rectangle(display,(x,y),(x+w,y+h),(255,255,0),2)
                gray_face = gray_frame[y:y+h, x:x+w]
        except Exception as exc:
            print(exc)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(display, "FPS : " + str(int(fps)), (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 170, 50), 2)

        cv2.imshow("display", display)

        if gray_face:
            cv2.imshow("gray_face", gray_face)

        k = cv2.waitKey(1) & 0xff
        
        if k == 27: return -1 # ESC pressed, check nel main loop -> break
        elif k == 122:
            # z pressed
            pass

        elif k == 115:
            # s pressed
            img_count += 1
            fname = os.path.join(path, "Face_{}.jpg".format(img_count))
            cv2.imwrite(fname, gray_face)

            print(fname + " saved!")

        elif k != 255: print(k)

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    path = "./dataset/custom"
    record_loop(path)