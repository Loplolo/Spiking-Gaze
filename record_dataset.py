
##
# gaze_recorder.py
#
# Implementation of the GazeRecorder class to track and record screen gaze points using Tkinter.
#
#

import cv2
import os
import re
import tkinter as tk
import numpy as np
import argparse
import sys
from utils import load_calibration
import dlib

class GazeRecorder:

    def __init__(self):
        self.screen_gaze_point = None
        self._window_created = False
        self._initialize_tkinter()

    def _initialize_tkinter(self):
        if not self._window_created:
            self.root = tk.Tk()
            self.root.attributes('-fullscreen', True)
            self.canvas = tk.Canvas(self.root, bg='#D3D3D3', highlightthickness=0)
            self.canvas.pack(fill=tk.BOTH, expand=True)
            self.canvas.bind("<Motion>", self._draw_circle)
            self.canvas.bind("<Button-1>", self._on_click)
            self.canvas.bind('<Button-3>', sys.exit)
            self._window_created = True
            self._fade_duration = 3000
            self._steps = 100  
            self._step_duration = self._fade_duration // (2 * self._steps)  
            self._fade_in_progress = False
            self._start_fading()

    def _start_fading(self):
        if not self._fade_in_progress:
            self._fade_in_progress = True
            self._fade_cycle()

    def _fade_cycle(self):
        self._fade_in('black', 'gray', self._steps, self._fade_out)

    def _fade_in(self, start_color, end_color, steps, callback):
        start_rgb = self._color_to_rgb(start_color)
        end_rgb = self._color_to_rgb(end_color)
        for step in range(steps):
            t = step / (steps - 1)
            interpolated_color = self._interpolate_color(start_rgb, end_rgb, t)
            self.root.after(step * self._step_duration, lambda color=self._rgb_to_color(interpolated_color): self.canvas.config(bg=color))
        self.root.after(steps * self._step_duration, callback)

    def _fade_out(self):
        self._fade_in('gray', 'black', self._steps, self._fade_cycle)

    def _color_to_rgb(self, color):
        return tuple(int(self.root.winfo_rgb(color)[i] / 256) for i in range(3))

    def _rgb_to_color(self, rgb):
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

    def _interpolate_color(self, start_rgb, end_rgb, t):
        return (
            int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * t),
            int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * t),
            int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * t)
        )

    def _on_click(self, event):
        self.screen_gaze_point = (event.x, event.y)        
        self.root.quit()

    def _draw_circle(self, event):
        self.canvas.delete("circle")
        self.canvas.create_oval(event.x - 20, event.y - 20, event.x + 20, event.y + 20, outline='red', fill="red", width=2, tags="circle")

    def start_recording(self):
        self._initialize_tkinter()
        self.root.mainloop()
        return self.screen_gaze_point

    def stop_recording(self):
        if self._window_created:
            self.root.quit()
            self.root.destroy()
            self._window_created = False

class FaceProcessor:
    
    def __init__(self, path, calib_folder_path):
        self.path = path
        self.camera = load_calibration(os.path.join(calib_folder_path, "Camera.mat"))
        self.screenSize = load_calibration(os.path.join(calib_folder_path, "screenSize.mat"))
        self.monitorPose = load_calibration(os.path.join(calib_folder_path, "monitorPose.mat"))
        
        # Focal length in pixels from  calibrated camera matrix
        self.fx, self.fy = self.camera["cameraMatrix"][0][0], self.camera["cameraMatrix"][1][1]

        self.video_capture = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.img_index = self._get_initial_img_index()

        if not self.video_capture.isOpened():
            print("Cannot open video")
            sys.exit()

    def _get_initial_img_index(self):
        img_index = 0
        for root, _, files in os.walk(self.path):
            for file in files:
                match = re.search(r'(\d+).jpg', os.path.join(root, file))
                if match:
                    num = int(match.group(1))
                    img_index = max(num, img_index)
        return img_index

    def process_gaze_point(self, gaze_point):
        if gaze_point:
            
            # Adjust gaze point coordinates to be relative to the center of the screen
            sx = (self.screenSize["width_mm"] / self.screenSize["width_pixel"])[0][0]
            sy = (self.screenSize["height_mm"] / self.screenSize["height_pixel"])[0][0]

            # Pixel to mm conversion
            scaling_matrix = np.array([[sx, 0, 0],[0, sy, 0], [0, 0,  1]])
            rotation_matrix, _ = cv2.Rodrigues(self.monitorPose["rvects"])
            translation_vector = self.monitorPose["tvecs"]

            screen_point_3d = np.array([[gaze_point[0], gaze_point[1], 0]], dtype=np.float32)

            # Screen CRS -> Centered Camera CRS
            camera_gaze_point = np.dot(rotation_matrix, np.dot(scaling_matrix, screen_point_3d.T)) + translation_vector

            # Get camera frame dimensions
            cam_width_px = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            cam_height_px = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

            last_face = None
            face_detected = False  
            frame_skip_counter = 0  

            while not face_detected:
                ok, frame = self.video_capture.read()
                if not ok:
                    break

                # Skip some frames to avoid processing the same face multiple times
                frame_skip_counter += 1
                if frame_skip_counter % 10 != 0:
                    continue

                # Convert frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:

                    margin = 30  
                    x -= margin
                    y -= margin
                    w += 2 * margin
                    h += 2 * margin

                    x, y, w, h = max(x, 0), max(y, 0), min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)

                    face_rect = dlib.rectangle(x, y, x + w, y + h)
                    face = frame[y:y + h, x:x + w]

                    if face is not None:

                        # Check if the current face is similar to the last processed face
                        if last_face is not None and np.array_equal(face, last_face):
                            continue

                        last_face = face.copy()

                        # Detect facial landmarks
                        shape = self.predictor(gray, face_rect)

                        six_landmarks = [
                            (shape.part(36).x, shape.part(36).y),  # Left eye outer corner
                            (shape.part(39).x, shape.part(39).y),  # Left eye inner corner
                            (shape.part(42).x, shape.part(42).y),  # Right eye outer corner
                            (shape.part(45).x, shape.part(45).y),  # Right eye inner corner
                            (shape.part(48).x, shape.part(48).y),  # Mouth left corner
                            (shape.part(54).x, shape.part(54).y)   # Mouth right corner
                        ]

                        left_eye = ((six_landmarks[0][0] + six_landmarks[1][0]) / 2, (six_landmarks[0][1] + six_landmarks[1][1]) / 2)
                        right_eye = ((six_landmarks[2][0] + six_landmarks[3][0]) / 2, (six_landmarks[2][1] + six_landmarks[3][1]) / 2)
                        
                        # Approximate pixel -> mm conversion using average eyes distance
                        eye_distance_pixels = ((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)**0.5 
                        average_eye_distance_mm = 63
                        pixels_to_mm = average_eye_distance_mm / eye_distance_pixels

                        # Face center
                        center_x_pixels = sum([pt[0] for pt in six_landmarks]) / len(six_landmarks)
                        center_y_pixels = sum([pt[1] for pt in six_landmarks]) / len(six_landmarks)

                        # Calculate face center distance from the camera center
                        camera_center_pixels = (cam_width_px / 2, cam_width_px / 2)
                        face_center_pixels = (center_x_pixels - camera_center_pixels[0], center_y_pixels - camera_center_pixels[0]) 

                        # Pixel -> mm conversion usign the approximate value
                        face_center_x = face_center_pixels[0] * pixels_to_mm 
                        face_center_y = face_center_pixels[1] * pixels_to_mm 

                        # Approximate face depth
                        distance = ( self.fx * pixels_to_mm )                        

                        # Save the detected face image
                        self.img_index += 1
                        filename = os.path.join(self.path, str(self.img_index).zfill(4) + ".jpg")
                        cv2.imwrite(filename, face)
                        print(f"{filename} saved!")

                        # Save relative filename in the annotations file
                        relative_filename = os.path.relpath(filename, start=os.path.join(filename, '..', '..'))

                        # concatenate landmarks coordinates and format for writing

                        flattened_points = [str(coord) for pt in six_landmarks for coord in pt]
                        points_string = ' '.join(flattened_points)

                        with open(os.path.join(self.path, 'annotations.txt'), 'a') as file:
                            file.write(f"{relative_filename} " +           # Relative path
                                    f"{gaze_point[0]} {gaze_point[1]} " +  # Screen Coordinate 2D 
                                    points_string + " "                    # Flattened corner points
                                    "0 " * 6 +                             # No 3D model informations
                                    f"{face_center_x} {face_center_y} " +  # Face center coordinates
                                    f"{distance} " +                       # Distance
                                    f"{camera_gaze_point[0][0]} " +        # Camera gaze point X
                                    f"{camera_gaze_point[1][0]} " +        # Camera gaze point Y
                                    f"{camera_gaze_point[2][0]}\n")        # Camera gaze point Z
                            
                        print(f"Face center at ({face_center_x}, {face_center_y}, {distance}), Gaze target at ({camera_gaze_point[0][0]}, {camera_gaze_point[1][0]}, {camera_gaze_point[2][0]})")

                        face_detected = True 
                        break  # Break out of the for loop once a face is saved

def main():

    parser = argparse.ArgumentParser(description="Face gaze tracking with distance estimation.")

    parser.add_argument('--dataset_dir', type=str, default='./dataset/custom', help='Directory to save faces.')
    parser.add_argument('--id', type=str, default="p00", help="Person's identificative." )
    parser.add_argument('--day', type=str, default="day01", help="Same day identificative." )

    args = parser.parse_args()

    # Separate different people's and days images with MPIIFaceGaze format
    dataset_path = os.path.join(args.dataset_dir , args.id, args.day)

    os.makedirs(dataset_path, exist_ok=True)

    # Get calibration folder path (MPIIFaceGaze format)
    calib_path =  os.path.join(args.dataset_dir, args.id, "Calibration")

    face_processor = FaceProcessor(dataset_path, calib_path)
    gaze_recorder = GazeRecorder()
        
    while True:
        print("Please click on the screen to record a new gaze point.")
        gaze_point= gaze_recorder.start_recording()

        if gaze_point:
            face_processor.process_gaze_point(gaze_point)

        gaze_recorder.stop_recording()

if __name__ == "__main__":
    main()
