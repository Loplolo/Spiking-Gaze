import os
import numpy as np
import pandas as pd
import cv2
from keras.utils import Sequence
import matplotlib.pyplot as plt
import tensorflow as tf
import gc
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import json 
import scipy.io

##
# utils.py
#
# Data generation from dataset and useful methods used by the main program
#
#

class MPIIFaceGazeGenerator(Sequence):
    """ Defines a data generator for the MPIIFaceGaze dataset adapted to
        both nengo_dl and keras input specifications and applying preprocessing."""

    def __init__(self, image_paths, annotations, batch_size, image_size=(224, 224), nengo=False, n_steps=1, shuffle=True):

        """
        Generator initialization

        image_paths : Dataset Images paths list
        annotations : Dataset Annotations list
        batch_size  : Dimensions of batches
        image_size  : (Width, Height) of images 
        nengo       : true if its a Nengo_dl model
        n_steps     : n_steps for Nengo_dl simulator
        shuffle     : true if the data should be shuffled

        """
        self.image_paths = image_paths
        self.annotations = annotations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.nengo = nengo
        self.n_steps = n_steps
        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches"""
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        """
        Gets the next batch

        index : Current batch index

        returns single batch of images and annotations in a tuple
        """
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_annotations = self.annotations[index * self.batch_size:(index + 1) * self.batch_size]
        images, annotations = self._generate_batch(batch_image_paths, batch_annotations)
        return images, annotations

    def on_epoch_end(self):
        """Shuffles data on epoch end and frees memory"""
        if self.shuffle:
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.annotations = self.annotations[indices]
            gc.collect()
            tf.keras.backend.clear_session()

    def _generate_batch(self, batch_image_paths, batch_annotations):
        """
        Generates the next batch applying preprocessing to the images

        batch_image_paths : Batch containing image path strings
        batch_annotations : Batch containing annotations 

        images            : Correctly shaped batch with images
        batch_annotations : Correctly shaped batch with annotations

        """
        images = []
        for image_path in batch_image_paths:
            image = cv2.imread(image_path)

            # Cut out black pixel
            y_nonzero, x_nonzero, _ = np.nonzero(image)
            image = image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

            # Load camera calibration
            calib_path = os.path.join(os.path.dirname(os.path.dirname(image_path)), "Calibration", "Camera.mat") 
            calib_data = load_calibration(calib_path)
            image = undistort_image(image, calib_data["cameraMatrix"], calib_data["distCoeffs"])

            # Image pre-processing
            image = preprocess_image(image, self.image_size)
            images.append(image)
            
        images = np.array(images)

        # Reshaping to match input
        images.reshape(self.batch_size, images.shape[1], images.shape[2], 1)
        batch_annotations = np.array(batch_annotations)

        if self.nengo:
            # Reshape images and annotations for NengoDL 
            images = images.reshape((self.batch_size, 1, -1)) 
            batch_annotations = batch_annotations.reshape((self.batch_size, 1, -1)) 

            # Tiling images for each step
            images = np.tile(images, (1, self.n_steps, 1))
            batch_annotations = np.tile(batch_annotations, (1, self.n_steps, 1))

            if self.nengo:

                return {"input_1":images,
                        "n_steps":np.ones((self.batch_size, self.n_steps), dtype=np.int32), 
                        "conv2d.0.bias":np.zeros((self.batch_size, 96, 1), dtype=np.int32),
                        "conv2d_1.0.bias":np.zeros((self.batch_size, 256, 1), dtype=np.int32),
                        "conv2d_2.0.bias":np.zeros((self.batch_size, 384, 1), dtype=np.int32),
                        "conv2d_3.0.bias":np.zeros((self.batch_size, 384, 1), dtype=np.int32),
                        "conv2d_4.0.bias":np.zeros((self.batch_size, 256, 1), dtype=np.int32),
                        "dense_2.0.bias":np.zeros((self.batch_size, batch_annotations.shape[-1], 1), dtype=np.int32)
                        }, {'probe': batch_annotations}
                
        return images, batch_annotations


def load_data(dataset_dir, train_split, seed=42, load_percentage=1.0):
    """
        Loads dataset informations and extracts annotations

        dataset_dir      Path to dataset directory
        train_split      Percentage of train data to split with eval data
        seed             Shuffle seed to avoid data contaminations
        load_percentage  Percentage of the dataset to load

        Returns the path to train images, train annotations, path to evaluation images
        and evaluation annotations in a tuple
    """
    image_paths = []
    annotations = []

    # Save paths of jpg files in subfolders
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    total_samples = len(image_paths)
    num_samples_to_load = int(total_samples * load_percentage)
    image_paths = image_paths[:num_samples_to_load]

    # Open .txt annotation files and extract gaze vector informations
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.txt'):
                df = pd.read_csv(os.path.join(root, file), sep=" ", header=None)

                # ( 21, 22, 23 ) = Face center = fc
                # ( 24, 25, 26 ) = Gaze Target = gt
                # Gaze direction = gt - fc in CAMERA COORDINATES

                fc = df.iloc[:, 21:24].values
                gt = df.iloc[:, 24:27].values

                annotation = gt - fc
                annotations.extend(annotation)

    annotations = np.array(annotations)
    
    # Load only the wanted percentage of data
    annotations = annotations[:num_samples_to_load]
    image_paths, annotations = shuffle(image_paths, annotations, random_state=seed)

    # Compute train/eval split
    split_index = int(train_split * num_samples_to_load)
    train_image_paths = image_paths[:split_index]
    train_annotations = annotations[:split_index]
    eval_image_paths = image_paths[split_index:]
    eval_annotations = annotations[split_index:]

    return train_image_paths, train_annotations, eval_image_paths, eval_annotations

def load_calibration(calibration_file):
    """ 
    Loads calibration file

    calibration_file :  Path to calibration file

    returns a dictionary containing the loaded calibration informations
    """

    _ , filetype = os.path.splitext(calibration_file)

    if filetype == '.mat':
        with open(calibration_file, 'rb') as file:
            calib_data = scipy.io.loadmat(file)

        return calib_data

def undistort_image(image, camera_matrix, dist_coeffs):
    """
    Fixes distortion in images caused by camera differences

    image         : Input image
    camera_matrix : Camera's intrinsic matrix
    dist_coeffs   : Camera's distortion coefficients
    
    returns the undistorted image
    """
    height, width = image.shape[:2]
    new_camera_matrix, roi  = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted_image = undistorted_image[y : y + h, x : x + w]

    return undistorted_image

def preprocess_image(image, new_size):
    """
    Image preprocessing

    image    : Input image
    new_size : Input image resize goal
    
    returns the processed image
    """
    image = cv2.resize(image, new_size)
    
    # Crop to get a square image
    height, width, _ = image.shape
    square_size = min(height, width)

    if width > height:
        x_start = (width - square_size) // 2
        y_start = 0
    else:
        x_start = 0
        y_start = (height - square_size) // 2

    image = image[y_start:y_start + square_size, x_start:x_start + square_size]

    # Convert in grayscale and apply histogram equalization
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.equalizeHist(image)
    
    # Normalization
    image = image.astype('float32') / 255.0
    
    return image
