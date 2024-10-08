import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from keras.utils import Sequence
import gc
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import scipy.io

##
# utils.py
#
# Data generation from dataset and useful methods used by the main program
#
#

class MPIIFaceGazeGenerator(Sequence):
    """ Defines a data generator for the MPIIFaceGaze dataset adapted to
        nengo_dl input specifications and applying preprocessing."""

    def __init__(self, image_paths, gaze_vectors, landmarks, batch_size, image_size, eyes_only=False, n_steps=1, shuffle=True):

        """
        Generator initialization

        image_paths  : Dataset Images paths list
        gaze_vectors : Dataset gaze vectors list
        gaze_vectors : Dataset 6 landmarks list
        batch_size   : Dimension of batches
        image_size   : Goal (Width, Height) of images 
        n_steps      : n_steps for Nengo_dl simulator (SNN only)
        shuffle      : true if the data should be shuffled

        """
        self.image_paths = image_paths
        self.gaze_vectors = gaze_vectors
        self.batch_size = batch_size
        self.landmarks = landmarks
        self.image_size = image_size
        self.shuffle = shuffle
        self.n_steps = n_steps
        self.eyes_only = eyes_only
        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches"""
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        """
        Gets the next batch

        index : Current batch index

        returns single batch of images and gaze_vectors in a tuple
        """
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_gaze_vectors = self.gaze_vectors[index * self.batch_size:(index + 1) * self.batch_size]
        batch_landmarks = self.landmarks[index * self.batch_size:(index + 1) * self.batch_size]

        return self._generate_batch(batch_image_paths, batch_gaze_vectors, batch_landmarks)
         
    def on_epoch_end(self):
        """Shuffles data on epoch end and frees memory"""
        if self.shuffle:
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.gaze_vectors = [self.gaze_vectors[i] for i in indices] 
            self.landmarks = [self.landmarks[i] for i in indices]      
            gc.collect()
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            
    def _generate_batch(self, batch_image_paths, batch_gaze_vectors, batch_landmarks):
        """
        Generates the next batch applying preprocessing to the images

        batch_image_paths  : Batch containing image path strings
        batch_gaze_vectors : Batch containing gaze_vectors 
        batch_landmarks    : Batch containing 6 landmarks list 
        
        """
        images = []

        for image_path, landmarks in zip(batch_image_paths, batch_landmarks):
            img = cv2.imread(image_path)
          
            # Cut out black pixel
            y_nonzero, x_nonzero, _ = np.nonzero(img)
            cut_y = [np.min(y_nonzero), np.max(y_nonzero)]
            cut_x = [np.min(x_nonzero), np.max(x_nonzero)]
            img = img[cut_y[0]:cut_y[1], cut_x[0]:cut_x[1]]

            # Image pre-processing
            img = preprocess_image(img, image_path, self.image_size, landmarks, self.eyes_only, cut_x[0], cut_y[0])
            images.append(img)
            
        images = np.array(images)

        # Reshape images for NengoDL 
        images = images.reshape((self.batch_size, 1, -1))
        images = np.tile(images, (1, self.n_steps, 1))

        # Reshape gaze_vectors for NengoDL
        batch_gaze_vectors = np.array(batch_gaze_vectors) 
        batch_gaze_vectors = batch_gaze_vectors.reshape((self.batch_size, 1, -1)) 
        batch_gaze_vectors = np.tile(batch_gaze_vectors, (1, self.n_steps, 1))

        return {"n_steps": np.ones((self.batch_size, 1), dtype=np.int32), 
                "input_1":images,
                "conv2d.0.bias":np.ones((self.batch_size, 96, 1), dtype=np.int32),
                "conv2d_1.0.bias":np.ones((self.batch_size, 256, 1), dtype=np.int32),
                "conv2d_2.0.bias":np.ones((self.batch_size, 384, 1), dtype=np.int32),
                "conv2d_3.0.bias":np.ones((self.batch_size, 384, 1), dtype=np.int32),
                "conv2d_4.0.bias":np.ones((self.batch_size, 256, 1), dtype=np.int32),
                "dense_2.0.bias":np.ones((self.batch_size, batch_gaze_vectors.shape[-1], 1), dtype=np.int32)
                }, {'probe': batch_gaze_vectors}

def load_data(dataset_dir, test_split, train_split, seed=0, load_percentage=1.0):
    """
    Loads dataset information and extracts gaze_vectors

    dataset_dir     : Path to dataset directory
    train_split     : Percentage of train data out of the train+evaluation data
    test_split      : Percentage of total data to use for testing
    seed            : Shuffle seed
    load_percentage : Percentage of the dataset to load

    Returns the path to train images, train gaze_vectors, path to evaluation images,
    evaluation gaze_vectors, path to test images, and test gaze_vectors in a tuple
    """
    image_paths = []
    gaze_vectors = []
    landmarks_list = []

    # Save paths of jpg files in subfolders
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    total_samples = len(image_paths)
    num_samples_to_load = int(total_samples * load_percentage)
    image_paths = image_paths[:num_samples_to_load]

    # Open .txt annotation files and extract gaze vector information
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.txt'):
                df = pd.read_csv(os.path.join(root, file), sep=" ", header=None)

                # From MPIIFaceGaze
                # http://dx.doi.org/10.1109/TPAMI.2017.2778103
                #
                # Dimension 1: image file path and name.
                # Dimension 2~3: Gaze location on the screen coordinate in pixels, 
                # the actual screen size can be found in the "Calibration" folder.
                # Dimension 4~15: (x,y) position for the six facial landmarks, 
                # which are four eye corners and two mouth corners.
                # Dimension 16~21: The estimated 3D head pose in the camera coordinate system 
                # Dimension 22~24 (fc): Face center in the camera coordinate system, which is averaged 3D location of the 6 focal landmarks face model.
                # Dimension 25~27 (gt): The 3D gaze target location in the camera coordinate system. 
                # The gaze direction can be calculated as gt - fc.
                # Dimension 28: Which eye (left or right) is used for the evaluation subset.
                #

                fc = df.iloc[:, 21:24].values
                gt = df.iloc[:, 24:27].values

                landmarks = df.iloc[:, 3:15].values

                annotation = gt - fc
                gaze_vectors.extend(annotation)
                landmarks_list.extend(landmarks)

    gaze_vectors = np.array(gaze_vectors)

    # Load only the desired percentage of data
    gaze_vectors = gaze_vectors[:num_samples_to_load]
    landmarks_list = landmarks_list[:num_samples_to_load]

    image_paths, gaze_vectors, landmarks_list = shuffle(image_paths, gaze_vectors, landmarks_list, random_state=seed)

    # Train+eval / test split
    test_index = int((1 - test_split) * num_samples_to_load)
    train_and_eval_image_paths = image_paths[:test_index]
    train_and_eval_gaze_vectors = gaze_vectors[:test_index]
    train_and_eval_landmarks = landmarks_list[:test_index]

    test_image_paths = image_paths[test_index:]
    test_gaze_vectors = gaze_vectors[test_index:]
    test_landmarks = landmarks_list[test_index:]

    # Train/eval split
    train_index = int(train_split * len(train_and_eval_image_paths))

    train_image_paths = train_and_eval_image_paths[:train_index]
    train_gaze_vectors = train_and_eval_gaze_vectors[:train_index]
    train_landmarks = train_and_eval_landmarks[:train_index]

    eval_image_paths = train_and_eval_image_paths[train_index:]
    eval_gaze_vectors = train_and_eval_gaze_vectors[train_index:]
    eval_landmarks = landmarks_list[train_index:]

    return train_image_paths, train_gaze_vectors, train_landmarks, \
           eval_image_paths,  eval_gaze_vectors,  eval_landmarks,  \
           test_image_paths,  test_gaze_vectors,  test_landmarks

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

def preprocess_image(image, image_path, new_shape, landmarks, eyes_only, cut_x=0, cut_y=0):
    """
    Image preprocessing

    image     : Input image
    new_shape : Input images resize goal
    landmarks : Input images facial landmarks
    eyes_only : Consider eyes only
    cut_x     : Width cutting offset from original image 
    cut_y     : Height cutting offset from original image 
    returns the processed image
    """
    
    # Align image
    image = align_eyes(image, landmarks, cut_x, cut_y)

    # Undistort image
    calib_path = os.path.join(os.path.dirname(os.path.dirname(image_path)), "Calibration", "Camera.mat") 
    calib_data = load_calibration(calib_path)
    image = undistort_image(image, calib_data["cameraMatrix"], calib_data["distCoeffs"])

    if eyes_only:

        # Get eyes bounding boxes
        left_x1, left_y1, left_x2, left_y2 = landmarks[0:4]
        left_min_x, left_max_x, left_min_y, left_max_y = min(left_x1, left_x2), max(left_x1, left_x2), min(left_y1, left_y2), max(left_y1, left_y2)
        left_width = int((left_max_x - left_min_x) * 1.5)
        left_mid = ((left_min_x + left_max_x) // 2 -cut_x, (left_min_y + left_max_y) // 2-cut_y)
        left_height = int(left_width)

        l_bbox = [
            left_mid[0] - left_width // 2, 
            left_mid[1] - left_height // 2, 
            left_mid[0] + left_width // 2, 
            left_mid[1] + left_height // 2
        ]

        l_bbox = [max(0, x) for x in l_bbox]

        right_x1, right_y1, right_x2, right_y2 = landmarks[4:8]
        right_min_x, right_max_x, right_min_y, right_max_y= min(right_x1, right_x2), max(right_x1, right_x2), min(right_y1, right_y2), max(right_y1, right_y2)
        right_width = int((right_max_x - right_min_x )* 1.5)  
        right_mid = ((right_min_x + right_max_x) // 2-cut_x, (right_min_y + right_max_y) // 2 - cut_y)
        right_height = int(right_width)

        r_bbox = [
            right_mid[0] - right_width // 2, 
            right_mid[1] - right_height // 2, 
            right_mid[0] + right_width // 2, 
            right_mid[1] + right_height // 2
        ]

        r_bbox = [max(0, x) for x in r_bbox]

        # Combine left and right bounding boxes and crop image to
        combined_bbox = [
            min(l_bbox[0], r_bbox[0]),  # x_min
            min(l_bbox[1], r_bbox[1]),  # y_min
            max(l_bbox[2], r_bbox[2]),  # x_max
            max(l_bbox[3], r_bbox[3])   # y_max
        ]
        cropped_image = image[combined_bbox[1]:combined_bbox[3], combined_bbox[0]:combined_bbox[2]]
        cropped_height, cropped_width, _ = cropped_image.shape

        scale = min(new_shape[1] / cropped_height, new_shape[0] / cropped_width)
        new_width = int(cropped_width * scale)
        new_height = int(cropped_height * scale)

        # Resize the cropped image while maintaining aspect ratio
        resized_image = cv2.resize(cropped_image, (new_width, new_height))
        delta_w = new_shape[0] - new_width
        delta_h = new_shape[1] - new_height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # Use edge padding to reduce the impact of padding on the model
        image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_REPLICATE)

    else:

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
    
    # Resize image
    image = cv2.resize(image, new_shape)

    # Convert in grayscale and apply histogram equalization
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.equalizeHist(image)

    # Normalization
    image = image.astype('float32') / 255.0
    return image

def align_eyes(image, landmarks, cut_x, cut_y):
    """
    Aligns the eyes to be at the same position in the image using the provided landmarks

    image     : Input image
    landmarks : Providing the position for each eye in the image
    cut_x     : X cut offset
    cut_y     : Y cut offset
    return the aligned image
    """       

    # Coordinates for the left and right eye centers
    left_eye_coords = ((landmarks[0] + landmarks[2]) / 2  - cut_x, (landmarks[1] + landmarks[3]) / 2 - cut_y)
    right_eye_coords = ((landmarks[4] + landmarks[6]) / 2  - cut_x, (landmarks[5] + landmarks[7]) / 2 - cut_y)

    # Angle between the eye coordinates
    dy = right_eye_coords[1] - left_eye_coords[1]
    dx = right_eye_coords[0] - left_eye_coords[0]
    angle = np.arctan2(dy, dx) * 180. / np.pi

    # Rotate the image around it's center
    rotation_center = (image.shape[1]/2, image.shape[0]/2)

    # Rotation matrix for rotating the face around its center
    M = cv2.getRotationMatrix2D(rotation_center, angle, 1)
    aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    return aligned_image