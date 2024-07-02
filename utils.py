import os
import numpy as np
import pandas as pd
import cv2
from keras.utils import Sequence
import matplotlib.pyplot as plt
import tensorflow as tf

class MPIIFaceGazeGenerator(Sequence):
    '''Generatore di dati per Keras'''
    def __init__(self, image_paths, annotations, batch_size, image_size=(448, 448), nengo=False, n_steps=1, shuffle=True):
        self.image_paths = image_paths
        self.annotations = annotations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.nengo = nengo

        self.n_steps = n_steps
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_annotations = self.annotations[index * self.batch_size:(index + 1) * self.batch_size]
        images, annotations = self._generate_batch(batch_image_paths, batch_annotations)
        return images, annotations

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.annotations = self.annotations[indices]

    def _generate_batch(self, batch_image_paths, batch_annotations):
        images = []
        for image_path in batch_image_paths:

            image = cv2.imread(image_path)
            # Crop out black pixels -> non preciso
            y_nonzero, x_nonzero, _ = np.nonzero(image)

            image = image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
            image = cv2.resize(image, self.image_size)

            #Crop to get a square
            height, width, _ = image.shape
            square_size = min(height, width)
            if width > height:
                x_start = (width - square_size) // 2
                y_start = 0
            else:
                x_start = 0
                y_start = (height - square_size) // 2

            image = image[y_start:y_start + square_size, x_start:x_start + square_size]

            # Convert to grayscale
            '''
            Illumination also influences the appearance of the human eye.
            To handle this, researchers usually take gray-scale images rather
            than RGB images as input and apply histogram equalization in the
            gray-scale images to enhance the image.
            '''
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.equalizeHist(image)

            image = image.astype('float32') / 255.0
            images.append(image)

        images = np.array(images)
        images.reshape(self.batch_size, images.shape[1], images.shape[2], 1)
        batch_annotations = np.array(batch_annotations)

        if self.nengo:
            # Reshape images and annotations for NengoDL
            images = images.reshape((self.batch_size, 1, -1)) 
            batch_annotations = batch_annotations.reshape((self.batch_size, 1, -1)) 
            images = np.tile(images, (1, self.n_steps, 1))
            batch_annotations = np.tile(batch_annotations, (1, self.n_steps, 1))
            if self.nengo:
                return ({"input_1":images,
                        "n_steps":np.ones((self.batch_size, self.n_steps), dtype=np.int32), 
                        "conv2d.0.bias":np.ones((self.batch_size, 96, 1), dtype=np.int32),
                        "conv2d_1.0.bias":np.ones((self.batch_size, 256, 1), dtype=np.int32),
                        "conv2d_2.0.bias":np.ones((self.batch_size, 384, 1), dtype=np.int32),
                        "conv2d_3.0.bias":np.ones((self.batch_size, 384, 1), dtype=np.int32),
                        "conv2d_4.0.bias":np.ones((self.batch_size, 256, 1), dtype=np.int32),
                        "dense_2.0.bias":np.ones((self.batch_size, batch_annotations.shape[-1], 1), dtype=np.int32)
                        },
                        {'probe': batch_annotations} )
        
        # Hack from
        # https://github.com/tensorflow/tensorflow/issues/39523#issuecomment-914352213
        def getitem(self, index):
            return self.__getitem__(index)
                
        return images, batch_annotations

def load_data(dataset_dir, train_split):
    image_paths = []
    annotations = []

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

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

    split_index = int(train_split * len(image_paths))
    train_image_paths = image_paths[:split_index]
    train_annotations = annotations[:split_index]
    eval_image_paths = image_paths[split_index:]
    eval_annotations = annotations[split_index:]

    return train_image_paths, train_annotations, eval_image_paths, eval_annotations


def model_plot_informations(history):

    plt.plot(history.history['accuracy'], label='Accuracy (training data)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
    plt.title('Accuracy')
    plt.ylabel('Accuracy value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()
    plt.plot(history.history['loss'], label='Error (training data)')
    plt.plot(history.history['val_loss'], label='Error (validation data)')
    plt.title('Error')
    plt.ylabel('Error value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()

