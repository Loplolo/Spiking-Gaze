import numpy as np
import matplotlib.pyplot as plt
import datetime
import cv2
import random  
import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, MaxPool2D

import nengo_dl
import nengo

import keras
import keras.backend as K
from keras.metrics import Mean

from keras_spiking import ModelEnergy

from utils import MPIIFaceGazeGenerator, preprocess_image, undistort_image, load_calibration

##
# models.py
#
# Wrappers for Neural Network models usage
#
# Classes:
#
# KerasGazeModel class to predict, visualize and evaluate gaze direction vector 
# prediction using a keras model
#
# NengoGazeModel class to predict, visualize and evaluate gaze direction vector 
# prediction using a nengo model
#

class KerasGazeModel():
    """
    Wrapper for Gaze prediction, training and evaluation using a Keras model
    """

    def __init__(self, input_shape, output_shape, batch_size):
        """!
        input_shape  : Model's input shape
        output_shape : Model's output shape
        batch_size   : Training and Evaluation batch size
        
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size

    def create_model(self):
        """! 
        Wrapper constructor for the estimation model
        """
        self.gaze_estimation_model = alexNet(self.input_shape, self.output_shape)
        self.gaze_estimation_model.summary()
        return self.gaze_estimation_model

    def compile(self, optimizer='adam', loss='mean_squared_error'):
        """!
        Wrapper for keras compile function
        
        optimizer : Optimizer to be used by the model, default is Adam
        loss      : Loss function to be used by the model, default is mean_squared_error
        """
        self.gaze_estimation_model.compile(optimizer=optimizer, loss=loss, metrics=['mae', 'mse', R2Score(), AngularDistance()])

    def train(self, dataset, n_epochs):
        """!
        Creates dataset generator and trains the model

        dataset  :  Tuple containing train image paths, train annotations, eval image_paths and eval annotations
        n_epochs :  Number of epochs for training
        """

        log_dir = "logs/keras_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Tensorboard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

        # Create data generator
        train_image_paths, train_annotations, eval_image_paths, eval_annotations  = dataset

        train_generator = MPIIFaceGazeGenerator(train_image_paths, train_annotations, self.batch_size)
        eval_generator  = MPIIFaceGazeGenerator(eval_image_paths, eval_annotations, self.batch_size)

        # Training
        self.gaze_estimation_model.fit(train_generator, 
                                        validation_data=eval_generator,
                                        verbose=1,
                                        epochs=n_epochs, 
                                        callbacks=[tensorboard_callback], 
                                        shuffle=True)
        
    def load(self, filename='keras.model'):
        """ 
        Wrapper for load_model

        filename                   : Name of the file to load
        self.gaze_estimation_model : Keras model
        """
        self.gaze_estimation_model = keras.models.load_model(filename)
        return self.gaze_estimation_model
    
    def save(self, filename):
        """!
        Wrapper for save_model

        filename : Name of the saved file
        """
        if (self.gaze_estimation_model):
            self.gaze_estimation_model.save(filename)
        
    def eval(self, dataset, batch_size):
        """
        Model evaluation wrapper and energy estimates

        dataset    : Tuple containing train image paths, train annotations, eval image_paths and eval annotations
        batch_size : Evaluation batch size
        """

        # Create data generator
        _, _, eval_image_paths, eval_annotations  = dataset
        eval_generator  = MPIIFaceGazeGenerator(eval_image_paths, eval_annotations, batch_size)

        # Evaluation
        results = self.gaze_estimation_model.evaluate( eval_generator, verbose = 1)
        print(results)

        # Energy estimates
        energy = ModelEnergy(self.gaze_estimation_model, example_data=eval_generator[0][0])
        energy.summary(
            columns=(
                "name",
                "rate",
                "synop_energy cpu",
                "synop_energy gpu",
                "synop_energy loihi",
                "neuron_energy cpu",
                "synop_energy gpu",
                "neuron_energy loihi",
            ),
            print_warnings=False,)

    def show_predictions(self, dataset):
        """
        Plots predictions and ground truth with images side by side

        dataset : Tuple containing train image paths, train annotations, eval image_paths and eval annotations
        """
        _, _, eval_image_paths, eval_annotations = dataset

        fig = plt.figure(figsize=(20, 15))

        for i in range(3):
            for j in range(3):
                index = i * 3 + j

                # Random image selection
                rand_index = random.randint(0, len(eval_image_paths) - 1)
                im_path = eval_image_paths[rand_index]
                img = cv2.imread(im_path)

                # Image undistortion
                calib_path = os.path.join(os.path.dirname(os.path.dirname(im_path)), "Calibration", "Camera.mat") 
                calib_data = load_calibration(calib_path)
                img = undistort_image(img, calib_data["cameraMatrix"], calib_data["distCoeffs"])

                # Cut out black pixels  
                y_nonzero, x_nonzero, _ = np.nonzero(img)
                img = img[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

                # Image preprocessing and reshape
                img = preprocess_image(img, (self.input_shape[0], self.input_shape[1]))
                inp_img = img.reshape(1, self.input_shape[0], self.input_shape[1], 1)

                img_ax = fig.add_subplot(3, 6, 2 * index + 2)
                img_ax.imshow(img, cmap='gray') 
                img_ax.axis('off')

                ax = fig.add_subplot(3, 6, 2 * index + 1, projection='3d')

                # Ground truth vector normalization
                vector = eval_annotations[rand_index] / np.linalg.norm(eval_annotations[rand_index])
                ax.quiver(0, 0, 0,
                        vector[0], vector[1], vector[2],
                        color='black', arrow_length_ratio=0.1, linewidth=1)

                # Infer gaze vector
                predicted_vector = self.gaze_estimation_model.predict(inp_img)[-1]

                # Predicted truth vector normalization
                predicted_vector = predicted_vector / np.linalg.norm(predicted_vector)
                ax.quiver(0, 0, 0,
                        predicted_vector[0], predicted_vector[1], predicted_vector[2],
                        color='red', arrow_length_ratio=0.1, linewidth=1)

                print(vector)
                print(predicted_vector)

                ax.view_init(elev=-90, azim=-90)  

                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

        plt.tight_layout()
        plt.show()


    def predict(self, image):
        """
        Wrapper for gaze vector infer

        image : Input image

        returns normalized predicted gaze vector
        """
        image = image.reshape(1, self.input_shape[0], self.input_shape[1], 1)
        prediction = self.gaze_estimation_model.predict(image, verbose = 0)
        prediction = prediction[-1] / np.linalg.norm(prediction)

        print(prediction)
        return prediction
    
    def getModel(self):
        """
        Get and returns keras model
        """
        return self.gaze_estimation_model

class NengoGazeModel():
    """
    Wrapper for Gaze prediction, training and evaluation using a Nengo_dl model
    """

    def __init__(self, input_shape, output_shape, batch_size):
        """!
        input_shape  : Model's input shape
        output_shape : Model's output shape
        batch_size   : Training and Evaluation batch size
        
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.sim = None

    def compile(self, optimizer='adam', loss='mean_squared_error', ):
        """
        Wrapper for nengo_dl compile function
        
        optimizer : Optimizer to be used by the model, default is Adam
        loss      : Loss function to be used by the model, default is mean_squared_error
        """
        self.sim.compile(loss=loss, optimizer=optimizer, metrics=['mae', 'mse', R2Score(), AngularDistance()])

    def convert(self, model, scale_fr=1, synapse=None, inference_only=False):
        """
        Convert keras model in a nengo_dl one
        
        model          : Keras model to be converted
        scale_fr       : Scales the inputs of neurons 
        synapse        : Synaptic filter to be applied on the output of all neurons
        inference_only : Boolean to save memory in case training is not needed

        returns the nengo_dl.Converter object
        """       
        converter = nengo_dl.Converter(model, 
                                    scale_firing_rates=scale_fr, 
                                    synapse=synapse,
                                    inference_only = inference_only,
                                    max_to_avg_pool=max_to_avg_pool,                        
                                    swap_activations=swap_activations,
                                    allow_fallback=True
                                    )
        self.gaze_estimation_model_net = converter.net
        return converter
    
    def train(self, dataset, n_epochs):
        """
        Creates dataset generator and trains the model

        dataset  : Tuple containing train image paths, train annotations, eval image_paths and eval annotations
        n_epochs : Number of epochs for training
        """

        log_dir = "logs/nengo_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Tensorboard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Create data generator
        train_image_paths, train_annotations, eval_image_paths, eval_annotations  = dataset

        train_generator = MPIIFaceGazeGenerator(train_image_paths, train_annotations, self.batch_size, nengo=True)
        eval_generator  = MPIIFaceGazeGenerator(eval_image_paths, eval_annotations, self.batch_size, nengo=True)

        # Training
        with self.gaze_estimation_model_net:

            self.sim.fit(train_generator, 
                        validation_data=eval_generator,
                        verbose=1,
                        epochs=n_epochs, 
                        callbacks=[tensorboard_callback],
                        shuffle=True)

    def create_simulator(self):
        """
        Creates and returns a nengo_dl simulator 
        """
        self.sim = nengo_dl.Simulator(self.gaze_estimation_model_net, minibatch_size=self.batch_size)
        return self.sim
    
    def __del__(self):
        """
        Closes the simulator on object destruction
        """
        if(self.sim):
            self.sim.close()

    def load(self, filename):
        """ 
        Wrapper for load_params

        filename : Name of the file to load
        """
        self.sim.load_params(filename)
        
    def save(self, filename):
        """
        Wrapper for save_params

        filename : Name of the file to load
        """
        self.sim.save_params(filename)

    def eval(self, dataset, batch_size):
        """
        Model evaluation wrapper and energy estimates

        dataset    : Tuple containing train image paths, train annotations, eval image_paths and eval annotations
        batch_size : Evaluation batch size
        """
        
        # Create data generator
        _, _, eval_image_paths, eval_annotations  = dataset
        eval_generator  = MPIIFaceGazeGenerator(eval_image_paths, eval_annotations, batch_size, nengo=True)

        # Evaluation
        results = self.sim.evaluate( eval_generator, verbose = 1)
        print(results)

    def show_predictions(self, dataset):
        """
        Plot predictions and ground truth with images side by side

        dataset : Tuple containing train image paths, train annotations, eval image_paths and eval annotations
        """
        _, _, eval_image_paths, eval_annotations = dataset

        fig = plt.figure(figsize=(20, 15))

        for i in range(3):
            for j in range(3):

                index = i * 3 + j

                # Random image selection
                rand_index = random.randint(0, len(eval_image_paths) - 1)
                im_path = eval_image_paths[rand_index]
                img = cv2.imread(im_path)

                # Image undistortion
                calib_path = os.path.join(os.path.dirname(os.path.dirname(im_path)), "Calibration", "Camera.mat") 
                calib_data = load_calibration(calib_path)
                img = undistort_image(img, calib_data["cameraMatrix"], calib_data["distCoeffs"])

                # Cut out black pixels
                y_nonzero, x_nonzero, _ = np.nonzero(img)
                img = img[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

                # Image preprocessing and reshape
                img = preprocess_image(img, (self.input_shape[0], self.input_shape[1]))
                inp_img = img.reshape(1, 1, self.input_shape[0] * self.input_shape[1])

                img_ax = fig.add_subplot(3, 6, 2 * index + 2)
                img_ax.imshow(img, cmap='gray') 
                img_ax.axis('off')

                ax = fig.add_subplot(3, 6, 2 * index + 1, projection='3d')

                # Ground truth vector normalization
                vector = eval_annotations[rand_index] / np.linalg.norm(eval_annotations[rand_index])
         
                ax.quiver(0, 0, 0,
                        vector[0], vector[1], vector[2],
                        color='black', arrow_length_ratio=0.1, linewidth=1)
                
                # Infer gaze vector
                predicted_vector = self.sim.predict({"input_1" : inp_img})

                # Extract only the last probed value
                predicted_vector = predicted_vector[self.gaze_estimation_model_net.probes[0]]
                predicted_vector = predicted_vector[0][-1]

                # Predicted truth vector normalization
                predicted_vector = predicted_vector / np.linalg.norm(predicted_vector)

                print(vector)
                print(predicted_vector)

                ax.quiver(0, 0, 0,
                        predicted_vector[0], predicted_vector[1], predicted_vector[2],
                        color='red', arrow_length_ratio=0.1, linewidth=1)

                ax.view_init(elev=-90, azim=-90)  

                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

        plt.tight_layout()
        plt.show()
        self.sim.close()

    def predict(self, image):
        """
        Wrapper for gaze vector infer

        image : Input image

        returns the normalized predicted gaze vector
        """
        image = image.reshape(1, 1, self.input_shape[0]* self.input_shape[1])

        prediction = self.sim.predict({"input_1" : image}, verbose = 0)
        predicted_vector = prediction[self.gaze_estimation_model_net.probes[0]]
        predicted_vector = predicted_vector[0][-1]
        predicted_vector = predicted_vector / np.linalg.norm(predicted_vector)

        print(predicted_vector)
        return predicted_vector
    
    def getModel(self):
        """! 
        Get and returns the Nengo_dl network
        """
        return self.gaze_estimation_model_net

def alexNet(input_shape, output_shape):
    """
    AlexNet keras model

    Implementation for the AlexNet model using Keras

    input_shape  : model's input shape
    output_shape : model's output shape
    """
    inp = Input(shape=input_shape)

    conv1 = Conv2D(96, (11, 11), strides=4, padding='same', activation='relu', bias_initializer='zeros')(inp)
    maxpool1 = MaxPool2D(pool_size=(3, 3), strides=2)(conv1)

    conv2 = Conv2D(256, (5, 5), padding='same', activation='relu', bias_initializer='zeros')(maxpool1)
    maxpool2 = MaxPool2D(pool_size=(3, 3), strides=2)(conv2)

    conv3 = Conv2D(384, (3, 3), padding='same', activation='relu', bias_initializer='zeros')(maxpool2)
    conv4 = Conv2D(384, (3, 3), padding='same', activation='relu', bias_initializer='zeros')(conv3)
    conv5 = Conv2D(256, (3, 3), padding='same', activation='relu', bias_initializer='zeros')(conv4)
    conv6 = MaxPool2D(pool_size=(3, 3), strides=2)(conv5)

    flat = Flatten()(conv6)
    dense1 = Dense(4096, activation='relu', bias_initializer='zeros')(flat)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(4096, activation='relu', bias_initializer='zeros')(drop1)
    drop2 = Dropout(0.5)(dense2)

    out = Dense(output_shape, activation='linear', bias_initializer='zeros')(drop2) 
    model = Model(inputs=inp, outputs=out)

    return model

class AngularDistanceSD(tf.keras.metrics.Metric):
    """
    Keras metric to compute the standard deviation of the angular distances
    """
    def __init__(self, name='angular_distance_sd', **kwargs):
        super(AngularDistanceSD, self).__init__(name=name, **kwargs)
        self.sum_angles = self.add_weight(name='sum_angles', initializer='zeros')
        self.sum_squared_angles = self.add_weight(name='sum_squared_angles', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_norm = K.l2_normalize(y_true, axis=-1)
        y_pred_norm = K.l2_normalize(y_pred, axis=-1)
        cosine_similarity = K.sum(y_true_norm * y_pred_norm, axis=-1)
        
        cosine_similarity = tf.clip_by_value(cosine_similarity, -1.0, 1.0)
        
        angles = tf.math.acos(cosine_similarity)
        
        self.sum_angles.assign_add(K.sum(angles))
        self.sum_squared_angles.assign_add(K.sum(K.square(angles)))
        
        if sample_weight is not None:
            self.count.assign_add(K.sum(sample_weight))
        else:
            self.count.assign_add(K.cast(K.shape(y_true)[0], tf.float32))

    def result(self):
        mean_angle = self.sum_angles / self.count
        mean_square_angle = self.sum_squared_angles / self.count
        variance_angle = mean_square_angle - K.square(mean_angle)
        
        variance_angle = K.maximum(variance_angle, 0.0)
        
        return K.sqrt(variance_angle)

    def reset_states(self):
        K.batch_set_value([(v, K.zeros_like(v)) for v in self.variables])

class AngularDistance(Mean):
    """
    Keras metric for Angular Distance (from Cosine Similarity)
    """
    def __init__(self, name='angular_distance', **kwargs):
        super(AngularDistance, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_norm = K.l2_normalize(y_true, axis=-1)
        y_pred_norm = K.l2_normalize(y_pred, axis=-1)
        cosine_similarity = K.sum(y_true_norm * y_pred_norm, axis=-1)
        
        cosine_similarity = tf.clip_by_value(cosine_similarity, -1.0, 1.0)

        angular_distance = tf.math.acos(cosine_similarity)
        
        return super(AngularDistance, self).update_state(angular_distance, sample_weight)

class R2Score(tf.keras.metrics.Metric):
    """
    Keras metric for the coefficient of determination, R2 Score
    """
    def __init__(self, name='r2_score', **kwargs):
        super(R2Score, self).__init__(name=name, **kwargs)
        self.ss_res = self.add_weight(name='ss_res', initializer='zeros')
        self.ss_tot = self.add_weight(name='ss_tot', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        residuals = y_true - y_pred
        ss_res = K.sum(K.square(residuals))
        self.ss_res.assign_add(ss_res)

        y_true_mean = K.mean(y_true)
        total = y_true - y_true_mean
        ss_tot = K.sum(K.square(total))
        self.ss_tot.assign_add(ss_tot)

        if sample_weight is not None:
            self.count.assign_add(K.sum(sample_weight))
        else:
            self.count.assign_add(K.cast(K.shape(y_true)[0], tf.float32))

    def result(self):
        ss_res = self.ss_res / self.count
        ss_tot = self.ss_tot / (self.count - 1)
        r2_score = K.constant(1, dtype=tf.float32) - ss_res / ss_tot
        return r2_score

    def reset_states(self):
        K.batch_set_value([(v, K.zeros_like(v)) for v in self.variables])