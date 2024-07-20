import tensorflow as tf
import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, MaxPool2D
from utils import MPIIFaceGazeGenerator, preprocess_image, undistort_image
from keras_spiking import ModelEnergy
import nengo_dl
import nengo
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import datetime
import cv2
import random  
import os

class KerasGazeModel():
    """ANN models using keras"""

    def __init__(self, input_shape, output_shape, batch_size):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size

    def create_model(self):
        """Wrapper constructor for the estimation model(s)"""
        self.gaze_estimation_model = alexNet(self.input_shape, self.output_shape)
        self.gaze_estimation_model.summary()
        return self.gaze_estimation_model

    def compile(self, optimizer='adam', loss='mean_absolute_error'):
        """Wrapper for keras compile function"""
        self.gaze_estimation_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def train(self, dataset, n_epochs, train_split=0.8):
        """Create dataset generator and trains the model"""
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        log_dir = "logs/keras_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

        train_image_paths, train_annotations, eval_image_paths, eval_annotations  = dataset

        train_generator = MPIIFaceGazeGenerator(train_image_paths, train_annotations, self.batch_size)
        eval_generator  = MPIIFaceGazeGenerator(eval_image_paths, eval_annotations, self.batch_size)

        self.gaze_estimation_model.fit(train_generator, 
                                        validation_data=eval_generator,
                                        verbose=1,
                                        epochs=n_epochs, 
                                        callbacks=[early_stopping, tensorboard_callback],
                                        shuffle=True)
        
    def load(self, filename='keras.model'):
        """Wrapper for load_model"""
        self.gaze_estimation_model = keras.models.load_model(filename)
        return self.gaze_estimation_model
    
    def save(self, filename):
        """Wrapper for save_model"""
        if (self.gaze_estimation_model):
            self.gaze_estimation_model.save(filename)
        
    def eval(self, dataset, batch_size):

        _, _, eval_image_paths, eval_annotations  = dataset
        eval_generator  = MPIIFaceGazeGenerator(eval_image_paths, eval_annotations, batch_size)

        results = self.gaze_estimation_model.evaluate( eval_generator, verbose = 1)
        print(results)

        energy = ModelEnergy(self.gaze_estimation_model, example_data=eval_generator[0][0])
        energy.summary(
            columns=(
                "name",
                "rate",
                "synop_energy cpu",
                "synop_energy loihi",
                "neuron_energy cpu",
                "neuron_energy loihi",
            ),
            print_warnings=False,)

    def show_predictions(self, dataset):
        """Plot predictions and ground truth with images side by side"""
        _, _, eval_image_paths, eval_annotations = dataset

        fig = plt.figure(figsize=(20, 15))

        for i in range(3):
            for j in range(3):
                index = i * 3 + j
                rand_index = random.randint(0, len(eval_image_paths) - 1)
                im_path = eval_image_paths[rand_index]
                img = cv2.imread(im_path)


                calib_path = os.path.dirname(os.path.dirname(im_path)) + "\Calibration\Camera.mat"
                camera_matrix, dist_coeffs, rvecs, tvecs = load_camera_calibration(calib_path)
                img = undistort_image(img, camera_matrix, dist_coeffs)

                # Cut out black pixels  
                y_nonzero, x_nonzero, _ = np.nonzero(img)
                img = img[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

                img = preprocess_image(img, (self.input_shape[0], self.input_shape[1]))

                inp_img = img.reshape(1, self.input_shape[0], self.input_shape[1], 1)

                img_ax = fig.add_subplot(3, 6, 2 * index + 2)
                img_ax.imshow(img, cmap='gray') 
                img_ax.axis('off')

                ax = fig.add_subplot(3, 6, 2 * index + 1, projection='3d')

                vector = eval_annotations[rand_index] / np.linalg.norm(eval_annotations[rand_index])
                ax.quiver(0, 0, 0,
                        vector[0], vector[1], vector[2],
                        color='black', arrow_length_ratio=0.1, linewidth=1)

                predicted_vector = self.gaze_estimation_model.predict(inp_img)[-1]

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
        image = image.reshape(1, self.input_shape[0], self.input_shape[1], 1)
        prediction = self.gaze_estimation_model.predict(image, verbose = 0)
        prediction = prediction[-1] / np.linalg.norm(prediction)

        print(prediction)
        return prediction
    
    def getModel(self):
        return self.gaze_estimation_model

class NengoGazeModel():
    """SNN models using nengo_dl"""

    def __init__(self, input_shape, output_shape, batch_size):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.sim = None
        
    def create_model(self):
        with nengo.Network() as net:
            self.gaze_estimation_model_net = SpikingAlexNet(self.input_shape, self.output_shape)
            return self.gaze_estimation_model_net

    def convert(self, model, scale_fr=1, synapse=None, inference_only=False):
        """Convert Keras model to nengo_dl network"""
        converter = nengo_dl.Converter(model, 
                                    scale_firing_rates=scale_fr, 
                                    synapse=synapse,
                                    inference_only = inference_only,
                                    )
        self.gaze_estimation_model_net = converter.net
        return converter
   
    def create_simulator(self):
        """Construct simulator"""
        self.sim = nengo_dl.Simulator(self.gaze_estimation_model_net, minibatch_size=self.batch_size)
        return self.sim
    
    def __del__(self):
        """Destructor"""
        if(self.sim):
            self.sim.close()

    def compile(self, optimizer='adam', loss='mean_absolute_error', ):
        self.sim.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def train(self, dataset, n_epochs):
        """Create generator and train model"""

        early_stopping = EarlyStopping(monitor='val_probe_loss', patience=10, restore_best_weights=True)
        log_dir = "logs/nengo_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        train_image_paths, train_annotations, eval_image_paths, eval_annotations  = dataset
        
        train_generator = MPIIFaceGazeGenerator(train_image_paths, train_annotations, self.batch_size, nengo=True)
        eval_generator  = MPIIFaceGazeGenerator(eval_image_paths, eval_annotations, self.batch_size, nengo=True)

        with self.gaze_estimation_model_net:

            self.sim.fit(train_generator, 
                        validation_data=eval_generator,
                        verbose=1,
                        epochs=n_epochs, 
                        callbacks=[early_stopping, tensorboard_callback],
                        shuffle=True)

    def load(self, filename):
        self.sim.load_params(filename)

    def save(self, filename):
        self.sim.save_params(filename)

    def eval(self, dataset, batch_size):

        _, _, eval_image_paths, eval_annotations  = dataset
        eval_generator  = MPIIFaceGazeGenerator(eval_image_paths, eval_annotations, batch_size, nengo=True)

        results = self.sim.evaluate( eval_generator, verbose = 1)
        print(results)

    def show_predictions(self, dataset):
        """Plot predictions and ground truth with images side by side"""

        _, _, eval_image_paths, eval_annotations = dataset

        fig = plt.figure(figsize=(20, 15))

        if(self.batch_size != 1):
            print("Couldn't predict value, batch_size must be 1 for inference with nengo_dl models")
            return

        for i in range(3):
            for j in range(3):

                index = i * 3 + j
                rand_index = random.randint(0, len(eval_image_paths) - 1)
                im_path = eval_image_paths[rand_index]
                img = cv2.imread(im_path)

                calib_path = os.path.dirname(os.path.dirname(im_path)) + "\Calibration\Camera.mat"
                camera_matrix, dist_coeffs, rvecs, tvecs = load_camera_calibration(calib_path)
                img = undistort_image(img, camera_matrix, dist_coeffs)
                
                # Cut out black pixels
                y_nonzero, x_nonzero, _ = np.nonzero(img)
                img = img[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

                img = preprocess_image(img, (self.input_shape[0], self.input_shape[1]))

                inp_img = img.reshape(1, 1, self.input_shape[0] * self.input_shape[1])

                img_ax = fig.add_subplot(3, 6, 2 * index + 2)
                img_ax.imshow(img, cmap='gray') 
                img_ax.axis('off')

                ax = fig.add_subplot(3, 6, 2 * index + 1, projection='3d')

                vector = eval_annotations[rand_index] / np.linalg.norm(eval_annotations[rand_index])
         
                ax.quiver(0, 0, 0,
                        vector[0], vector[1], vector[2],
                        color='black', arrow_length_ratio=0.1, linewidth=1)
                
                predicted_vector = self.sim.predict({"input_1" : inp_img})

                # Extract only the last probed value
                predicted_vector = predicted_vector[self.gaze_estimation_model_net.probes[0]]
                predicted_vector = predicted_vector[0][-1]
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

        image = image.reshape(1, 1, self.input_shape[0]* self.input_shape[1])

        prediction = self.sim.predict({"input_1" : image}, verbose = 0)
        predicted_vector = prediction[self.gaze_estimation_model_net.probes[0]]
        predicted_vector = predicted_vector[0][-1]
        predicted_vector = predicted_vector / np.linalg.norm(predicted_vector)

        print(predicted_vector)
        return predicted_vector
    

    def getModel(self):
        return self.gaze_estimation_model_net

def alexNet(input_shape, output_shape):

    """Keras gaze estimation model, AlexNet"""

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

def SpikingAlexNet(input_shape, output_shape):

    with nengo.Network() as net:

        net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
        net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
        net.config[nengo.Connection].synapse = None

        neuron_type = nengo.SpikingRectifiedLinear()

        inp = nengo.Node(np.zeros(np.prod(input_shape)))

        x = nengo_dl.Layer(tf.keras.layers.Conv2D(
            filters=96, kernel_size=11, strides=4, padding='same'))(inp, shape_in=input_shape)
        x = nengo_dl.Layer(neuron_type)(x)

        x = nengo_dl.Layer(tf.keras.layers.AveragePooling2D(
            pool_size=(3, 3), strides=2))(x, shape_in=(56, 56, 96))

        x = nengo_dl.Layer(tf.keras.layers.Conv2D(
            filters=256, kernel_size=5, padding='same'))(x, shape_in=(28, 28, 96))
        x = nengo_dl.Layer(neuron_type)(x)

        x = nengo_dl.Layer(tf.keras.layers.AveragePooling2D(
            pool_size=(3, 3), strides=2))(x, shape_in=(28, 28, 256))

        x = nengo_dl.Layer(tf.keras.layers.Conv2D(
            filters=384, kernel_size=3, padding='same'))(x, shape_in=(14, 14, 256))
        x = nengo_dl.Layer(neuron_type)(x)

        x = nengo_dl.Layer(tf.keras.layers.Conv2D(
            filters=384, kernel_size=3, padding='same'))(x, shape_in=(14, 14, 384))
        x = nengo_dl.Layer(neuron_type)(x)

        x = nengo_dl.Layer(tf.keras.layers.Conv2D(
            filters=256, kernel_size=3, padding='same'))(x, shape_in=(14, 14, 384))
        x = nengo_dl.Layer(neuron_type)(x)

        x = nengo_dl.Layer(tf.keras.layers.AveragePooling2D(
            pool_size=(3, 3), strides=2))(x, shape_in=(14, 14, 256))

        x = nengo_dl.Layer(tf.keras.layers.Flatten())(x, shape_in=(6, 6, 256))

        x = nengo_dl.Layer(tf.keras.layers.Dense(4096))(x, shape_in=(6*6*256,))
        x = nengo_dl.Layer(neuron_type)(x)
        x = nengo_dl.Layer(tf.keras.layers.Dropout(0.5))(x, shape_in=(4096,))

        x = nengo_dl.Layer(tf.keras.layers.Dense(4096))(x, shape_in=(4096,))
        x = nengo_dl.Layer(neuron_type)(x)
        x = nengo_dl.Layer(tf.keras.layers.Dropout(0.5))(x, shape_in=(4096,))

        out = nengo_dl.Layer(tf.keras.layers.Dense(output_shape))(x, shape_in=(4096,))
        out_p = nengo.Probe(out, label="out_p")

    return net
