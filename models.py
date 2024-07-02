import tensorflow as tf
import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, MaxPool2D
from utils import MPIIFaceGazeGenerator, model_plot_informations, load_data
from keras_spiking import ModelEnergy
import nengo_dl
import nengo
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

class KerasGazeModel():
    """ANN models using keras"""

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def create_model(self):
        """Wrapper constructor for the estimation model(s)"""
        self.gaze_estimation_model = alexNet(self.input_shape, self.output_shape)
        self.gaze_estimation_model.summary()
        return self.gaze_estimation_model

    def compile(self, optimizer='adam', loss='mean_absolute_error'):
        """Wrapper for keras compile function"""
        self.gaze_estimation_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def train(self, dataset_dir, batch_size, n_epochs, train_split=0.8):
        """Loads dataset and trains the model"""
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)

        train_image_paths, train_annotations, eval_image_paths, eval_annotations  = load_data(dataset_dir, train_split)

        train_generator = MPIIFaceGazeGenerator(train_image_paths, train_annotations, batch_size)
        eval_generator  = MPIIFaceGazeGenerator(eval_image_paths, eval_annotations, batch_size)

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

    def eval(self, dataset_dir, batch_size, train_split=0.8):

        _, _, eval_image_paths, eval_annotations  = load_data(dataset_dir, train_split)
        eval_generator  = MPIIFaceGazeGenerator(eval_image_paths, eval_annotations, batch_size)

        results = self.gaze_estimation_model.evaluate( eval_generator, verbose = 1)
        print(results)
        
        energy = ModelEnergy(self.gaze_estimation_model, example_data=eval_generator)
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

        # TO BE TESTED
        fig, axes = plt.subplots(3, 3, subplot_kw={'projection': '3d'}, figsize=(15, 15))
        for i in range(3):
            for j in range(3):
                ax = axes[i, j]

                img = eval_image_paths[i * 3 + j]
                vector = eval_annotations[i * 3 + j]
                predicted_vector = self.gaze_estimation_model.predict(eval_image_paths[i * 3 + j])

                ax.imshow(img, extent=[0, 1, 0, 1], aspect='auto')
                ax.plot(
                    vector[0], 
                    vector[1], 
                    vector[2], 
                    color='black', linewidth=2)   

                ax.imshow(img, extent=[0, 1, 0, 1], aspect='auto')
                ax.plot(
                    predicted_vector[0], 
                    predicted_vector[1], 
                    predicted_vector[2], 
                    color='red', linewidth=2)   

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_zlim(0, 1)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

        plt.tight_layout()
        plt.show()

    def predict(self, image):
        prediction = self.gaze_estimation_model.predict(image, verbose = 0)
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

    def convert(self, model, scale_fr=1, syn_time=None):
        """Convert Keras model to nengo_dl network"""
        converter = nengo_dl.Converter(model, 
                                    scale_firing_rates=scale_fr, 
                                    synapse=syn_time,
                                    max_to_avg_pool=True, #max_to_avg_pool=True nota
                                    swap_activations={tf.keras.activations.relu: nengo.SpikingRectifiedLinear()})

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

    def train(self, dataset_dir, n_epochs, train_split=0.8):
        """Load dataset and train model"""

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)

        train_image_paths, train_annotations, eval_image_paths, eval_annotations  = load_data(dataset_dir, train_split)
        
        train_generator = MPIIFaceGazeGenerator(train_image_paths, train_annotations, self.batch_size, nengo=True)
        eval_generator  = MPIIFaceGazeGenerator(eval_image_paths, eval_annotations, self.batch_size, nengo=True)

        #Out of Memory problems
        with self.gaze_estimation_model_net:
            nengo_dl.configure_settings(stateful=False)
            nengo_dl.configure_settings(use_loop=False)

            self.sim.fit(train_generator, 
                        validation_data=eval_generator,
                        verbose=1,
                        epochs=n_epochs, 
                        callbacks=[early_stopping, tensorboard_callback],
                        shuffle=True)

    def load(self, filename):
        self.sim.load_params(filename)

    def save(self, filename):
        self.gaze_estimation_model_net.save_params(filename)

    def eval(self, dataset_dir, batch_size, train_split=0.8):

        _, _, eval_image_paths, eval_annotations  = load_data(dataset_dir, train_split)
        eval_generator  = MPIIFaceGazeGenerator(eval_image_paths, eval_annotations, batch_size, nengo=True)
        results = self.sim.evaluate( eval_generator, verbose = 1)
        print(results)

    def predict(self, image):
        #Ricorda di disabilitare Training (inference_only)
        pass

    def getModel(self):
        return self.gaze_estimation_model_net

def alexNet(input_shape, output_shape):
    """Keras gaze estimation model, AlexNet"""
    inp = Input(shape=input_shape)

    conv1 = Conv2D(96, (11, 11), strides=4, padding='same', activation='relu')(inp)
    maxpool1 = MaxPool2D(pool_size=(3, 3), strides=2)(conv1)

    conv2 = Conv2D(256, (5, 5), padding='same', activation='relu')(maxpool1)
    maxpool2 = MaxPool2D(pool_size=(3, 3), strides=2)(conv2)

    conv3 = Conv2D(384, (3, 3), padding='same', activation='relu')(maxpool2)
    conv4 = Conv2D(384, (3, 3), padding='same', activation='relu')(conv3)
    conv5 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv4)
    conv6 = MaxPool2D(pool_size=(3, 3), strides=2)(conv5)


    flat = Flatten()(conv6)
    dense1 = Dense(4096, activation='relu')(flat)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(4096, activation='relu')(drop1)
    drop2 = Dropout(0.5)(dense2)

    out = Dense(output_shape, activation='linear')(drop2) 

    model = Model(inputs=inp, outputs=out)

    return model

def SpikingAlexNet(input_shape, output_shape):

    with nengo.Network(seed=0) as net:

        net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
        net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
        net.config[nengo.Connection].synapse = None

        neuron_type = nengo.SpikingRectifiedLinear()
        nengo_dl.configure_settings(stateful=False)

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

