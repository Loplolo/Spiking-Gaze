import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')	

import datetime
import cv2
import random  
import os

import tensorflow as tf

import nengo
import nengo_dl

import keras.backend as K
from keras.metrics import Mean
from keras.callbacks import EarlyStopping

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
# prediction using a nengo_dl model
#

class NengoGazeModel():
    """
    Wrapper for Gaze prediction, training and evaluation using a Nengo_dl model
    """

    def __init__(self, input_shape, output_shape, batch_size, n_steps=1):
        """!
        input_shape  : Model's input shape
        output_shape : Model's output shape
        batch_size   : Training and Evaluation batch size
        
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.sim = None
        self.layer_objs_lst = []

    def create_model(self):
        self.model, self.layer_objs_lst = gaze_estimation_model(self.input_shape, self.output_shape)
        self.model.summary()
        return self.model

    def compile(self, optimizer, loss='mean_squared_error', ):
        """
        Wrapper for nengo_dl compile function
        
        optimizer : Optimizer to be used by the model, default is Adam
        loss      : Loss function to be used by the model, default is mean_squared_error
        """
        self.sim.compile(loss={ "probe" : loss } , optimizer=optimizer, metrics={ 'probe' : ['mae', 'mse', R2Score() ,  AngularDistance(),  AngularDistanceSD()]})

    def convert(self, model, scale_fr=1, synapse=None, inference_only=False, swap_activations=None):
        """
        Convert keras model in a nengo_dl one
        
        model            : Keras model to be converted
        scale_fr         : Scales the firing rate of neurons 
        synapse          : Synaptic filter to be applied on the output of all neurons
        inference_only   : Boolean to save memory in case training is not needed
        swap_activations : Swaps activations functions

        returns the nengo_dl.Converter object
        """       
        self.converter = nengo_dl.Converter(model, 
                                    scale_firing_rates=scale_fr, 
                                    synapse=synapse,
                                    inference_only = inference_only,
                                    swap_activations=swap_activations,
                                    allow_fallback=False
                                    )

        self.probes = [self.converter.inputs[self.layer_objs_lst[0]]] # Input layer probe.
        with self.converter.net:
            nengo_dl.configure_settings(stateful=False)

            # Probe for the first Conv layer.
            first_conv_probe = nengo.Probe(self.converter.layers[self.layer_objs_lst[1]])
            self.probes.append(first_conv_probe)

            # Probe for penultimate dense layer.
            penltmt_dense_probe = nengo.Probe(self.converter.layers[self.layer_objs_lst[-2]])
            self.probes.append(penltmt_dense_probe)

        self.probes.append(self.converter.outputs[self.layer_objs_lst[-1]]) # Output layer probe.

        return self.converter
    
    def train(self, dataset, n_epochs, patience):
        """
        Creates dataset generator and trains the model

        dataset  : Tuple containing train image paths, train annotations, eval image_paths and eval annotations
        n_epochs : Number of epochs for training
        """

        log_dir = "logs/nengo_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Tensorboard callback
        tensorboard_summaries_callback =  nengo_dl.callbacks.NengoSummaries(log_dir=log_dir, 
                                                                            sim=self.sim, 
                                                                            objects=self.converter.net.connections)
        
        early_stopping_callback = EarlyStopping(
                                        monitor='val_loss',
                                        patience=patience,
                                        restore_best_weights=True
                                    )
        tensorboard_callback = nengo_dl.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

        # Create data generator
        train_image_paths, train_annotations, train_landmarks, eval_image_paths, eval_annotations, eval_landmarks  = dataset

        train_generator = MPIIFaceGazeGenerator(train_image_paths, train_annotations, train_landmarks , self.batch_size, n_steps=self.n_steps, image_size=self.input_shape[:2])
        eval_generator  = MPIIFaceGazeGenerator(eval_image_paths, eval_annotations, eval_landmarks, self.batch_size, n_steps=self.n_steps, image_size=self.input_shape[:2])

        # Training
        with self.converter.net:
            nengo_dl.configure_settings(stateful=False)

            self.sim.fit(train_generator, 
                        validation_data=eval_generator,
                        verbose = 1,
                        epochs=n_epochs, 
                        callbacks=[tensorboard_callback, tensorboard_summaries_callback, early_stopping_callback],
                        shuffle=True)

    def create_simulator(self):
        """
        Creates and returns a nengo_dl simulator 
        """
        self.sim = nengo_dl.Simulator(self.converter.net, minibatch_size=self.batch_size)
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

    def eval(self, dataset, sfr, collect_spikes_output=True):
        """
        Model evaluation loop

        dataset               : Tuple containing test image paths and test annotations
        collect_spikes_output : boolean true if spikes need to be collected

        returns the collected spikes
        """
        # Create data generator
        test_image_paths, test_annotations  = dataset
        test_generator  = MPIIFaceGazeGenerator(test_image_paths, test_annotations, self.batch_size, n_steps=self.n_steps, image_size=self.input_shape[:2])

        n_images = len(test_image_paths)

        with self.converter.net:
            nengo_dl.configure_settings(stateful=False, trainable=False, keep_history=True)

        # Istantiate metrics
        mae = tf.keras.metrics.MeanAbsoluteError()
        mse = tf.keras.metrics.MeanSquaredError()
        angular_distance = AngularDistance()
        angular_distance_sd = AngularDistanceSD()
        r2_score = R2Score()

        metrics = [mae, mse, angular_distance, angular_distance_sd, r2_score]

        spikes = [] # store spikes to plot later
        curr = 0

        for x_batch_dict, y_batch_dict in test_generator:
            curr += 1
            print("Batch: " + str(curr) + "/" + str(n_images//self.batch_size))

            x_input = x_batch_dict["input_1"]
            y_true = y_batch_dict["probe"][:, -1, :]

            sim_data = self.sim.predict_on_batch({"input_1": x_input})

            y_pred = sim_data[self.probes[-1]][:, -1, :]

            for metric in metrics:
                metric.update_state(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32))
                print(str(metric.name) + " : " + str(tf.keras.backend.get_value(metric.result())))

            if collect_spikes_output:
                # Collecting spikes for each image in the first batch.
                for i in range(self.batch_size):
                    spikes.append({
                        self.probes[1].obj.ensemble.label: sim_data[self.probes[1]][i],
                        self.probes[2].obj.ensemble.label: sim_data[self.probes[2]][i]
                    })

                # Not collecting the spikes for rest of the batches to save memory.
                collect_spikes_output = False

                for probe in self.probes[1:-1]:
                    self.plot_spikes(probe, spikes, sfr)

                # Example plot for the first image
                self.plot_prediction(y_batch_dict["probe"][0], sim_data[self.probes[-1]][0])
                self.plot_metrics_timesteps(tf.cast(y_batch_dict["probe"][0], tf.float32), tf.cast(sim_data[self.probes[-1]][0], tf.float32), metrics)

                plt.show()

    def energyEstimates(self, dataset):
        """
        Model energy estimates

        dataset : Tuple containing test image paths and test annotations
        """
        # Create data generator
        test_image_paths, test_annotations  = dataset
        test_generator  = MPIIFaceGazeGenerator(test_image_paths, test_annotations, self.batch_size, n_steps=self.n_steps, image_size=self.input_shape[:2])

        example_data = test_generator[0][0]['input_1'][:, -1, :]
        example_data = example_data.reshape((self.batch_size, self.input_shape[0], self.input_shape[1], 1)) 

        # Estimate energy
        energy = ModelEnergy(self.model, example_data=example_data)
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
        
    def plot_spikes(self, probe, spikes, sfr, test_data_idx=0, num_neurons=64, dt=0.001):
        """
        Plots the spikes of the layer corresponding to the `probe`.

        probe         : The probe object of the layer whose spikes are to be plotted.
        spikes        : The collected spikes
        test_data_idx : Test image's index for which spikes were generated.
        num_neurons   : Number of random neurons for which spikes are to be plotted.
        dt            : The duration of each timestep. Nengo-DL's default duration is 0.001s.
        """
        lyr_name = probe.obj.ensemble.label
        spikes_matrix = spikes[test_data_idx][lyr_name] * sfr * dt
        neurons = np.random.choice(spikes_matrix.shape[1], num_neurons, replace=False)
        spikes_matrix = spikes_matrix[:, neurons]

        fig, ax = plt.subplots(figsize=(14, 12))
        timesteps = np.arange(self.n_steps)
        for i in range(num_neurons):
            for t in timesteps[np.where(spikes_matrix[:, i] != 0)]:
                ax.plot([t, t], [i+0.5, i+1.5], color="red")

        ax.set_ylim(0.5, num_neurons+0.5)
        ax.set_yticks(list(range(1, num_neurons+1, int(np.ceil(num_neurons/50)))))
        ax.set_xticks(list(range(1, self.n_steps+1, 10)))
        ax.set_ylabel("Neuron Index")
        ax.set_xlabel("Time in $ms$")
        ax.set_title("Layer: %s" % lyr_name)
    
        plt.tight_layout()

    def plot_prediction(self, y_true, y_pred):
        """
        Plots the temporal variability in predicted value.

        y_true   : true annotation repeated for each timestep
        y_pred   : predicted value tiled for each timestep
        """
        fig, ax = plt.subplots(figsize=(14, 12))

        ax.set_title("True value: %s, Pred value: %s" % (y_true[-1], y_pred[-1] ))
        ax.set_xlabel("Time in $ms$")
        ax.set_ylabel("Predicted Value")

        ax.plot(y_true, color='black')
        ax.plot(y_pred, color='red',)

        plt.tight_layout()
        


    def plot_metrics_timesteps(self, y_true, y_pred, metrics):
        metrics_over_time = {metric.name: [] for metric in metrics}
        
        for t in range(y_true.shape[0]):
            for metric in metrics:
                metric.reset_state()  # Reset each time to get singular metric
                metric.update_state(y_true[t], y_pred[t]) 
                metrics_over_time[metric.name].append(metric.result().numpy())  
        
        for metric_name, metric_values in metrics_over_time.items():
            plt.figure(figsize=(10, 6))
            
            plt.plot(metric_values, label=f'{metric_name} over time', color='blue')
            
            final_value = metric_values[-1]
            plt.axhline(y=final_value, color='red', linestyle='--', label=f'Final {metric_name}: {final_value:.4f}')
            
            plt.xlabel("Timesteps")
            plt.ylabel("Metric Value")
            plt.title(f"{metric_name} Over Time")
            plt.legend()
            plt.grid(True)
            

    def show_predictions(self, dataset):
        """
        Plot predictions and ground truth with images side by side

        dataset : Tuple containing test image paths, test annotations
        """
        test_image_paths, test_annotations, test_landmarks = dataset

        fig = plt.figure(figsize=(20, 15))

        for i in range(3):
            for j in range(3):

                index = i * 3 + j

                # Random image selection
                rand_index = random.randint(0, len(test_image_paths) - 1)
                im_path = test_image_paths[rand_index]
                img = cv2.imread(im_path)

                # Image undistortion
                calib_path = os.path.join(os.path.dirname(os.path.dirname(im_path)), "Calibration", "Camera.mat") 
                calib_data = load_calibration(calib_path)
                img = undistort_image(img, calib_data["cameraMatrix"], calib_data["distCoeffs"])

                # Cut out black pixel
                y_nonzero, x_nonzero, _ = np.nonzero(img)
                cut_y = [np.min(y_nonzero), np.max(y_nonzero)]
                cut_x = [np.min(x_nonzero), np.max(x_nonzero)]
                img = img[cut_y[0]:cut_y[1], cut_x[0]:cut_x[1]]

                # Image preprocessing and reshape
                img = preprocess_image(img, (self.input_shape[0], self.input_shape[1]), test_landmarks[rand_index], cut_x[0], cut_y[0])
                inp_img = img.reshape(1, 1, self.input_shape[0] * self.input_shape[1])

                inp_img = np.tile(inp_img, (1, self.n_steps, 1))

                img_ax = fig.add_subplot(3, 6, 2 * index + 2)
                img_ax.imshow(img, cmap='gray') 
                img_ax.axis('off')

                ax = fig.add_subplot(3, 6, 2 * index + 1, projection='3d')

                # Ground truth vector normalization
                vector = test_annotations[rand_index] / np.linalg.norm(test_annotations[rand_index])
         
                ax.quiver(0, 0, 0,
                        vector[0], vector[1], vector[2],
                        color='black', arrow_length_ratio=0.1, linewidth=1)
                
                # Infer gaze vector
                predicted_vector = self.sim.predict({"input_1" : inp_img}, stateful=False, verbose = 0)

                # Extract only the last probed value
                predicted_vector = predicted_vector[self.probes[-1]]
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


    def predict(self, image):
        """
        Wrapper for gaze vector infer

        image   : Input image
        
        returns the normalized predicted gaze vector
        """
        image = image.reshape(1, 1, self.input_shape[0]* self.input_shape[1])
        image = np.tile(image, (1, self.n_steps, 1))

        prediction = self.sim.predict({"input_1" : image}, stateful=False, verbose = 0, n_steps = self.n_steps)
        predicted_vector = prediction[self.probes[-1]]
        predicted_vector = predicted_vector[0][-1]
        predicted_vector = predicted_vector / np.linalg.norm(predicted_vector)

        print(predicted_vector)
        return predicted_vector

def gaze_estimation_model(input_shape, output_shape):
    """
    Keras model for 3D gaze estimation

    Implementation for the AlexNet model using Keras

    input_shape  : model's input shape
    output_shape : model's output shape

    returns the model and a list with the layer objects
    """
    layer_objs_lst = [] # store the layer objects to probe

    inp = tf.keras.layers.Input(shape=input_shape)
    layer_objs_lst.append(inp)

    conv1 = tf.keras.layers.Conv2D(96, (11, 11), strides=4, padding='same', activation='relu', kernel_initializer="he_uniform")(inp)
    avgpool1 = tf.keras.layers.AvgPool2D()(conv1)
    layer_objs_lst.append(conv1)

    conv2 = tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='relu', kernel_initializer="he_uniform")(avgpool1)
    avgpool2 = tf.keras.layers.AvgPool2D()(conv2)
    layer_objs_lst.append(conv2)

    conv3 = tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu', kernel_initializer="he_uniform")(avgpool2)
    layer_objs_lst.append(conv3)

    conv4 = tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu', kernel_initializer="he_uniform")(conv3)
    layer_objs_lst.append(conv4)

    conv5 = tf.keras.layers.Conv2D(64, (1, 1), padding='same', activation='relu', kernel_initializer="he_uniform")(conv4)
    avgpool3 = tf.keras.layers.AvgPool2D()(conv5)
    layer_objs_lst.append(conv5)

    flat = tf.keras.layers.Flatten()(avgpool3)

    dense1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer="he_uniform")(flat)
    layer_objs_lst.append(dense1)

    dense2 = tf.keras.layers.Dense(64, activation='relu', kernel_initializer="he_uniform")(dense1)
    layer_objs_lst.append(dense2)

    out = tf.keras.layers.Dense(output_shape, activation='linear')(dense2) 
    layer_objs_lst.append(out)

    model = tf.keras.Model(inputs=inp, outputs=out)

    return model, layer_objs_lst

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

    def reset_state(self):
        K.batch_set_value([(v, K.zeros_like(v)) for v in self.variables])

class AngularDistance(Mean):
    """
    Keras metric for Angular Distance (from Cosine Similarity),
    considering only the last few steps of the input.
    """
    def __init__(self,  name='angular_distance', **kwargs):
        super(AngularDistance, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Normalize the vectors
        y_true_norm = K.l2_normalize(y_true, axis=-1)
        y_pred_norm = K.l2_normalize(y_pred, axis=-1)
        
        # Compute the cosine similarity
        cosine_similarity = K.sum(y_true_norm * y_pred_norm, axis=-1)
        
        # Clip values to ensure they are within the valid range
        cosine_similarity = tf.clip_by_value(cosine_similarity, -1.0, 1.0)

        # Calculate the angular distance
        angular_distance = tf.math.acos(cosine_similarity)
        
        # Update the state with the computed angular distance
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

    def reset_state(self):
        K.batch_set_value([(v, K.zeros_like(v)) for v in self.variables])