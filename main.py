import argparse
import tensorflow as tf
import numpy as np
from models import NengoGazeModel
from camera_loop import infer_loop
from utils import load_data

from nengo import SpikingRectifiedLinear
import keras

##
# main.py
#
# Program to train, infer, and evaluate neural network models for 
# Full Face Deep Appearance-Based Gaze Estimation
#
#

# Get repeatable results
import tensorflow as tf
tf.random.set_seed(879372)
np.random.seed(879372) 

# Set memory growth on GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  
def main(args):
    """Main program entry"""

    IMAGE_SIZE = (224, 224, 1) # Half size for less memory usage

    dataset = load_data(args.dataset_dir, args.test_split, args.train_split, seed=879372)
    train_dataset, test_dataset = dataset[:4], dataset[4:]

    if (args.action == "train"):
        args.n_steps = 1   # we present the images only once since it's a non-spiking network being trained

    gazeModel = NengoGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size, n_steps=args.n_steps)
    gazeModel.create_model()

    if (args.type == 'snn' and args.action == "train") or (args.type == "ann"):
        gazeModel.convert(gazeModel.model, scale_fr=args.sfr)
    else:
        gazeModel.convert(gazeModel.model,
                            synapse=args.synapse, 
                            scale_fr=args.sfr, 
                            swap_activations={tf.keras.activations.relu: SpikingRectifiedLinear()})
    
    if (args.action == "show" or args.action == "webcam"):
        gazeModel.batch_size = 1
    gazeModel.create_simulator()
        
    if(args.load):
        gazeModel.load(args.load)

    gazeModel.compile(keras.optimizers.Adam(learning_rate=args.lr))
    
    if args.action == 'train':
        gazeModel.train(train_dataset, n_epochs=args.epochs, patience=args.patience)
        if args.save:
            gazeModel.save(args.save)

    elif args.action == 'test':
        gazeModel.energyEstimates(test_dataset)
        gazeModel.eval(test_dataset, sfr=args.sfr)

    elif args.action == 'show':
        gazeModel.show_predictions(test_dataset)

    elif args.action == 'webcam':
        infer_loop(gazeModel, IMAGE_SIZE, calib_path=args.calib_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Nengo_dl model on the MPIIFaceGaze dataset.')
    parser.add_argument('type', choices=['ann', 'snn'], help='Type of model to use')
    parser.add_argument('action', choices=['train', 'test', 'webcam', 'show'], help='Action to perform')

    parser.add_argument('--dataset_dir', type=str, default="./dataset/MPIIFaceGaze", help='Path to the dataset directory')
    parser.add_argument('--calib_path', type=str, default="./dataset/custom/p00/Calibration/Camera.mat", help='Path to calibration file for webcam infer')

    parser.add_argument('--train_split', type=float, default=0.8, help='Proportion of training+eval data to use for training')
    parser.add_argument('--test_split', type=float, default=0.2, help='Proportion of total data to use for testing')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs with no change before early stopping')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    parser.add_argument('--n_steps', type=int, default=100, help='Nengo_dl number of steps for each image')
    parser.add_argument('--synapse', type=float, default=0.01, help='Nengo_dl synapse filter')
    parser.add_argument('--sfr', type=int, default=100, help='Nengo_dl scale firing rate')

    parser.add_argument('--save', type=str, help='Path to save the model')
    parser.add_argument('--load', type=str, help='Path to load a pre-trained model')

    args = parser.parse_args()
    main(args)
