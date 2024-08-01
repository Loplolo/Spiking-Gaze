import argparse
import tensorflow as tf
from models import KerasGazeModel, NengoGazeModel
from camera_loop import infer_loop
from utils import load_data

##
# @file main.py
#
# @brief Model training and evaluation for artificial and spiking neural networks for Gaze estimation
#
# @mainpage Spiking Gaze
#
# @section description_main Description
# Program to train, infer, and evaluate neural network models for 
# Full Face Deep Appearance-Based Gaze Estimation

def main(args):
    """! Main program entry"""

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
   
    IMAGE_SIZE = (224, 224, 1) # Half size for less memory usage
    
    dataset = load_data(args.dataset_dir, args.train_split, seed=42)
    gazeModel = None

    if args.type in ['keras', 'nengo', 'converted']:
        kerasModel = KerasGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size)
        kerasModel.create_model()

    if args.type in ['nengo', 'converted']:
        gazeModel = NengoGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size)
        gazeModel.convert(kerasModel.getModel(), inference_only=args.type == 'converted')

    if args.type == 'keras':
        gazeModel = kerasModel

    if gazeModel and (args.action == "show" or args.action == "webcam"):
        gazeModel.batch_size = 1

    if args.type in ['nengo', 'converted']:
        gazeModel.create_simulator()

    gazeModel.compile()

    if (args.load and not args.type == 'converted'):
        gazeModel.load(args.load)
    
    if args.action == 'train':
        gazeModel.train(dataset, n_epochs=args.epochs)
        if args.save:
            gazeModel.save(args.save)

    elif args.action == 'eval':
        gazeModel.eval(dataset, args.batch_size)

    elif args.action == 'show':
        gazeModel.show_predictions(dataset)

    elif args.action == 'webcam':
        infer_loop(gazeModel, IMAGE_SIZE, calib_path=args.calib_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Nengo_dl or Keras model on the MPIIFaceGaze dataset.')
    parser.add_argument('type', choices=['nengo', 'keras', 'converted'], help='Type of model to use')
    parser.add_argument('action', choices=['train', 'eval', 'webcam', 'show'], help='Action to perform')
    
    parser.add_argument('--dataset_dir', type=str, default="./dataset/MPIIFaceGaze", help='Path to the dataset directory')
    parser.add_argument('--calib_path', type=str, default="./dataset/custom/p00/Calibration/Camera.mat", help='Path to calibration file for webcam infer')
    parser.add_argument('--train_split', type=float, default=0.8, help='Proportion of data to use for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--save', type=str, help='Path to save the model')
    parser.add_argument('--load', type=str, help='Path to load a pre-trained model')

    args = parser.parse_args()
    main(args)
