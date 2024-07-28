import argparse
from models import KerasGazeModel, NengoGazeModel
from camera_loop import infer_loop
import tensorflow as tf 
from utils import load_data

##
# @file main.py
#
# @brief Model training and evaluation for artificial and spiking neural networks for Gaze estimation
#
# @mainpage Spiking Gaze
#
# @section description_main Description
# Program to train, infer and evaluate neural network models for 
# Full Face Deep Appearance-Based Gaze Estimation

def main(args):
    """! Main program entry"""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    IMAGE_SIZE = (224, 224, 1) # Half size for less memory usage

    # Fixed seed to avoid data contamination after saving
    dataset = load_data(args.dataset_dir, args.train_split, seed=42)

    # Argument parsing
    match args.type:

        case 'keras':
            # Keras neural network model
            gazeModel = KerasGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size)
            gazeModel.create_model()

        case 'nengo':
            # Trained converted keras model in nengo_dl
            kerasModel = KerasGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size)
            kerasModel.create_model()
            gazeModel = NengoGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size)
            gazeModel.convert(kerasModel.getModel())

            if args.action == "show" or args.action == "webcam":
                gazeModel.batch_size = 1

            gazeModel.create_simulator()

        case 'converted':
            # Keras trained and then converted to nengo_dl
            kerasModel = KerasGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size)
            kerasModel.create_model()

            if (args.load):
                kerasModel.load(args.load)

            gazeModel = NengoGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size)
            gazeModel.convert(kerasModel.getModel(), inference_only=True)

            if args.action == "show" or args.action == "webcam":
                gazeModel.batch_size = 1

            gazeModel.create_simulator()

    gazeModel.compile()

    if (args.load and not args.type == 'converted'):
        gazeModel.load(args.load)

    match args.action:
        case 'train':
            # Train model
            gazeModel.train(dataset, n_epochs=args.epochs)
            if args.save:
                gazeModel.save(args.save)

        case 'eval':
            # Evaluate model
            gazeModel.eval(dataset, args.batch_size)

        case 'show':
            # Show examples from the evaluation dataset
            # With 3d vector plots
            gazeModel.show_predictions(dataset)

        case 'webcam':
            # Start webcam stream to infer
            # gaze direction
            infer_loop(gazeModel, IMAGE_SIZE, calib_path=args.calib_path)

if __name__ == '__main__':

    types = ['nengo', 'keras', 'converted', 'nengo_alt']
    actions = ['train', 'eval', 'webcam', 'show']

    parser = argparse.ArgumentParser(description='Train a Nengo_dl or Keras model on the MPIIFaceGaze dataset.')

    parser.add_argument('type', choices=types,
                        help='choose one of {}'.format(types))
    parser.add_argument('action', choices=actions,
                        help='choose one of {}'.format(actions))
    
    parser.add_argument('--dataset_dir', type=str, default="./dataset/MPIIFaceGaze", help='Path to the dataset directory')
    parser.add_argument('--calib_path', type=str, default="./dataset/custom/p00/Calibration/Camera.mat", help='Path to calibration file for webcam infer')

    parser.add_argument('--train_split', type=float, default=0.8, help='Train split')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')

    parser.add_argument('--save', type=str, help='Save the model to a file')
    parser.add_argument('--load', type=str, help='Path to the model file to load')

    args = parser.parse_args()
    main(args)
