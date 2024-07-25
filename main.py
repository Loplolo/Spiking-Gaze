import argparse
from models import KerasGazeModel, NengoGazeModel
from camera_loop import infer_loop
import tensorflow as tf 
from utils import load_data

def main(args):

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    IMAGE_SIZE = (224, 224, 1)

    #load dataset, fixed seed to avoid data contaminations after saving
    dataset = load_data(args.dataset_dir, args.train_split, seed=42)

    match args.type:
        case 'nengo_alt':
            gazeModel = NengoGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size)
            gazeModel.create_model()
            gazeModel.create_simulator()

        case 'keras':
            gazeModel = KerasGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size)
            gazeModel.create_model()

        case 'nengo':
            kerasModel = KerasGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size)
            kerasModel.create_model()
            gazeModel = NengoGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size)
            gazeModel.convert(kerasModel.getModel())
            gazeModel.create_simulator()

        case 'converted':
            kerasModel = KerasGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size)
            kerasModel.create_model()

            if (args.load):
                kerasModel.load(args.load)

            gazeModel = NengoGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size)
            gazeModel.convert(kerasModel.getModel(), inference_only=True)
            gazeModel.create_simulator()

    gazeModel.compile()

    if (args.load and not args.type == 'converted'):
        gazeModel.load(args.load)

    match args.action:
        case 'train':
            gazeModel.train(dataset, n_epochs=args.epochs)
            if args.save:
                gazeModel.save(args.save)

        case 'eval':
            gazeModel.eval(dataset, args.batch_size)

        case 'show':
            gazeModel.batch_size = 1
            gazeModel.show_predictions(dataset)

        case 'webcam':
            gazeModel.batch_size = 1
            infer_loop(gazeModel, IMAGE_SIZE, calib_path="calibration/calibration.json")

if __name__ == '__main__':

    types = ['nengo', 'keras', 'converted', 'nengo_alt']
    actions = ['train', 'eval', 'webcam', 'show']

    parser = argparse.ArgumentParser(description='Train a Nengo_dl or Keras model on the MPIIFaceGaze dataset.')

    parser.add_argument('type', choices=types,
                        help='choose one of {}'.format(types))
    parser.add_argument('action', choices=actions,
                        help='choose one of {}'.format(actions))
    
    parser.add_argument('--dataset_dir', type=str, default=".\dataset\MPIIFaceGaze", help='Path to the dataset directory')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train split')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')

    parser.add_argument('--save', type=str, help='Save the model to a file')
    parser.add_argument('--load', type=str, help='Path to the model file to load')

    args = parser.parse_args()
    main(args)
