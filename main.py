import argparse
from models import KerasGazeModel, NengoGazeModel
from camera_loop import infer_loop


def main(args):
    IMAGE_SIZE = (224, 224, 1)

    match args.type:
        case 'keras':
            gazeModel = KerasGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size)
            gazeModel.create_model()

        case 'nengo':
            gazeModel = NengoGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size)
            gazeModel.create_model()
            gazeModel.create_simulator()

        case 'converted':
            if (not args.load):
                print("Attention: If you are trying to infer using a converted model remember to set --load")

            kerasModel = KerasGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size)
            kerasModel.create_model()

            if (args.load):
                kerasModel.load(args.load)

            gazeModel = NengoGazeModel(input_shape=IMAGE_SIZE, output_shape=3, batch_size=args.batch_size)
            gazeModel.convert(kerasModel.getModel())
            gazeModel.create_simulator()

    gazeModel.compile()

    if (args.load and not args.type == 'converted'):
        gazeModel.load(args.load)

    match args.action:
        case 'train':
            gazeModel.train(args.dataset_dir, n_epochs=args.epochs)
            if args.save:
                gazeModel.save(args.save)

        case 'eval':
            gazeModel.eval(args.dataset_dir, args.batch_size)

        case 'webcam':
            infer_loop(gazeModel, IMAGE_SIZE)

if __name__ == '__main__':

    types = ['nengo', 'keras', 'converted']
    actions = ['train', 'eval', 'webcam']

    parser = argparse.ArgumentParser(description='Train a Nengo_dl or Keras model on the MPIIFaceGaze dataset.')

    parser.add_argument('type', choices=types,
                        help='choose one of {}'.format(types))
    parser.add_argument('action', choices=actions,
                        help='choose one of {}'.format(actions))
    
    parser.add_argument('--dataset_dir', type=str, default=".\dataset\MPIIFaceGaze", help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')

    parser.add_argument('--save', type=str, help='Save the model to a file')
    parser.add_argument('--load', type=str, help='Path to the model file to load')

    args = parser.parse_args()
    main(args)
