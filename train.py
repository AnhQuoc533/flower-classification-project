from model import FlowerClassifier
import argparse
from utils import load_img_data

def get_parser():
    parser = argparse.ArgumentParser(description='Train a neural network on an image dataset (and save its information).')

    # Load and save
    parser.add_argument('data_dir', help="Input directory which contains two sub-folders of dataset: 'train' and 'val'.")
    parser.add_argument('--save_dir', nargs='?', const='', 
                        help='Input directory where information of trained model will be saved.\
                              No argument means saving at the current working directory.')
    parser.add_argument('--arch', choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'], default='vgg19', 
                        help='Choose a VGG model architecture. Default is vgg19.')

    # Set hyper-parameters
    parser.add_argument('--learning_rate', metavar='ALPHA', type=float, default=.001, help='Set learning rate value. Default is 0.001.')
    parser.add_argument('--hidden_units', type=int, nargs='*', default=[], 
                        help='Input multiple integers separated by a single space to design the hidden layers for \
                              the classification part of the model.')
    parser.add_argument('--epochs', type=int, default=10, help='Set the number of epochs. Default is 10.')
    parser.add_argument('--batch_size', type=int, default=32, help='Set the size of each batch. Default is 32.')
    parser.add_argument('--drop_p', metavar='P', type=float, default=.0, help='Set probability for dropout regularization. Default is 0.')

    # Use gpu
    parser.add_argument('--gpu', action='store_true', help='Allow the program to use GPU to train the model. No arguments needed.')

    return parser


if __name__ == '__main__':
    arg_parser = get_parser()
    args = arg_parser.parse_args()
    # print(args)

    # Load data
    data_dir = args.data_dir if args.data_dir[-1] in "/\\" else args.data_dir + '/'
    train = load_img_data(data_dir + 'train', kind='train')
    val = load_img_data(data_dir + 'val', kind='val')

    # Load model
    model = FlowerClassifier(args.arch, args.hidden_units + [len(train.class_to_idx)], args.drop_p)
    
    # Train model
    print('Training model...')
    model.train(train, val, args.learning_rate, args.epochs, args.batch_size, args.gpu, plot_loss=True)

    if args.save_dir is not None:
        print('Saving model...')
        model.save('checkpoint.pth', args.save_dir)
