from model import VGGModel
import argparse
import json
from PIL import Image
from utils import get_transforms, plot_prediction

def get_parser():
    parser = argparse.ArgumentParser(description='Train a new VGG neural network on a image dataset.')

    # Load image and checkpoint
    parser.add_argument('img_path', help="Input the path to the image which will be predicted by the model.")
    parser.add_argument('checkpoint', help="Input the path to the checkpoint file which contains trained model's information.")

    # Optional arguments
    parser.add_argument('--topk', metavar='K', type=int, default=1, help='Input the number of top classes to be displayed.')
    parser.add_argument('--category_names', metavar='JSON', help="Input the path to the JSON file which is a mapping of categories to real names.")
    parser.add_argument('--gpu', action='store_true', help='Allow the program to use GPU to train the model. No arguments needed.')

    return parser


if __name__ == '__main__':
    arg_parser = get_parser()
    args = arg_parser.parse_args()
    # print(args)

    # Load and pre-process image
    preprocess = get_transforms('test')
    img = Image.open(args.img_path)
    data = preprocess(img)

    # Load model
    model = VGGModel(file=args.checkpoint)
    
    # Get prediction from model
    top_p, top_cls = model.predict(data, args.topk, args.gpu)

    # Get image name
    img_name = args.img_path.rsplit('.', maxsplit=1)[0]
    img_name = img_name.replace('\\', '/')
    img_name = img_name.rsplit('/', maxsplit=1)[-1]

    # Map classes to real names
    if args.category_names is not None:
        with open(args.category_names, 'r') as f:
            cls_to_name = json.load(f)
        top_cls = [cls_to_name[cls] for cls in top_cls]

    # Display result
    plot_prediction(top_p, top_cls, img, img_name)
