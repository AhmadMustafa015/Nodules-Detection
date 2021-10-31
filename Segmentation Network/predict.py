import argparse
import logging
import sys
from config import config
import torch
this_module = sys.modules[__name__]

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=config['preprocessed_data_dir'],
                    help="Input image to predict the nodules in it")
parser.add_argument('--out_dir', type=str, default=config['out_dir'],
                    help="path to save the results")
parser.add_argument('--weight', type=str, default=config['initial_checkpoint'],
                    help="path to model weights to be used")
parser.add_argument('--device', type=str, default='GPU',
                    help="Run the model on CPU or GPU")
def main():
    loggingLevel = logging.DEBUG
    logFileDir = './predict.log'
    logging.basicConfig(filename=logFileDir,format='[%(levelname)s][%(asctime)s] %(message)s',
                        level=loggingLevel)
    # Passing input argument
    args = parser.parse_args()
    inputImage = args.input
    logging.info('Input Image Directory: ', inputImage)
    outputDir = args.out_dir
    logging.info('Output Image Directory: ', outputDir)
    weightDir = args.weight
    logging.info('Weights Directory: ', weightDir)
    net = config['net']
    logging.info('Network chosen: ', net)
    device = args.device
    # Get our neural network
    net = getattr(this_module, net)
    if weightDir:
        logging.info('Loading model from: %s' % weightDir)
        checkpoint = torch.load(weightDir, map_location=torch.device(device))




if __name__ == '__main__':
    main()