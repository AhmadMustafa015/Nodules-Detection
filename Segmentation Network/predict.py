import argparse
import logging
import sys
from config import config
import torch
import os
from dataset.mask_reader import MaskReader
this_module = sys.modules[__name__]
logger = logging.getLogger("my logger")
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
    logging.info('Output Directory: ', outputDir)
    weightDir = args.weight
    logging.info('Weights Directory: ', weightDir)
    net = config['net']
    logging.info('Network chosen: ', net)
    device = args.device
    logging.info("Device used: %s" %device)
    # Get the neural network
    net = getattr(this_module, net)
    # Load the weights
    if weightDir:
        logging.info('Loading model from: %s' % weightDir)
        checkpoint = torch.load(weightDir, map_location=torch.device(device))
        epoch = checkpoint['epoch']
        logging.info('Loaded weights epoch number: %s' % epoch)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        print('No model weight file specified')
        logging.error('No model weight file specified')
        return
    # Prepare the output directory
    logging.info('Output directory: ', outputDir)
    print('Input Image directory: ', inputImage)
    print('Output directory: ', outputDir)
    save_dir = os.path.join(outputDir, 'results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Read the input data
    dataset = MaskReader(inputImage, None, config, mode='eval')


if __name__ == '__main__':
    main()