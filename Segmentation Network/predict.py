import argparse
import logging
import sys
from config import config
import torch
import os
import SimpleITK as sitk
import numpy as np
from dataset.mask_reader import MaskReader
import scipy.ndimage as ndimage
from skimage import measure, morphology, segmentation
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
    predict(net, dataset, save_dir)

def predict(net, dataset, save_dir):
    net.set_model('eval') #TODO: check net.set_model and added predict type
    net.use_mask = True
    net.use_rcnn = True
    aps = []
    dices = []

def load_image(path_to_img):
    """
    This function load the raw dicom images; we support dcm and mhd extension
    INPUT:
        path_to_image :: the directory that contains the dcm slices OR the mhd file path.
    OUTPUT:
        3D numpy image
    """
    slices = []
    scan_extension = ""
    totalNumSlices = 0
    for filename in os.listdir(path_to_img):
        if filename.endswith(".dcm"):
            logging.debug('Reading file in .dcm format')
            slices.append(os.path.join(path_to_img, filename))
            scan_extension = "dcm"
            totalNumSlices += 1
            logging.info('Add the slice to be precessed in the directory:\t', os.path.join(path_to_img, filename))
        elif filename.endswith(".mhd"):
            logging.debug('Reading file in .mhd format')
            slices.append(os.path.join(path_to_img, filename))
            scan_extension = "mhd"
            logging.info('Add the raw mhd image to be precessed in the directory:\t', os.path.join(path_to_img, filename))
        else:
            logging.error('Unknown input file format; File:\t', os.path.join(path_to_img, filename))
    if len(slices) < 1:
        logging.error('No images found in the directory:\t',path_to_img)

    if scan_extension == "dcm":
        patient_id = path_to_img.basename(path_to_img)
        logging.info('Patient ID:\t',patient_id)
        reader = sitk.ImageSeriesReader()
        itkimage = sitk.ReadImage(reader.GetGDCMSeriesFileNames(path_to_img, patient_id))
        logging.debug('Loaded the dicom image')
    elif scan_extension == "mhd":
        itkimage = sitk.ReadImage(path_to_img)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    logging.debug('Convert the image to numpy array [z,y,x]')
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    logging.info('Dicom Image origin [z,y,x]:\t', numpyOrigin)
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    logging.info('Dicom Image spacing [z,y,x]:\t', numpySpacing)
    return numpyImage # return numpy image


def HU2uint8(image, HU_min=-1200.0, HU_max=600.0, HU_nan=-2000.0):
    """
    Convert HU unit into uint8 values. First bound HU values by predfined min
    and max, and then normalize
    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    HU_min: float, min HU value.
    HU_max: float, max HU value.
    HU_nan: float, value for nan in the raw CT image.
    """
    logging.info("[HyperParameter] HU minimum limit is:\t", HU_min)
    logging.info("[HyperParameter] HU maximum limit is:\t", HU_max)
    logging.debug("HU NAN filling is:\t", HU_nan)
    image_new = np.array(image)
    image_new[np.isnan(image_new)] = HU_nan

    # normalize to [0, 1]
    image_new = (image_new - HU_min) / (HU_max - HU_min)
    image_new = np.clip(image_new, 0, 1)
    logging.info("Normalize the image to [0,1] based on HU_max and HU_min.")
    image_new = (image_new * 255).astype('uint8')
    logging.debug("Multiply the normalized image with 255.")
    return image_new


def generate_markers(image):
    # Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    # Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    # Creation of the Watershed Marker matrix
    marker_watershed = np.zeros((512, 512), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128

    return marker_internal, marker_external, marker_watershed


def watershed_segmentation(image):
    # Creation of the markers:
    marker_internal, marker_external, marker_watershed = generate_markers(image)

    # Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    # Watershed algorithm
    watershed = segmentation.watershed(sobel_gradient, marker_watershed)

    # Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3, 3))
    outline = outline.astype(bool)

    # Performing Black-Tophat Morphology for reinclusion
    # Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    # Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)

    # Use the internal marker and the Outline that was just created to generate the lung filter
    lungfilter = np.bitwise_or(marker_internal, outline)
    # Close holes in the lung filter
    # fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5, 5)), iterations=3)

    # Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image, -2000 * np.ones((512, 512)))
    return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed

def preprocessing(imagePixels):
    """
    The preprocessing code is divided into four main tasks
    First: Convert the numpy images from HU to 8 bit unsigned int uint8
    Second: Creat a lung mask for the dicom image
    Third: Segment the image using the mask
    Fourth: Resample the image to (1*1*1) mm
    ...............................................................
    INPUT: 3D numpy image
    """
    imageNew = HU2uint8(imagePixels)

    masks = []
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    imagePixels[imagePixels == -2000] = 0
    for i in range(imagePixels.shape[0]):
        lung_segmented, lung_lungfilter, lung_outline, lung_watershed, lung_sobel_gradient, \
        lung_marker_internal, lung_marker_external, lung_marker_watershed = watershed_segmentation(imagePixels[i])
        masks.append(lung_lungfilter)
    segMask = np.array(masks, dtype=np.int16)

if __name__ == '__main__':
    main()