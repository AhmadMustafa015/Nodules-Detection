import argparse
import logging
import math
import ntpath
import sys

import cv2
from net.nodule_net import NoduleNet
from config import config
import torch
import os
import SimpleITK as sitk
import numpy as np
from dataset.mask_reader import MaskReader
import scipy.ndimage as ndimage
from skimage import measure, morphology, segmentation
import torch.nn.functional as F
from utils.util import average_precision, crop_boxes2mask_single
from config import train_config, data_config, net_config, config
from lungmask import mask
import datetime
from PIL import Image, ImageDraw
from pydicom.encaps import encapsulate
from pydicom.uid import generate_uid, JPEGExtended
from pydicom._storage_sopclass_uids import SecondaryCaptureImageStorage
import glob
import pydicom as dicom


this_module = sys.modules[__name__]
logger = logging.getLogger("my-logger")
logger.propagate = False

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=config['preprocessed_data_dir'],
                    help="Input image to predict the nodules in it")
parser.add_argument('--out_dir', type=str, default=config['out_dir'],
                    help="path to save the results")
parser.add_argument('--weight', type=str, default=config['initial_checkpoint'],
                    help="path to model weights to be used")
parser.add_argument('--device', type=str, default='gpu',
                    help="Run the model on CPU or GPU")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #force using one GPU only


def main():
    loggingLevel = logging.ERROR
    logFileDir = './predict.log'
    logging.basicConfig(filename=logFileDir, format='[%(levelname)s][%(asctime)s] %(message)s',
                        level=loggingLevel)
    ################### PASSING INPUT ARGUMENT #######################
    args = parser.parse_args()
    inputImage = args.input
    logging.info('Input Image Directory: %s' % inputImage)
    outputDir = args.out_dir
    logging.info('Output Directory: %s' % outputDir)
    weightDir = args.weight
    logging.info('Weights Directory: %s' % weightDir)
    net = config['net']
    logging.info('Network chosen: %s' % net)
    device = args.device
    logging.info("Device used: %s" % device)
    ################### GET THE NEURAL NETWORK #######################
    net = getattr(this_module, net)(net_config)
    if device == 'gpu':
        net = net.cuda()
    ################### LOAD NUERAL NETWORK WIEGHTS #######################
    if weightDir:
        logging.info('Loading model from: %s' % weightDir)
        checkpoint = torch.load(weightDir, map_location=torch.device('cuda'))
        epoch = checkpoint['epoch']
        logging.info('Loaded weights epoch number: %s' % epoch)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        print('No model weight file specified')
        logging.error('No model weight file specified')
        return
    # Prepare the output directory
    ################### PREPARE THE OUTPUT DIRECTORY #######################
    logging.info('Output directory: %s' % outputDir)
    print('Input Image directory: %s' % inputImage)
    print('Output directory: %s' % outputDir)
    save_dir = os.path.join(outputDir, 'results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Read the input data
    ################### PREPARE THE INPUT DATA #######################
    logging.info("LOADING THE INPUT IMAGE ...")
    imageNumpy, imageSpacing, patient_id, itkimage = load_image(inputImage)
    logging.info("START PREPROCESSING THE INPUTTED IMAGE ...")
    preprocessedImage = preprocessing(imageNumpy, imageSpacing,itkimage,inputImage)


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
            logging.info('Add the slice to be precessed in the directory:\t %s' % os.path.join(path_to_img, filename))
        elif filename.endswith(".mhd"):
            logging.debug('Reading file in .mhd format')
            slices.append(os.path.join(path_to_img, filename))
            scan_extension = "mhd"
            logging.info('Add the raw mhd image to be precessed in the directory:\t %s' %
                         os.path.join(path_to_img, filename))
        else:
            logging.error('Unknown input file format; File:\t %s' % os.path.join(path_to_img, filename))
    if len(slices) < 1:
        logging.error('No images found in the directory:\t %s' % path_to_img)

    if scan_extension == "dcm":
        patient_id = path_to_img.split('/')[-1]
        logging.info('Patient ID:\t %s' % patient_id)
        reader = sitk.ImageSeriesReader()
        itkimage = sitk.ReadImage(reader.GetGDCMSeriesFileNames(path_to_img, patient_id))
        logging.debug('Loaded the dicom image')
    elif scan_extension == "mhd":
        itkimage = sitk.ReadImage(path_to_img)
        patient_id = ntpath.basename(path_to_img).replace('.mhd','')
    numpyImage = sitk.GetArrayFromImage(itkimage)
    logging.debug('Convert the image to numpy array [z,y,x]')
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    logging.info('Dicom Image origin [z,y,x]:\t %s' % numpyOrigin)
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    logging.info('Dicom Image spacing [z,y,x]:\t %s' % numpySpacing)
    return numpyImage, numpySpacing, patient_id,itkimage  # return numpy image


def HU2uint8(image, HU_min=-1200.0, HU_max=600.0, HU_nan=-2000.0):
    """
    Convert HU unit into uint8 values. First bound HU values by predfined min
    and max, and then normalize
    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    HU_min: float, min HU value.
    HU_max: float, max HU value.
    HU_nan: float, value for nan in the raw CT image.
    """
    logging.info("[HyperParameter] HU minimum limit is:\t %f" % HU_min)
    logging.info("[HyperParameter] HU maximum limit is:\t %f" % HU_max)
    logging.debug("HU NAN filling is:\t %f" % HU_nan)
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
    logging.info("Start creating Internal Marker by threshold the image. Any HU > -400 will be removed")
    marker_internal = segmentation.clear_border(marker_internal)
    logging.info("Clear objects connected to the internal marker border")
    marker_internal_labels = measure.label(marker_internal)
    logging.info("Label connected regions of the internal marker")
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    logging.info("Calculate the number of pixels in each labeled region of the internal marker")
    areas.sort()
    logging.info("Make any labeled region in the internal marker equal to ZERO"
                 " if its area less than the 2nd biggest area in areas list")
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    marker_internal_labels[coordinates[0], coordinates[1]] = 0
    else:
        logging.debug("areas list size is <= 2")
    marker_internal = marker_internal_labels > 0
    # Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    logging.info("Create external_a marker morphological dilation of the internal marker with 10 iterations")
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    logging.info("Create external_b marker morphological dilation of the internal marker with 55 iterations")
    marker_external = external_b ^ external_a
    logging.info("Create external marker by external_b XOR external_a")
    # Creation of the Watershed Marker matrix
    logging.info("Internal marker and external marker values are either 1 or 0")
    marker_watershed = np.zeros((512, 512), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    logging.info("watershed marker = internal * 255 + external * 128")
    return marker_internal, marker_external, marker_watershed


def watershed_segmentation(image):
    # Creation of the markers:
    logging.info("Generate internal and external marker for watershed algorithm")
    marker_internal, marker_external, marker_watershed = generate_markers(image)

    # Creation of the Sobel-Gradient
    logging.info("Apply sobel filter to the input image x axis")
    sobel_filtered_dx = ndimage.sobel(image, 1)
    logging.info("Apply sobel filter to the input image y axis")
    sobel_filtered_dy = ndimage.sobel(image, 0)
    logging.info("Calculate sobel gradiant using sqrt(sobel_filtered_dx^2+sobel_filtered_dy^2)")
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    logging.info("Normalize then multiply sobel gradient by 255")
    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    # Watershed algorithm
    logging.info("Find watershed basins in image flooded from given markers")
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
    logging.info("fill holes that is not used, since in some slices the heart it would be included by accident")
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5, 5)), iterations=3)

    # Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image, -2000 * np.ones((512, 512)))
    return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed











def preprocessing(imagePixels, spacing,itkimage,inputDir):
    """
    The preprocessing code is divided into four main tasks
    First: Convert the numpy images from HU to 8 bit unsigned int uint8
    Second: Creat a lung mask for the dicom image
    Third: Segment the image using the mask
    Fourth: Resample the image to (1*1*1) mm
    ...............................................................
    INPUT:
    imagePixels: 3D numpy image [z, y, x]
    spacing: float * 3, raw CT spacing in [z, y, x] order.

    """
    imageNew = HU2uint8(imagePixels)
    a = datetime.datetime.now()


    segmentation = mask.apply(itkimage)
    segmentation[np.where(segmentation == 2)] = 1;
    masks=[]
    for i,segment in enumerate(segmentation):
        #out_img = HU2uint8(segment)
        segmented = np.where(segment == 1, imagePixels[i], -2000 * np.ones((512, 512)))
        out_img = HU2uint8(segmented)
        cv2.imwrite("./OUTPUT/deep/watershed_code_" + str(i).zfill(3) + "_.png", out_img)
        masks.append(segment)
    b = datetime.datetime.now()
    c = b - a
    print("Time needed to process deep learning based segmentation is in milliSecond:", c)
    segMask = np.array(masks, dtype=np.int16)

    binary_mask_dilated = np.array(segMask)
    dilate_factor = 1.2
    logging.info("[HyperParameter] Dilate factor = %s; factor of increased area after dilation" % dilate_factor)
    for i in range(segMask.shape[0]):
        slice_binary = segMask[i]

        if np.sum(slice_binary) > 0:
            logging.info("Apply convex hull to mask slice number %s" % i)
            slice_convex = morphology.convex_hull_image(slice_binary)
            # The convex hull is the set of pixels included in the
            # smallest convex polygon that surround all white pixels in the input image.

            if np.sum(slice_convex) <= dilate_factor * np.sum(slice_binary):
                logging.info("Replace slice number %s of the mask with the convexed slice" % i)
                binary_mask_dilated[i] = slice_convex
            else:
                logging.warning("convex slice is > dilate factor * summation of the original slice; keep the original "
                                "mask slice")
        else:
            logging.info("Slice %s doesn't have any white pixels" % i)

    struct = ndimage.morphology.generate_binary_structure(3, 1)
    binary_mask_dilated = ndimage.morphology.binary_dilation(
        binary_mask_dilated, structure=struct, iterations=10)
    logging.debug("Apply binary dilation")
    binary_mask_extra = binary_mask_dilated ^ segMask  # binary_mask_dilated XOR original mask
    pad_value = 170
    logging.info("[HyperParameter] pad_value = %s" % pad_value)
    logging.info("replace image values outside binary_mask_dilated with pad value = %s" % pad_value)
    image_new = imageNew * binary_mask_dilated + \
                pad_value * (1 - binary_mask_dilated).astype('uint8')

    for i in range(image_new.shape[0]):
        cv2.imwrite("./OUTPUT/deep_plus/dilation_DL" + str(i).zfill(3) + "_.png", image_new[i])
    save_dir = './dicom_nodules'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_as_dicom(image_new,inputDir)


    """
    ################### HU TO UINT8 #######################
    logging.info("PREPROCESSING: convert HU to 8 bit unsigned int")
    imageNew = HU2uint8(imagePixels)
    ################### LUNG MASK CREATION #######################
    masks = []
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    logging.info("PREPROCESSING: Creat a lung mask for the input dicom image")
    a = datetime.datetime.now()
    imagePixels[imagePixels == -2000] = 0
    for i in range(imagePixels.shape[0]):
        logging.info("Apply watershed algorithm to slice number %d out of %d slices" % (i, imagePixels.shape[0]))
        lung_segmented, lung_lungfilter, lung_outline, lung_watershed, lung_sobel_gradient, \
        lung_marker_internal, lung_marker_external, lung_marker_watershed = watershed_segmentation(imagePixels[i])
        masks.append(lung_lungfilter)
        out_img = HU2uint8(lung_segmented)
        cv2.imwrite("./OUTPUT/watershed/watershed_code_" + str(i).zfill(3) + "_.png", out_img)
    b = datetime.datetime.now()
    c = b - a
    print("Time needed to process watershed based segmentation is in milliSecond:", c)
    segMask = np.array(masks, dtype=np.int16)

    binary_mask_dilated = np.array(segMask)
    dilate_factor = 1.5
    logging.info("[HyperParameter] Dilate factor = %s; factor of increased area after dilation" % dilate_factor)
    for i in range(segMask.shape[0]):
        slice_binary = segMask[i]

        if np.sum(slice_binary) > 0:
            logging.info("Apply convex hull to mask slice number %s" % i)
            slice_convex = morphology.convex_hull_image(slice_binary)
            # The convex hull is the set of pixels included in the
            # smallest convex polygon that surround all white pixels in the input image.

            if np.sum(slice_convex) <= dilate_factor * np.sum(slice_binary):
                logging.info("Replace slice number %s of the mask with the convexed slice" % i)
                binary_mask_dilated[i] = slice_convex
            else:
                logging.warning("convex slice is > dilate factor * summation of the original slice; keep the original "
                                "mask slice")
        else:
            logging.info("Slice %s doesn't have any white pixels" % i)

    struct = ndimage.morphology.generate_binary_structure(3, 1)
    binary_mask_dilated = ndimage.morphology.binary_dilation(
        binary_mask_dilated, structure=struct, iterations=10)
    logging.debug("Apply binary dilation")
    binary_mask_extra = binary_mask_dilated ^ segMask  # binary_mask_dilated XOR original mask
    pad_value = 170
    logging.info("[HyperParameter] pad_value = %s" % pad_value)
    logging.info("replace image values outside binary_mask_dilated with pad value = %s" % pad_value)
    image_new = imageNew * binary_mask_dilated + \
                pad_value * (1 - binary_mask_dilated).astype('uint8')
    for i in range(image_new.shape[0]):
        cv2.imwrite("./OUTPUT/WS_plus/dilation_WS" + str(i).zfill(3) + "_.png", image_new[i])
        
    """

def ensure_even(stream):
    # Very important for some viewers
    if len(stream) % 2:
        return stream + b"\x00"
    return stream


def save_as_dicom(pred_masks, inputDir, invert_order=False):
    for count in range(pred_masks.shape[0]):
        pred_mask = pred_masks[count]

        # display_img = normalize(img)
        # pixels = get_pixels_hu([slice_])
        # image = patient_img[slice_]
        # img = image[0]
        # print(img.shape)
        # ds = slices[count]
        def load_patient(src_dir):
            # slices = [dicom.read_file(src_dir + '/' + s) for s in os.listdir(src_dir)]
            slices = []
            # print(src_dir)
            for files in glob.glob(src_dir + "/*.dcm"):
                slices.append(dicom.read_file(files))
            # print(len(slices))
            slices.sort(key=lambda x: int(x.InstanceNumber))
            try:
                slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
            except:
                # print(len(slices))
                slice_thickness = np.abs(slices[5].SliceLocation - slices[6].SliceLocation)
            if slice_thickness == 0:
                slice_thickness = np.abs(slices[4].ImagePositionPatient[2] - slices[5].ImagePositionPatient[2])
            for s in slices:
                s.SliceThickness = slice_thickness
            return slices

        slices = load_patient(inputDir)
        if not invert_order:
            save_count = pred_mask.shape[0] - count
        #ds = slices[save_count - 1]
        ds = slices[0]
        pred_mask = pred_mask.astype('uint8')
        pred_mask = Image.fromarray(pred_mask)
        pred_mask = pred_mask.convert('RGB')
        pred_mask = np.asarray(pred_mask)
        # modify DICOM tags

        ds.PhotometricInterpretation = 'RGB'
        ds.SamplesPerPixel = 3
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.SeriesNumber = 53
        # ds.add_new(0x00280006, 'US', 0)
        ds.is_little_endian = True
        # ds.SliceLocation = str(save_count)
        ds.InstanceNumber = str(save_count)
        ds.is_implicit_VR = True
        # ds.Modality = "OT"
        ds.SOPInstanceUID = generate_uid()
        HEIGHT = pred_mask.shape[1]
        WIDTH = pred_mask.shape[0]
        ds.Rows = HEIGHT
        ds.Columns = WIDTH
        ds.PixelRepresentation = 0
        # ds.AcquisitionNumber = '9'
        # ds.SeriesInstanceUID = '1.3.6.1.4.1.14519.5.2.1.6279.6001.1763629124204912627830645853331'
        # ds.StudyInstanceUID = '1.3.6.1.4.1.14519.5.2.1.6279.6001.101493103577576219860121359500'

        ds.SOPClassUID = SecondaryCaptureImageStorage

        ds.fix_meta_info()
        # ds.fix_meta_info()

        # save pixel data and dicom file
        ds.PixelData = encapsulate([ensure_even(pred_mask.tobytes())])
        ds.save_as('dicom_nodules/' + str(save_count) + '.dcm')
        # ds = dicom.read_file('dicom_nodules/'+str(save_count)+'.dcm',force=True)
        # pixels = get_pixels_hu([ds])
        # image = pixels
        # img = image[0]

if __name__ == '__main__':
    main()