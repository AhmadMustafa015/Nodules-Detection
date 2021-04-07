import os
COMPUTER_NAME = os.environ['COMPUTERNAME']
print("Computer: ", COMPUTER_NAME)

WORKER_POOL_SIZE = 6

TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 0
SEGMENTER_IMG_SIZE = 320

BASE_DIR_SSD = "F:/Cengiz/Nodules-Detection/output/"
BASE_DIR = "C:/Users/RadioscientificOne/PycharmProjects/NoduleDetect/Nodules-Detection/Data/"
EXTRA_DATA_DIR = "resources/"
LIDC_RAW_SRC_DIR = "D:/Datasets/manifest-1600709154662/LIDC-IDRI/"

LIDC_EXTRACTED_IMAGE_DIR = "F:/Cengiz/Lung Nodules Detection/Nodule Detection Project 2020/output/luna16_extracted_images/"
LIDC_LABEL = "C:/Users/RadioscientificOne/PycharmProjects/NoduleDetect/Nodules-Detection/tcia-lidc-xml/"
LIDC_PREDICTION_DIR = "F:/Cengiz/Nodules-Detection/prediction/"
