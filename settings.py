import os
COMPUTER_NAME = os.environ['COMPUTERNAME']
print("Computer: ", COMPUTER_NAME)

WORKER_POOL_SIZE = 6

TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 0
SEGMENTER_IMG_SIZE = 320

BASE_DIR_SSD = "F:/Lung_Nodule_Detection/output/"
BASE_DIR = "F:/Lung_Nodule_Detection/Data/"
EXTRA_DATA_DIR = "resources/"
NDSB3_RAW_SRC_DIR = BASE_DIR + "ndsb_raw/stage12/"
LUNA16_RAW_SRC_DIR = BASE_DIR + "luna_raw/"
LIDC_RAW_SRC_DIR = "F:/manifest-1600709154662/LIDC-IDRI/"

NDSB3_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "ndsb3_extracted_images/"
LUNA16_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "luna16_extracted_images/"
NDSB3_NODULE_DETECTION_DIR = BASE_DIR_SSD + "ndsb3_nodule_predictions/"
LIDC_EXTRACTED_IMAGE_DIR = "F:/Lung_Nodule_Detection/" + "lidc_extracted_images/"
LIDC_LABEL = "F:/Lung_Nodule_Detection/tcia-lidc-xml/"
