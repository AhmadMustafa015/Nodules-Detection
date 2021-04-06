import os
COMPUTER_NAME = os.environ['COMPUTERNAME']
print("Computer: ", COMPUTER_NAME)

WORKER_POOL_SIZE = 6

TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 0
SEGMENTER_IMG_SIZE = 320

BASE_DIR_SSD = "C:/Users/RadioscientificOne/PycharmProjects/NoduleDetect/Nodules-Detection/output/"
BASE_DIR = "C:/Users/RadioscientificOne/PycharmProjects/NoduleDetect/Nodules-Detection/Data/"
EXTRA_DATA_DIR = "resources/"
LIDC_RAW_SRC_DIR = "D:/Datasets/manifest-1600709154662/LIDC-IDRI/"

LIDC_EXTRACTED_IMAGE_DIR = "C:/Users/RadioscientificOne/PycharmProjects/NoduleDetect/Nodules-Detection/" + "lidc_extracted_images_cubic_interpolation/"
LIDC_LABEL = "C:/Users/RadioscientificOne/PycharmProjects/NoduleDetect/Nodules-Detection/tcia-lidc-xml/"
<<<<<<< HEAD
<<<<<<< HEAD
LIDC_PREDICTION_DIR = "C:/Users/RadioscientificOne/PycharmProjects/NoduleDetect/Nodules-Detection/prediction/"
=======
>>>>>>> 5908c8233f4f4d2b3d959197b6b8e687f65111d0
=======
LIDC_PREDICTION_DIR = "C:/Users/RadioscientificOne/PycharmProjects/NoduleDetect/Nodules-Detection/prediction/"
>>>>>>> 5f0925181973786c61497a07bb161b2d00efea53
