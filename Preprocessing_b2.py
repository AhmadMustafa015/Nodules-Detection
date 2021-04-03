import settings
import helpers
import glob
import pandas
import ntpath
import numpy
import cv2
import os
from multiprocessing import Pool

CUBE_IMGTYPE_SRC = "_i"


def save_cube_img(target_path, cube_img, rows, cols):
    assert rows * cols == cube_img.shape[0]
    img_height = cube_img.shape[1]
    img_width = cube_img.shape[1]
    res_img = numpy.zeros((rows * img_height, cols * img_width), dtype=numpy.uint8)

    for row in range(rows):
        for col in range(cols):
            target_y = row * img_height
            target_x = col * img_width
            res_img[target_y:target_y + img_height, target_x:target_x + img_width] = cube_img[row * cols + col]

    cv2.imwrite(target_path, res_img)


def get_cube_from_img(img3d, center_x, center_y, center_z, block_size):
    start_x = max(center_x - block_size / 2, 0)
    if start_x + block_size > img3d.shape[2]:
        start_x = img3d.shape[2] - block_size

    start_y = max(center_y - block_size / 2, 0)
    start_z = max(center_z - block_size / 2, 0)
    if start_z + block_size > img3d.shape[0]:
        start_z = img3d.shape[0] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    res = img3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    return res

def make_annotation_images_lidc(compined_list):
    csv_file = compined_list[1]
    patient_index = compined_list[0]
    dst_dir = settings.BASE_DIR_SSD + "generated_traindata2/lidc_train_cubes/"
    patient_id = ntpath.basename(csv_file).replace("_annos_pos_lidc.csv", "")
    df_annos = pandas.read_csv(csv_file)
    if len(df_annos) == 0:
        return
    try:
        images = helpers.load_patient_images(patient_id, settings.LIDC_EXTRACTED_IMAGE_DIR, "*" + CUBE_IMGTYPE_SRC + ".png")
    except:
        print("Error in patient ID: ", patient_id)

    for index, row in df_annos.iterrows():
        coord_x = int(row["coord_x"] * images.shape[2])
        coord_y = int(row["coord_y"] * images.shape[1])
        coord_z = int(row["coord_z"] * images.shape[0])
        malscore = int(row["malscore"])
        anno_index = row["anno_index"]
        anno_index = str(anno_index).replace(" ", "xspacex").replace(".", "xpointx").replace("_", "xunderscorex")
        cube_img = get_cube_from_img(images, coord_x, coord_y, coord_z, 64)
        if cube_img.sum() < 5:
            print(" ***** Skipping ", coord_x, coord_y, coord_z)
            continue

        if cube_img.mean() < 10:
            print(" ***** Suspicious ", coord_x, coord_y, coord_z)

        if cube_img.shape != (64, 64, 64):
            print(" ***** incorrect shape !!! ", str(anno_index), " - ",(coord_x, coord_y, coord_z))
            continue

        save_cube_img(dst_dir + patient_id + "_" + str(anno_index) + "_" + str(malscore * malscore) + "_1_pos.png", cube_img, 8, 8)
    helpers.print_tabbed([patient_index, patient_id, len(df_annos)], [5, 64, 8])

def make_candidate_auto_images(compined_list):
    index = compined_list[0]
    csv_file = compined_list[1]
    candidate_type = compined_list[2]
    dst_dir = settings.BASE_DIR_SSD + "generated_traindata2/lidc_train_cubes_auto/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    if candidate_type == "neg_lidc":
        patient_id = ntpath.basename(csv_file).replace("_annos_" + candidate_type + ".csv", "")
    else:
        patient_id = ntpath.basename(csv_file).replace("_candidates_" + candidate_type + ".csv", "")
    if candidate_type == "neg_lidc":
        scan_path = settings.LIDC_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_candidates_luna.csv"
        # if os.path.exists(scan_path):
    print(index, ",patient: ", patient_id, " type:", candidate_type)
    # if not "148229375703208214308676934766" in patient_id:
    #     continue
    df_annos = pandas.read_csv(csv_file)
    if len(df_annos) == 0:
        return
    images = helpers.load_patient_images(patient_id, settings.LIDC_EXTRACTED_IMAGE_DIR, "*" + CUBE_IMGTYPE_SRC + ".png", exclude_wildcards=[])

    row_no = 0
    for index, row in df_annos.iterrows():
        coord_x = int(row["coord_x"] * images.shape[2])
        coord_y = int(row["coord_y"] * images.shape[1])
        coord_z = int(row["coord_z"] * images.shape[0])
        anno_index = int(row["anno_index"])
        cube_img = get_cube_from_img(images, coord_x, coord_y, coord_z, 48)
        if cube_img.sum() < 10:
            print("Skipping ", coord_x, coord_y, coord_z)
            continue
        # print(cube_img.sum())
        try:
            save_cube_img(dst_dir + patient_id + "_" + str(anno_index) + "_0_" + candidate_type + ".png", cube_img, 6, 8)
        except Exception as ex:
            print(ex)

        row_no += 1
        max_item = 240 if candidate_type == "white" else 200
        if candidate_type == "luna":
            max_item = 500
        if row_no > max_item:
            break

def make_candidate_auto_images_p(candidate_types=[]):
    dst_dir = settings.BASE_DIR_SSD + "generated_traindata2/lidc_train_cubes_auto/"
    for candidate_type in candidate_types:
        for file_path in glob.glob(dst_dir + "*_" + candidate_type + ".png"):
            os.remove(file_path)

    for candidate_type in candidate_types:
        if candidate_type == "falsepos":
            src_dir = "C:/Users/RadioscientificOne/PycharmProjects/NoduleDetect/Nodules-Detection/resources/luna16_falsepos_labels/"
        else:
            src_dir = settings.LIDC_EXTRACTED_IMAGE_DIR + "_labels/"
        patient_inx_csv = []
        if candidate_type == "neg_lidc":
            for index, csv_file in enumerate(glob.glob(src_dir + "*_annos_" + candidate_type + ".csv")):
                patient_inx_csv.append([index, csv_file, candidate_type])
        else:
            for index, csv_file in enumerate(glob.glob(src_dir + "*_candidates_" + candidate_type + ".csv")):
                patient_inx_csv.append([index, csv_file, candidate_type])
        pool = Pool(settings.WORKER_POOL_SIZE)
        pool.map(make_candidate_auto_images, patient_inx_csv)
def make_annotation_images_lidc_p():
    src_dir = settings.LIDC_EXTRACTED_IMAGE_DIR + "_labels/"
    dst_dir = settings.BASE_DIR_SSD + "generated_traindata2/lidc_train_cubes/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for file_path in glob.glob(dst_dir + "*.*"):
        os.remove(file_path)
    patient_inx_csv = []
    for patient_index, csv_file in enumerate(glob.glob(src_dir + "*_annos_pos_lidc.csv")):
        patient_inx_csv.append([patient_index, csv_file])

    pool = Pool(settings.WORKER_POOL_SIZE)
    pool.map(make_annotation_images_lidc, patient_inx_csv)

if __name__ == "__main__":
    if not os.path.exists(settings.BASE_DIR_SSD + "generated_traindata2/"):
        os.mkdir(settings.BASE_DIR_SSD + "generated_traindata2/")

    if True:
        make_annotation_images_lidc_p()
    if True:
        make_candidate_auto_images_p(["falsepos", "edge", "luna"])