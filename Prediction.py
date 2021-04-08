import settings
import helpers
import sys
import os
import glob
import random
import pandas
import ntpath
import cv2
import numpy
from tensorflow.keras import backend as K
import math


# limit memory usage..
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
import Train
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.compat.v1.Session(config=config))

# zonder aug, 10:1 99 train, 97 test, 0.27 cross entropy, before commit 573
# 3 pools istead of 4 gives (bigger end layer) gives much worse validation accuray + logloss .. strange ?
# 32 x 32 x 32 lijkt het beter te doen dan 48 x 48 x 48..

#K.common.set_image_dim_ordering("tf")
CUBE_SIZE = Train.CUBE_SIZE
MEAN_PIXEL_VALUE = settings.MEAN_PIXEL_VALUE_NODULE
NEGS_PER_POS = 20
P_TH = 0.6

PREDICT_STEP = 12
USE_DROPOUT = False


def prepare_image_for_net3D(img):
    img = img.astype(numpy.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img


def predict_cubes(model_path, continue_job, only_patient_id=None, lidc=True, magnification=1, flip=False, ext_name="",input_format = "dicom",evaluate = True):
    # choose which directory you should store output at
    dst_dir = settings.LIDC_PREDICTION_DIR
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    holdout_ext = ""
    # if holdout_no is not None:
    #     holdout_ext = "_h" + str(holdout_no) if holdout_no >= 0 else ""
    flip_ext = ""
    if flip:
        flip_ext = "_flip"
    # some big nodules can't be detected very well, solution is to down sample the images by 2 that is the magnification
    dst_dir += "predictions" + str(int(magnification * 10)) + holdout_ext + flip_ext + "_" + ext_name + "/"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    sw = helpers.Stopwatch.start_new()
    # get the pre trained model
    #model = Train.get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), load_weight_path=model_path)
    model = tf.keras.models.load_model(model_path)
    patient_ids = []
    if evaluate:
        cubic_images = pandas.read_csv(settings.BASE_DIR_SSD + "Test_data.csv", sep=',').values.tolist()
        model.summary()
        random.shuffle(cubic_images)
        for row in cubic_images:
            root_dir = os.path.basename(row[0])
            parts = root_dir.split("_")
            if parts[0] in patient_ids:
                continue
            patient_ids.append(parts[0])
            print("Test Case ID: ", parts[0])
        print("start evaluation on: ", len(patient_ids), " Patient")
    else:
        patient_ids = only_patient_id

    all_predictions_csv = []
    all_metric = []
    for patient_index, patient_id in enumerate(reversed(patient_ids)):
        # AM: TODO find the mean of metadata
        if "metadata" in patient_id:
            continue
        if only_patient_id is not None and only_patient_id != patient_id:
            continue
        # TODO: real use of holdout

        print(patient_index, ": ", patient_id)
        if lidc and evaluate:
            csv_label_path = settings.LIDC_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_pos_lidc.csv"
            #csv_label_path = "F:/Cengiz/Nodules-Detection/" + patient_id + "_annos_pos_lidc.csv"
            csv_gt_label = pandas.read_csv(csv_label_path)
            gt_x = list(csv_gt_label["coord_x"].values.tolist())
            gt_y = list(csv_gt_label["coord_y"].values.tolist())
            gt_z = list(csv_gt_label["coord_z"].values.tolist())
        csv_target_path = dst_dir + patient_id + ".csv"
        if continue_job:
            if os.path.exists(csv_target_path):
                continue
        if lidc:
            patient_img = helpers.load_patient_images(patient_id, settings.LIDC_EXTRACTED_IMAGE_DIR, "*_i.png", [])
        else:
            patient_img = helpers.load_patient_images(patient_id, settings.NDSB3_EXTRACTED_IMAGE_DIR, "*_i.png", [])
        if magnification != 1:
            patient_img = helpers.rescale_patient_images(patient_img, (1, 1, 1), magnification)
        if lidc:
            patient_mask = helpers.load_patient_images(patient_id, settings.LIDC_EXTRACTED_IMAGE_DIR, "*_m.png", [])
        else:
            patient_mask = helpers.load_patient_images(patient_id, settings.NDSB3_EXTRACTED_IMAGE_DIR, "*_m.png", [])
        if magnification != 1:
            patient_mask = helpers.rescale_patient_images(patient_mask, (1, 1, 1), magnification, is_mask_image=True)

            # patient_img = patient_img[:, ::-1, :]
            # patient_mask = patient_mask[:, ::-1, :]

        step = PREDICT_STEP
        CROP_SIZE = CUBE_SIZE
        # CROP_SIZE = 48

        predict_volume_shape_list = [0, 0, 0]
        for dim in range(3):
            dim_indent = 0
            while dim_indent + CROP_SIZE < patient_img.shape[dim]:
                predict_volume_shape_list[dim] += 1
                dim_indent += step

        predict_volume_shape = (predict_volume_shape_list[0], predict_volume_shape_list[1], predict_volume_shape_list[2])
        predict_volume = numpy.zeros(shape=predict_volume_shape, dtype=float)
        print("Predict volume shape: ", predict_volume.shape)
        done_count = 0
        skipped_count = 0
        batch_size = 128
        batch_list = []
        batch_list_coords = []
        patient_predictions_csv = []
        cube_img = None
        annotation_index = 0
        iteration = 0
        if evaluate:
            unique_nodules = []
            for tx, ty, tz in zip(gt_x, gt_y, gt_z):
                exist = False
                if len(unique_nodules) == 0:
                    exist = True
                    unique_nodules.append([tx,ty,tz])
                for index in range(len(unique_nodules)):
                    if abs(tx - unique_nodules[index][0]) < 0.08 and abs(ty - unique_nodules[index][1]) < 0.08 and abs(
                            tz - unique_nodules[index][2]) < 0.08:
                        exist = True
                if not exist:
                    unique_nodules.append([tx, ty, tz])
        for z in range(0, predict_volume_shape[0]):
            for y in range(0, predict_volume_shape[1]):
                for x in range(0, predict_volume_shape[2]):
                    #if cube_img is None:
                    cube_img = patient_img[z * step:z * step+CROP_SIZE, y * step:y * step + CROP_SIZE, x * step:x * step+CROP_SIZE]
                    cube_mask = patient_mask[z * step:z * step+CROP_SIZE, y * step:y * step + CROP_SIZE, x * step:x * step+CROP_SIZE]

                    if cube_mask.sum() < 2000:
                        skipped_count += 1
                    else:
                        if flip:
                            cube_img = cube_img[:, :, ::-1]

                        if CROP_SIZE != CUBE_SIZE:
                            cube_img = helpers.rescale_patient_images2(cube_img, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
                            # helpers.save_cube_img("c:/tmp/cube.png", cube_img, 8, 4)
                            # cube_mask = helpers.rescale_patient_images2(cube_mask, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
                        #print(cube_img.shape)
                        #helpers.save_cube_img(dst_dir + patient_id + "__" + str(annotation_index) + "__"+str(iteration) + ".png", cube_img, 4, 8)
                        img_prep = prepare_image_for_net3D(cube_img)
                        batch_list.append(img_prep)
                        batch_list_coords.append((z, y, x))
                        if len(batch_list) % batch_size == 0:
                            batch_data = numpy.vstack(batch_list)
                            p = model.predict(batch_data, batch_size=batch_size)
                            #model.summary()
                            #print("Output shape: ",len(p[0]))
                            #print("Output:", p)

                            for i in range(len(p)):
                                p_z = batch_list_coords[i][0]
                                p_y = batch_list_coords[i][1]
                                p_x = batch_list_coords[i][2]
                                nodule_chance = p[i][0]
                                print("nodule_chance:", nodule_chance)
                                predict_volume[p_z, p_y, p_x] = nodule_chance
                                if nodule_chance > P_TH:
                                    p_z = p_z * step + CROP_SIZE / 2
                                    p_y = p_y * step + CROP_SIZE / 2
                                    p_x = p_x * step + CROP_SIZE / 2
                                    #diameter_mm = round(p[1][i][0], 4)
                                    diameter_mm = 32.0
                                    #print (patient_img.shape[0], patient_img.shape[1], patient_img.shape[2])

                                    p_z_perc = round(p_z / patient_img.shape[0], 4)
                                    p_y_perc = round(p_y / patient_img.shape[1], 4)
                                    p_x_perc = round(p_x / patient_img.shape[2], 4)

                                    #p_x,p_y,slice_num = helpers.percentage_to_pixels(p_x_perc, p_y_perc, p_z_perc, patient_img)
                                    #src_img_paths = settings.NDSB3_EXTRACTED_IMAGE_DIR + patient_id + "/" + "img_" + str(
                                    #    slice_num).rjust(4, '0') + "_i.png"
                                    #print(settings.NDSB3_EXTRACTED_IMAGE_DIR + patient_id + "/" + "img_" + str(
                                    #    slice_num).rjust(4, '0') + "_i.png")
                                    #slice_img = cv2.imread(src_img_paths, cv2.IMREAD_GRAYSCALE)
                                    #xymax = (int(p_x + diameter_mm / 2), int(p_y + diameter_mm / 2))
                                    #xymin = (int(p_x - diameter_mm / 2), int(p_y - diameter_mm / 2))
                                    #colorRGB = (255, 255, 0)
                                    #print(xymin, xymax, diameter_mm)
                                    #gray_BGR = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)
                                    #image = cv2.rectangle(gray_BGR, xymin, xymax, colorRGB, 1)
                                    #cv2.imwrite(dst_dir + "/Annotation_V2/" + "Positive_nodule_" + str(
                                    #    nodule_chance) + patient_id + "__" + str(annotation_index) + "__" + str(
                                    #    iteration) + ".png", image)

                                    # diameter_perc = round(2 * step / patient_img.shape[2], 4)
                                    status = ""
                                    if evaluate:
                                        status = "FP"
                                        for tx,ty,tz in zip(gt_x,gt_y,gt_z):
                                            dx = abs(p_x_perc - tx)
                                            dy = abs(p_y_perc - ty)
                                            dz = abs(p_z_perc - tz)
                                            if dx < 0.07 and dy < 0.07 and dz < 0.07:
                                                status = "TP"
                                                break
                                    nodule_chance = round(nodule_chance, 4)
                                    #helpers.save_cube_img(dst_dir + "Positive_nodule_" + str(nodule_chance) +  patient_id + "__" + str(annotation_index) + "__" + str(
                                     #       iteration) + ".png", cube_img, 4, 8)
                                    patient_predictions_csv_line = [annotation_index, p_x_perc, p_y_perc, p_z_perc, nodule_chance,status]
                                    patient_predictions_csv.append(patient_predictions_csv_line)
                                    all_predictions_csv.append([patient_id] + patient_predictions_csv_line)
                                    annotation_index += 1
                            batch_list = []
                            batch_list_coords = []
                    done_count += 1
                    if done_count % 10000 == 0:
                        print("Done: ", done_count, " skipped:", skipped_count)
                    iteration = iteration + 1
        if evaluate:
            TP = FP = 0
            unique_nodules_2 = []
            for row in patient_predictions_csv:
                exist = False
                tx = row[1]
                ty = row[2]
                tz = row[3]
                if len(unique_nodules_2) == 0:
                    exist = True
                    unique_nodules_2.append([tx, ty, tz])
                for index in range(len(unique_nodules_2)):
                    if abs(tx - unique_nodules_2[index][0]) < 0.08 and abs(
                            ty - unique_nodules_2[index][1]) < 0.08 and abs(
                            tz - unique_nodules_2[index][2]) < 0.08:
                        exist = True
                if not exist:
                    unique_nodules_2.append([tx, ty, tz])
                    TP += 1 if row[5] == "TP" else 0
                    FP += 1 if row[5] == "FP" else 0

            FN = len(unique_nodules) - TP
            TN = done_count - FP
            if TP != 0:
                sensitivity = TP / (TP + FN)
            else:
                sensitivity = -1
            print("Sensetivity: ",sensitivity)
            if TN != 0:
                specificity = TN / (TN + FP)
            else:
                specificity = -1
            print("Specificity: ",specificity)
            all_metric.append([patient_id,FP,FN,sensitivity,specificity])
        df = pandas.DataFrame(patient_predictions_csv, columns=["anno_index", "coord_x", "coord_y", "coord_z", "nodule_chance","status"])
        df.to_csv(csv_target_path, index=False)
        print(predict_volume.mean())
        print("Done in : ", sw.get_elapsed_seconds(), " seconds")
    df = pandas.DataFrame(all_predictions_csv,
                          columns=["patient_id", "anno_index", "coord_x", "coord_y", "coord_z", "nodule_chance",
                                   "status"])
    df.to_csv(settings.LIDC_PREDICTION_DIR + "Detected_Nodules_All_Test_Candidates.csv.csv", index=False)
    if evaluate:
        total_FP = total_FN = total_sens = total_spec =0
        for row in all_metric:
            total_FP += row[1]
            total_FN += row[2]
            total_sens += row[3]
            total_spec += row[4]
        total_FP /= len(all_metric)
        total_FN /= len(all_metric)
        total_sens /= len(all_metric)
        total_spec /= len(all_metric)
        print("Total FP: ", total_FP)
        print("Total FN: ", total_FN)
        print("Total Sensetivity: ", total_sens)
        print("Total Specificity: ", total_spec)
        all_metric.append(["Total Mean Value",total_FP,total_FN,total_sens,total_spec])
        df = pandas.DataFrame(all_metric,columns=["patient_id", "false_positive", "false_negative", "sensitivity", "specificity"])
        df.to_csv(settings.LIDC_PREDICTION_DIR + "Metric_All_Test_Candidates.csv", index=False)


if __name__ == "__main__":

    CONTINUE_JOB = True
    only_patient_id = None

    if not CONTINUE_JOB or only_patient_id is not None:
        for file_path in glob.glob("c:/tmp/*.*"):
            if not os.path.isdir(file_path):
                remove_file = True
                if only_patient_id is not None:
                    if only_patient_id not in file_path:
                        remove_file = False
                        remove_file = False

                if remove_file:
                    os.remove(file_path)

    if True:
        for magnification in [1]:  #
            predict_cubes("models/model_luna16_full__fs_best.hd5", CONTINUE_JOB, only_patient_id=only_patient_id, magnification=magnification, flip=False, ext_name="luna16_fs")
            #predict_cubes("models/model_luna16_full__fs_best.hd5", CONTINUE_JOB, only_patient_id=only_patient_id, magnification=magnification, flip=False, train_data=False, holdout_no=None, ext_name="luna16_fs")