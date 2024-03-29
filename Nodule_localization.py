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
from typing import List, Tuple
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPool3D, Flatten, Dense, Dropout, AveragePooling3D, Activation
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler,TensorBoard
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import math
import shutil
import datetime


print("Here")
# limit memory usage..
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.compat.v1.Session(config=config))

# zonder aug, 10:1 99 train, 97 test, 0.27 cross entropy, before commit 573
# 3 pools istead of 4 gives (bigger end layer) gives much worse validation accuray + logloss .. strange ?
# 32 x 32 x 32 lijkt het beter te doen dan 48 x 48 x 48..

#K.common.set_image_dim_ordering("tf")
CUBE_SIZE = 32
MEAN_PIXEL_VALUE = settings.MEAN_PIXEL_VALUE_NODULE
POS_WEIGHT = 2
NEGS_PER_POS = 20
P_TH = 0.6
# POS_IMG_DIR = "luna16_train_cubes_pos"
LEARN_RATE = 0.001

USE_DROPOUT = False
normalizeNetwork = False
def prepare_image_for_net3D(img):
    img = img.astype(numpy.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img


def get_train_holdout_files(fold_count, train_percentage=85, test_percentage=5, logreg=True, ndsb3_holdout=0, manual_labels=False, full_luna_set=False):
    print("Get train/holdout files.")
    # pos_samples = glob.glob(settings.BASE_DIR_SSD + "luna16_train_cubes_pos/*.png")
    pos_samples = glob.glob(settings.LIDC_POS_CUBES_DIR+"*.png")
    print("Pos samples: ", len(pos_samples))

    random.shuffle(pos_samples)
    train_pos_count = int((len(pos_samples) * train_percentage) / 100)
    test_pos_count = int((len(pos_samples) * test_percentage) / 100)
    number_of_test = 0
    pos_samples_test = []
    while number_of_test < test_pos_count:
        break
        index = random.randint(0,len(pos_samples))
        test_case = pos_samples[index]
        file_name = ntpath.basename(test_case)
        parts = file_name.split("_")
        patient_id = parts[0]
        for root, dirs, files in os.walk(settings.LIDC_RAW_SRC_DIR):
            if patient_id in dirs and len(files) > 10:
                root_dir =root
                root_dir = root_dir.split("/")[:-1]
                root_dir2 = ""
                for i in root_dir:
                    root_dir2 += str(i)+ '/'
                if len(os.listdir(root_dir2)) > 2:
                    continue

        images = glob.glob(settings.LIDC_POS_CUBES_DIR+patient_id +"*.png")
        number_of_test += len(images)
        for image in images:
            pos_samples_test.append(image)
            pos_samples.remove(image)

    print("Number of Test positive: ", number_of_test)
    pos_samples_train = pos_samples[:train_pos_count]
    pos_samples_holdout = pos_samples[train_pos_count:]
    if full_luna_set:
        pos_samples_train += pos_samples_holdout + pos_samples_test
        if manual_labels:
            pos_samples_holdout = []
    train_res = []
    holdout_res = []
    test_res = pos_samples_test
    test_annos = pandas.DataFrame(test_res,columns=["image_dir"])
    test_annos.to_csv(settings.BASE_DIR_SSD + "Test_data_nodules_localization.csv",index=False)
    #for index, neg_sample_path in enumerate(neg_samples):
    #class_label = int(parts[-2])
    #size_label = int(parts[-3])
    sets = [(train_res, pos_samples_train), (holdout_res, pos_samples_holdout)]
    for set_item in sets:
        pos_idx = 0
        res = set_item[0]
        pos_samples = set_item[1]
        print("Pos", len(pos_samples))
            # res.append(sample_path + "/")
        for pos_sample in pos_samples:
            pos_sample_path = pos_sample
            file_name = ntpath.basename(pos_sample_path)
            parts = file_name.split("_")

            class_label = int(parts[-2])
            size_label = int(parts[-3]) / 32.0
            #TODO: double check
            assert class_label == 1
            assert parts[-1] == "pos.png"
            #assert size_label >= 1

            res.append((pos_sample_path, size_label))
            pos_idx += 1


    print("Train count: ", len(train_res), ", holdout count: ", len(holdout_res))
    return train_res, holdout_res


def data_generator(batch_size, record_list, train_set):
    batch_idx = 0
    means = []
    random_state = numpy.random.RandomState(1301)
    while True:
        img_list = []
        size_list = []
        center_list_x = []
        center_list_y = []
        center_list_z = []
        if train_set:
            random.shuffle(record_list)
        CROP_SIZE = CUBE_SIZE
        # CROP_SIZE = 48
        for record_idx, record_item in enumerate(record_list):
            #rint patient_dir
            size_label = record_item[1]
            cube_image = helpers.load_cube_img(record_item[0], 8, 8, 64)
            if train_set:
                pass

            current_cube_size = cube_image.shape[0]
            indent_x = (current_cube_size - CROP_SIZE) / 2
            indent_y = (current_cube_size - CROP_SIZE) / 2
            indent_z = (current_cube_size - CROP_SIZE) / 2
            wiggle_indent = 0
            wiggle = current_cube_size - CROP_SIZE - 1
            if wiggle > (CROP_SIZE / 2):
                wiggle_indent = CROP_SIZE / 4
                wiggle = current_cube_size - CROP_SIZE - CROP_SIZE / 2 - 1
            if train_set:
                indent_x = wiggle_indent + random.randint(0, wiggle)
                indent_y = wiggle_indent + random.randint(0, wiggle)
                indent_z = wiggle_indent + random.randint(0, wiggle)

            indent_x = int(indent_x)
            indent_y = int(indent_y)
            indent_z = int(indent_z)
            center_x = center_y = center_z = CROP_SIZE
            cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE, indent_x:indent_x + CROP_SIZE]
            center_x = center_x - indent_x
            center_y = center_y - indent_y
            center_z = center_z - indent_z
            #print(center_z)
            if normalizeNetwork:
                center_x /= 32.0
                center_y /= 32.0
                center_z /= 32.0
            if CROP_SIZE != CUBE_SIZE:
                cube_image = helpers.rescale_patient_images2(cube_image, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
            assert cube_image.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

            if False:
                if random.randint(0, 100) > 50:
                    cube_image = numpy.fliplr(cube_image)
                if random.randint(0, 100) > 50:
                    cube_image = numpy.flipud(cube_image)
                if random.randint(0, 100) > 50:
                    cube_image = cube_image[:, :, ::-1]
                if random.randint(0, 100) > 50:
                    cube_image = cube_image[:, ::-1, :]


            means.append(cube_image.mean())
            img3d = prepare_image_for_net3D(cube_image)
            if train_set:
                if len(means) % 1000000 == 0:
                    print("Mean: ", sum(means) / len(means))
            img_list.append(img3d)
            size_list.append(size_label)
            center_list_x.append(center_x)
            center_list_y.append(center_y)
            center_list_z.append(center_z)
            batch_idx += 1
            if batch_idx >= batch_size:
                x = numpy.vstack(img_list)
                y_size = numpy.vstack(size_list)
                y_centerX = numpy.vstack(center_list_x)
                y_centerY = numpy.vstack(center_list_y)
                y_centerZ = numpy.vstack(center_list_z)
                #yield x, {"out_diameter": y_size, "out_centerX": y_centerX, "out_centerY": y_centerY, "out_centerZ": y_centerZ}
                yield x, {"out_centerZ": y_centerZ}
                img_list = []
                center_list_x = []
                size_list = []
                center_list_y = []
                center_list_z = []
                batch_idx = 0


def get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), load_weight_path=None, features=False, mal=False) -> Model:
    inputs = Input(shape=input_shape, name="input_1")
    x = inputs
    x = AveragePooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), padding="same")(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same', name='conv1', strides=(1, 1, 1))(x)
    x = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(x)

    # 2nd layer group
    x = Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same', name='conv2', strides=(1, 1, 1))(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.3)(x)

    # 3rd layer group
    x = Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same', name='conv3a', strides=(1, 1, 1))(x)
    x = Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same', name='conv3b', strides=(1, 1, 1))(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.4)(x)

    # 4th layer group
    x = Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same', name='conv4a', strides=(1, 1, 1))(x)
    x = Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same', name='conv4b', strides=(1, 1, 1),)(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.5)(x)

    last64 = Conv3D(64, kernel_size=(2, 2, 2), activation="relu", name="last_64")(x)
    #out_diameter = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid", name="out_diameter_last")(last64)
    #out_diameter = Flatten(name="out_diameter")(out_diameter)

    if normalizeNetwork:
        #out_centerX = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid", name="out_centerX_last")(last64)
        #out_centerX = Flatten(name="out_centerX")(out_centerX)

        #out_centerY = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid", name="out_centerY_last")(last64)
        #out_centerY = Flatten(name="out_centerY")(out_centerY)

        out_centerZ = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid", name="out_centerZ_last")(last64)
        out_centerZ = Flatten(name="out_centerZ")(out_centerZ)
    else:
        #out_centerX = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid", name="out_centerX_last")(last64)
        #out_centerX = Flatten(name="out_centerX")(out_centerX)

        #out_centerY = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid", name="out_centerY_last")(last64)
        #out_centerY = Flatten(name="out_centerY")(out_centerY)

        out_centerZ = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid", name="out_centerZ_last")(last64)
        out_centerZ = Flatten(name="out_centerZ")(out_centerZ)

    #model = Model(inputs=inputs, outputs=[out_diameter, out_centerX, out_centerY, out_centerZ])
    model = Model(inputs=inputs, outputs=[out_centerZ])
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
    if normalizeNetwork:
        model.compile(optimizer=SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True), loss={"out_centerZ": "binary_crossentropy"}, metrics={"out_centerZ": [binary_accuracy, binary_crossentropy]})

        #model.compile(optimizer=SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True), loss={"out_diameter": "binary_crossentropy", "out_centerY": "binary_crossentropy","out_centerX": "binary_crossentropy","out_centerZ": "binary_crossentropy"}, metrics={"out_diameter": [binary_accuracy, binary_crossentropy], "out_centerZ": [binary_accuracy, binary_crossentropy],"out_centerY": [binary_accuracy, binary_crossentropy],"out_centerX": [binary_accuracy, binary_crossentropy]})
    else:
        #model.compile(optimizer=SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True), loss={"out_diameter": "binary_crossentropy", "out_centerY": mean_absolute_error,"out_centerX": mean_absolute_error,"out_centerZ": mean_absolute_error}, metrics={"out_diameter": [binary_accuracy, binary_crossentropy], "out_centerZ": mean_absolute_error,"out_centerY": mean_absolute_error,"out_centerX": mean_absolute_error})
        model.compile(optimizer=SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True),
                      loss={"out_centerZ": mean_absolute_error},
                      metrics={"out_centerZ": mean_absolute_error})

    if features:
        model = Model(inputs=inputs, outputs=[last64])
    model.summary(line_length=140)

    return model


def step_decay(epoch):
    res = 0.001
    if epoch > 5:
        res = 0.0001
    print("learnrate: ", res, " epoch: ", epoch)
    return res


def train(model_name, fold_count, train_full_set=False, load_weights_path=None, ndsb3_holdout=0, manual_labels=True):
    batch_size = 16
    train_files, holdout_files = get_train_holdout_files(train_percentage=80, ndsb3_holdout=ndsb3_holdout, manual_labels=manual_labels, full_luna_set=train_full_set, fold_count=fold_count)
    print(holdout_files)

    # train_files = train_files[:100]
    # holdout_files = train_files[:10]
    train_gen = data_generator(batch_size, train_files, True)
    holdout_gen = data_generator(batch_size, holdout_files, False)
    for i in range(0, 10):
        tmp = next(holdout_gen)
        cube_img = tmp[0][0].reshape(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1)
        cube_img = cube_img[:, :, :, 0]
        cube_img *= 255.
        cube_img += MEAN_PIXEL_VALUE
        # helpers.save_cube_img("c:/tmp/img_" + str(i) + ".png", cube_img, 4, 8)
        # print(tmp)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    learnrate_scheduler = LearningRateScheduler(step_decay)
    model = get_net(load_weight_path=load_weights_path)
    holdout_txt = "_h" + str(ndsb3_holdout) if manual_labels else ""
    if train_full_set:
        holdout_txt = "_fs" + holdout_txt

    checkpoint = ModelCheckpoint(os.path.join("workdir\\model_" + model_name + "_" + holdout_txt + "_e" + "{epoch:02d}-{val_loss:.4f}.hd5"), monitor='val_loss', verbose=1, save_best_only=not train_full_set, save_weights_only=False, mode='auto', period=1)
    checkpoint_fixed_name = ModelCheckpoint(os.path.join("workdir\\model_" + model_name + "_" + holdout_txt + "_best.hd5"), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    #model.fit_generator(train_gen, len(train_files) / batch_size, 12, validation_data=holdout_gen, validation_steps=len(holdout_files) / batch_size, callbacks=[checkpoint, checkpoint_fixed_name, learnrate_scheduler])
    log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logdir = os.path.join(log_dir)
    tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)
    #tensorboard_callback,
    call_back = [tensorboard_callback, checkpoint, checkpoint_fixed_name, learnrate_scheduler]
    model.fit(train_gen,steps_per_epoch =  len(train_files) / batch_size,epochs = 25, validation_data=holdout_gen, validation_steps=len(holdout_files) / batch_size, callbacks= call_back,workers=1)

    model.save(os.path.join("workdir_loc\\model_" + model_name + "_" + holdout_txt + "_end.hd5"))

def compute_class_sens_spec(pred, label):
    """
    Compute sensitivity and specificity for a particular example
    for a given class.

    Args:
        pred (np.array): binary arrary of predictions, shape is
                         (num classes, height, width, depth).
        label (np.array): binary array of labels, shape is
                          (num classes, height, width, depth).
        class_num (int): number between 0 - (num_classes -1) which says
                         which prediction class to compute statistics
                         for.

    Returns:
        sensitivity (float): precision for given class_num.
        specificity (float): recall for given class_num
    """

    # extract sub-array for specified class
    class_pred = pred
    class_label = label

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # compute:

    # true positives
    tp = numpy.sum((class_pred == 1) & (class_label == 1))

    # compute sensitivity and specificity

    # true negatives
    tn = numpy.sum((class_pred == 0) & (class_label == 0))

    # false positives
    fp = numpy.sum((class_pred == 1) & (class_label == 0))

    # false negatives
    fn = numpy.sum((class_pred == 0) & (class_label == 1))

    # compute sensitivity and specificity
    sensitivity = tp / (tp + fn)
    print(sensitivity)
    specificity = tn / (tn + fp)
    print(specificity)
    ### END CODE HERE ###
    return fp,fn
    #return sensitivity, specificity

if __name__ == "__main__":
    if True:
        pre_trained = "F:/Cengiz/Nodules-Detection/models/model_luna16_full__fs_best.hd5"
        if not os.path.exists("workdir_loc/"):
            os.mkdir("workdir_loc")
        train(train_full_set=True, load_weights_path=None, model_name="lidc_loc", fold_count=-1, manual_labels=False)
        if not os.path.exists("models/"):
            os.mkdir("models")
        shutil.copy("workdir/model_lidc_loc__fs_best.hd5", "models/model_lidc_loc__fs_best.hd5")
    # This part to calculate metrics from the model
    if False:
        evaluate(image_label=settings.BASE_DIR_SSD + "Test_data.csv",model_path="models/model_luna16_full__fs_best.hd5")
