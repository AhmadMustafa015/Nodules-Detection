import itk

import settings
import helpers
import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import numpy as np
import pandas
import ntpath
import cv2  # conda install -c https://conda.anaconda.org/menpo opencv3
import shutil
import random
import math
from multiprocessing import Pool
from bs4 import BeautifulSoup  # conda install beautifulsoup4, coda install lxml
import os
import glob
import nrrd
from matplotlib import pyplot as plt
import scipy.misc
import pydicom as dicom  # pip install pydicom
only_slice_thik = True
random.seed(1321)
np.random.seed(1321)
slices_thick_info = []
def find_mhd_file(patient_id):
    # for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
    src_dir = settings.LUNA16_RAW_SRC_DIR  # + "subset" + str(subject_no) + "/"
    src_paths = glob.glob(src_dir + "*.mhd")
    for src_path in glob.glob(src_dir + "*.mhd"):
        if patient_id in src_path:
            return src_path
    return None
def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def process_image(src_path):
    patient_id = ntpath.basename(src_path).replace(".mhd", "")
    print("Patient: ", patient_id)

    dst_dir = settings.LNDB_EXTRACTED_IMAGE_DIR + patient_id + "/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)

    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)

    direction = np.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)


    spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    img_array = helpers.rescale_patient_images(img_array, spacing, settings.TARGET_VOXEL_MM)
    img_list = []
    segimg_list = []
    for i in range(img_array.shape[0]):
        img = img_array[i]
        seg_img, mask = helpers.get_segmented_lungs(img.copy())
        segimg_list.append(seg_img)
        img = normalize(img)
        img_list.append(img)
        cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_i.png", img * 255)
        cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_m.png", mask * 255)
    if settings.save_image_as == "original":
        nrrd.write(os.path.join(dst_dir,"img_" + '_i.nrrd'), np.array(img_list))
        nrrd.write(os.path.join(dst_dir,"img_" +'_m.nrrd'), np.array(segimg_list))


def process_images(delete_existing=False, only_process_patient=None,datatype="Train"):
    if datatype == "Train":
        src_dir = settings.LNDB_RAW_TRAIN_DIR
    else:
        src_dir = settings.LNDB_RAW_TEST_DIR

    if delete_existing and os.path.exists(settings.LNDB_EXTRACTED_IMAGE_DIR):
        print("Removing old stuff..")
        if os.path.exists(settings.LNDB_EXTRACTED_IMAGE_DIR):
            shutil.rmtree(settings.LNDB_EXTRACTED_IMAGE_DIR)
    if not os.path.exists(settings.LNDB_EXTRACTED_IMAGE_DIR):
        os.mkdir(settings.LNDB_EXTRACTED_IMAGE_DIR)
        os.mkdir(settings.LNDB_EXTRACTED_IMAGE_DIR + "_labels/")

    # for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
    src_paths = glob.glob(src_dir + "*.mhd")

    if only_process_patient is None and True:
        pool = Pool(settings.WORKER_POOL_SIZE)
        pool.map(process_image, src_paths)
    else:
        for src_path in src_paths:
            print(src_path)
            if only_process_patient is not None:
                if only_process_patient not in src_path:
                    continue
            process_image(src_path)


def nodEqDiam(vol):
    # Calc nodule equivalent diameter from volume vol
    return 2 * (vol * 3 / (4 * math.pi)) ** (1 / 3)


def joinNodules(nodules, verb=False):
    # join nodules from different radiologists (if within radiusor 3mm from each other)
    header = nodules[0]
    lines = nodules[1:]
    lndind = header.index('LNDbID')
    radind = header.index('RadID')
    fndind = header.index('FindingID')
    xind = header.index('x')
    yind = header.index('y')
    zind = header.index('z')
    nodind = header.index('Nodule')
    volind = header.index('Volume')
    texind = header.index('Text')
    LND = [int(line[lndind]) for line in lines]

    # Match nodules
    nodules = [['LNDbID', 'RadID', 'RadFindingID', 'FindingID', 'x', 'y', 'z', 'AgrLevel', 'Nodule', 'Volume', 'Text']]
    for lndU in np.unique(LND):  # within each CT
        nodlnd = [line for lnd, line in zip(LND, lines) if lnd == lndU]
        dlnd = [nodEqDiam(float(n[volind])) for n in nodlnd]
        nodnew = []  # create merged nodule list
        dnew = []
        for iself, n in enumerate(nodlnd):  # for each nodule
            dself = dlnd[iself] / 2
            rself = n[radind]
            nodself = n[nodind]
            match = False
            for inew, nnew in enumerate(nodnew):  # check distance with every nodule in merged list
                if not float(rself) in nnew[radind] and float(nodself) in nnew[nodind]:
                    dother = max(dnew[inew]) / 2
                    dist = ((float(n[xind]) - np.mean(nnew[xind])) ** 2 +
                            (float(n[yind]) - np.mean(nnew[yind])) ** 2 +
                            (float(n[zind]) - np.mean(nnew[zind])) ** 2) ** .5
                    if dist < max(max(dself, dother),
                                  3):  # if distance between nodules is smaller than maximum radius or 3mm
                        match = True
                        for f in range(len(n)):
                            nnew[f].append(float(n[f]))  # append to existing nodule in merged list
                        dnew[inew].append(np.mean([nodEqDiam(v) for v in nodnew[inew][volind]]))
                        break
            if not match:
                nodnew.append([[float(l)] for l in n])  # otherwise append new nodule to merged list
                dnew.append([np.mean([nodEqDiam(v) for v in nodnew[-1][volind]])])

            if verb:
                print(iself)
                for inew, nnew in enumerate(nodnew):
                    print(nnew, dnew[inew])

        # Merge matched nodules
        for ind, n in enumerate(nodnew):
            agrlvl = n[nodind].count(1.0)
            if agrlvl > 0:  # nodules
                nod = 1
                vol = np.mean(n[volind])  # volume is the average of all radiologists
                tex = np.mean(n[texind])  # texture is the average of all radiologists
            else:  # non-nodules
                nod = 0
                vol = 4 * math.pi * 1.5 ** 3 / 3  # volume is the minimum for equivalent radius 3mm
                tex = 0
            nodules.append([int(n[lndind][0]),
                            ','.join([str(int(r)) for r in n[radind]]),  # list radiologist IDs
                            ','.join([str(int(f)) for f in n[fndind]]),  # list radiologist finding's IDs
                            ind + 1,  # new finding ID
                            np.mean(n[xind]),  # centroid is the average of centroids
                            np.mean(n[yind]),
                            np.mean(n[zind]),
                            agrlvl,  # number of radiologists that annotated the finding (0 if non-nodule)
                            nod,
                            vol,
                            tex])
    if verb:
        for n in nodules:
            print(n)
    return nodules
def process_lndb_annotations(only_patient=None, agreement_threshold=0,nodules_less_3mm = "False"):
    # lines.append(",".join())
    file_no = 0
    pos_count = 0
    neg_count = 0
    all_lines = []
    pos_lines = []
    # for anno_dir in [d for d in glob.glob(settings.LIDC_RAW_SRC_DIR+"*/*/*") if os.path.isdir(d)]:
    dispFlag = True

    # Read nodules csv
    csvlines = helpers.readCsv(settings.LNDB_EXTRACTED_IMAGE_DIR+'_labels/trainNodules_gt.csv')
    header = csvlines[0]
    nodules = csvlines[1:]

    lndloaded = -1

    for n in nodules:
        vol = float(n[header.index('Volume')])
        if nodules_less_3mm == "True" or (nodules_less_3mm == "False" and nodEqDiam(vol) > 3):  # only get nodule cubes for nodules>3mm
            ctr = np.array([float(n[header.index('x')]), float(n[header.index('y')]), float(n[header.index('z')])])
            lnd = int(n[header.index('LNDbID')])
            rads = list(map(int, list(n[header.index('RadID')].split(','))))
            radfindings = list(map(int, list(n[header.index('RadFindingID')].split(','))))
            finding = int(n[header.index('FindingID')])
            agrlevel= int(n[header.index('AgrLevel')])
            #print(lnd, finding, rads, radfindings, agrlevel)


            # Read scan
            if lnd != lndloaded:
                if not only_patient == None and not only_patient == 'LNDb-{:04}'.format(lnd):
                    continue
                print(file_no," Reading scan: ", 'LNDb-{:04}'.format(lnd))
                if len(pos_lines) > 0:
                    df_annos = pandas.DataFrame(pos_lines, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter"])
                    df_annos.to_csv(settings.LNDB_EXTRACTED_IMAGE_DIR + "_labels/" + 'LNDb-{:04}'.format(lndloaded) + "_annos_pos_lndb.csv",
                                    index=False)
                    pos_lines = []
                [scan, spacing, origin, transfmat] = helpers.readMhd('D:/LNDB/Train/LNDb-{:04}.mhd'.format(lnd))
                print("Origin: ", origin, "\t Spacing: ", spacing, "\t Orientation: ", transfmat)
                image_arry = scan
                transfmat_toimg, transfmat_toworld = helpers.getImgWorldTransfMats(spacing, transfmat)
                lndloaded = lnd
                file_no+=1


            # Convert coordinates to image
            ctr = helpers.convertToImgCoord(ctr, origin, transfmat_toimg)
            x_center_perc = round(ctr[0] / image_arry.shape[2], 4)
            y_center_perc = round(ctr[1] / image_arry.shape[1], 4)
            z_center_perc = round(ctr[2] / image_arry.shape[0], 4)
            diameter_perc = nodEqDiam(vol)/max(image_arry.shape[2],image_arry.shape[1])
            line = [finding,x_center_perc,y_center_perc,z_center_perc,diameter_perc]
            if agreement_threshold > 1:
                if agreement_threshold >= agrlevel:
                    pos_lines.append(line)
                    pos_count += 1
                    all_lines.append(line)
            else:
                pos_lines.append(line)
                pos_count +=1
                all_lines.append(line)
            """
            for rad, radfinding in zip(rads, radfindings):
                # Read segmentation mask
                [mask, _, _, _] = helpers.readMhd('D:/LNDB/Train/Extra/masks/masks/LNDb-{:04}_rad{}.mhd'.format(lnd, rad))

                # Extract cube around nodule
                scan_cube = helpers.extractCube(scan, spacing, ctr)
                file_name = 'nodule_segment_LNDb-' + str(lnd).rjust(4, '0') + "_" + str(rad)
                # Extract cube around nodule
                scan_cube = helpers.extractCube(scan, spacing, ctr)
                masknod = helpers.copy.copy(mask)
                masknod[masknod != radfinding] = 0
                masknod[masknod > 0] = 1
                mask_cube = helpers.extractCube(masknod, spacing, ctr)
                file_name = 'nodule_segment_LNDb-' + str(lnd).rjust(4, '0') + "_" + str(rad)
                # Display mid slices from resampled scan/mask
                if dispFlag:
                    fig, axs = plt.subplots(2, 3)
                    axs[0, 0].imshow(scan_cube[int(scan_cube.shape[0] / 2), :, :])
                    axs[1, 0].imshow(mask_cube[int(mask_cube.shape[0] / 2), :, :])
                    axs[0, 1].imshow(scan_cube[:, int(scan_cube.shape[1] / 2), :])
                    axs[1, 1].imshow(mask_cube[:, int(mask_cube.shape[1] / 2), :])
                    axs[0, 2].imshow(scan_cube[:, :, int(scan_cube.shape[2] / 2)])
                    axs[1, 2].imshow(mask_cube[:, :, int(mask_cube.shape[2] / 2)])
                    plt.savefig('F:/Cengiz/Nodules-Detection/LNDB_nodule_pic/' + file_name + '.png')
                    plt.show()
            """
            #diameter = max(x_diameter, y_diameter)
            #diameter_perc = round(max(x_diameter / img_array.shape[2], y_diameter / img_array.shape[1]), 4)
            # extended_line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy, sphericiy, margin, spiculation, texture, calcification, internal_structure, lobulation, subtlety ]
    print("Total number of positive nodules: ", pos_count,"\t Total number of negatives: ", neg_count)
    df_annos = pandas.DataFrame(all_lines, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter"])
    df_annos.to_csv(settings.LNDB_EXTRACTED_IMAGE_DIR + "_labels/" + "all_nodules.csv",index=False)

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


if __name__ == "__main__":
    if False:
        print("step 1 Process images...")
        process_images(delete_existing=False, only_process_patient=None,datatype="Train")
        if only_slice_thik == True:
            df_annos = pandas.DataFrame(slices_thick_info,columns=["patient_id", "slice_thickness"])
            df_annos.to_csv(settings.BASE_DIR + "slices_thickness.csv", index=False)
    if False:
        print("Step 2 merge nodules that within 3mm radius from different radiologist: ")
        # Merge nodules from train set
        prefix = 'train'
        fname_gtNodulesFleischner = settings.LNDB_RAW_TRAIN_DIR +"_labels/" + '{}Nodules.csv'.format(prefix)
        gtNodules = pandas.readcsv(fname_gtNodulesFleischner)
        for line in gtNodules:
            print(line)
        gtNodules = joinNodules(gtNodules)
        df = pandas.DataFrame(gtNodules[1:],columns=gtNodules[0])
        df.to_csv(settings.LNDB_EXTRACTED_IMAGE_DIR + '_labels/'+'{}Nodules_gt.csv'.format(prefix), index=False)
    if True:
        print("step 3 Generate labels: ")
        process_lndb_annotations(only_patient=None, agreement_threshold=0,nodules_less_3mm = "False")
