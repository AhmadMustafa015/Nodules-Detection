import csv
import os
import SimpleITK
import settings
import glob
import ntpath

#name = input("Enter your name: ")
path = 'C:/Users/RadioscientificOne/PycharmProjects/NoduleDetect/Nodules-Detection/Data/lidc_annotations.csv'
name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.303494235102183795724852353824'
patientid_loc = []
slice_x = []
slice_y = []
slice_z = []

with open(path, 'r') as file:
    reader = csv.reader(file)
    a = 0
    for row in reader:
        if a == 0:
            a += 1
            continue

        patient_id = row[0]
        if patient_id == name:
            patientid_loc.append(a)
            slice_x.append(row[2])
            slice_y.append(row[3])
            slice_z.append(row[4])

        a += 1
        #print(row)
        '''
        if a == 10:
            break
        '''
b = 1

src_dir = settings.LIDC_RAW_SRC_DIR
src_path = []

for src_p in glob.glob(src_dir + "*/*/*/*30.dcm"):
    src_p = os.path.split(src_p)[0]
    src_path.append(src_p)

for src_p in src_path:
    b += 1
    patient_id2 = ntpath.basename(src_p)
    if name == patient_id2:
        files_dir = os.listdir(src_p)
        _,file_extention = os.path.splitext(files_dir[0])
        if file_extention == ".dcm":
            reader = SimpleITK.ImageSeriesReader()
            # Use the functional interface to read the image series.
            itk_img = SimpleITK.ReadImage(reader.GetGDCMSeriesFileNames(src_p, patient_id2))
        elif file_extention == ".mhd":
            itk_img = SimpleITK.ReadImage(src_p)
        img_array = SimpleITK.GetArrayFromImage(itk_img)
c =1

"""

        extract_dicom_images_patient(src_p)

        dir_path = src_dir

        print("Dir_Path: ", src_dir)
        patient_id = os.path.basename(src_dir)
        search_dirs = os.listdir(settings.LIDC_EXTRACTED_IMAGE_DIR)
        scan_path = settings.LIDC_EXTRACTED_IMAGE_DIR + patient_id + "/img_0033_i.png"
        if os.path.exists(scan_path):
            return


        
"""