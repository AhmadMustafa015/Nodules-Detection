import csv
import os
import SimpleITK
import settings
import glob
import ntpath
import PySimpleGUI as sg
import cv2
import numpy as np
#name = input("Enter your name: ")
path = settings.BASE_DIR +'1.3.6.1.4.1.14519.5.2.1.6279.6001.330643702676971528301859647742_annos_pos_lidc.csv'
name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.330643702676971528301859647742'
direc = "D:/LIDC-IDRI/LIDC-IDRI-0802/1.3.6.1.4.1.14519.5.2.1.6279.6001.322126192251489550021873181090/1.3.6.1.4.1.14519.5.2.1.6279.6001.330643702676971528301859647742"
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
        if True:
            patientid_loc.append(a)
            slice_x.append(row[1])
            slice_y.append(row[2])
            slice_z.append(row[3])

        a += 1
        #print(row)
        '''
        if a == 10:
            break
        '''
b = 1

src_dir = settings.LIDC_RAW_SRC_DIR
src_path = []
"""
for src_p in glob.glob(src_dir + "*/*/*/*30.dcm"):
    src_p = os.path.split(src_p)[0]
    src_path.append(src_p)

for src_p in src_path:
    b += 1
    patient_id2 = ntpath.basename(src_p)
    if name == patient_id2:
        files_dir = os.listdir(src_p)
        _,file_extention = os.path.splitext(files_dir[0])
        #if file_extention == ".dcm":
        reader = SimpleITK.ImageSeriesReader()
        # Use the functional interface to read the image series.
        itk_img = SimpleITK.ReadImage(reader.GetGDCMSeriesFileNames(src_p, patient_id2))
        #elif file_extention == ".mhd":
        #itk_img = SimpleITK.ReadImage(src_p)
        img_array = SimpleITK.GetArrayFromImage(itk_img)
c =1
"""
reader = SimpleITK.ImageSeriesReader()
itk_img = SimpleITK.ReadImage(reader.GetGDCMSeriesFileNames(direc, name))
img_array = SimpleITK.GetArrayFromImage(itk_img)
def main():
    sg.theme("LightGreen")

    # Define the window layout
    layout = [
        [sg.Text("OpenCV Demo", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="-IMAGE-")],
        [sg.Radio("None", "Radio", True, size=(10, 1))],
        [
            sg.Radio("threshold", "Radio", size=(10, 1), key="-THRESH-"),
            sg.Slider(
                (0, 255),
                128,
                1,
                orientation="h",
                size=(40, 15),
                key="-THRESH SLIDER-",
            ),
        ],
        [
            sg.Radio("canny", "Radio", size=(10, 1), key="-CANNY-"),
            sg.Slider(
                (0, 255),
                128,
                1,
                orientation="h",
                size=(20, 15),
                key="-CANNY SLIDER A-",
            ),
            sg.Slider(
                (0, 255),
                128,
                1,
                orientation="h",
                size=(20, 15),
                key="-CANNY SLIDER B-",
            ),
        ],
        [
            sg.Radio("blur", "Radio", size=(10, 1), key="-BLUR-"),
            sg.Slider(
                (1, 11),
                1,
                1,
                orientation="h",
                size=(40, 15),
                key="-BLUR SLIDER-",
            ),
        ],
        [
            sg.Radio("hue", "Radio", size=(10, 1), key="-HUE-"),
            sg.Slider(
                (0, 225),
                0,
                1,
                orientation="h",
                size=(40, 15),
                key="-HUE SLIDER-",
            ),
        ],
        [
            sg.Radio("enhance", "Radio", size=(10, 1), key="-ENHANCE-"),
            sg.Slider(
                (1, 255),
                128,
                1,
                orientation="h",
                size=(40, 15),
                key="-ENHANCE SLIDER-",
            ),
        ],
        [sg.Button("Exit", size=(10, 1))],
    ]

    # Create the window and show it without the plot
    window = sg.Window("OpenCV Integration", layout, location=(800, 400))

    cap = cv2.VideoCapture(0)
    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        ret, frame = cap.read()

        if values["-THRESH-"]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
            frame = cv2.threshold(
                frame, values["-THRESH SLIDER-"], 255, cv2.THRESH_BINARY
            )[1]
        elif values["-CANNY-"]:
            frame = cv2.Canny(
                frame, values["-CANNY SLIDER A-"], values["-CANNY SLIDER B-"]
            )
        elif values["-BLUR-"]:
            frame = cv2.GaussianBlur(frame, (21, 21), values["-BLUR SLIDER-"])
        elif values["-HUE-"]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame[:, :, 0] += int(values["-HUE SLIDER-"])
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        elif values["-ENHANCE-"]:
            enh_val = values["-ENHANCE SLIDER-"] / 40
            clahe = cv2.createCLAHE(clipLimit=enh_val, tileGridSize=(8, 8))
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

    window.close()

main()

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