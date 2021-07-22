import sys
sys.path.append('../../')
from pylung.annotation import *
from tqdm import tqdm
import sys
import nrrd
import SimpleITK as sitk
import cv2
from config import config
import glob

def load_itk_image(filename,scan_extension,patient_id):
    """Return img array and [z,y,x]-ordered origin and spacing
    """
    #TODO: Distinguash if it's DCM or mhd
    if scan_extension == "dcm":     # Use the functional interface to read the image series.
        reader = sitk.ImageSeriesReader()
        itkimage = sitk.ReadImage(reader.GetGDCMSeriesFileNames(filename, patient_id))
    else:
        itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing


def xml2mask(xml_file):
    header, annos = parse(xml_file)
    """
    This part extract the nodules information from LIDC xml files
    annos --> have info about nodules found per doctor (4 doctors mean the size of this list will be 4)
    reader.(nodules or non_nodules or small_nodules) --> info about each finding
    roi --> info per slice that the nodule spread to it.
    """
    ctr_arrs = [] #only contains nodules > 3 mm
    #TODO: Added a list for small nodules
    s_ctr_arrs = [] #contains info about small nodules < 3mm
    for i, reader in enumerate(annos):
        for j, nodule in enumerate(reader.nodules):
            ctr_arr = []
            for k, roi in enumerate(nodule.rois):
                z = roi.z
                for roi_xy in roi.roi_xy:
                    ctr_arr.append([z, roi_xy[1], roi_xy[0]])
            ctr_arrs.append(ctr_arr)
        for j, nodule in enumerate(reader.small_nodules):
            s_ctr_arr = []
            for k, roi in enumerate(nodule.rois):
                z = roi.z
                for roi_xy in roi.roi_xy:
                    s_ctr_arr.append([z, roi_xy[1], roi_xy[0]])
            s_ctr_arrs.append(s_ctr_arr)

    seriesuid = header.series_instance_uid
    return seriesuid, ctr_arrs, s_ctr_arrs


def annotation2masks(annos_dir, save_dir):
    files = find_all_files(annos_dir, '.xml')
    for f in tqdm(files, total=len(files)): #tqdm to show progress par
        try:
            seriesuid, masks,s_masks = xml2mask(f)
            np.save(os.path.join(save_dir, '%s' % (seriesuid)), masks)
            np.save(os.path.join(save_dir, 'small_%s' % (seriesuid)), s_masks)
        except:
            print("Unexpected error:", sys.exc_info()[0])
    
def arr2mask(arr, reso):
    mask = np.zeros(reso)
    arr = arr.astype(np.int32)
    mask[arr[:, 0], arr[:, 1], arr[:, 2]] = 1
    
    return mask

def arrs2mask(img_dir, ctr_arr_dir, save_dir,scan_extension):
    #TODO: CHANGE TO DCM FORMAT
    if scan_extension == "dcm":
        src_path = []
        pids = []
        for src_p in glob.glob(img_dir + "*/*/*/*30.dcm"):
            src_p = os.path.split(src_p)[0]
            id_p = os.path.basename(src_p)
            if id_p not in pids:
                src_path.append(src_p)
                pids.append(id_p)
    else:
        pids = [f[:-4] for f in os.listdir(img_dir) if f.endswith('.mhd')]
    cnt = 0
    consensus = {1: 0, 2: 0, 3: 0, 4: 0} # represent the agreement level between doctors
    
    for k in consensus.keys():
        if not os.path.exists(os.path.join(save_dir, str(k))):
            os.makedirs(os.path.join(save_dir, str(k)))

    for counter, pid in enumerate(tqdm(pids, total=len(pids))): #loop per CT scan
        if scan_extension == "dcm":
            img_dir = src_path[counter]
            img, origin, spacing = load_itk_image(img_dir,scan_extension,pid)
        else:
            img, origin, spacing = load_itk_image(os.path.join(img_dir, '%s.mhd' % (pid)),scan_extension,pid)
        ctr_arrs = np.load(os.path.join(ctr_arr_dir, '%s.npy' % (pid)),allow_pickle=True)
        cnt += len(ctr_arrs) #ctr give you the total number of processed nodules

        nodule_masks = [] # a list of 3D (image size) info for all nodules in a slice
        for ctr_arr in ctr_arrs: #loop nodules per CT scans
            #convert z from voxel to pixel
            z_origin = origin[0]
            z_spacing = spacing[0]
            ctr_arr = np.array(ctr_arr) # [z, roi_xy[1], roi_xy[0]]
            ctr_arr[:, 0] = np.absolute(ctr_arr[:, 0] - z_origin) / z_spacing
            ctr_arr = ctr_arr.astype(np.int32)

            mask = np.zeros(img.shape) # nodules mask per slice
            #iterate over each slice and creat a mask by filling the mask array to 1
            for z in np.unique(ctr_arr[:, 0]): # loop per unique slice
                ctr = ctr_arr[ctr_arr[:, 0] == z][:, [2, 1]] # get X and Y coordinate only for nodules in slice z
                ctr = np.array([ctr], dtype=np.int32)
                mask[z] = cv2.fillPoly(mask[z], ctr, color=(1,) * 1) # create a white point in the mask for the nodules
            nodule_masks.append(mask)

        i = 0 # when the loop will finish
        visited = []
        d = {}
        masks = [] # list of lists of the same or close nodules
        #calculate the iou if it's > 0.4 then we saw this nodule before (exclude)
        #Advantage of this part:  1- save important info about nodules 1- how many times it occures 2- the iou
        while i < len(nodule_masks): # iterate over nodules masks per a slice, size of nodules masks will be the number of slices that had nodules
            # If matched before, then no need to create new mask
            if i in visited:
                i += 1
                continue
            same_nodules = [] # the nodule and the other nodules that are close to it
            mask1 = nodule_masks[i]
            same_nodules.append(mask1)
            d[i] = {}
            d[i]['count'] = 1 # how many times the nodules repeated in the scan
            d[i]['iou'] = []

            # Find annotations pointing to the same nodule
            for j in range(i + 1, len(nodule_masks)): # iterate over nodules masks per a slice AFTER i
                # if not overlapped with previous added nodules
                if j in visited:
                    continue
                mask2 = nodule_masks[j]
                # mask1 is the first loop mask2 second loop
                iou = float(np.logical_and(mask1, mask2).sum()) / np.logical_or(mask1, mask2).sum() # to

                if iou > 0.4:
                    visited.append(j)
                    same_nodules.append(mask2)
                    d[i]['count'] += 1
                    d[i]['iou'].append(iou)

            masks.append(same_nodules) # list of lists of the same or close nodules
            i += 1

        for k, v in d.items(): # k is the number of times that we repeated the nodule in the scan, v is the iou for each nodule with its pairs
            if v['count'] > 4: # To keep the count 4 because the total number of radioligst is 4
                print('WARNING:  %s: %dth nodule, iou: %s' % (pid, k, str(v['iou'])))
                v['count'] = 4
            consensus[v['count']] += 1

        # number of consensus
        num = np.array([len(m) for m in masks])
        num[num > 4] = 4 # To keep the count 4 because the total number of radioligst is 4

        if len(num) == 0: # a list is empty skip
            continue
        # Iterate from the nodules with most consensus
        for n in range(num.max(), 0, -1):
            mask = np.zeros(img.shape, dtype=np.uint8)

            for i, index in enumerate(np.where(num >= n)[0]):
                same_nodules = masks[index] # extract all nodules list that occurs n times
                m = np.logical_or.reduce(same_nodules) # will apply or to all similar nodules that are agreed by n doctors
                mask[m] = i + 1 # will save each nodule places in the mask with a unique number starting from 1
            nrrd.write(os.path.join(save_dir, str(n), pid), mask) # save each nodule in a separate nrrd file
        
#         for i, same_nodules in enumerate(masks):
#             cons = len(same_nodules)
#             if cons > 4:
#                 cons = 4
#             m = np.logical_or.reduce(same_nodules)
#             mask[m] = i + 1
#             nrrd.write(os.path.join(save_dir, str(cons), pid), mask)
        
    print(consensus)
    print("Total number of processed nodules in all the scans (number of scans ",len(pids),") is: ",cnt)
if __name__ == '__main__':
    annos_dir = config['annos_dir']
    img_dir = config['data_dir']
    ctr_arr_save_dir = config['ctr_arr_save_dir']
    mask_save_dir = config['mask_save_dir']
    scan_extension = config['scan_extension']
    os.makedirs(ctr_arr_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    #annotation2masks(annos_dir, ctr_arr_save_dir)
    arrs2mask(img_dir, ctr_arr_save_dir, mask_save_dir,scan_extension)
