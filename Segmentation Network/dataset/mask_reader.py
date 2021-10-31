import numpy as np
import torch
from torch.utils.data import Dataset
import os
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate
import math
import time
from scipy.ndimage.measurements import label
import nrrd
from utils.util import masks2bboxes_masks_one, pad2factor

class MaskReader(Dataset):
    def __init__(self, data_dir, set_name, cfg, mode='train', split_combiner=None):
        self.mode = mode
        self.cfg = cfg
        self.r_rand = cfg['r_rand_crop']
        self.augtype = cfg['augtype']
        self.pad_value = cfg['pad_value']
        self.data_dir = data_dir
        self.stride = cfg['stride']
        self.blacklist = cfg['blacklist']
        self.set_name = set_name

        labels = [] #contains all the labels loaded from "bboxes.npy"
        self.source = []
        if mode != 'predict':
            if set_name.endswith('.csv'):
                self.filenames = np.genfromtxt(set_name, dtype=str)
            elif set_name.endswith('.npy'):
                self.filenames = np.load(set_name)
        else:
            self.filenames = data_dir.split("/")[-1] #TODO: Complete file name

        if mode != 'test':
            #self.filenames = [self.filenames]
            #pass
            self.filenames = [f for f in self.filenames if (f not in self.blacklist)]
            self.filenames = [f for f in self.filenames]

        for fn in self.filenames:
            # For nodules > 3mm
            if mode != 'predict':
                is_found = True
                if os.path.isfile(os.path.join(data_dir, '%s_bboxes.npy' % fn)):
                    l = np.load(os.path.join(data_dir, '%s_bboxes.npy' % fn))
                else:
                    is_found = False
                if os.path.isfile(os.path.join(data_dir, 'small_%s_bboxes.npy' % fn)):
                    is_found = True
                    l_small = np.load(os.path.join(data_dir, 'small_%s_bboxes.npy' % fn))
                    if l_small != []:
                        l = np.concatenate((l, l_small), axis=0)
                if np.all(l==0):
                    l=np.array([])
                labels.append(l)
                if not is_found:
                    self.filenames.remove(fn)
        print("Total Training Data is: ", len(self.filenames))
        if mode != 'predict':
            self.sample_bboxes = labels
        if self.mode in ['train', 'val', 'eval']:
            self.bboxes = []
            for i, l in enumerate(labels):
                if len(l) > 0 :
                    for t in l:
                        self.bboxes.append([np.concatenate([[i],t])]) #Concatenate all the boxes coordinate with the number of the label
            self.bboxes = np.concatenate(self.bboxes,axis = 0).astype(np.float32) # concatenate based on the labels number
        if mode != 'predict':
            self.crop = Crop(cfg)
        self.split_combiner = split_combiner

    def __getitem__(self, idx):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time
        is_random_img  = False
        if self.mode in ['train', 'val']:
            if idx >= len(self.bboxes):
                is_random_crop = True
                idx = idx % len(self.bboxes)
                is_random_img = np.random.randint(2)
            else:
                is_random_crop = False
        else:
            is_random_crop = False

        if self.mode in ['train', 'val']:
            if not is_random_img:
                bbox = self.bboxes[idx] #imageID,z,y,x,d
                filename = self.filenames[int(bbox[0])]
                imgs = self.load_img(filename)
                masks = self.load_mask(filename)
                    
                bboxes = self.sample_bboxes[int(bbox[0])]
                #if filename == "1.3.6.1.4.1.14519.5.2.1.6279.6001.183184435049555024219115904825":
                #    print("ok")
                do_sacle = self.augtype['scale'] and (self.mode=='train')
                sample, target, masks = self.crop(imgs, bbox[1:], masks, do_sacle, is_random_crop)
                if self.mode == 'train' and not is_random_crop:
                     sample, target, masks = augment(sample, target, masks, 
                                                             do_flip=self.augtype['flip'], do_rotate=self.augtype['rotate'],
                                                             do_swap=self.augtype['swap'])
            else:
                randimid = np.random.randint(len(self.filenames))
                filename = self.filenames[randimid]
                imgs = self.load_img(filename)
                bboxes = self.sample_bboxes[randimid]
                isScale = self.augtype['scale'] and (self.mode=='train')
                print("Section 2 ****** BBOX: ", bboxes)
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes,isScale=False,isRand=True)
                print("Section 2 ****** BBOX: ", bboxes)

            if sample.shape[1] != self.cfg['crop_size'][0] or sample.shape[2] != \
                self.cfg['crop_size'][1] or sample.shape[3] != self.cfg['crop_size'][2]:
                print(filename, sample.shape)

            input = (sample.astype(np.float32) - 128) / 128

            bboxes, truth_masks = masks2bboxes_masks_one(masks, border=self.cfg['bbox_border'])
            truth_masks = np.array(truth_masks).astype(np.uint8)
            bboxes = np.array(bboxes)
            #print(filename, "\t\t\t", bboxes, "\t\t\t", bbox)
            if bboxes is None:
                print(filename)
            try:
                truth_labels = bboxes[:, -1]
            except:
                print(filename)
            truth_bboxes = bboxes[:, :-1]
            masks = np.expand_dims(masks, 0).astype(np.float32)

            return [torch.from_numpy(input).float(), truth_bboxes, truth_labels, truth_masks, masks]

        if self.mode in ['eval']:
            image = self.load_img(self.filenames[idx])
            
            original_image = image[0]

            image = pad2factor(image[0])
            image = np.expand_dims(image, 0)

            mask = self.load_mask(self.filenames[idx])
            mask = pad2factor(mask)
            bboxes, truth_masks = masks2bboxes_masks_one(mask, border=self.cfg['bbox_border'])
            truth_masks = np.array(truth_masks).astype(np.uint8)
            bboxes = np.array(bboxes)
            truth_labels = bboxes[:, -1]
            truth_bboxes = bboxes[:, :-1]
            masks = np.expand_dims(mask, 0).astype(np.float32)

            input = (image.astype(np.float32) - 128.) / 128.

            return [torch.from_numpy(input).float(), truth_bboxes, truth_labels, truth_masks, masks, original_image]


    def __len__(self):
        if self.mode == 'train':
            return int(len(self.bboxes) / (1-self.r_rand))
        elif self.mode =='val':
            return len(self.bboxes)
        else:
            return len(self.filenames)


    def load_img(self, path_to_img):
        path_to_img = str(path_to_img)
        if path_to_img.startswith('LKDS'):
            img = np.load(os.path.join(self.data_dir, '%s_clean.npy' % (path_to_img)))
        else:
            img, _ = nrrd.read(os.path.join(self.data_dir, '%s_clean.nrrd' % (path_to_img)))
        img = img[np.newaxis,...]

        return img


    def load_mask(self, filename):
        mask, _ = nrrd.read(os.path.join(self.data_dir, '%s_mask.nrrd' % (filename)))

        return mask


def pad_to_factor(image, factor=16, pad_value=170):
    _, depth, height, width = image.shape
    d = int(math.ceil(depth / float(factor))) * factor
    h = int(math.ceil(height / float(factor))) * factor
    w = int(math.ceil(width / float(factor))) * factor

    pad = []
    pad.append([0, 0])
    pad.append([0, d - depth])
    pad.append([0, h - height])
    pad.append([0, w - width])

    image = np.pad(image, pad, 'constant', constant_values=pad_value)

    return image

def fillter_box(bboxes, size):
    res = []
    for box in bboxes:
        if np.all(box[:3] - box[-1] / 2 > 0) and np.all(box[:3] + box[-1] / 2 < size):
            res.append(box)
    return np.array(res)

def augment(sample, target, masks, do_flip = True, do_rotate=True, do_swap = True):
    masks = (masks > 0).astype(np.int32)
    if do_rotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand()*180
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1/180*np.pi),-np.sin(angle1/180*np.pi)],[np.sin(angle1/180*np.pi),np.cos(angle1/180*np.pi)]])
            newtarget[1:3] = np.dot(rotmat,target[1:3]-size/2)+size/2
            if np.all(newtarget[:3]>target[3]) and np.all(newtarget[:3]< np.array(sample.shape[1:4])-newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample,angle1,axes=(2,3),reshape=False)
                masks = rotate(masks, angle1, axes=(1,2), reshape=False)
            else:
                counter += 1
                if counter ==3:
                    break
    if do_swap:
        if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
            coord = np.transpose(coord,np.concatenate([[0],axisorder+1]))
            target[:3] = target[:3][axisorder]
            bboxes[:,:3] = bboxes[:,:3][:,axisorder]

    if do_flip:
#         flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2-1
        sample = np.ascontiguousarray(sample[:,::flipid[0],::flipid[1],::flipid[2]])
        masks = np.ascontiguousarray(masks[::flipid[0],::flipid[1],::flipid[2]])

        for ax in range(3):
            if flipid[ax]==-1:
                target[ax] = np.array(sample.shape[ax+1])-target[ax]

    masks, num = label((masks > 0.5).astype(np.int32))
    return sample, target, masks

class Crop(object):
    def __init__(self, config):
        self.crop_size = config['crop_size'] # Box size
        self.bound_size = config['bound_size']
        self.stride = config['stride']
        self.pad_value = config['pad_value'] #The mask color

    def __call__(self, imgs, target, masks, do_scale=False, isRand=False):
        masks = (masks > 0).astype(np.int32)
        if do_scale and target[3] > 6:
            radiusLim = [8.,120.]
            scaleLim = [0.75,1.25]
            scaleRange = [np.min([np.max([(radiusLim[0]/target[3]),scaleLim[0]]),1])
                         ,np.max([np.min([(radiusLim[1]/target[3]),scaleLim[1]]),1])]
            scale = np.random.rand()*(scaleRange[1]-scaleRange[0])+scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float') / scale).astype('int')
        else:
            crop_size=self.crop_size
        bound_size = self.bound_size
        target = np.copy(target)
        #print(masks.max())
        tt_tt = masks[masks >0].shape
        start = []
        debug_var_start_rand = [0,0]
        debug_var_start_rand_2 = 0
        for i in range(3): #z,y,x
            #s and e are used to select the starting point for each axes
            if not isRand:
                r = target[3] / 2
                s = np.floor(target[i] - r) + 1 - bound_size
                e = np.ceil(target[i] + r) + 1 + bound_size - crop_size[i]
                debug_var_start_rand = [0,1]
            else:
                s = np.max([imgs.shape[i+1]-crop_size[i]/2,imgs.shape[i+1]/2+bound_size]) #since the images are list of 3D
                e = np.min([crop_size[i]/2,              imgs.shape[i+1]/2-bound_size])
                target = np.array([np.nan,np.nan,np.nan,np.nan])
                debug_var_start_rand = [1,0]
            if s>e: #s is the center - radius | e is the center + radius - crop size
                start.append(np.random.randint(e,s))# the start point is a random number btw center and (center - crop size)
                debug_var_start_rand_2 = [0,debug_var_start_rand]
            else:
                start.append(int(target[i])-crop_size[i]/2+np.random.randint(-bound_size/2,bound_size/2))
                debug_var_start_rand_2 = [1, debug_var_start_rand]
        debug_var_all_index_1 = np.argwhere(masks > 0)

        pad = []
        pad.append([0,0]) # we are dealing with list of 3D this is for the list idx
        for i in range(3): # Added padding if the cropped image frame is outside the original image area
            leftpad = max(0,-start[i])
            rightpad = max(0,start[i]+crop_size[i]-imgs.shape[i+1])
            pad.append([leftpad,rightpad])
        crop = imgs[:,
            max(start[0],0):min(start[0] + crop_size[0], imgs.shape[1]),
            max(start[1],0):min(start[1] + crop_size[1], imgs.shape[2]),
            max(start[2],0):min(start[2] + crop_size[2], imgs.shape[3])]
        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)
        masks = masks[
            max(start[0],0):min(start[0] + crop_size[0], imgs.shape[1]),
            max(start[1],0):min(start[1] + crop_size[1], imgs.shape[2]),
            max(start[2],0):min(start[2] + crop_size[2], imgs.shape[3])]
        masks = np.pad(masks, pad[1:], 'constant', constant_values=0) # Added padding so the size will be = crop_size
        #print(masks.shape, '\t\t\t',masks.max())# should be equal to crop_size , 1
        for i in range(3):
            target[i] = target[i] - start[i]
        tt_tt_2 = np.count_nonzero(masks)
        if do_scale and target[3] > 6:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop, [1, scale, scale, scale], order=1) #The array is zoomed using spline interpolation of the requested order.
                masks = zoom(masks, [scale, scale, scale], order=1) #WARNING: Small nodule may disappear
                #print(masks.shape, '\t\t\t', masks.max())
            newpad = self.crop_size[0] - crop.shape[1:][0] # To make sure the image size is the same as the original crop_size
            if newpad<0:
                crop = crop[:,:-newpad,:-newpad,:-newpad]
                masks = masks[:-newpad,:-newpad,:-newpad]
                #print(masks.shape, '\t\t\t', masks.max())
            elif newpad>0:
                pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)
                masks = np.pad(masks, pad2[1:], 'constant', constant_values=0)
                #print(masks.shape, '\t\t\t', masks.max())

            for i in range(4):
                target[i] = target[i]*scale
        masks, num = label((masks > 0.5).astype(np.int32))
        if num == 0:
            print("Size of the images before: ",tt_tt,"\tSize of the nodules after: ",tt_tt_2)
            print("Error",scale,"\tstarting point: ",start)
            print(masks.shape, '\t\t\t', masks.max())
            print("Target ", target)
            print("coordinate for the cropped image X: ",max(start[0],0),"\tTO\t",min(start[0] + crop_size[0], imgs.shape[1]))
            print("coordinate for the cropped image Y: ", max(start[1], 0), "\tTO\t",
                  min(start[1] + crop_size[1], imgs.shape[2]))
            print("coordinate for the cropped image Z: ", max(start[2], 0), "\tTO\t",
                  min(start[2] + crop_size[2], imgs.shape[3]))
            print("Coordinate for the mask > 1 before cropping: ",debug_var_all_index_1)
            print("starting code selecting: ",debug_var_start_rand_2)

        return crop, target, masks


# def collate(batch):
#     if torch.is_tensor(batch[0]):
#         return [b.unsqueeze(0) for b in batch]
#     elif isinstance(batch[0], np.ndarray):
#         return batch
#     elif isinstance(batch[0], int):
#         return torch.LongTensor(batch)
#     elif isinstance(batch[0], collections.Iterable):
#         transposed = zip(*batch)
#         return [collate(samples) for samples in transposed]
#
# def collate2(batch):
#     batch_size = len(batch)
#     #for b in range(batch_size): print (batch[b][0].size())
#     inputs    = torch.stack([batch[b][0]for b in range(batch_size)], 0)
#     boxes     =             [batch[b][1]for b in range(batch_size)]
#     labels    =             [batch[b][2]for b in range(batch_size)]
#     target    =   torch.stack([batch[b][3]for b in range(batch_size)], 0)
#     coord    =   torch.stack([batch[b][4]for b in range(batch_size)], 0)
#
#     return [inputs, boxes, labels, target, coord]
#
# def eval_collate(batch):
#     batch_size = len(batch)
#     #for b in range(batch_size): print (batch[b][0].size())
#     inputs    = torch.stack([batch[b][0] for b in range(batch_size)], 0)
#     boxes     =             [batch[b][1] for b in range(batch_size)]
#     labels    =             [batch[b][2] for b in range(batch_size)]
#     images    =             [batch[b][3] for b in range(batch_size)]
#     coord    =   torch.stack([batch[b][4]for b in range(batch_size)], 0)
#
#     return [inputs, boxes, labels, images, coord]
