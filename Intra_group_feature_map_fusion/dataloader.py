import os, pdb
import pickle
import numpy as np
import copy
import functools
import torch
from torch.utils.data import Dataset

def patch_loader(path, pyramid): #, folder_patch):
    slice_ft_maps = None
    for name in ['cluster_0.pickle','cluster_3.pickle','cluster_6.pickle']:
        path_tmp = os.path.join(path, name)
        slice_ft_map = None
        if name in os.listdir(path):
            f = open(path_tmp, 'rb')
            slice_ft_map = pickle.load(f)
            f.close()
        else:
            # import pdb; pdb.set_trace()
            slice_ft_map = np.zeros((1024*pyramid), dtype='float32')
            # print(slice_ft_map.shape)

        slice_ft_maps = slice_ft_map if slice_ft_maps is None else np.concatenate((slice_ft_maps, slice_ft_map), axis=0)
    # print(slice_ft_maps.shape, slice_ft_maps)
    return slice_ft_maps

def dcm_loader(image_loader, pat_dir, patch_name, pyramid): #, folder_phase):
    dcm = []
    # pdb.set_trace()
    img_path = os.path.join(pat_dir, patch_name)
    image_loader=patch_loader

    return image_loader(img_path, pyramid) #, folder_phase)

def get_default_dcmimage_loader():
    return patch_loader

def get_default_dcm_loader():
    image_loader = patch_loader#get_default_dcmimage_loader()
    return functools.partial(dcm_loader, image_loader=image_loader)

def one_hot_label(label, label_num):
    one_hot = np.zeros((1,label_num),dtype=np.int64)
    one_hot[0,label] = 1
    return one_hot

def make_dataset(train_val_dir):
    dataset = []
    pat_labels = {'AD': 0,
                  'MCI':0,
                   'NC': 1}

    for pat_label in os.listdir(train_val_dir):
        pat_dir = os.path.join(train_val_dir, pat_label)
        pat_label_num = pat_labels[pat_label]

        sample = {
            'pat_label': pat_label_num,   #pat_label_one_hot, #
            'pat_dir': pat_dir
        }

        for patch_name in os.listdir(pat_dir):
            sample_i = copy.deepcopy(sample)
            sample_i['patch_name'] = patch_name
            dataset.append(sample_i)
        '''
        sample = {
            'pat_label': pat_label,  # patient label---normal/abnormal dir
            'pat_dir':, pat_dir      # os.path.join(train_val_dir, pat_label)
            'patch_name': patch_name # name of patient
        }
        pat_dcm_dir = os.path.join(pat_dir, patch_name)
        '''
    # print(train_val_dir, len(dataset))
    return dataset   # dataset is a dictionary, where keys are integers and values are instances of the sample

class DCM(Dataset):
    def __init__(self, pyramid, train_val_dir, spatial_transform=None, get_loader=dcm_loader):#get_default_dcm_loader):
        self.pyramid = pyramid
        self.data = make_dataset(train_val_dir)
        self.loader = get_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        pat_path = self.data[index]['pat_dir']
        patch_name = self.data[index]['patch_name']
        dcm = self.loader(None, pat_path, patch_name, self.pyramid)
        # print('pat_dir: {}'.format(os.path.join(pat_path, patch_name)))
        # if self.spatial_transform is not None:
        #     dcm = self.spatial_transform(dcm)

        pat_label = self.data[index]['pat_label']
        # print('pat_path:{}\npatch_name:{}'.format(pat_path, patch_name))

        return dcm, torch.tensor(pat_label), os.path.join(pat_path, patch_name)

    def __len__(self):
        return len(self.data)
