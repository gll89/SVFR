import os, pdb
import pickle
import numpy as np
import copy
import functools
import torch
from torch.utils.data import Dataset

def patch_loader(path): #, folder_patch):
    # pdb.set_trace()
    # print('Read {}'.format(path))
    # path = os.path.join(path, os.listdir(path)[0])
    # f = open(path, 'rb')
    # patches_valid = pickle.load(f)
    # f.close()
    # return patches_valid
    return np.load(path)

def dcm_loader(image_loader, pat_dir, patch_name): #, folder_phase):
    dcm = []
    # pdb.set_trace()
    img_path = os.path.join(pat_dir, patch_name)
    image_loader=patch_loader

    return image_loader(img_path) #, folder_phase)


def get_default_dcmimage_loader():
    # pdb.set_trace()
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
    label_num = len(pat_labels)

    for pat_label in os.listdir(train_val_dir):
        pat_dir = os.path.join(train_val_dir, pat_label)
        pat_label_num = pat_labels[pat_label]
        pat_label_one_hot = one_hot_label(pat_label_num, label_num)
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
    def __init__(self, train_val_dir, spatial_transform=None, get_loader=dcm_loader):#get_default_dcm_loader):
        # self.folder_phase = folder_phase
        self.data = make_dataset(train_val_dir)
        # print(train_val_dir, len(self.data))
        # pdb.set_trace()
        self.spatial_transform = spatial_transform
        self.loader = get_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # pdb.set_trace()
        pat_path = self.data[index]['pat_dir']
        patch_name = self.data[index]['patch_name']
        dcm = self.loader(None, pat_path, patch_name)
        # print('pat_dir: {}'.format(os.path.join(pat_path, patch_name)))
        if self.spatial_transform is not None:
            dcm = self.spatial_transform(dcm)

        pat_label = self.data[index]['pat_label']
        # print('pat_path:{}\npatch_name:{}'.format(pat_path, patch_name))

        return dcm, torch.tensor(pat_label), os.path.join(pat_path, patch_name)

    def __len__(self):
        return len(self.data)
