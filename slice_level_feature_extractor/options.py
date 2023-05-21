import os, time, pdb
import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='resnet18', type=str, help='options contain resnet18, 34, 50, 152, 22')
    parser.add_argument('--param_filename', default='resnet18.pth', type=str, help='parameters of pre-trained 3D ResNet')
    parser.add_argument('--batch_size', default=80, type=int, help='Batch size')
    parser.add_argument('--epoch_num', default=60, type=int, help='Epoch number')
    parser.add_argument('--n_threads', default=0, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--lr_init', default=1e-3, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay_epoch', default=15, type=int, help='Initial decay of epoch')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of category')
    parser.add_argument('--input_size', default = (1, 75, 360, 300), type=tuple, help='Input size with the shape of 4')
    parser.add_argument('--in_channels', default=1, type=int, help='Number of the channels of input data')
    parser.add_argument('--categry_f', default='AD_NC', type=str, help='category folder')
    parser.add_argument('--cluster_ids', default=['cluster_0','cluster_1','cluster_2','cluster_3','cluster_4','cluster_5','cluster_6','cluster_7','cluster_8'], type=str, help='cluster id') # 'cluster_5',
    parser.add_argument('--cuda_id', default=1, type=int, help='cuda id')
    parser.add_argument('--input_img_f', default='cluster_train_val_test_norm_slice_6', help='input_dataset_folder')
    # parser.add_argument('--input_img_f', default='normalize_cluster_train_val_6_v1_normalizeslices', help='input_dataset_folder')


    args = parser.parse_args()

    return args
