import os, time, pdb
import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='resnet18', type=str, help='options contain resnet18, 34, 50, 152, 22')
    parser.add_argument('--param_filename', default='Pretrain_resnet18_cluster_4_50_7_acc0.7438.pt', type=str, help='parameters of pre-trained 3D ResNet')
    parser.add_argument('--batch_size', default=300, type=int, help='Batch size')
    parser.add_argument('--epoch_num', default=60, type=int, help='Epoch number')
    parser.add_argument('--n_threads', default=0, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--lr_init', default=1e-3, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay_epoch', default=10, type=int, help='Initial decay of epoch')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of category')
    parser.add_argument('--categry_f', default='AD_NC', type=str, help='category folder')
    parser.add_argument('--cuda_id', default=1, type=int, help='cuda id')
    parser.add_argument('--input_img_f', default='pat_sp2_ft_vector_11', help='input_dataset_folder')

    args = parser.parse_args()

    return args
