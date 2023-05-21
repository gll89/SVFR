import numpy as np
import pdb, os
import pickle
import torch
from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader

class Count_mean_st():

    def __init__(self, root, cluster_id, categry_f, num_cluster_f):
        '''
        Calculate the mean and std of the training and validation datasets
        :param root: root of the training and validation datasets
        '''
        self.root = root
        self.cluster_id = cluster_id
        self.categry_f = categry_f
        self.num_cluster_f = num_cluster_f
        self.write_filename = 'mean_std_' + self.cluster_id + '.pkl'
        self.write_dir = os.path.abspath(os.path.join(os.curdir, 'output_files', self.categry_f, self.num_cluster_f))

    def count_mean_std(self):
        print('='*15, "Count mean and std",'='*15)
        sum, sum_sq, img_num = 0.0, 0.0, 0.0
        # import pdb;
        # pdb.set_trace()
        for phase_folder in ['train', 'val']:
            print(phase_folder)
            dir = os.path.join(self.root, self.categry_f+'_NC', self.cluster_id, phase_folder)
            for categry_f_tmp in os.listdir(dir):
                for filename in os.listdir(os.path.join(dir, categry_f_tmp)):
                    if not filename.endswith('.npy'):
                        continue

                    # print(os.path.join(dir, categry_f_tmp, filename))
                    img = np.load(os.path.join(dir, categry_f_tmp, filename))
                    # import pdb;pdb.set_trace()
                    # sum_tmp = np.sum(img != 0)
                    # if sum_tmp > 0:
                    #     img_num += sum_tmp

                    tmp = img.astype(np.float64)  # patch.dtype=int16, patches.max()=900. This means 900**2 is far greater than int16
                    if np.sum(tmp) > 0:
                        sum += np.sum(tmp)
                        sum_sq += np.sum(tmp**2)
                        img_num += np.sum(tmp != 0)


        self.mean = 1.0*sum/img_num
        sq_mean = 1.0*sum_sq/img_num
        # import pdb;
        # pdb.set_trace()

        self.std = np.sqrt(sq_mean - self.mean**2)
        print('mean: {}\tstd: {}'.format(self.mean, self.std))
        return self.mean, self.std

    def write_mean_std(self):
        norm = {
            'mean': self.mean,
            'std': self.std
        }
        # import pdb; pdb.set_trace()
        if not os.path.exists(self.write_dir):
            os.makedirs(self.write_dir)

        with open(os.path.join(self.write_dir, self.write_filename), 'wb') as f:
            pickle.dump(norm, f)
            f.close()

    def read_mean_std(self):
        # import pdb; pdb.set_trace()
        if os.path.exists(os.path.join(self.write_dir, self.write_filename)):
            with open(os.path.join(self.write_dir, self.write_filename), 'rb') as f:
                norm = pickle.load(f)
                self.mean = norm['mean']
                self.std = norm['std']
            print('mean:{}\tstd:{}'.format(self.mean, self.std))
        else:
            self.mean, self.std = self.count_mean_std()
            self.write_mean_std()
        return self.mean, self.std

# if __name__ == '__main__':
#     root = '/media/gll/Data/brain/9_clusters_axis/normalize_cluster_train_val_6_v1_normalizeslices'
#     # get writing path
#     cluster_id, categry_f =  'AD'
#     for categry_f in ['AD', 'MCI']:
#         for id in range(9): #['cluster_0', 'cluster_1']:
#             cluster_id = 'cluster_'+str(id)
#             count_mean_std = Count_mean_st(root, cluster_id, categry_f, 'normalize_cluster_train_val_6_v1_normalizeslices')
#             mean, std=count_mean_std.read_mean_std()
