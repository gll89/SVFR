import os, pickle

def fusion_ft_vector(root):
    cluster_dict = {}
    for cluster_id in os.listdir(root):
        cluster_dir = os.path.join(root, cluster_id)
        pat_path = os.path.join(root, cluster_id, 'pat_path.pickle')
        f = open(pat_path, 'rb')
        pat_path = pickle.load(f)
        f.close()
        import pdb;
        pdb.set_trace()
        pat_path_n = [path.split('/')[-1] for path in pat_path]
        pat_path = os.path.join(root, cluster_id, 'labels.pickle')
        f = open(pat_path, 'rb')
        labels = pickle.load(f)
        f.close()
        pat_path = os.path.join(root, cluster_id, 'ft_vectors.pickle')
        f = open(pat_path, 'rb')
        ft_vectors = pickle.load(f)
        f.close()

        zip_list = zip(pat_path, ft_vectors, labels)
        zip_list = sorted(zip_list, key=lambda  x: x[0])
        pat_path, labels, ft = zip(*zip_list)

def fusion_ft_vector(root_tmp, pyramid_f, input_f, output_f, categry_f):
    for phase in ['train','test', 'val']:
        root = os.path.join(root_tmp, input_f, pyramid_f, categry_f, phase)
        root_dst = root.replace(input_f, output_f)
        # import pdb; pdb.set_trace()
        for cluster_id in os.listdir(root):
            pat_path = os.path.join(root, cluster_id, 'pat_path.pickle')
            f = open(pat_path, 'rb')
            pat_paths = pickle.load(f)
            f.close()
            pat_path_strs = [path.split('/')for path in pat_paths]

            pat_path = os.path.join(root, cluster_id, 'labels.pickle')
            f = open(pat_path, 'rb')
            labels = pickle.load(f)
            f.close()

            pat_path = os.path.join(root, cluster_id, 'ft_vectors.pickle')
            f = open(pat_path, 'rb')
            ft_vectors = pickle.load(f)
            f.close()

            label_dict = {'AD': 0, 'NC': 1}
            for i, path in enumerate(pat_paths):
                pat_str = pat_path_strs[i]
                if labels[i] != label_dict[pat_str[-2]]:
                    print(root_dst, pat_str[-2], 'two labels')
                ft = ft_vectors[i]

                path_dst_n = os.path.join(root_dst, pat_str[-2], pat_str[-1])
                if not os.path.exists(path_dst_n):
                    os.makedirs(path_dst_n)
                path_w = os.path.join(path_dst_n, cluster_id+'.pickle')
                # print(path_w)
                f = open(path_w, 'wb')
                pickle.dump(ft, f)
                f.close()

if __name__ == '__main__':
    root = '/media/gll/Data/brain/Datasets/9_clusters_axis/'
    input_f, output_f = 'sp2_ft_vector_10_3', 'pat_sp2_ft_vector_11_3'
    categry_f = 'AD_NC'
    for pyramid in [2]:
        pyramid_f = str(pyramid)+"_pyramid_layer"
        # root_t = os.path.join(root, pyramid_f)
        fusion_ft_vector(root, pyramid_f, input_f, output_f, categry_f)



