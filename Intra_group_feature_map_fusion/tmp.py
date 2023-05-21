import os

train_list = []
val_list = []
test_list = []

file_root = '/media/gll/Data/brain/Datasets/9_clusters_axis/cluster_train_val_test_5/AD_NC'

for root, drs, files in os.walk(file_root):
    for file in files:
        temp = os.path.join(root,file)
        image_id = file.split('_I')[1].split('_')[0]
        if '/train/' in temp:
            if image_id not in train_list:
                train_list.append(image_id)
        if '/val/' in temp:
            if image_id not in val_list:
                val_list.append(image_id)
        if '/test/' in temp:
            if image_id not in test_list:
                test_list.append(image_id)
print(len(train_list))
print(len(val_list))
print(len(test_list))
print(len(train_list)+len(val_list)+len(test_list))
print(train_list)
print(val_list)
print(test_list)
