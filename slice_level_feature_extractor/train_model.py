from options import parse_opts
from count_mean_std import Count_mean_st
from dataloader import DCM
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152  #ResNet #,
from metrices import conf_matrx_cal
import torch, os, time, copy, pickle
import numpy as np
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision import models
from torchsummary import summary
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

class LRScheduler():
    def __init__(self, init_lr=1e-1, lr_decay_epoch=20):
        self.init_lr = init_lr
        self.lr_decay_epoch = lr_decay_epoch

    def __call__(self, optimizer, epoch):
        lr = self.init_lr * (0.1 ** (epoch // self.lr_decay_epoch))
        lr = max(lr, 1e-8)
        if epoch % self.lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

class ClassifyModel():
    def __init__(self, dataset, datasets_len, cluster_id):  # , cuda_id=0 ):
        # print('Initialize model')
        self.datasets_len = datasets_len
        self.dataset = dataset
        self.cluster_id = cluster_id
        self.args = parse_opts()
        self.cuda_id = self.args.cuda_id
        self.categry_f = self.args.categry_f
        self.num_classes = self.args.num_classes
        self.model_name = self.args.model_name
        self.in_channels = self.args.in_channels
        self.input_size = self.args.input_size
        print('Model: {}'.format(self.args.model_name))

        self.model = self._build_model()
        self.best_epoch_num = 0
        self.best_acc = 0.0
        self.best_model = None
        self.input_img_f = self.args.input_img_f
        self.save_dir = os.path.abspath(os.path.join(os.getcwd(), 'best_models', self.input_img_f, self.categry_f))


    def _build_model(self ):

        if self.model_name == 'resnet18':
            model = resnet18(pretrained=False)
        elif self.model_name == 'resnet34':
            model = resnet34(pretrained=True)
        elif self.model_name == 'resnet50':
            model = resnet50(pretrained=True)
        elif self.model_name == 'resnet101':
            model = resnet101(pretrained=True)
        elif self.model_name == 'inceptionv3':
            model = models.inception_v3(pretrained=True)
        weights = torch.load(os.path.join(os.getcwd(), 'pretrained_models', self.model_name+'.pth'))
        model.load_state_dict(weights)

        ft_num = model.fc.in_features
        model.fc = nn.Linear(ft_num, self.args.num_classes)

        ws_tmp = torch.empty([self.args.num_classes, ft_num])
        ws = nn.init.normal_(ws_tmp)
        bs_tmp = torch.empty(self.args.num_classes)
        bs = nn.init.constant_(bs_tmp, 0)

        model.fc.weight.data = ws
        model.fc.bias.data = bs

        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
        return model

    def _train_model(self, epoch):
        print('Train--')
        start = time.time()
        running_loss, running_corrects = .0, .0
        labels_gt, labels_pred = [], []
        self.optimizer = self.lr_scheduler(self.optimizer, epoch)
        self.model.train(True)
        # self.model.to(self.cuda_id)
        self.model.cuda()

        for data in self.dataset['train']:
            inputs, labels, _ = data
            # import pdb; pdb.set_trace()
            inputs = torch.cat((inputs, inputs, inputs),1)
            inputs_new = inputs.cuda()
            labels_new = labels.cuda()

            self.optimizer.zero_grad()  # zero the parameter gradients
            outputs = self.model(inputs_new)  # 1*2


            loss = self.criterion(outputs, labels_new)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()  # running_loss is a Python data
            preds = torch.argmax(outputs.data, dim=1)  # preds is still a tensor
            preds = preds.cpu().numpy()
            labels_new = labels_new.cpu().numpy()
            running_corrects += np.sum(preds == labels_new)

            labels_gt.extend(labels_new)
            labels_pred.extend(preds)
            loss = 0
            torch.cuda.empty_cache()

        data_len = self.datasets_len['train']
        epoch_loss = running_loss / data_len
        epoch_acc = running_corrects / data_len
        end = time.time() - start

        print('Loss: {:.4f}\t Acc: {:.4f}\t{}m{:.2f}s'.format(epoch_loss, epoch_acc,int(end//60), end%60))

        return epoch_acc, epoch_loss, running_loss

    def _eval_process(self, phase, epoch):
        # import pdb; pdb.set_trace()
        print('Evaluation--')
        start = time.time()
        running_loss, running_corrects = .0, .0
        labels_gt, labels_pred = [], []
        with torch.no_grad():
            self.model.eval()
        for data in self.dataset[phase]:
            inputs, labels, _ = data
            inputs = torch.cat((inputs, inputs, inputs), 1)
            # with torch.cuda.device(self.cuda_id):
            #     inputs_new = inputs.to(self.cuda_id)
            #     labels_new = labels.to(self.cuda_id)
            inputs_new = inputs.cuda() #to(self.cuda_id)
            labels_new = labels.cuda() #to(self.cuda_id)

            outputs = self.model(inputs_new)  # 1*2
            del inputs_new
            loss = self.criterion(outputs, labels_new)
            running_loss += loss.item()  # running_loss is a Python data
            preds = torch.argmax(outputs.data, dim=1)  # preds is still a tensor
            preds = preds.cpu().numpy()
            labels_new = labels_new.cpu().numpy()
            running_corrects += np.sum(preds == labels_new)

            labels_gt.extend(labels_new)
            labels_pred.extend(preds)
            preds, labels_new = None, None
            torch.cuda.empty_cache()
            del labels_new, preds

        data_len = self.datasets_len['val']
        epoch_loss = running_loss / data_len
        # epoch_acc = running_corrects / data_len
        epoch_acc,conf_matrx,sen,spe,pre,f1=conf_matrx_cal(labels_gt, labels_pred)
        end = time.time() - start
        print('Loss: {:.4f}\t Acc: {:.4f}\t{}m{:.2f}s'.format(epoch_loss,epoch_acc,int(end//60),end%60))
        print('Conf_matrix: \n', conf_matrx)

        if phase == 'val' and epoch_acc >= self.best_acc:
            self.best_acc = epoch_acc
            self.best_epoch_num = epoch
            self.best_model = copy.deepcopy(self.model)
        return epoch_acc, epoch_loss, running_loss

    def _test_process(self, ):
        print('-'*10, 'Loading model', '-'*10)
        self._load_model('Epoch1_0_acc0.49.pt')
        self.model.cuda() #to(self.cuda_id)
        epoch_acc, epoch_loss, running_loss=self._eval_process( 'val', 0)

    def train_model(self):
        os.environ['CUDA_VISIBLE_DEVICES']=str(self.cuda_id)
        torch.backends.cudnn.enabled = True
        # CUDA_VISIBLE_DEVICES = 1
        self.lr_scheduler = LRScheduler(self.args.lr_init, self.args.lr_decay_epoch)
        writer = SummaryWriter()
        for epoch in range(self.args.epoch_num):
            print('Epoch {}/{}{}'.format(epoch, self.args.epoch_num - 1, '===' * 10))
            epoch_acc_train, epoch_loss_train, running_loss_train = self._train_model(epoch)
            epoch_acc_val, epoch_loss_val, running_loss_val = self._eval_process('val', epoch)

            writer.add_scalars('eval', {'train-loss': epoch_loss_train, 'train-whole-loss':running_loss_train, 'train-acc': epoch_acc_train,
                            'val-loss':epoch_loss_val, 'val-whole-loss':running_loss_val,'val-acc':epoch_acc_val}, epoch)
            # save_dir = os.path.abspath(os.path.join(os.getcwd(), 'best_models', self.input_img_f, self.categry_f))
            # import pdb; pdb.set_trace()
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            modelname='Pretrain_'+self.model_name+'_'+self.cluster_id+'_'+str(self.args.epoch_num)+'_'+str(self.best_epoch_num)+'_acc'+str(round(self.best_acc,4))+'.pt'
            save_path = os.path.join(self.save_dir, modelname)
            torch.save(self.best_model.state_dict(), save_path)
            # with open(save_path, 'wb') as f:
                # pickle.dump(self.best_model.state_dict(), f)
                # f.close()
        # import pdb; pdb.set_trace()
        print('Best val acc: {:4f}\t Best epoch: {}'.format(self.best_acc, self.best_epoch_num))
        writer.close()

        return self.best_acc, self.best_epoch_num




    def _build_ft_extractor(self, ):
        print('-'*10, 'Loading model', '-'*10)
        best_model_filename = None
        pre_filename = modelname='Pretrain_'+self.model_name+'_'+self.cluster_id
        print('pre_filename: ', pre_filename)
        best_model_names = os.listdir(self.save_dir)
        for model_name in best_model_names:
            if pre_filename in model_name:
                best_model_filename = model_name
                break;

        path = os.path.join(self.save_dir, best_model_filename)
        self.model.load_state_dict(torch.load(path))

        modules = list(self.model.children())[:8]   #output shape: [batch_size, 512, 6,7]
        self.ft_extractor = nn.Sequential(*modules)
        return self.ft_extractor

    def _extract_ft_map(self, phase):
        print(phase, ' data--')
        start = time.time()
        self.ft_extractor.to(self.cuda_id)
        ft_maps_all, labels_all, pat_path_all = None, None, []
        for data in self.dataset[phase]:
            inputs, labels, pat_path = data
            inputs = torch.cat((inputs, inputs, inputs),1)
            with torch.cuda.device(self.cuda_id):
                inputs_new = inputs.to(self.cuda_id)

            ft_maps = self.ft_extractor(inputs_new)  # 1*2
            ft_maps = ft_maps.cpu().data.numpy()
            ft_maps_all = ft_maps if ft_maps_all is None else np.concatenate((ft_maps_all, ft_maps), axis=0)
            labels_all = labels if labels_all is None else np.concatenate((labels_all, labels), axis=0)
            pat_path_all.extend(pat_path)
        return ft_maps_all, labels_all, pat_path_all

    def ft_extraction(self, cluster_id, train_val_dir, ft_map_f):
        self.cluster_id = cluster_id
        self._build_ft_extractor()
        for phase in ['val']:  #, 'train'
            print('='*10, phase, '='*10)
            ft_maps_all, labels_all, pat_path_all = self._extract_ft_map(phase)
            write_dir_tmp = train_val_dir.replace(self.input_img_f, ft_map_f)
            write_dir = os.path.join(write_dir_tmp, phase)
            filename = 'ft_map.pickle'
            if not os.path.exists(write_dir):
                os.makedirs(write_dir)
            f = open(os.path.join(write_dir, filename), 'wb')
            pickle.dump(ft_maps_all, f)
            f = open(os.path.join(write_dir, 'labels.pickle'), 'wb')
            pickle.dump(labels_all, f)
            f = open(os.path.join(write_dir,  'pat_path.pickle'), 'wb')
            pickle.dump(pat_path_all, f)

if __name__ == '__main__':
    root = '/media/gll/Data/brain/9_clusters_axis'
    ft_map_f = 'ft_map_folder_7_normalizeslices'

    opts = parse_opts()
    input_img_f = opts.input_img_f
    categry_f, cluster_ids = opts.categry_f, opts.cluster_ids
    cuda = opts.cuda_id
    # import pdb; pdb.set_trace()
    cluster_ids = [ 'cluster_3','cluster_4']  #['cluster_6', 'cluster_0',
    for cluster_id in cluster_ids:
        train_val_dir = os.path.join(root, input_img_f, categry_f, cluster_id)
        print('='*10,categry_f, cluster_id, 'cuda_'+str(cuda), '='*10)

        mean_std = Count_mean_st(train_val_dir, cluster_id, categry_f, input_img_f)
        mean, std = mean_std.read_mean_std()
        spatial_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        # print("="*15, "Loading Data",'='*15)
        data_train = DCM(os.path.join(train_val_dir, 'train'), spatial_transform=spatial_transform)
        print('Train size: ' + str(len(data_train)))
        dataloader_train = DataLoader(data_train, batch_size=opts.batch_size, shuffle=True,
                                      num_workers=opts.n_threads, pin_memory=False)

        data_val = DCM(os.path.join(train_val_dir, 'val'), spatial_transform=spatial_transform)
        print('Validation size: ' + str(len(data_val)))
        dataloader_val = DataLoader(data_val, batch_size=opts.batch_size, shuffle=False,
                                    num_workers=opts.n_threads, pin_memory=False)
        datasets = {'train': dataloader_train, 'val': dataloader_val}
        datasets_len = {'train': len(data_train), 'val': len(data_val)}


        for i in range(5):
            print('--'*5, i, '--'*5)
            classifier = ClassifyModel(datasets,datasets_len, cluster_id)
            best_acc, best_epoch_num = classifier.train_model()

            out_str = '{}  Best val acc: {:4f}   Best epoch: {}'.format(self.cluster_id, self.best_acc,
                                                                        self.best_epoch_num)
            bestmodelname = 'Pretrain_' + self.model_name + '_' + cluster_id + '_' + str(i) + '_best_model.pickle'
            f = open(os.path.join(self.save_dir, bestmodelname), 'wb')
            pickle.dump(out_str, f)
            f.close()

        # classifier._test_process()
        # classifier.ft_extraction(cluster_id, train_val_dir, ft_map_f)
