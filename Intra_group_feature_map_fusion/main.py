from options import parse_opts
from dataloader import DCM
from metrices import conf_matrx_cal
import torch, os, time, copy, pickle
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from FusionModel import FusionModel

print('Fusion final model')
# class LRScheduler():
#     def __init__(self, init_lr=1e-1, lr_decay_epoch=20):
#         self.init_lr = init_lr
#         self.lr_decay_epoch = lr_decay_epoch
#
#     def __call__(self, optimizer, epoch):
#         lr = self.init_lr * (0.1 ** (epoch // self.lr_decay_epoch))
#         lr = max(lr, 1e-8)
#         if epoch % self.lr_decay_epoch == 0:
#             print('LR is set to {}'.format(lr))
#
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#
#         return optimizer

class LRScheduler():
    def __init__(self, init_lr=1e-1, lr_decay_epoch=20):
        self.init_lr = init_lr
        self.lr_decay_epoch = lr_decay_epoch

    def __call__(self, optimizer, epoch):
        if epoch < 30:
            lr = self.init_lr * (10**(epoch//self.lr_decay_epoch))
            if epoch % self.lr_decay_epoch == 0:
                print('LR is set to {}'.format(lr))
        else:
            lr = self.init_lr * (0.1 ** (epoch // self.lr_decay_epoch))
            lr = max(lr, 1e-8)
            if epoch % self.lr_decay_epoch == 0:
                print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer


class TrainModel():
    def __init__(self, model, dataset, datasets_len, pyramid_f): #cluster_id
        # print('Initialize model')
        self.model = model
        self.datasets_len = datasets_len
        self.dataset = dataset
        # self.cluster_id = cluster_id
        self.pyramid_f = pyramid_f
        self.args = parse_opts()
        self.cuda_id = self.args.cuda_id
        self.input_img_f = self.args.input_img_f
        self.best_epoch_num = 0
        self.best_acc = 0.0
        self.best_model = None
        self.categry_f = self.args.categry_f
        self.save_dir = os.path.abspath(os.path.join(os.getcwd(), 'best_models', self.input_img_f, self.categry_f, self.pyramid_f, 'best_model'))
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3) #, momentum=0.9, weight_decay=1e-3)

    def _train_model(self, epoch):
        print('Train--')
        start = time.time()
        running_loss, running_corrects = .0, .0
        labels_gt, labels_pred = [], []
        # import pdb; pdb.set_trace()
        self.optimizer = self.lr_scheduler(self.optimizer, epoch)
        self.model.train(True)
        self.model.cuda()
        # import pdb; pdb.set_trace()
        for data in self.dataset['train']:
            inputs, labels,_ = data
            # print("inputs.shape: ", inputs.shape)
            # with torch.cuda.device(self.cuda_id):
            inputs_new = inputs.cuda()
            labels_new = labels.cuda()
            # print('train, input_size:', inputs_new.shape)

            self.optimizer.zero_grad()  # zero the parameter gradients
            # import pdb;
            # pdb.set_trace()
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
        epoch_acc, conf_matrx, sen, spe, pre, f1 = conf_matrx_cal(labels_gt, labels_pred)
        print(conf_matrx)
        # print('Conf_matrix: \n', conf_matrx)

        return epoch_acc, epoch_loss, running_loss

    def _eval_process(self, phase, epoch):
        # import pdb; pdb.set_trace()
        # print(pnhase, '--')
        start = time.time()
        running_loss, running_corrects = .0, .0
        labels_gt, labels_pred = [], []
        with torch.no_grad():
            self.model.eval()
        for data in self.dataset[phase]:
            inputs, labels, _ = data
            inputs_new = inputs.cuda()
            labels_new = labels.cuda()
            # print('train, input_size:', inputs_new.shape)

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
        epoch_acc,conf_matrx,sen,spe,pre,f1=conf_matrx_cal(labels_gt, labels_pred)
        end = time.time() - start
        print('Loss: {:.4f}\t Acc: {:.4f}\t{}m{:.2f}s'.format(epoch_loss,epoch_acc,int(end//60),end%60))
        print(conf_matrx)
        # print('Conf_matrix: \n', conf_matrx)

        if phase == 'val' and epoch_acc >= self.best_acc:
            self.best_acc = epoch_acc
            self.best_epoch_num = epoch
            self.best_model = copy.deepcopy(self.model)
        return epoch_acc, epoch_loss, running_loss


    def _test_process(self, ):
        # print('-'*10, 'Loading model', '-'*10)
        best_model_filename = None
        pre_filename = 'FusionModel'
        print('pre_filename: ', pre_filename)
        best_model_names = os.listdir(self.save_dir)
        for model_name in best_model_names:
            if pre_filename in model_name:
                best_model_filename = model_name
                break;
        print('best_model_filename: ', best_model_filename)

        path = os.path.join(self.save_dir, best_model_filename)
        self.model.load_state_dict(torch.load(path))

        self.model.cuda()
        # import pdb; pdb.set_trace()
        print('='*10, 'Val', '='*10)
        self._eval_process( 'val')
        print('=' * 10, 'Test', '=' * 10)
        self._eval_process('test')

    def train_model(self):
        os.environ['CUDA_VISIBLE_DEVICES']=str(self.cuda_id)
        self.lr_scheduler = LRScheduler(self.args.lr_init, self.args.lr_decay_epoch)
        for epoch in range(self.args.epoch_num):
            print('Epoch {}/{}{}'.format(epoch, self.args.epoch_num - 1, '===' * 10))
            epoch_acc_train, epoch_loss_train, running_loss_train = self._train_model(epoch)
            epoch_acc_val, epoch_loss_val, running_loss_val = self._eval_process('val', epoch)

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            modelname='FusionModel_'+str(self.args.epoch_num)+'_'+str(self.best_epoch_num)+'_acc'+str(round(self.best_acc,4))+'.pt'
            save_path = os.path.join(self.save_dir, modelname)
            torch.save(self.best_model.state_dict(), save_path)

        print('Best val acc: {:4f}\t Best epoch: {}'.format(self.best_acc, self.best_epoch_num))



if __name__ == '__main__':
    root = '/media/gll/Data/brain/Datasets/9_clusters_axis'
    opts = parse_opts()
    input_img_f = opts.input_img_f
    categry_f, cuda = opts.categry_f, opts.cuda_id


    for pyramid in [2]:  #1, 2,3
        pyramid_f = str(pyramid)+"_pyramid_layer"

        train_val_dir = os.path.join(root, input_img_f, pyramid_f, categry_f)
        print('='*10, pyramid_f, categry_f, 'cuda_'+str(cuda), '='*10)

        for i in range(1):
            print('*'*10, i, '*'*10)
            data_train = DCM(pyramid, os.path.join(train_val_dir, 'train'))
            print('Train size: ' + str(len(data_train)))
            dataloader_train = DataLoader(data_train, batch_size=opts.batch_size, shuffle=True,
                                          num_workers=opts.n_threads, pin_memory=False)

            data_val = DCM(pyramid, os.path.join(train_val_dir,  'val'))
            print('Validation size: ' + str(len(data_val)))
            dataloader_val = DataLoader(data_val, batch_size=opts.batch_size, shuffle=False,
                                        num_workers=opts.n_threads, pin_memory=False)
            data_test = DCM(pyramid, os.path.join(train_val_dir, 'test'))
            print('Test size: ' + str(len(data_test)))
            dataloader_test = DataLoader(data_test, batch_size=opts.batch_size, shuffle=False,
                                        num_workers=opts.n_threads, pin_memory=False)

            datasets = {'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test}
            datasets_len = {'train': len(data_train), 'val': len(data_val), 'test': len(data_test)}
            # datasets = {'val': dataloader_val, 'test': dataloader_test}
            # datasets_len = { 'val': len(data_val), 'test': len(data_test)}

            model = FusionModel(pyramid)
            train_model = TrainModel(model,datasets,datasets_len, pyramid_f)
            train_model.train_model()
            # train_model._test_process()