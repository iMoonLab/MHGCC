import yaml
import argparse
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import sys
import traceback
import time
import shutil
import inspect
from collections import OrderedDict
import pickle
import glob
import utils
from tqdm import tqdm
from loss import *  # coxph_loss, mse_loss
import json


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='multi-modal survival prediction')
    parser.add_argument('--config', default='config/gsz/wsi.yaml', help='path to the configuration file')
    parser.add_argument('--work_dir', default='./work_dir/', help='the work folder for storing results')

    parser.add_argument('--phase', default='train', help='must be train or test')
    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--print_log', default=True, help='print logging or not')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=0, help='the start epoch to save model (#iteration)')
    parser.add_argument('--draw', default=False, help='if draw KM curve')

    # data_loader
    parser.add_argument('--n_fold', type=int, default=5, help='the num of fold for cross validation')
    parser.add_argument('--start_fold', type=int, default=0, help='the start fold for cross validation')
    parser.add_argument('--dataset', default='dataset.WSI_Dataset.SlidePatch', help='data set will be used')
    parser.add_argument('--data_seed', type=int, default=1, help='random seed for n_fold dataset')
    parser.add_argument('--drop_sample_num', type=int, default=None, nargs='+',
                        help='the num of dropping uncensored sample')
    parser.add_argument('--WSI_patch_ft_dir', help='path to the feature of WSI patch')
    parser.add_argument('--WSI_patch_coor_dir', help='path to the feature of WSI patch coor file')
    parser.add_argument('--CT_3d_feature_file', help='path to the feature of CT file')
    parser.add_argument('--CT_2d_feature_file', help='path to the feature of CT file')
    parser.add_argument('--Cli_feature_file', help='path to the feature of Cli file')

    parser.add_argument('--num_worker', type=int, default=4, help='the number of worker for data loader')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size')

    # model
    parser.add_argument('--H_coors', default=False, help='if use the coors of patches to create H')
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model_args', default=dict(), help='the arguments of model')
    parser.add_argument('--model_hyconv', default=None, help='the model will be used')
    parser.add_argument('--model_hyconv_args', default=dict(), help='the arguments of model')
    parser.add_argument('--model_wsi', default=None, help='the model will be used')
    parser.add_argument('--model_wsi_args', default=dict(), help='the arguments of model')
    parser.add_argument('--model_hyconv_wsi', default=None, help='the model will be used')
    parser.add_argument('--model_hyconv_wsi_args', default=dict(), help='the arguments of model')
    parser.add_argument('--model_HC', default=None, help='the model will be used')
    parser.add_argument('--model_HC_args', default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+',
                        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--optimizer_hyconv', default='Adam', help='type of optimizer')
    parser.add_argument('--optimizer_wsi', default='SGD', help='type of optimizer')
    parser.add_argument('--optimizer_hyconv_wsi', default='Adam', help='type of optimizer')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--step', type=int, default=100,
                        help='the epoch where optimizer reduce the learning rate')  # , nargs='+'
    parser.add_argument('--step_hyconv', type=int, default=100,
                        help='the epoch where optimizer reduce the learning rate')  # , nargs='+'
    parser.add_argument('--step_wsi', type=int, default=100,
                        help='the epoch where optimizer reduce the learning rate')  # , nargs='+'
    parser.add_argument('--step_hyconv_wsi', type=int, default=100,
                        help='the epoch where optimizer reduce the learning rate')  # , nargs='+'
    parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num_epoch', type=int, default=300, help='stop training in which epoch')
    parser.add_argument('--num_epoch_hyconv', type=int, default=300, help='stop training in which epoch')
    parser.add_argument('--num_epoch_wsi', type=int, default=300, help='stop training in which epoch')
    parser.add_argument('--num_epoch_hyconv_wsi', type=int, default=300, help='stop training in which epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--weight_decay_hyconv_wsi', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--weight_decay_hyconv', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--loss', type=str, default='loss.mse_loss', help='the type of loss function')
    parser.add_argument('--lamb', type=float, default=0.0, help='the weight of loss function')
    parser.add_argument('--lamb_ds', type=float, default=1, help='the weight of loss function')
    parser.add_argument('--lamb_hc', type=float, default=1, help='the weight of loss function')

    return parser


class Processor():
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()

        self.load_data()
        self.load_model()
        self.load_optimizer()

        self.lr = self.arg.base_lr
        self.best_i_fold_acc = 0
        self.best_i_fold_acc_epoch = 0
        self.best_acc = 0
        self.best_i_fold = 0
        self.best_epoch = 0

        self.best_i_fold_acc_hyconv = 0
        self.best_i_fold_acc_epoch_hyconv = 0
        self.best_acc_hyconv = 0
        self.best_i_fold_hyconv = 0
        self.best_epoch_hyconv = 0

        self.lr_wsi = self.arg.base_lr
        self.best_i_fold_acc_wsi = 0
        self.best_i_fold_acc_epoch_wsi = 0
        self.best_i_fold_loss_wsi = float('inf')
        self.best_i_fold_loss_epoch_wsi = 0
        self.best_i_fold_wsi = 0
        self.best_acc_wsi = 0
        self.best_epoch_wsi = 0

        self.best_i_fold_acc_hyconv_wsi = 0
        self.best_i_fold_acc_epoch_hyconv_wsi = 0
        self.best_i_fold_loss_hyconv_wsi = float('inf')
        self.best_i_fold_loss_epoch_hyconv_wsi = 0
        self.best_i_fold_hyconv_wsi = 0
        self.best_acc_hyconv_wsi = 0
        self.best_epoch_hyconv_wsi = 0

        self.model = self.model.cuda(self.output_device)
        self.model_hyconv = self.model_hyconv.cuda(self.output_device)
        self.model_wsi = self.model_wsi.cuda(self.output_device)
        self.model_hyconv_wsi = self.model_hyconv_wsi.cuda(self.output_device)
        self.loss = nn.CrossEntropyLoss()

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def load_data(self):
        dataset = import_class(self.arg.dataset)
        self.data_loader = dict()
        WSI_info_list = utils.get_WSI_sample_list(self.arg.WSI_patch_ft_dir, self.arg.WSI_patch_coor_dir,
                                                  self.arg.CT_3d_feature_file,
                                                  self.arg.CT_2d_feature_file,
                                                  self.arg.Cli_feature_file)  # , multi_label=True
        n_fold_train_list, n_fold_val_list = utils.get_n_fold_data_list(WSI_info_list, self.arg.n_fold,
                                                                        self.arg.data_seed)

        self.data_loader['train'] = []
        self.data_loader['val'] = []
        for i in range(len(n_fold_train_list)):
            self.data_loader['train'].append(torch.utils.data.DataLoader(
                dataset=dataset(n_fold_train_list[i]),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed))
            self.data_loader['val'].append(torch.utils.data.DataLoader(
                dataset=dataset(n_fold_val_list[i]),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed))

    def load_model(self, i=0):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device

        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        if isinstance(self.arg.model_args, str):
            self.arg.model_args = json.loads(self.arg.model_args)
        self.model = Model(**self.arg.model_args)
        print(self.model)

        Model_hyconv = import_class(self.arg.model_hyconv)
        shutil.copy2(inspect.getfile(Model_hyconv), self.arg.work_dir)
        print(Model_hyconv)
        if isinstance(self.arg.model_hyconv_args, str):
            self.arg.model_hyconv_args = json.loads(self.arg.model_hyconv_args)
        self.model_hyconv = Model_hyconv(**self.arg.model_hyconv_args)
        print(self.model_hyconv)

        Model_wsi = import_class(self.arg.model_wsi)
        shutil.copy2(inspect.getfile(Model_wsi), self.arg.work_dir)
        print(Model_wsi)
        if isinstance(self.arg.model_wsi_args, str):
            self.arg.model_wsi_args = json.loads(self.arg.model_wsi_args)
        self.model_wsi = Model_wsi(**self.arg.model_wsi_args)
        print(self.model_wsi)

        Model_hyconv_wsi = import_class(self.arg.model_hyconv_wsi)
        shutil.copy2(inspect.getfile(Model_hyconv_wsi), self.arg.work_dir)
        print(Model_hyconv_wsi)
        if isinstance(self.arg.model_hyconv_wsi_args, str):
            self.arg.model_hyconv_wsi_args = json.loads(self.arg.model_hyconv_wsi_args)
        self.model_hyconv_wsi = Model_hyconv_wsi(**self.arg.model_hyconv_wsi_args)
        print(self.model_hyconv_wsi)

        Model_HC = import_class(self.arg.model_HC)
        shutil.copy2(inspect.getfile(Model_HC), self.arg.work_dir)
        print(Model_HC)
        if isinstance(self.arg.model_HC_args, str):
            self.arg.model_HC_args = json.loads(self.arg.model_HC_args)
        self.model_HC = Model_HC(**self.arg.model_HC_args)
        print(self.model_HC)

        if self.arg.weights:
            #### load model weight starts
            weights = os.path.join(self.arg.weights, str(i) + '_fold_best_model.pt')
            self.print_log('Load weights from {}.'.format(weights))
            if '.pkl' in weights:
                with open(weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
            #### load model weight ends

            #### load model_hyconv weight starts
            weights_hyconv = os.path.join(self.arg.weights, str(i) + '_fold_best_model_hyconv.pt')
            self.print_log('Load weights from {}.'.format(weights_hyconv))
            if '.pkl' in weights_hyconv:
                with open(weights_hyconv, 'r') as f:
                    weights_hyconv = pickle.load(f)
            else:
                weights_hyconv = torch.load(weights_hyconv)

            keys_hyconv = list(weights_hyconv.keys())
            for w in self.arg.ignore_weights:
                for key in keys_hyconv:
                    if w in key:
                        if weights_hyconv.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))
            try:
                self.model_hyconv.load_state_dict(weights_hyconv)
            except:
                state_hyconv = self.model_hyconv.state_dict()
                diff_hyconv = list(set(state_hyconv.keys()).difference(set(weights_hyconv.keys())))
                print('Can not find these weights:')
                for d in diff_hyconv:
                    print('  ' + d)
                state_hyconv.update(weights_hyconv)
                self.model_hyconv.load_state_dict(state_hyconv)
            #### load model_hyconv weight ends

            #### load model_wsi weight starts
            weights_wsi = os.path.join(self.arg.weights, str(i) + '_fold_best_model_wsi.pt')
            self.print_log('Load weights from {}.'.format(weights_wsi))
            if '.pkl' in weights_wsi:
                with open(weights_wsi, 'r') as f:
                    weights_wsi = pickle.load(f)
            else:
                weights_wsi = torch.load(weights_wsi)

            keys_wsi = list(weights_wsi.keys())
            for w in self.arg.ignore_weights:
                for key in keys_wsi:
                    if w in key:
                        if weights_wsi.pop(key, None) is not None:
                            print('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            print('Can Not Remove Weights: {}.'.format(key))
            try:
                self.model_wsi.load_state_dict(weights_wsi)
            except:
                state_wsi = self.model_wsi.state_dict()
                diff_wsi = list(set(state_wsi.keys()).difference(set(weights_wsi.keys())))
                print('Can not find these weights:')
                for d in diff_wsi:
                    print('  ' + d)
                state_wsi.update(weights_wsi)
                self.model_wsi.load_state_dict(state_wsi)
            #### load model_wsi weight ends

            #### load model_hyconv_wsi weight starts
            weights_hyconv_wsi = os.path.join(self.arg.weights, str(i) + '_fold_best_model_hyconv_wsi.pt')
            self.print_log('Load weights from {}.'.format(weights_hyconv_wsi))
            if '.pkl' in weights_hyconv_wsi:
                with open(weights_hyconv_wsi, 'r') as f:
                    weights_hyconv_wsi = pickle.load(f)
            else:
                weights_hyconv_wsi = torch.load(weights_hyconv_wsi)

            keys_hyconv_wsi = list(weights_hyconv_wsi.keys())
            for w in self.arg.ignore_weights:
                for key in keys_hyconv_wsi:
                    if w in key:
                        if weights_hyconv_wsi.pop(key, None) is not None:
                            print('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            print('Can Not Remove Weights: {}.'.format(key))
            try:
                self.model_hyconv_wsi.load_state_dict(weights_hyconv_wsi)
            except:
                state_hyconv_wsi = self.model_hyconv_wsi.state_dict()
                diff_hyconv_wsi = list(set(state_hyconv_wsi.keys()).difference(set(weights_hyconv_wsi.keys())))
                print('Can not find these weights:')
                for d in diff_hyconv_wsi:
                    print('  ' + d)
                state_hyconv_wsi.update(weights_hyconv_wsi)
                self.model_hyconv_wsi.load_state_dict(state_hyconv_wsi)
            #### load model_hyconv_wsi weight ends

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                weight_decay=self.arg.weight_decay)  # self.model.parameters(), filter(lambda p: p.requires_grad, self.model.parameters()),
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=self.arg.step,
                                                  gamma=self.arg.lr_decay_rate)

        if self.arg.optimizer_hyconv == 'SGD':
            self.optimizer_hyconv = optim.SGD(
                self.model_hyconv.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                weight_decay=self.arg.weight_decay_hyconv)  # self.model.parameters(), filter(lambda p: p.requires_grad, self.model.parameters()),
        elif self.arg.optimizer_hyconv == 'Adam':
            self.optimizer_hyconv = optim.Adam(
                self.model_hyconv.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay_hyconv)
        else:
            raise ValueError()
        self.scheduler_hyconv = lr_scheduler.MultiStepLR(self.optimizer_hyconv, milestones=self.arg.step_hyconv,
                                                  gamma=self.arg.lr_decay_rate)

        if self.arg.optimizer_wsi == 'SGD':
            self.optimizer_wsi = optim.SGD(
                self.model_wsi.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer_wsi == 'Adam':
            self.optimizer_wsi = optim.Adam(
                self.model_wsi.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        self.scheduler_wsi = lr_scheduler.StepLR(self.optimizer_wsi, step_size=self.arg.step_wsi,
                                                  gamma=self.arg.lr_decay_rate)

        if self.arg.optimizer_hyconv_wsi == 'SGD':
            self.optimizer_hyconv_wsi = optim.SGD(
                self.model_hyconv_wsi.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                weight_decay=self.arg.weight_decay_hyconv_wsi)
        elif self.arg.optimizer_hyconv_wsi == 'Adam':
            self.optimizer_hyconv_wsi = optim.Adam(
                self.model_hyconv_wsi.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay_hyconv_wsi)
        else:
            raise ValueError()
        self.scheduler_hyconv_wsi = lr_scheduler.MultiStepLR(self.optimizer_hyconv_wsi, milestones=self.arg.step_hyconv_wsi,
                                                  gamma=self.arg.lr_decay_rate)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def concat(self, a, b):
        if a is None:
            return b
        else:
            a = torch.concat((a, b), dim=0)
            return a

    def compute_loss(self, sorted_output, sorted_gt, sorted_status, model, features):
        if 'coxph_loss' in self.arg.loss:
            loss = (self.loss(sorted_output, sorted_status)).sum()  # coxph_loss
        elif 'bcr_with_mse_loss' in self.arg.loss:
            loss = (self.loss(sorted_output, sorted_gt, sorted_status, features)).sum()
        elif 'mse_loss' in self.arg.loss or 'coxph_with_mse_loss' in self.arg.loss:
            loss = (self.loss(sorted_output, sorted_gt, sorted_status)).sum()  # mse_loss
        else:
            loss = (self.loss(sorted_output, sorted_gt, sorted_status, model)).sum()  # nll
        return loss

    def contrastive_loss(self, x, x_aug, label, T=0.5):
        """
        :param x: the hidden vectors of original data
        :param x_aug: the positive vector of the auged data
        :param T: temperature
        :return: loss
        """
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        a = torch.einsum('ik,jk->ij', x, x_aug)
        b = torch.einsum('i,j->ij', x_abs, x_aug_abs)

        sim_matrix = a / b
        sim_matrix = torch.exp(sim_matrix / T)

        label_pos = torch.eq(label.unsqueeze(1), label.unsqueeze(1).T).int()
        label_neg = label_pos * -1 + 1
        pos_sim = sim_matrix * label_pos
        neg_sim = sim_matrix * label_neg
        loss = pos_sim.sum(1) / neg_sim.sum(1)

        loss = -torch.log(loss).mean()
        return loss

    def train(self, epoch, i_fold, save_model=False):
        self.model.train()
        self.print_log('Ct representation Training epoch: {} , n_fold: {}'.format(epoch + 1, i_fold))
        loader = self.data_loader['train'][i_fold]

        loss_value = []
        # acc_value = []
        all_label = None
        all_output = None
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        for batch_idx, (features, coors, ct_3d_features, axial, sagittal, coronal, clinical_fts, label, id) in enumerate(process):
            with torch.no_grad():
                ct_3d_features = ct_3d_features.float().cuda(self.output_device)
                axial = axial.float().cuda(self.output_device)
                sagittal = sagittal.float().cuda(self.output_device)
                coronal = coronal.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()
            _, output, output_fts = self.model(ct_3d_features, axial, sagittal, coronal)

            loss = self.loss(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            all_label = self.concat(all_label, label)
            all_output = self.concat(all_output, output)


            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            timer['statistics'] += self.split_time()

        _, predict_label = torch.max(all_output.data, 1)
        acc = utils.accuracy(all_label, predict_label)
        sen = utils.sensitivity(all_label, predict_label)
        spe = utils.specificity(all_label, predict_label)
        f1_score = utils.f1_score(all_label, predict_label)
        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }

        self.print_log(
            '\tCt representation Mean training loss: {:.4f}.  Mean acc: {:.2f}%. Mean sensitivity {:.2f}%. Mean specificity {:.2f}%. Mean f1_score {:.2f}%. lr: {:.8f}'.format(
                np.mean(loss_value), acc * 100, sen * 100, spe * 100, f1_score * 100, self.lr))
        # self.print_log(
        #     '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%. Mean training noise_acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100, np.mean(noise_acc_value)*100))
        self.print_log('\tCt representation Time consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k, v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, os.path.join(self.arg.work_dir, str(i_fold) + '-runs-' + str(epoch + 1) + '.pt'))


    def train_hyconv(self, epoch, i_fold, save_model=False, with_hc=True):
        if self.train_label is None:
            # load model
            weights_path = os.path.join(self.arg.work_dir, str(i_fold) + '_fold_best_model.pt')
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)
            # load model ends

            # load wsi model
            weights_path_wsi = os.path.join(self.arg.work_dir, str(i_fold) + '_fold_best_model_wsi.pt')
            weights_wsi = torch.load(weights_path_wsi)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights_wsi = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights_wsi.items()])
            self.model_wsi.load_state_dict(weights_wsi)
            # load wsi model ends

            # load wsi hyconv model
            weights_path_hyconv_wsi = os.path.join(self.arg.work_dir, str(i_fold) + '_fold_best_model_hyconv_wsi.pt')
            weights_hyconv_wsi = torch.load(weights_path_hyconv_wsi)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights_hyconv_wsi = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights_hyconv_wsi.items()])
            self.model_hyconv_wsi.load_state_dict(weights_hyconv_wsi)
            # load wsi hyconv model ends

        self.model.eval()
        self.model_wsi.eval()
        self.model_hyconv_wsi.eval()

        self.model_hyconv.train()
        self.print_log('CT Hyconv Training epoch: {} , n_fold: {}'.format(epoch + 1, i_fold))
        loader = self.data_loader['train'][i_fold]

        loss_value = []
        # acc_value = []
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        if self.train_label is None:
            for batch_idx, (features, coors, ct_3d_features, axial, sagittal, coronal, clinical_fts, label, id) in enumerate(process):
                with torch.no_grad():
                    ct_3d_features = ct_3d_features.float().cuda(self.output_device)
                    axial = axial.float().cuda(self.output_device)
                    sagittal = sagittal.float().cuda(self.output_device)
                    coronal = coronal.float().cuda(self.output_device)
                    clinical_fts = clinical_fts.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    timer['dataloader'] += self.split_time()
                    ct_features, output, output_fts = self.model(ct_3d_features, axial, sagittal, coronal)

                    features = features.float().cuda(self.output_device)
                    coors = coors.float().cuda(self.output_device)

                    if self.arg.H_coors:
                        _, wsi_features = self.model_wsi(features, coors, train=False)
                    else:
                        _, wsi_features = self.model_wsi(features, train=False)

                self.train_label = self.concat(self.train_label, label)
                self.train_cli_features = self.concat(self.train_cli_features, clinical_fts)
                self.train_ct_features = self.concat(self.train_ct_features, ct_features)
                self.train_wsi_features = self.concat(self.train_wsi_features, wsi_features)

        # forward
        with torch.no_grad():
            all_output_wsi, _ = self.model_hyconv_wsi(self.train_ct_features, self.train_wsi_features,
                                                      self.train_cli_features, train=False)

        all_output, all_output_fts = self.model_hyconv(self.train_ct_features, self.train_wsi_features,
                                                       self.train_cli_features, train=True)

        if self.model_HC.delta_e_ is None:
            tmpH = self.model_hyconv_wsi.train_H(self.train_ct_features, self.train_wsi_features)
            self.model_HC.preprocess(self.model_hyconv_wsi, self.train_wsi_features, tmpH, self.train_cli_features)

        loss_x = self.loss(all_output, self.train_label)
        loss_k = F.kl_div(F.log_softmax(all_output, dim=1), F.log_softmax(all_output_wsi, dim=1), reduction="batchmean",
                          log_target=True)
        if with_hc:
            tmpH = self.model_hyconv.train_H(self.train_ct_features, self.train_wsi_features)
            loss_h = self.model_HC(all_output, all_output_wsi, tmpH)
            loss_k = self.arg.lamb_ds * loss_k + self.arg.lamb_hc * loss_h

        loss = self.arg.lamb * loss_x + (1 - self.arg.lamb) * loss_k

        self.optimizer_hyconv.zero_grad()
        loss.backward()
        self.optimizer_hyconv.step()

        loss_value.append(loss.data.item())
        timer['model'] += self.split_time()

        self.lr = self.optimizer_hyconv.param_groups[0]['lr']
        timer['statistics'] += self.split_time()

        _, predict_label = torch.max(all_output.data, 1)
        acc = utils.accuracy(self.train_label, predict_label)
        sen = utils.sensitivity(self.train_label, predict_label)
        spe = utils.specificity(self.train_label, predict_label)
        f1_score = utils.f1_score(self.train_label, predict_label)
        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }

        self.print_log(
            '\t CT Hyconv Mean training loss: {:.4f}.  Mean acc: {:.2f}%. Mean sensitivity {:.2f}%. Mean specificity {:.2f}%. Mean f1_score {:.2f}%. lr: {:.8f}'.format(
                np.mean(loss_value), acc * 100, sen * 100, spe * 100, f1_score * 100, self.lr))
        # self.print_log(
        #     '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%. Mean training noise_acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100, np.mean(noise_acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k, v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, os.path.join(self.arg.work_dir, str(i_fold) + '-runs-' + str(epoch + 1) + '_hyconv.pt'))


    def train_wsi(self, epoch, i_fold, save_model=False):
        self.model_wsi.train()
        self.print_log('WSI representation Training epoch: {} , n_fold: {}'.format(epoch + 1, i_fold))
        loader = self.data_loader['train'][i_fold]

        loss_value = []
        sum_loss_value = []
        # acc_value = []
        all_label = None
        all_output = None
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        for batch_idx, (features, coors, ct_3d_features, axial, sagittal, coronal, clinical_fts, label, id) in enumerate(process):
            with torch.no_grad():
                features = features.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                coors = coors.float().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # forward
            if self.arg.H_coors:
                output, output_fts = self.model_wsi(features, coors, train=True)
            else:
                output, output_fts = self.model_wsi(features, train=True)

            loss = self.loss(output, label)  # + 1.0 * self.contrastive_loss(output,output,label)

            self.optimizer_wsi.zero_grad()
            loss.backward()
            self.optimizer_wsi.step()

            loss_value.append(loss.data.item())
            sum_loss_value.append(loss.data.item() * len(label))
            timer['model'] += self.split_time()

            all_label = self.concat(all_label, label)
            all_output = self.concat(all_output, output)

            # statistics
            self.lr_wsi = self.optimizer_wsi.param_groups[0]['lr']
            timer['statistics'] += self.split_time()

        _, predict_label = torch.max(all_output.data, 1)
        acc = utils.accuracy(all_label, predict_label)
        sen = utils.sensitivity(all_label, predict_label)
        spe = utils.specificity(all_label, predict_label)
        f1_score = utils.f1_score(all_label, predict_label)
        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }

        if acc > self.best_i_fold_acc_wsi:
            self.best_i_fold_acc_wsi = acc
            self.best_i_fold_acc_epoch_wsi = epoch + 1
            save_model = True

        if sum(sum_loss_value) < self.best_i_fold_loss_wsi:
            self.best_i_fold_loss_wsi = sum(sum_loss_value)
            self.best_i_fold_loss_epoch_wsi = epoch + 1
            save_model = True

        if acc > self.best_acc:
            self.best_acc_wsi = acc
            self.best_epoch_wsi = epoch + 1
            self.best_i_fold_wsi = i_fold

        self.print_log(
            '\tWSI representation Mean training loss: {:.4f}.  Mean acc: {:.2f}%. Mean sensitivity {:.2f}%. Mean specificity {:.2f}%. Mean f1_score {:.2f}%. lr: {:.8f}'.format(
                np.mean(loss_value), acc * 100, sen * 100, spe * 100, f1_score * 100, self.lr_wsi))
        self.print_log('\tWSI representation Time consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model_wsi.state_dict()
            weights = OrderedDict([[k, v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, os.path.join(self.arg.work_dir, str(i_fold) + '-runs-' + str(epoch + 1) + '_wsi.pt'))


    def train_hyconv_wsi(self, epoch, i_fold, save_model=False):
        if self.wsihyconv_train_label is None:
            # load model
            weights_path = os.path.join(self.arg.work_dir, str(i_fold) + '_fold_best_model.pt')
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)
            # load model ends
            # load wsi model
            weights_path_wsi = os.path.join(self.arg.work_dir, str(i_fold) + '_fold_best_model_wsi.pt')
            weights_wsi = torch.load(weights_path_wsi)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights_wsi = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights_wsi.items()])
            self.model_wsi.load_state_dict(weights_wsi)
            # load wsi model ends

        self.model.eval()
        self.model_wsi.eval()

        self.model_hyconv_wsi.train()
        self.print_log('Hyconv WSI Prediction Training epoch: {} , n_fold: {}'.format(epoch + 1, i_fold))
        loader = self.data_loader['train'][i_fold]

        loss_value = []
        all_output = None
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        if self.wsihyconv_train_label is None:
            for batch_idx, (features, coors, ct_3d_features, axial, sagittal, coronal, clinical_fts, label, id) in enumerate(process):
                with torch.no_grad():
                    ct_3d_features = ct_3d_features.float().cuda(self.output_device)
                    axial = axial.float().cuda(self.output_device)
                    sagittal = sagittal.float().cuda(self.output_device)
                    coronal = coronal.float().cuda(self.output_device)
                    clinical_fts = clinical_fts.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    timer['dataloader'] += self.split_time()
                    ct_features, output, output_fts = self.model(ct_3d_features, axial, sagittal, coronal)

                    features = features.float().cuda(self.output_device)
                    coors = coors.float().cuda(self.output_device)

                    if self.arg.H_coors:
                        _, wsi_features = self.model_wsi(features, coors, train=False)
                    else:
                        _, wsi_features = self.model_wsi(features, train=False)

                self.wsihyconv_train_label = self.concat(self.wsihyconv_train_label, label)
                self.wsihyconv_cli_features = self.concat(self.wsihyconv_cli_features, clinical_fts)
                self.wsihyconv_ct_features = self.concat(self.wsihyconv_ct_features, ct_features)
                self.wsihyconv_wsi_features = self.concat(self.wsihyconv_wsi_features, wsi_features)

        all_output, _ = self.model_hyconv_wsi(self.wsihyconv_ct_features, self.wsihyconv_wsi_features,
                                              self.wsihyconv_cli_features, train=True)
        loss = self.loss(all_output, self.wsihyconv_train_label)
        self.optimizer_hyconv_wsi.zero_grad()
        loss.backward()
        self.optimizer_hyconv_wsi.step()

        loss_value.append(loss.data.item())
        timer['model'] += self.split_time()
        self.lr_wsi = self.optimizer_hyconv_wsi.param_groups[0]['lr']
        timer['statistics'] += self.split_time()

        _, predict_label = torch.max(all_output.data, 1)
        acc = utils.accuracy(self.wsihyconv_train_label, predict_label)
        sen = utils.sensitivity(self.wsihyconv_train_label, predict_label)
        spe = utils.specificity(self.wsihyconv_train_label, predict_label)
        f1_score = utils.f1_score(self.wsihyconv_train_label, predict_label)
        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }

        if acc > self.best_i_fold_acc_hyconv_wsi:
            self.best_i_fold_acc_hyconv_wsi = acc
            self.best_i_fold_acc_epoch_hyconv_wsi = epoch + 1
            save_model = True

        if loss.data.item() < self.best_i_fold_loss_hyconv_wsi:
            self.best_i_fold_loss_hyconv_wsi = loss.data.item()
            self.best_i_fold_loss_epoch_hyconv_wsi = epoch + 1
            save_model = True

        if acc > self.best_acc_hyconv_wsi:
            self.best_acc_hyconv_wsi = acc
            self.best_epoch_hyconv_wsi = epoch + 1
            self.best_i_fold_hyconv_wsi = i_fold

        self.print_log(
            '\tHyconv WSI Prediction Mean training loss: {:.4f}.  Mean acc: {:.2f}%. Mean sensitivity {:.2f}%. Mean specificity {:.2f}%. Mean f1_score {:.2f}%. lr: {:.8f}'.format(
                np.mean(loss_value), acc * 100, sen * 100, spe * 100, f1_score * 100, self.lr_wsi))
        # self.print_log(
        #     '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%. Mean training noise_acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100, np.mean(noise_acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model_hyconv_wsi.state_dict()
            weights = OrderedDict([[k, v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, os.path.join(self.arg.work_dir, str(i_fold) + '-runs-' + str(epoch + 1) + '_hyconv_wsi.pt'))

    def eval(self, epoch, i_fold, train_gt_value=None, train_output_value=None, train_status_value=None,
             save_model=False, save_score=False):
        self.model.eval()
        self.print_log('Ct representation Eval epoch: {},  n_fold: {}'.format(epoch + 1, i_fold))
        loss_value = []
        # acc_value = []
        all_label = None
        all_output = None
        all_output_fts = None
        all_id = None
        step = 0
        process = tqdm(self.data_loader['val'][i_fold], ncols=40)
        for batch_idx, (features, coors, ct_3d_features, axial, sagittal, coronal, clinical_fts, label, id) in enumerate(process):
            with torch.no_grad():
                ct_3d_features = ct_3d_features.float().cuda(self.output_device)
                axial = axial.float().cuda(self.output_device)
                sagittal = sagittal.float().cuda(self.output_device)
                coronal = coronal.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                if all_id is None:
                    all_id = id
                else:
                    all_id = all_id + id

                _, output, output_fts = self.model(ct_3d_features, axial, sagittal, coronal)  # forward(data)# test_

                loss = self.loss(output, label)

                loss_value.append(loss.data.item())

                all_label = self.concat(all_label, label)
                all_output = self.concat(all_output, output)
                all_output_fts = self.concat(all_output_fts, output_fts)

                step += 1
        with torch.no_grad():
            loss = np.mean(loss_value)
            _, predict_label = torch.max(all_output.data, 1)
            acc = utils.accuracy(all_label, predict_label)
            sen = utils.sensitivity(all_label, predict_label)
            spe = utils.specificity(all_label, predict_label)
            f1_score = utils.f1_score(all_label, predict_label)
            auc = utils.area_under_the_curve(all_label, predict_label)

        if acc > self.best_i_fold_acc:
            self.best_i_fold_acc = acc
            self.best_i_fold_acc_epoch = epoch + 1
            save_model = True
            save_score = True
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch + 1
            self.best_i_fold = i_fold

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k, v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, os.path.join(self.arg.work_dir, str(i_fold) + '-runs-' + str(epoch + 1) + '.pt'))
        result_dict = None
        result_dict = {'id': all_id, 'label': all_label.detach().cpu().numpy(),
                       'predict': all_output.detach().cpu().numpy(),
                       'predict_fts': all_output_fts.detach().cpu().numpy()}

        self.print_log(
            '\tCt representation Mean val loss: {:.4f}. current epoch acc: {:.2f}%. current epoch sensitivity: {:.2f}%. current epoch specificity: {:.2f}%. current epoch f1_score: {:.2f}%. current epoch auc: {:.2f}. best acc: {:.2f}%.'.format(
                loss, acc * 100, sen * 100, spe * 100, f1_score * 100, auc, np.mean(self.best_i_fold_acc) * 100))
        return np.mean(acc) * 100, np.mean(sen) * 100, np.mean(spe) * 100, np.mean(f1_score) * 100, auc, result_dict


    def eval_hyconv(self, epoch, i_fold, train_gt_value=None, train_output_value=None, train_status_value=None,
             save_model=False, save_score=False):
        self.model.eval()
        self.model_hyconv.eval()
        self.print_log('CT Hyconv Eval epoch: {},  n_fold: {}'.format(epoch + 1, i_fold))

        if self.eval_label is None:
            process = tqdm(self.data_loader['val'][i_fold], ncols=40)
            for batch_idx, (features, coors, ct_3d_features, axial, sagittal, coronal, clinical_fts, label, id) in enumerate(process):
                with torch.no_grad():
                    ct_3d_features = ct_3d_features.float().cuda(self.output_device)
                    axial = axial.float().cuda(self.output_device)
                    sagittal = sagittal.float().cuda(self.output_device)
                    coronal = coronal.float().cuda(self.output_device)
                    clinical_fts = clinical_fts.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    ct_features, output, output_fts = self.model(ct_3d_features, axial, sagittal, coronal)

                if self.all_id_hyconv is None:
                    self.all_id_hyconv = id
                else:
                    self.all_id_hyconv = self.all_id_hyconv + id
                self.eval_label = self.concat(self.eval_label, label)
                self.eval_cli_features = self.concat(self.eval_cli_features, clinical_fts)
                self.eval_ct_features = self.concat(self.eval_ct_features, ct_features)

        all_output, output_fts = self.model_hyconv.test_forward(self.eval_ct_features, self.train_ct_features,
                                                       self.train_wsi_features,
                                                       self.eval_cli_features, self.train_cli_features)

        loss = self.loss(all_output, self.eval_label)

        with torch.no_grad():
            # loss = np.mean(loss_value)
            _, predict_label = torch.max(all_output.data, 1)
            acc = utils.accuracy(self.eval_label, predict_label)
            sen = utils.sensitivity(self.eval_label, predict_label)
            spe = utils.specificity(self.eval_label, predict_label)
            f1_score = utils.f1_score(self.eval_label, predict_label)
            auc = utils.area_under_the_curve(self.eval_label, predict_label)

        if acc > self.best_i_fold_acc_hyconv:
            self.best_i_fold_acc_hyconv = acc
            self.best_i_fold_acc_epoch_hyconv = epoch + 1
            save_model = True
            save_score = True
        if acc > self.best_acc_hyconv:
            self.best_acc_hyconv = acc
            self.best_epoch_hyconv = epoch + 1
            self.best_i_fold_hyconv = i_fold

        if save_model:
            state_dict = self.model_hyconv.state_dict()
            weights = OrderedDict([[k, v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, os.path.join(self.arg.work_dir, str(i_fold) + '-runs-' + str(epoch + 1) + '_hyconv.pt'))
        result_dict = None
        result_dict = {'id': self.all_id_hyconv, 'label': self.eval_label.detach().cpu().numpy(),
                       'predict': all_output.detach().cpu().numpy(), 'predict_fts': output_fts.detach().cpu().numpy()}

        self.print_log(
            '\tCT Hyconv Mean val loss: {:.4f}. current epoch acc: {:.2f}%. current epoch sensitivity: {:.2f}%. current epoch specificity: {:.2f}%. current epoch f1_score: {:.2f}%. current epoch auc: {:.2f}. best acc: {:.2f}%.'.format(
                loss, acc * 100, sen * 100, spe * 100, f1_score * 100, auc, np.mean(self.best_i_fold_acc_hyconv) * 100))
        return np.mean(acc) * 100, np.mean(sen) * 100, np.mean(spe) * 100, np.mean(f1_score) * 100, auc, result_dict

    def test_best_model(self, i_fold, epoch, save_model=False):
        weights_path = os.path.join(self.arg.work_dir, str(i_fold) + '-runs-' + str(epoch) + '.pt')
        weights = torch.load(weights_path)
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                weights = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights.items()])
        self.model.load_state_dict(weights)
        self.arg.print_log = False
        acc, sen, spe, f1_score, auc, result_dict = self.eval(epoch=0, i_fold=i_fold, save_score=True)
        self.arg.print_log = True
        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, os.path.join(self.arg.work_dir, str(i_fold) + '_fold_best_model.pt'))
        if result_dict is not None:
            with open(os.path.join(self.arg.work_dir, str(i_fold) + '_fold_best_model.pkl'), 'wb') as f:
                pickle.dump(result_dict, f)
        return acc, sen, spe, f1_score, auc

    def save_model_wsi(self, i_fold, epoch):
        weights_path = os.path.join(self.arg.work_dir, str(i_fold) + '-runs-' + str(epoch) + '_wsi.pt')
        weights = torch.load(weights_path)
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                weights = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights.items()])
        self.model_wsi.load_state_dict(weights)
        state_dict = self.model_wsi.state_dict()
        weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
        torch.save(weights, os.path.join(self.arg.work_dir, str(i_fold) + '_fold_best_model_wsi.pt'))

    def save_model_hyconv_wsi(self, i_fold, epoch):
        weights_path = os.path.join(self.arg.work_dir, str(i_fold) + '-runs-' + str(epoch) + '_hyconv_wsi.pt')
        weights = torch.load(weights_path)
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                weights = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights.items()])
        self.model_hyconv_wsi.load_state_dict(weights)
        state_dict = self.model_hyconv_wsi.state_dict()
        weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
        torch.save(weights, os.path.join(self.arg.work_dir, str(i_fold) + '_fold_best_model_hyconv_wsi.pt'))

    def test_best_model_hyconv(self, i_fold, epoch, epoch_hyconv, save_model=False):
        weights_path = os.path.join(self.arg.work_dir, str(i_fold) + '_fold_best_model.pt')
        weights = torch.load(weights_path)
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                weights = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights.items()])
        self.model.load_state_dict(weights)

        weights_path = None
        weights = None
        weights_path = os.path.join(self.arg.work_dir, str(i_fold) + '-runs-' + str(epoch_hyconv) + '_hyconv.pt')
        weights = torch.load(weights_path)
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                weights = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights.items()])
        self.model_hyconv.load_state_dict(weights)

        self.arg.print_log = False
        acc, sen, spe, f1_score, auc, result_dict = self.eval_hyconv(epoch=0, i_fold=i_fold, save_score=True)
        self.arg.print_log = True
        if save_model:
            state_dict = self.model_hyconv.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, os.path.join(self.arg.work_dir, str(i_fold) + '_fold_best_model_hyconv.pt'))
        if result_dict is not None:
            with open(os.path.join(self.arg.work_dir, str(i_fold) + '_fold_best_model_hyconv.pkl'), 'wb') as f:
                pickle.dump(result_dict, f)
        return acc, sen, spe, f1_score, auc

    def predict_final_fts(self, i_fold):
        self.model.eval()
        self.model_hyconv.eval()
        self.model_wsi.eval()
        self.print_log('CT Hyconv Eval n_fold: {}'.format(i_fold))

        train_id = None
        self.train_label = None
        self.train_ct_features = None
        self.train_wsi_features = None
        self.train_cli_features = None
        process = tqdm(self.data_loader['train'][i_fold], ncols=40)

        for batch_idx, (features, coors, ct_3d_features, axial, sagittal, coronal, clinical_fts, label, id) in enumerate(process):
            with torch.no_grad():
                ct_3d_features = ct_3d_features.float().cuda(self.output_device)
                axial = axial.float().cuda(self.output_device)
                sagittal = sagittal.float().cuda(self.output_device)
                coronal = coronal.float().cuda(self.output_device)
                clinical_fts = clinical_fts.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                ct_features, output, output_fts = self.model(ct_3d_features, axial, sagittal, coronal)

                features = features.float().cuda(self.output_device)
                coors = coors.float().cuda(self.output_device)

                if self.arg.H_coors:
                    _, wsi_features = self.model_wsi(features, coors, train=False)
                else:
                    _, wsi_features = self.model_wsi(features, train=False)

            train_id = list(id) if train_id is None else train_id + list(id)
            self.train_label = self.concat(self.train_label, label)
            self.train_ct_features = self.concat(self.train_ct_features, ct_features)
            self.train_cli_features = self.concat(self.train_cli_features, clinical_fts)
            self.train_wsi_features = self.concat(self.train_wsi_features, wsi_features)

        total_id = None
        total_lbl = None
        total_ft = None
        total_cli = None
        process = tqdm(self.data_loader['val'][i_fold], ncols=40)
        for batch_idx, (features, coors, ct_3d_features, axial, sagittal, coronal, clinical_fts, label, id) in enumerate(process):
            with torch.no_grad():
                ct_3d_features = ct_3d_features.float().cuda(self.output_device)
                axial = axial.float().cuda(self.output_device)
                sagittal = sagittal.float().cuda(self.output_device)
                coronal = coronal.float().cuda(self.output_device)
                clinical_fts = clinical_fts.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                ct_features, output, output_fts = self.model(ct_3d_features, axial, sagittal, coronal)

            total_id = list(id) if total_id is None else total_id + list(id)
            total_lbl = self.concat(total_lbl, label)
            total_ft = self.concat(total_ft, ct_features)
            total_cli = self.concat(total_cli, clinical_fts)

        total_output, total_ft = self.model_hyconv.test_forward(total_ft, self.train_ct_features, self.train_wsi_features,
                                                                total_cli, self.train_cli_features)

        return total_id, total_lbl, total_ft, total_output

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.print_log(f'# Parameters: {count_parameters(self.model)}, '
                           f'# Parameters of hyconv: {count_parameters(self.model_hyconv)}, '
                           f'# Parameters of wsi: {count_parameters(self.model_wsi)}, '
                           f'# Parameters of hyconv_wsi: {count_parameters(self.model_hyconv_wsi)}')

            n_fold_val_best_acc = []
            n_fold_val_best_sen = []
            n_fold_val_best_spe = []
            n_fold_val_best_f1_score = []
            n_fold_val_best_auc = []
            for i in range(len(self.data_loader['train'])):
                if i < self.arg.start_fold:
                    continue

                if i > 0:
                    self.load_model()
                    self.load_optimizer()
                    self.model = self.model.cuda(self.output_device)
                    self.model_hyconv = self.model_hyconv.cuda(self.output_device)
                    self.model_wsi = self.model_wsi.cuda(self.output_device)
                    self.model_hyconv_wsi = self.model_hyconv_wsi.cuda(self.output_device)
                    self.model_HC = self.model_HC.cuda(self.output_device)

                    self.best_i_fold_acc = 0
                    self.best_i_fold_acc_epoch = 0

                    self.best_i_fold_acc_hyconv = 0
                    self.best_i_fold_acc_epoch_hyconv = 0

                    self.best_i_fold_acc_wsi = 0
                    self.best_i_fold_acc_epoch_wsi = 0
                    self.best_i_fold_loss_wsi = float('inf')
                    self.best_i_fold_loss_epoch_wsi = 0

                    self.best_i_fold_acc_hyconv_wsi = 0
                    self.best_i_fold_acc_epoch_hyconv_wsi = 0
                    self.best_i_fold_loss_hyconv_wsi = float('inf')
                    self.best_i_fold_loss_epoch_hyconv_wsi = 0

                for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                    self.train(epoch, i_fold=i, save_model=False)
                    self.scheduler.step()
                    self.eval(epoch, i)
                acc, sen, spe, f1_score, auc = self.test_best_model(i, self.best_i_fold_acc_epoch, save_model=True)

                for epoch in range(self.arg.start_epoch, self.arg.num_epoch_wsi):
                    self.train_wsi(epoch, i_fold=i, save_model=False)
                    self.scheduler_wsi.step()
                self.save_model_wsi(i, self.best_i_fold_loss_epoch_wsi)

                self.wsihyconv_train_label = None
                self.wsihyconv_cli_features = None
                self.wsihyconv_ct_features = None
                self.wsihyconv_wsi_features = None
                for epoch in range(self.arg.start_epoch, self.arg.num_epoch_hyconv_wsi):
                    self.train_hyconv_wsi(epoch, i_fold=i, save_model=False)
                    self.scheduler_hyconv_wsi.step()
                self.save_model_hyconv_wsi(i, self.best_i_fold_loss_epoch_hyconv_wsi)

                self.all_id_hyconv = None
                self.train_label = None
                self.train_cli_features = None
                self.train_ct_features = None
                self.train_wsi_features = None
                self.eval_label = None
                self.eval_cli_features = None
                self.eval_ct_features = None

                for epoch in range(self.arg.start_epoch, self.arg.num_epoch_hyconv):
                    self.train_hyconv(epoch, i_fold=i, save_model=False, with_hc=True)
                    self.scheduler_hyconv.step()
                    self.eval_hyconv(epoch, i)


                acc, sen, spe, f1_score, auc = self.test_best_model_hyconv(i, self.best_i_fold_acc_epoch,
                                                                           self.best_i_fold_acc_epoch_hyconv, save_model=True)

                n_fold_val_best_acc.append(acc)
                n_fold_val_best_sen.append(sen)
                n_fold_val_best_spe.append(spe)
                n_fold_val_best_f1_score.append(f1_score)
                n_fold_val_best_auc.append(auc)

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            for i in range(len(n_fold_val_best_acc)):
                self.print_log(
                    'n_fold: {}, best acc: {}. best sen: {}. best spe: {}. best f1_score: {}. best auc: {}.'.format(i,
                                                                                                                    n_fold_val_best_acc[
                                                                                                                        i],
                                                                                                                    n_fold_val_best_sen[
                                                                                                                        i],
                                                                                                                    n_fold_val_best_spe[
                                                                                                                        i],
                                                                                                                    n_fold_val_best_f1_score[
                                                                                                                        i],
                                                                                                                    n_fold_val_best_auc[
                                                                                                                        i]))
            self.print_log(
                '{}_fold, best mean acc: {}. mean sen: {}. mean spe: {}. mean f1_score: {}. mean auc: {}.'.format(
                    self.arg.n_fold, np.mean(n_fold_val_best_acc), np.mean(n_fold_val_best_sen),
                    np.mean(n_fold_val_best_spe), np.mean(n_fold_val_best_f1_score), np.mean(n_fold_val_best_auc)))
            self.print_log(f'Best c-index: {self.best_acc}')
            self.print_log(f'Best i_fold: {self.best_i_fold}')
            self.print_log(f'Epoch number: {self.best_epoch}')
            # self.test_best_model(self.best_i_fold, self.best_epoch)
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            n_fold_val_best_acc = []
            n_fold_val_best_sen = []
            n_fold_val_best_spe = []
            n_fold_val_best_f1_score = []
            for i in range(len(self.data_loader['val'])):
                self.load_model(i=i)
                self.model.to(self.arg.device[0])
                self.model_hyconv.to(self.arg.device[0])
                acc, sen, spe, f1_score, auc, result_dict = self.eval_hyconv(epoch=0, i_fold=i, save_score=True)
                n_fold_val_best_acc.append(acc)
                n_fold_val_best_sen.append(sen)
                n_fold_val_best_spe.append(spe)
                n_fold_val_best_f1_score.append(f1_score)
                if result_dict is not None:
                    with open(os.path.join(self.arg.work_dir, str(i) + '_fold_best_model.pkl'), 'wb') as f:
                        pickle.dump(result_dict, f)

            for i in range(len(n_fold_val_best_acc)):
                self.print_log('n_fold: {}, best acc: {}. best sen: {}. best spe: {}. best f1_score: {}.'.format(i,
                                                                                                                 n_fold_val_best_acc[
                                                                                                                     i],
                                                                                                                 n_fold_val_best_sen[
                                                                                                                     i],
                                                                                                                 n_fold_val_best_spe[
                                                                                                                     i],
                                                                                                                 n_fold_val_best_f1_score[
                                                                                                                     i]))
            self.print_log(
                '{}_fold, best mean acc: {}. mean sen: {}. mean spe: {}. mean f1_score: {}.'.format(self.arg.n_fold,
                                                                                                    np.mean(
                                                                                                        n_fold_val_best_acc),
                                                                                                    np.mean(
                                                                                                        n_fold_val_best_sen),
                                                                                                    np.mean(
                                                                                                        n_fold_val_best_spe),
                                                                                                    np.mean(
                                                                                                        n_fold_val_best_f1_score)))
            self.print_log('Done.\n')


if __name__ == '__main__':
    parser = get_parser()

    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
