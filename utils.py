import json
import os.path as osp
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import glob
import numpy as np
import torch
import lifelines.utils.concordance as LUC
import random
import pickle
import pandas as pd

def get_WSI_sample_list(patch_ft_dir,WSI_patch_coor_dir,CT_3d_feature_file=None,CT_2d_feature_file=None,
                        cli_feature_file=None):

    if WSI_patch_coor_dir is not None:
        all_coor_list = []
        all_coor_list.extend(glob.glob(osp.join(WSI_patch_coor_dir, '*_coors.pkl')))

        coor_dict = {}
        def get_id(_dir):
            tmp_dir = _dir.split('/')[-1][:-10]
            return tmp_dir
        for _dir in all_coor_list:
            id = get_id(_dir)
            coor_dict[id] = _dir
    if patch_ft_dir is not None:
        all_ft_list = []
        all_ft_list.extend(glob.glob(osp.join(patch_ft_dir, '*_fts.npy')))
        ft_dict = {}
        def get_id(_dir):
            tmp_dir = _dir.split('/')[-1][:-8]
            return tmp_dir
        for _dir in all_ft_list:
            id = get_id(_dir)
            ft_dict[id] = _dir
    if CT_3d_feature_file is not None:
        with open(CT_3d_feature_file,'rb') as f:
            ct_3d_features = pickle.load(f)

    if CT_2d_feature_file is not None:
        with open(CT_2d_feature_file,'rb') as f:
            ct_2d_features = pickle.load(f)

    if cli_feature_file is not None:
        cli_feature = pd.read_csv(cli_feature_file, encoding='gbk')
        cli_feature = cli_feature[['index', 'Gender', 'Size']].copy()

        cli_feature['Gender'] = ((cli_feature['Gender'] - cli_feature['Gender'].min()) / (
                cli_feature['Gender'].max() - cli_feature['Gender'].min()) * 2 - 1) * 10
        cli_feature['Size'] = ((cli_feature['Size'] - cli_feature['Size'].min()) / (
                cli_feature['Size'].max() - cli_feature['Size'].min()) * 2 - 1) * 10

    
    all_dict = {}
    for key in ft_dict.keys():
        all_dict[key] = {}
        all_dict[key]['fts'] = ft_dict[key]
        all_dict[key]['coors'] = coor_dict[key]
        if CT_3d_feature_file is not None:
            all_dict[key]['ct_3d_feature'] = ct_3d_features[key]
        if CT_2d_feature_file is not None:
            all_dict[key]['axial'] = ct_2d_features['axial'][key]
            all_dict[key]['sagittal'] = ct_2d_features['sagittal'][key]
            all_dict[key]['coronal'] = ct_2d_features['coronal'][key]
        if cli_feature_file is not None:
            all_dict[key]['clinical_fts'] = np.array(cli_feature[cli_feature['index'] == key].values[0][1:]).astype(float)
        if 'LUAD' in key:
            all_dict[key]['label'] = 0
        else:
            all_dict[key]['label'] = 1
    
    return all_dict

def get_n_fold_data_list(data_dict,n_fold,random_seed):
    LUAD_keys = []
    LUSC_keys = []
    for key in data_dict.keys():
        if data_dict[key]['label'] == 1:
            LUSC_keys.append(key)
        else:
            LUAD_keys.append(key)
    print("LUAD length {}".format(len(LUAD_keys)))
    print("LUSC length {}".format(len(LUSC_keys)))

    n_fold_LUSC_train_list = []
    n_fold_LUSC_val_list = []
    n_fold_LUAD_train_list = []
    n_fold_LUAD_val_list = []
    n_fold_train_list = []
    n_fold_val_list = []
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=random_seed) #random_seed
    for train_idx, val_idx in kf.split(LUSC_keys):
        train_keys = [LUSC_keys[i] for i in train_idx]
        val_keys = [LUSC_keys[i] for i in val_idx]

        train_data_dict = {key: data_dict[key] for key in train_keys}
        val_data_dict = {key: data_dict[key] for key in val_keys}
        n_fold_LUSC_train_list.append(train_data_dict)
        n_fold_LUSC_val_list.append(val_data_dict)

    for train_idx, val_idx in kf.split(LUAD_keys):
        train_keys = [LUAD_keys[i] for i in train_idx]
        val_keys = [LUAD_keys[i] for i in val_idx]

        train_data_dict = {key: data_dict[key] for key in train_keys}
        val_data_dict = {key: data_dict[key] for key in val_keys}
        n_fold_LUAD_train_list.append(train_data_dict)
        n_fold_LUAD_val_list.append(val_data_dict)

    for i in range(n_fold):
        n_fold_train_list.append(dict(n_fold_LUAD_train_list[i],**n_fold_LUSC_train_list[i]))
        n_fold_val_list.append(dict(n_fold_LUAD_val_list[i],**n_fold_LUSC_val_list[i]))
    
    output = {}
    for i in range(n_fold):
        key = '{}_fold'.format(i)
        output[key] = {}
        output[key]['train'] = list(n_fold_train_list[i].keys())
        output[key]['val'] = list(n_fold_val_list[i].keys())
    with open("random_seed-{}.json".format(random_seed), "w") as file:
        json.dump(output, file)
    return n_fold_train_list, n_fold_val_list

def sort_survival_time(gt_survival_time,pre_risk,censore, output_fts,patch_ft=None,coors=None):
    ix = torch.argsort(gt_survival_time, dim= 0, descending=True)#
    gt_survival_time = gt_survival_time[ix]
    pre_risk = pre_risk[ix]
    censore = censore[ix]
    output_fts = output_fts[ix]
    if patch_ft is not None:
        patch_ft = patch_ft[ix]
        coors = coors[ix]
        return gt_survival_time,pre_risk,censore,output_fts,patch_ft,coors
    return gt_survival_time,pre_risk,censore,output_fts

def accuracytest(survivals, risk, censors):
    survlist = []
    risklist = []
    censorlist = []

    for riskval in risk:
        # riskval = -riskval
        risklist.append(riskval.cpu().detach().item())

    for censorval in censors:
        censorlist.append(censorval.cpu().detach().item())

    for surval in survivals:
        # surval = -surval
        survlist.append(surval.cpu().detach().item())

    C_value = LUC.concordance_index(survlist, risklist, censorlist)

    return C_value

def accuracy(labels, predictions):
    correct = (labels == predictions).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

def sensitivity(labels, predictions):
    true_positives = ((labels == 1) & (predictions == 1)).sum().item()
    actual_positives = (labels == 1).sum().item()
    sensitivity = true_positives / actual_positives
    return sensitivity

def specificity(labels, predictions):
    true_negatives = ((labels == 0) & (predictions == 0)).sum().item()
    actual_negatives = (labels == 0).sum().item()
    specificity = true_negatives / actual_negatives
    return specificity

def f1_score(labels, predictions):
    true_positives = ((labels == 1) & (predictions == 1)).sum().item()
    false_positives = ((labels == 0) & (predictions == 1)).sum().item()
    false_negatives = ((labels == 1) & (predictions == 0)).sum().item()
    
    try:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
    
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    except:
        return 0.5

def area_under_the_curve(labels, predictions):
    labels = labels.cpu().detach().numpy()
    predictions = predictions.cpu().detach().numpy()
    auc = roc_auc_score(labels, predictions)
    return auc


    