from torch.utils.data import Dataset

import torch
import numpy as np
import pickle


class SlidePatch(Dataset):
    def __init__(self, data_dict: dict, CT_ft_file=None, clinical_file=None):
        super().__init__()
        self.count = 0
        self.id_list = list(data_dict.keys())
        self.data_dict = data_dict

        self.CT_ft_file = CT_ft_file
        if CT_ft_file is not None:
            with open(CT_ft_file,'rb') as f:
                CT_feature = pickle.load(f)
            f.close()
            for key in self.id_list:
                self.data_dict[key]['CT_ft'] = CT_feature[self.data_dict[key]['radiology']]
        
        if clinical_file is not None:
            with open(clinical_file,'rb') as f:
                clinical_fts = pickle.load(f) 
            f.close()
            for key in self.id_list:
                self.data_dict[key]['clinical_fts'] = clinical_fts[key]['clinical_fts']


    def __getitem__(self, idx: int):
        id = self.id_list[idx]
        self.count += 1
        label = torch.tensor(self.data_dict[id]['label']).float()
        wsi_fts = torch.tensor(np.load(self.data_dict[id]['fts'])).float()
        

        with open(self.data_dict[id]['coors'], 'rb') as f:
            coors = pickle.load(f)
            coors = torch.Tensor(coors)

        if 'ct_3d_feature' in self.data_dict[id].keys():
            ct_3d_fts = torch.tensor(self.data_dict[id]['ct_3d_feature']).float()
            if 'axial' in self.data_dict[id].keys():
                axial = torch.tensor(self.data_dict[id]['axial']).float()
                sagittal = torch.tensor(self.data_dict[id]['sagittal']).float()
                coronal = torch.tensor(self.data_dict[id]['coronal']).float()

                if 'clinical_fts' in self.data_dict[id].keys():
                    clinical_fts = torch.tensor(self.data_dict[id]['clinical_fts']).float()
                    return wsi_fts, coors, ct_3d_fts, axial, sagittal, coronal, clinical_fts, label, id
                return wsi_fts, coors, ct_3d_fts, axial, sagittal, coronal, label, id
            else:
                return wsi_fts, coors, ct_3d_fts, label, id #, stage, stage_t, stage_m, stage_n
        else:
            return wsi_fts, coors, label, id

    def __len__(self) -> int:
        return len(self.id_list)



class  CWDataset(Dataset):
    def __init__(self, data_dict: dict, survival_time_max, survival_time_min, CT_ft_file=None): #intra_wsi_fts_file=None,
        super().__init__()
        self.st_max = float(survival_time_max)
        self.st_min = float(survival_time_min)
        self.count = 0
        self.id_list = list(data_dict.keys())
        self.data_dict = data_dict

        self.CT_ft_file = CT_ft_file


        with open(CT_ft_file,'rb') as f:
            CT_feature = pickle.load(f)
        for key in self.id_list:
            self.data_dict[key]['CT_ft'] = CT_feature[self.data_dict[key]['radiology']]

        
        print()

    #     self.get_data()
    def __getitem__(self, idx: int):
        id = self.id_list[idx]
        self.count += 1
        fts = torch.tensor(np.load(self.data_dict[id]['ft_dir'])).float()

        ti = torch.tensor(self.data_dict[id]['survival_time']).float()
        survival_time = ti/self.st_max
        status = torch.tensor(self.data_dict[id]['status'])
        with open(self.data_dict[id]['patch_coors'], 'rb') as f:
            coors = pickle.load(f)
            coors = torch.Tensor(coors)

        ct_fts = torch.tensor(self.data_dict[id]['CT_ft']).float()
        
        return fts, coors, ct_fts, survival_time, status, id #, stage, stage_t, stage_m, stage_n

    def __len__(self) -> int:
        # print(len(self.id_list))
        return len(self.id_list)

class  CTDataset(Dataset):
    def __init__(self, data_dict: dict, survival_time_max, survival_time_min, CT_ft_file=None):
        super().__init__()
        self.st_max = float(survival_time_max)
        self.st_min = float(survival_time_min)
        self.count = 0
        self.id_list = list(data_dict.keys())
        self.data_dict = data_dict

        self.CT_ft_file = CT_ft_file



        with open(CT_ft_file,'rb') as f:
            CT_feature = pickle.load(f)
        for key in self.id_list:
            self.data_dict[key]['CT_ft'] = CT_feature[self.data_dict[key]['radiology']]
        
        print()

    #     self.get_data()
    def __getitem__(self, idx: int):
        id = self.id_list[idx]
        self.count += 1
        ti = torch.tensor(self.data_dict[id]['survival_time']).float()
        survival_time = ti/self.st_max #(self.st_min * (self.st_max - ti))/ (ti * (self.st_max - self.st_min)) #ti/self.st_max #  /self.st_max #
        status = torch.tensor(self.data_dict[id]['status'])
        
        ct_fts = torch.tensor(self.data_dict[id]['CT_ft']).float()
        
        return ct_fts, survival_time, status, id #, stage, stage_t, stage_m, stage_n

    def __len__(self) -> int:
        # print(len(self.id_list))
        return len(self.id_list)
