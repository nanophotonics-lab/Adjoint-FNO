import os
import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import shutil
from tqdm import tqdm


adj_maxval, adj_minval = 0.388, -0.239
lab_maxval, lab_minval = 11.31, -11.31

def get_data_from_folder_path(fp, path_label_dict, data_num, data_num_flag):
    label = path_label_dict[os.path.basename(fp)]
    print(fp,label)
    file_names = os.listdir(fp)
    file_nums = len(file_names)
    structures = []
    adjoints = []
    cnt_1 = 0
    cnt_0 = 0
    
    for i, fn in enumerate(file_names):
        if data_num_flag and (data_num < (i+1)):
                break
        npzip = np.load(os.path.join(fp, fn))
        structures.append(npzip['geometry'])
        adjoints.append(npzip['adjoint'])
        if npzip['geometry'].shape == (1,):
            cnt_1 += 1
        elif npzip['geometry'].shape == (0,):
            cnt_0 += 1
            
        # print(npzip['geometry'].shape, npzip['geometry'], fn)
    # print("cnt 1 : ", cnt_1)
    # print("cnt 0 : ", cnt_0)
    # print(structures)
    structures = np.stack(structures)
    
    adjoints = np.stack(adjoints)
    labels = np.zeros(len(adjoints), dtype=np.int32) + label
    return structures, adjoints, labels

def for_print_dict(path_label_dict):
    new_path_label_dict = {}
    for k, v in path_label_dict.items():
        new_path_label_dict[os.path.basename(k)] = v
    return new_path_label_dict

def minmaxscaler(data, minval=None, maxval=None):
    if minval==None:
        minval = np.min(data)
    if maxval==None:
        maxval = np.max(data)
    return (data - minval) / (maxval - minval), minval, maxval


    
def get_npzs(path):
    file_names = os.listdir(path)
    structures = []
    adjoints = []
    for fn in file_names:
        fp = os.path.join(path, fn)
        npzip = np.load(fp)
        structures.append(npzip['geometry'])
        adjoints.append(npzip['adjoint'])
    structures = np.stack(structures)
    adjoints = np.stack(adjoints)
    return structures, adjoints

max_adj, min_adj = 0.388, -0.239
    

adj_maxval, adj_minval = 0.388, -0.239
class CategoricalStructureAdjointDataset(Dataset):
    def __init__(self, path, data_num=None, mode="train", hard=False):
        data_num_flag = not data_num == None
        # path : corning repository
        self.path_label_dict = {'data_-2' : -11.31, 'data_-1' : -5.71, 'data' : 0, 'data_1': 5.71, 'data_2' : 11.31}
        if hard:
            folder_list = ['data_-2', 'data', 'data_2']
        else:
            folder_list = ['data_-2','data_-1', 'data', 'data_1', 'data_2']
        
        if mode=="train":
            path = os.path.join(path, 'train')
        elif mode == "test":
            path = os.path.join(path, 'test')
        else:
            path = os.path.join(path, 'valid')
        
        # x - min / max - min 
        def folder_to_path(folder):
            return os.path.join(path, folder)
        folder_paths = list(map(folder_to_path, folder_list))
        self.structures = []
        self.adjoints = []
        self.labels = []
        for i, fp in enumerate(tqdm(folder_paths)):
            
            structures, adjoints, labels = get_data_from_folder_path(fp, self.path_label_dict, data_num, data_num_flag)
            self.structures.append(structures)
            self.adjoints.append(adjoints)
            self.labels.append(labels)
            
        self.structures = np.concatenate(self.structures)
        self.adjoints = np.concatenate(self.adjoints)
        self.labels = np.concatenate(self.labels)
        
        # normalization. structure는 이미 0~1
        self.adjoints, minval, maxval = minmaxscaler(self.adjoints, minval=adj_minval, maxval=adj_maxval)
        self.labels, _, _ = minmaxscaler(self.labels)
        print(minval, maxval)
        
        self.print_dict = for_print_dict(self.path_label_dict)
        
        self.structures = torch.tensor(self.structures, dtype=torch.float32)
        self.adjoints = torch.tensor(self.adjoints, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32) # continuous conditional
        
    
    def get_class_info(self):
        print(f"Class Information : {self.print_dict}")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index) : 
        return self.structures[index], self.adjoints[index], self.labels[index]
            