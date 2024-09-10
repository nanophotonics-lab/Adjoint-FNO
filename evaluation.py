import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from tqdm import tqdm
from model import *
from data.dataset import CategoricalStructureAdjointDataset
import argparse
from utils import *
import os
import datetime
adj_maxval, adj_minval = 0.388, -0.239
lab_maxval, lab_minval = 11.31, -11.31

conditions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
angles = [1.15, 2.29, 3.43, 4.57, 5.71, 6.84, 7.97, 9.09, 10.20, 11.31]

label_dict = {"0.0" : 0.0}
for c, a in zip(conditions, angles):
    label_dict[str(c)] = a
    label_dict[str(-c)] = -a
# 현재 파일 경로
ABS_PATH = os.path.abspath(__file__)
ABS_PATH = "/".join(ABS_PATH.split('/')[:-1]) # ~~/joon/corning_

def gd_norm_string(grads):
    this_string = ""
    for i in range(len(grads)):
        this_string += f"layer:{i}, grad norm : {grads[i]}\n"
    return this_string

def create_geo(opt, fixed=False, path=None):
    x0 = None
    geometry_array = None
    initial_geometry_array = None
    if fixed:
        if os.path.exists(path):
            geometry_array = np.load(path)
        else:
            rng = np.random.default_rng()
            x0 = rng.random(100)
            geometry_array = np.round(x0, 3)
            np.save(path, geometry_array)

    else:
        rng = np.random.default_rng()
        x0 = rng.random(100)
        geometry_array = np.round(x0, 3)
 
    initial_geometry_array = geometry_array
    
    try:
        opt.update_design([initial_geometry_array])
        print("opt.update_design successfully executed")

    except Exception as e:
        print(f"An error occurred: {e}")
    
    return opt, initial_geometry_array


# validation split도 필요.
def J(fields): 
    return npa.mean(npa.abs(fields[:,1]) ** 2)
# simulator를 선언하는 함수입니다.
def define_simulator():
    # mp.verbosity(0)
    # seed = 240
    # np.random.seed(seed)
    
    LC_up = mp.Medium(epsilon=3.5)
    LC_lb = mp.Medium(epsilon=2.5)
    Air = mp.Medium(index=1.0)
    
    resolution = 20
    
    design_region_width = 5
    design_region_height = 0.5
    pml_size = 1.0

    Sx = 2 * pml_size + design_region_width
    Sy = 2 * pml_size + design_region_height + 8
    cell_size = mp.Vector3(Sx, Sy)

    pml_layers = [mp.PML(pml_size)]
    Nx = 100
    Ny = 1

    # Source parameters
    fcen = 1 / 1.55
    width = 0.2
    fwidth = width * fcen
    source_center = [0, -Sy/2 + 2, 0]
    source_size = mp.Vector3(Sx, 0, 0)
    src = mp.GaussianSource(frequency=fcen, fwidth=fwidth, is_integrated=True)
    source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]

    design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), LC_up, LC_lb, grid_type="U_MEAN")
    design_region = mpa.DesignRegion(
        design_variables,
        volume=mp.Volume(
            center=mp.Vector3(0,-Sy/2 + 3,),
            size=mp.Vector3(design_region_width, design_region_height, 0),
        ),
    )

    geometry = [
        mp.Block(
            center=design_region.center, size=design_region.size, material=design_variables
        )
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=source,
        eps_averaging=False,
        resolution=resolution,
    )
    
    return sim, design_region, fcen

# Meep simulator의 optimizer를 선언하는 함수입니다.
def define_opt(sim, fcen, design_region, condition):  
    design_region_width = 5
    design_region_height = 0.5
    pml_size = 1.0

    Sx = 2 * pml_size + design_region_width
    Sy = 2 * pml_size + design_region_height + 8
    cell_size = mp.Vector3(Sx, Sy)
    
    monitor_position, monitor_size = mp.Vector3(condition, Sy/2 - 2,), mp.Vector3(0.1,0)
    FourierFields = mpa.FourierFields(sim,mp.Volume(center=monitor_position,size=monitor_size),mp.Ez,yee_grid=True)
    ob_list = [FourierFields]

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=[J],
        objective_arguments=ob_list,
        design_regions=[design_region],
        fcen=fcen,
        df=0,
        nf=1,
    )
    
    return opt

def make_unseen_data(data_num, condition):
    structures = []
    adjoints = []
    for i in range(data_num):
        sim, design_region, fcen = define_simulator()
        opt = define_opt(sim=sim, fcen=fcen, design_region=design_region, condition=condition)
        opt, geo = create_geo(opt, fixed=False, path=None)
        f0, dJ_du = opt()
        structures.append(geo)
        adjoints.append(dJ_du)
    structures = np.stack(structures, axis=0) # N, 100
    adjoints = np.stack(adjoints, axis=0) # N, 100
    return structures, adjoints
        
        

"""
    Dataset
        - 각도 5개에 따른 결과 보여주기
        - Interpolation에 대한 결과도 필요
            - interpolation 결과 : -10 ~ 10에 대한 결과
"""
def minmaxscaler(data, minval=None, maxval=None):
    if minval==None:
        minval = np.min(data)
    if maxval==None:
        maxval = np.max(data)
    return (data - minval) / (maxval - minval), minval, maxval

def get_data(path, path_label_dict, data_num):
    label = path_label_dict[os.path.basename(path)]
    file_names = os.listdir(path)
    structures = []
    adjoints = []
    
    for i, fn in enumerate(file_names):
        if data_num < (i+1):
            break
        npzip = np.load(os.path.join(path, fn))
        structures.append(npzip['geometry'])
        adjoints.append(npzip['adjoint'])
    structures = np.stack(structures)
    
    adjoints = np.stack(adjoints)
    labels = np.zeros(len(adjoints), dtype=np.int32) + label
    return structures, adjoints, labels

def get_seen_data(data_num):
    data_path = '/home/joon/joon/corning_/data/dataset/valid'
    path_label_dict = {'data_-2' : -11.31, 'data_-1' : -5.71, 'data' : 0, 'data_1': 5.71, 'data_2' : 11.31}
    data_dict = {}
    for k in path_label_dict.keys():
        structures, adjoints, labels = get_data(os.path.join(data_path, k), path_label_dict, data_num)
        adjoints, _, _ = minmaxscaler(adjoints, minval=adj_minval, maxval=adj_maxval)
        labels, _, _ = minmaxscaler(labels, minval=lab_minval, maxval=lab_maxval)

        structures = torch.tensor(structures, dtype=torch.float32)
        adjoints = torch.tensor(adjoints, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        data_dict[k] = (structures, adjoints, labels)
    return data_dict
    

def define_model(log_path, epoch=500):
    args = open_args(log_path)
    device = f"cuda:1"
    model_path = os.path.join(log_path,  f'{args.model_name}_{epoch}.pt')
    print(args)
    if args.model_name == "VAE":
        if args.vae_cat:
            args.dim[0][-1] = args.dim[0][-1] - args.latent_dim
        model = VariationalDeepAdjointPredictor(input_dim=args.in_dim, dim=args.dim, latent_dim=args.latent_dim, layer_num=args.layer_num, condition=args.condition_num, p=args.p, concat=args.vae_cat).to(device)
    elif args.model_name == "MLP":
        model = LinearModel(input_dim=args.in_dim, dim=args.dim, layer_num=args.layer_num, condition=args.condition_num, p=args.p, fourier_embed=args.fourier_embedding).to(device)
    elif args.model_name == "resMLP":
        model = SingleResidualLinearModel(input_dim=args.in_dim, dim=args.dim, layer_num=args.layer_num).to(device)
    elif args.model_name == "FNO":
        model = FNO(indim=args.in_dim, dim=args.dim, layer_num=args.layer_num, condition_num=args.condition_num).to(device)
    elif args.model_name == "NewFNO":
        model = NewFNO(indim=args.in_dim, dim=args.dim, mode=args.mode, layer_num=args.layer_num, condition_num=args.condition_num).to(device)
    else:
        model = Unet1d(input_dim=1, dim=args.dim, condition=args.condition_num).to(device)
    model = load_model(model_path, model)
    model.eval()
    return model, device, args


def eval_seendata(save_path, log_path, data_num=500):
    args = open_args(log_path)
    model, device, args = define_model(log_path, epoch=500)

    datas_dict = get_seen_data(data_num)
    keys = datas_dict.keys()
    
    valid_dict = {}
    for k in keys:
        structures, adjoints, labels = datas_dict[k]
        structures, labels = structures.to(device), labels.to(device)
        if args.model_name == "VAE" or args.model_name == "new_VAE":
                adj, recons, mu, log_var, z = model(structures, labels)
        elif args.model_name == "resMLP" or args.model_name == "FNO" or args.model_name == "ResidualFNO" or args.model_name == 'NewFNO': # non_condition....
            adj = model([structures, labels])
        else:
            adj, _ = model([structures, labels])
        
        
        valid_dict[k] = {'pred_adj':adj.detach().cpu(), 'gt_adj':adjoints}
        
    torch.save(
        valid_dict, save_path
    )
    
    
def eval_unseendata(save_path, log_path, data_num=500):
    """
        mp | angle
        0.1	1.15
        0.2	2.29
        0.3	3.43
        0.4	4.57
        0.5	5.71
        0.6	6.84
        0.7	7.97
        0.8	9.09
        0.9	10.20
        1.0	11.31
    """
    
    
    args = open_args(log_path)
    model, device, args = define_model(log_path, epoch=500)
    
    results = {}
    
    for i in range(-10, 11, 1):
        this_results = {}
        # condition이 정해져야함.
        condition = i/10
        angle = label_dict[str(condition)]
        dir_prefix = int(condition*10)
        
        
        this_label = angle # [0, +-5.71, +-11.31]
        this_label, _, _ = minmaxscaler(this_label, minval=lab_minval, maxval=lab_maxval) # label scaler
        this_label = np.array([this_label])
        labels = np.ones(shape=(data_num,)) * this_label
        structures, adjoints = make_unseen_data(data_num=data_num, condition=condition)
        adjoints, _, _ = minmaxscaler(adjoints, minval=adj_minval, maxval=adj_maxval)
        structures = torch.tensor(structures, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        
        if args.model_name == "VAE" or args.model_name == "new_VAE":
            pred_adj, recons, mu, log_var, z = model(structures, labels)
        elif args.model_name == "resMLP" or args.model_name == "FNO" or args.model_name == "ResidualFNO" or args.model_name == 'NewFNO': # non_condition....
            pred_adj = model([structures, labels])
        else:
            pred_adj, _ = model([structures, labels])

        this_results['pred_adj']=pred_adj.detach().cpu()
        this_results['gt_adj']=adjoints

        results[str(angle)] = this_results
    
    torch.save(
        results, save_path
    )
        
if __name__ == "__main__":
    log_path ='/home/joon/joon/corning_/logs_2000/Unet1drelu__32_4_0.0/2024_08_21_05_47_01'
    
    save_path = '/home/joon/joon/corning_/unseen_angle/Unet1drelu__32_4_0.0'
    os.makedirs(save_path, exist_ok=True)
    save_path = '/home/joon/joon/corning_/unseen_angle/Unet1drelu__32_4_0.0/valid_data.pt'
    eval_unseendata(save_path, log_path, 500)