'''

### 코드 로직

1. .npz 파일에서 geometry, gradient 값 read
    1. iter = 1 이라면 geometry 생성 
2. geometry, gradient 값을 array 저장
3. geometry 값에 gradient * n 만큼 +
4. f0, djdu = opt()
5. f0 값 저장 
    1. fom.txt 파일 읽어와서 맨 마지막 줄에 값 저장
6. 업데이트 된 geometry, 업데이트 된 구조의 adjoint gradient를  .npz로 저장

→ 모델에 넣고 값 출력해서 1번부터 반복


'''

"""
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

import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from matplotlib import pyplot as plt
import secrets
import torch
import torch.nn as nn
from model import *
from utils import *
import time

mp.verbosity(0)

ABS_PATH = os.path.abspath(__file__)
ABS_PATH = "/".join(ABS_PATH.split('/')[:-1]) 


# Load pretrained model
def define_model(log_path, epoch=500):
    args = open_args(log_path)
    device = f"cuda:0"
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



# First order taylor approximation of FoM
def get_norm(adj, geo, update_parameter):
    negative_indices = np.where(geo+adj*update_parameter < 0)    
    overpos_indices = np.where(geo+adj*update_parameter > 1)    
    adj[negative_indices] = -geo[negative_indices] / update_parameter
    adj[overpos_indices] = (1-geo[overpos_indices]) / update_parameter
    return np.mean(np.abs(adj**2)) * update_parameter

    
# Define a simulator.
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

# Optimizer of Meep simlator (FDTD)
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

def inv_scaler(adj):
    return adj * (adj_maxval - adj_minval) + adj_minval
    
def minmaxscaler(x, value_range=(-11.31, 11.31)): # 7.13, 0
    minval, maxval = value_range
    x = (x-minval)/(maxval-minval)
    return x

# FoM
def J(fields): 
    return npa.mean(npa.abs(fields[:,1]) ** 2)

def read_npz(file_path):
    data = np.load(file_path)
    geometry_from_npz = data['geometry']
    gradient_from_npz = data['adjoint']
    return geometry_from_npz, gradient_from_npz

# structure or geometry initialization
# def create_geometry():
#     if iteration == 1:
#         initial_geometry_array = None
#         x0 = None
        
#         rng = np.random.default_rng()
#         x0 = rng.random(100)
#         initial_geometry_array = np.round(x0, 3)
        
#         try:
#             opt.update_design([initial_geometry_array])
#             print("opt.update_design successfully executed")

#         except Exception as e:
#             print(f"An error occurred: {e}")

#     else:
#         return initial_geometry_array
    
# structure or geometry initialization
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


def write_to_file(file_name, data):
    
    with open(file_name, 'a') as file: 
        file.write(f"{data}\n")  

# Update n pixels
def n_pixel_update_geometry(opt, n, geometry, gradient, update_parameter, ranking=True):
    if n != 100:
        if ranking:
            # n% ranking
            indices= np.array(sorted(range(len(gradient)),key=lambda i: gradient[i])[-n:])
        else:
            indices = np.random.randint(0, 100, n)
    else:
        indices = np.arange(0, 100)

    print("index : ", indices)
    updated_geometry = geometry.copy()
    updated_geometry[indices] = geometry[indices] + gradient[indices] * update_parameter
    
    updated_geometry = np.clip(updated_geometry, 0, 1)
    
    negative_indices = np.where(geometry[indices]+gradient[indices]*update_parameter < 0)    
    overpos_indices = np.where(geometry[indices]+gradient[indices]*update_parameter > 1)
    
    negative_indices = indices[negative_indices]
    overpos_indices = indices[overpos_indices]
    
    grad_ = gradient.copy()
    grad_[negative_indices] = -geometry[negative_indices] / update_parameter
    grad_[overpos_indices] = (1-geometry[overpos_indices]) / update_parameter
    diff_norm = np.mean(grad_[indices]**2) * update_parameter
    
    try:
        opt.update_design([updated_geometry])
        print("opt.update_design successfully executed")

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"{updated_geometry.shape}")
    
    return opt, updated_geometry, diff_norm


adj_maxval, adj_minval = 0.388, -0.239
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


conditions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
angles = [1.15, 2.29, 3.43, 4.57, 5.71, 6.84, 7.97, 9.09, 10.20, 11.31]

label_dict = {"0.0" : 0.0}
for c, a in zip(conditions, angles):
    label_dict[str(c)] = a
    label_dict[str(-c)] = -a


def iteration_func(iterations, log_path, epoch, flip_num=100, ranking=True, condition=0.0, fixed=False, init_structure_path=None):
    angle = label_dict[str(condition)]
    dir_prefix = int(condition*10)
    
    
    this_label = angle # [0, +-5.71, +-11.31]
    this_label = minmaxscaler(this_label, value_range=(-11.31, 11.31)) # label scaler
    this_label = np.array([this_label])

    """
        The created log path during training phase. 
        e.g. logs/CondFNO_512_8/2023_12_03_15_16_16
    """

    log_path = os.path.join(ABS_PATH, log_path)
    model, device, args = define_model(log_path=log_path, epoch=epoch)
    model.eval()
    ranking_str = "_ranking" if ranking else ""
    sub_folder_name = log_path.split(os.path.sep)[-2] + f"_{args.data_num}" + f"_flip{flip_num}"+ranking_str

    txt_file_path = os.path.join(ABS_PATH, "last_main_iteration_txt_files", sub_folder_name)
    npz_file_path = os.path.join(ABS_PATH, "npzs", sub_folder_name)
    
    os.makedirs(txt_file_path, exist_ok=True)
    os.makedirs(npz_file_path, exist_ok=True)
    '''
    --------------------- main ---------------------
    '''
    cond_npz_file_path = os.path.join(npz_file_path, f"{dir_prefix}_data_n5")
    os.makedirs(cond_npz_file_path, exist_ok=True)
    for iteration in range(0, iterations):
        
        file_path = os.path.join(cond_npz_file_path, f'{iteration}.npz') 
        torch_file_path = os.path.join(cond_npz_file_path, f'torch_{iteration}.npz') 
        update_parameter = 1

        sim, design_region, fcen = define_simulator()
        torch_sim, torch_design_region, torch_fcen = define_simulator()

        opt = define_opt(sim=sim, fcen=fcen, design_region=design_region, condition=condition)
        torch_opt = define_opt(sim=torch_sim, fcen=torch_fcen, design_region=torch_design_region, condition=condition)
        # geometry read or create
        # Simulator based.
        
        if iteration == 0:
            opt, geo = create_geo(opt, fixed=fixed, path=init_structure_path)
            torch_geo = geo.copy() # 1000
            print(f"Torch Fisrt : {torch_geo}")
            try:
                torch_opt.update_design([torch_geo])
                print("opt.update_design successfully executed")

            except Exception as e:
                print(f"An error occurred: {e}")
                print(f"torch_issue, {torch_geo.shape}")
        
        else:
            geo, sim_adj = read_npz(file_path)
            print(f"Before GEO : {geo}")
            print(f"Before ADJ : {sim_adj}")
            
            opt, geo, sim_adj_norm = n_pixel_update_geometry(opt, flip_num, geo, sim_adj, update_parameter, ranking=ranking) 
            print(f"Before ADJ Norm : {sim_adj_norm}")
            print(f"GEO : {geo}")
            
            torch_geo, torch_adj = read_npz(torch_file_path)
            print(f"Torch Before GEO : {torch_geo}")
            print(f"Torch Before ADJ : {torch_adj}")
            
            torch_opt, torch_geo, torch_adj_norm = n_pixel_update_geometry(torch_opt, flip_num, torch_geo, torch_adj, update_parameter, ranking=ranking)
            print(f"Torch Before ADJ Norm : {torch_adj_norm}")
            print(f"Torch : {torch_geo}")
            # torch_geo = torch_geo[::100]
        # Learning based
        print(torch_geo)
        
        ai_start = time.time()
        with torch.no_grad():
            tensor_aa = torch.tensor(torch_geo, dtype=torch.float32)[None, :].to(device)
            label = torch.tensor(this_label, dtype=torch.float32).to(device)
            print(tensor_aa.shape, label.shape)
            if args.model_name == "VAE" or args.model_name == "new_VAE":
                adj, recons, mu, log_var, z = model(tensor_aa, label)
            elif args.model_name == "resMLP" or args.model_name == "FNO" or args.model_name == "ResidualFNO" or args.model_name == 'NewFNO': # non_condition....
                adj = model([tensor_aa, label])
            else:
                adj, _ = model([tensor_aa, label])
            adj = inv_scaler(adj)
            
            torch_adj = adj[0].cpu().numpy() # adjoint

        ai_end = time.time()
        # forward, backward simulation
        sim_start = time.time()
        f0, dJ_du = opt()
        sim_end = time.time()
        torch_f0, torch_dJ_du = torch_opt()
        
        
        
        write_to_file(os.path.join(txt_file_path, f'{dir_prefix}_time.txt'), sim_end - sim_start)
        write_to_file(os.path.join(txt_file_path, f'{dir_prefix}_torch_time.txt'), ai_end - ai_start)
        
        write_to_file(os.path.join(txt_file_path, f'{dir_prefix}_fom_n5.txt'), f0)
        write_to_file(os.path.join(txt_file_path, f'{dir_prefix}_torch_fom_n5.txt'), torch_f0)
        if iteration > 0:
            write_to_file(os.path.join(txt_file_path, f"{dir_prefix}_sim_fom_norm_n5.txt"), sim_adj_norm)
            write_to_file(os.path.join(txt_file_path, f"{dir_prefix}_torch_fom_norm_n5.txt"), torch_adj_norm)
        
        
            
            
        updated_geometry_array = []
        updated_gradient_array = []

        for i in range(0, len(geo)):
            updated_geometry_array.append(geo[i])
            
        for i in range(0, len(dJ_du)):
            updated_gradient_array.append(dJ_du[i])

        geometry_values_np = np.array(updated_geometry_array)
        adjoint_values_np = np.array(updated_gradient_array)

        file_name = os.path.join(cond_npz_file_path, f'{iteration+1}.npz') 
        torch_file_name = os.path.join(cond_npz_file_path, f'torch_{iteration+1}.npz') 
        np.savez(file_name, geometry=geometry_values_np, adjoint=adjoint_values_np)
        np.savez(torch_file_name, geometry=torch_geo, adjoint=torch_adj)

        sim.reset_meep()
        torch_sim.reset_meep()
        
    geometry_dict= {'torch_geo':torch_geo, 'geometry_values_np':geometry_values_np}
    
    torch.save(geometry_dict, f'last_structures/{condition}_last_data.pt')
        
        
        
if __name__=="__main__":
    """
        The value of condition should be in [-1.0, -0.9, ...., 1.0]
    """
    log_path = 'logs_2000/CondNewFNO_24_4_30_0.001/2024_08_21_05_45_21'
    epoch = 500
    fixed_init_structure = True
    init_structure_path = "data/dataset/init_structure.npy"
    default_iterations = 50 # 100 update
    flip_num=20
    # iter_mult = 100//flip_num
    # iterations = default_iterations * iter_mult
    for i in range(-10, 11, 1):
        iteration_func(default_iterations, log_path, epoch, flip_num=flip_num, ranking=True, condition=i/10, fixed=fixed_init_structure, init_structure_path=init_structure_path)
        

