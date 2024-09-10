import torch
import pprint
from model import *
from thop import profile
import yaml
import os
import argparse
from functools import partial
import time
import timeit
import pickle


class Timer:
    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self

    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start


# 272, 206
def add_config_to_parser(parser, config):
    for key, value in config.items():
        if isinstance(value, dict) and key != "aux_params":
            # If the value is a nested dictionary, recursively add it to the parser
            subparser = parser.add_argument_group(key)
            add_config_to_parser(subparser, value)
        elif key == "aux_params":
            parser.add_argument(f'--{key}', type=dict, default=value, help=f'{key} (default: {value})')
        else:
            # Assume the value is a string (you can modify this based on your YAML structure)
            parser.add_argument(f'--{key}', type=str, default=value, help=f'{key} (default: {value})')

def yaml_to_args(model='fno2d'):
    parser = argparse.ArgumentParser(description='YAML to argparse', add_help=False)

    ABS_REP_PATH = os.path.abspath(__file__).split(os.path.sep)[:-1]

    yaml_path = os.path.sep + os.path.join(*ABS_REP_PATH, 'configs', model+".yaml")

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    add_config_to_parser(parser, config)
    
    return parser.parse_args()

if __name__ == '__main__':
    device= 'cuda:0'
    compile = False
    mlp = LinearModel(input_dim=100, dim=128, layer_num=4, condition=1).to(device)
    # fno = FNO(indim=100, dim=512, layer_num=8, condition_num=1).to(device)
    newfno = NewFNO(indim=1, dim=24, mode=30, layer_num=4, condition_num=1).to(device)
    
    VAE = VariationalDeepAdjointPredictor(input_dim=100, dim=[[256, 256, 256, 256, 128], 256], latent_dim = 128, layer_num=4, condition=1, p=0.5, concat=True).to(device)
    
    
    results = {}
    ft_dict = {}
    # precision_megabytes = (32 / 8.0) * 1e-6
    mill = 1000000
    # for k, model in zip(['FNO2d', 'ORIGUNET', 'MyFNO', 'Unet'],[fno2d, origunet, myfno, unet]):
    # for k, model in zip(['MyFNO', 'FNO2d', 'ORIGUNET', 'FNO2dFactor', 'NeurOLight', 'Unet'],[myfno, fno2d, origunet, f_fno , neurol, unet]):
    # for k, model in zip(['FNO2d'],[fno2d]):
    for k, model in zip(['mlp', 'newfno', 'VAE'],[mlp, newfno, VAE]):
    # for k, model in zip(['UNET'],[unet]):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_parameters = sum(p.numel() for p in model.parameters())
        # model_size = total_parameters * precision_megabytes
        model_size = total_parameters / mill
        
        
        print(f"{k} - MODEL SIZE (M) : {model_size}, TRAINABLE PARAMS : {trainable_params}, TOTAL PARAMS : {total_parameters}")
        
        if compile:
            model = torch.compile(model)
        bs = 16
        n_warmups = 10
        n_repeats = 10
        sleep = True
        
        shape = (100, )
        structure = torch.rand(*(bs, *shape), device=device)
        c = torch.rand(*(bs,), device=device)
        
        data = (structure, c)
        for _ in range(n_warmups):
            if k == 'VAE':
                _ = model(structure, c)
            else:
                _ = model(data)
            
        if device != 'cpu':
            with torch.cuda.device(device):
                torch.cuda.synchronize()
        pred_ft = 0
        with Timer() as ft:
            for i in range(n_repeats):
                with torch.set_grad_enabled(False):
                    if k == 'VAE':
                        _ = model(structure, c)
                    else:
                        _ = model(data)
                        
                if device != 'cpu':
                    with torch.cuda.device(device):
                        torch.cuda.synchronize()


        print(f"{k} forward time: {ft.dt/n_repeats:.3f}")
        # this_model_ft = np.array(model_ft[1:])
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_parameters = sum(p.numel() for p in model.parameters())
        # model_size = total_parameters * precision_megabytes
        model_size = total_parameters/mill
        results[k] = {"fwd_time": ft.dt / n_repeats, "num_params": trainable_params, "model_size": model_size}
        del model
        if device != 'cpu':
            with torch.cuda.device(device):
                torch.cuda.synchronize()
        # torch.cuda.empty_cache()
        if sleep:
            time.sleep(1)
            
    with open('/home/joon/InterpolationProject/MyProject/adj_save_dir/after0418/model_complexity/data.pickle', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
            
    pprint.pprint(results)
