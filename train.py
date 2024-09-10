import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from model import *
from data.dataset import CategoricalStructureAdjointDataset
import argparse
from utils import *
import os
import datetime

ABS_PATH = os.path.abspath(__file__)
ABS_PATH = "/".join(ABS_PATH.split('/')[:-1]) # ~~/joon/corning_
    
def gd_norm_string(grads):
    this_string = ""
    for i in range(len(grads)):
        this_string += f"layer:{i}, grad norm : {grads[i]}\n"
    return this_string

def main(args):
    device = f"cuda:{args.device}"
    # device = 'cpu'

    print("Loading... : Training set")
    dataset = CategoricalStructureAdjointDataset(path=args.data_folder, data_num=args.data_num, mode='train')
    print(len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    print("Loading... : Validation set")
    validdataset = CategoricalStructureAdjointDataset(path=args.data_folder, mode='valid')
    valid_dataloader = DataLoader(dataset=validdataset, batch_size=args.batch_size, shuffle=True)
    if args.model_name == "MLP":
        model = LinearModel(input_dim=args.in_dim, dim=args.dim, layer_num=args.layer_num, condition=args.condition_num, p=args.p).to(device)
    else:
        model = Unet1d(input_dim=1, dim=args.dim, condition=args.condition_num).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.exp_decay)
    
    criterion = nn.L1Loss()
    
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    residual_str = ""
    if args.residual:
        residual_str = "_res"
        
    bn_str = ""
    if args.bn:
        bn_str = "_bn"
    
    fe_str = ""
    if args.fourier_embedding:
        fe_str = "_fe"
    
    log_dir_postpix = os.path.join(args.model_name+f"relu_{bn_str}{fe_str}{residual_str}_{args.dim}_{args.layer_num}_{args.p}", date_time)
    if args.data_num!= None:
        log_dir = os.path.join(args.log_path + f"_{args.data_num}", log_dir_postpix)
    else:
        log_dir = os.path.join(args.log_path, log_dir_postpix)


    os.makedirs(log_dir, exist_ok=True)
    log_file_dir = os.path.join(log_dir, 'train.log')
    
    logger = initialize_logger(log_file_dir)
    save_args(args, log_dir)
    for epoch in range(1, args.epoch+1):
        model.train()
        print(f"EPOCH : {epoch}/{args.epoch}")
        losses = AverageMeter()
        
        for st, adj, cond in tqdm(dataloader):
            st = st.to(device)
            adj = adj.to(device)
            cond = cond.to(device)
            if args.condition_num > 0 :
                pred_adj, _ = model([st, cond])
            else:
                pred_adj, _ = model(st)
            
            loss = criterion(pred_adj, adj)
            
            optimizer.zero_grad()
            loss.backward()
            
            # grads = []
            # for i in range(len(model.layers)):
            #     this_grad = []
            #     for p in model.layers[i].parameters():
            #         this_grad.append(np.linalg.norm(p.grad.cpu()))
            #     grads.append(np.mean(this_grad))            

            optimizer.step()
            losses.update(loss.data)
        # lr_scheduler.step()
        
        model.eval()
        valid_l = []
        valid_losses = AverageMeter()
        with torch.no_grad():
            for vst, vadj, vcond in tqdm(valid_dataloader):
                vst = vst.to(device)
                vadj = vadj.to(device)
                vcond = vcond.to(device)
                if args.condition_num > 0 :
                    pred_adj, _ = model([vst, vcond])
                else:
                    pred_adj, _ = model(vst)
                    
                val_loss = criterion(pred_adj, vadj)
                valid_losses.update(val_loss.data)
                valid_l.append(val_loss.data)
        # grad_string = gd_norm_string(grads)
        # print("Gradient Norm Monitor")
        # print(grad_string)
        # for va in valid_l:
        #     logger.info(" Epoch[%06d], VALID: %.9f" % (epoch, va))    
        logger.info(" Epoch[%06d], Train Loss: %.9f, Valid Loss: %.9f" % (epoch, losses.avg, valid_losses.avg))
        # logger.info(f"\n{grad_string}")
        if epoch % args.save_frequency==0:
            save_model(args, model, optimizer, lr_scheduler, epoch, log_dir)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', type=int, help='device', default=1)
    argparser.add_argument('--epoch', type=int, help='number of training epochs', default=500)
    argparser.add_argument('--in_dim', type=int, help='number of input channels of the network', default=100)
    argparser.add_argument('--out_dim', type=int, help='number of output channels of the network', default=100)
    argparser.add_argument('--condition_num', type=int, help='number of conditions of the network', default=1)
    argparser.add_argument('--dim', type=int, help='dimensions of the network', default=32)
    argparser.add_argument('--layer_num', type=int, help='number of layer of the network', default=4)
    argparser.add_argument('--residual', type=int, help='Residual scheme', default=False)
    argparser.add_argument('--bn', type=int, help='Batch Normalization flag', default=False)
    argparser.add_argument('--neural_representation', type=int, help='neural_representation', default=False)
    argparser.add_argument('--p', type=float, help='dropout prob of layer of the network', default=0.0)
    argparser.add_argument('--batch_size', type=int, help='batch size', default=512)
    argparser.add_argument('--lr', type=float, help='initial learning rate', default=2e-4)
    argparser.add_argument("--data_folder", type=str, help='path of dataset', default=os.path.join(ABS_PATH, "data/dataset"))
    argparser.add_argument('--data_num', type=int, help='the number of data', default=2000)
    argparser.add_argument("--model_save_path", type=str, help="the root dir \
                           to save checkpoints", default="./network_weights/")
    argparser.add_argument("--model_name", type=str, help="name for the model. : Unet1d / MLP \
                           used for storing under the model_save_path", \
                           default="Unet1d")
    argparser.add_argument("--exp_decay", type=float, help="exponential decay of \
                            learning rate, update per epoch", default=1.)
    argparser.add_argument("--continue_train", type=bool, help = "if true, continue \
                            train from continue_epoch", default=False)
    argparser.add_argument("--weight_decay", type=float, help="l2 regularization coeff", default=3e-3)
    argparser.add_argument("--log_path", type=str, help="Logger Path", default=os.path.join(ABS_PATH, "logs"))
    argparser.add_argument("--save_frequency", type=int, help="Save Frequency", default=100)
    args = argparser.parse_args()
    
    main(args)
    