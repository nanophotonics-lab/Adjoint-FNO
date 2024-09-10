import torch
import torch.nn as nn
import logging
import numpy as np
import os
import json
import argparse

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

def save_checkpoint_best_psnr(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'best_psnr_%depoch.pth' % epoch))
    
def save_checkpoint_for_finetune(model_path, epoch, iteration, model, optimizer, train_losses_rmse, train_losses_psnr, valid_losses_rmse, valid_losses_psnr):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_losses_rmse' : train_losses_rmse,
        'train_losses_psnr' : train_losses_psnr,
        'valid_losses_rmse' : valid_losses_rmse,
        'valid_losses_psnr' : valid_losses_psnr
    }

    torch.save(state, os.path.join(model_path, 'fine_tune_result_%depoch.pth' % epoch))


class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        mrae = torch.mean(error.contiguous().view(-1))
        return mrae
    
class Loss_MAE(nn.Module):
    def __init__(self):
        super(Loss_MAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label)
        mae = torch.mean(error.contiguous().view(-1))
        return mae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close
    
    
def save_model(args, model, optimizer, scheduler, epoch, path):
    file_name = f'{args.model_name}_{epoch}.pt'
    torch.save({
                "model": model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "scheduler" : scheduler.state_dict()
            }, os.path.join(path, file_name))
    
def load_model(model_path, model):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model"])
    return model

    
def save_dict(dict_data, path):
    with open(path, 'w') as f:
        json.dump(dict_data, f, indent=2)
        
def read_dict(path):
    with open(path, 'r') as f:
        mydict = json.load(f)
    return mydict

def save_args(args, path):
    file_name = 'args.txt'
    args_path = os.path.join(path, file_name)
    with open(args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def open_args(path, ipykernel=True):
    argparser = argparse.ArgumentParser()
    
    args = argparser.parse_args(args=[])
    file_name = 'args.txt'
    args_path = os.path.join(path, file_name)
    with open(args_path, 'r') as f:
        args.__dict__ = json.load(f)
    return args

    