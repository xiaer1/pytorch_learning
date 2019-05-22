from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SuperResolutionNet
from dataset import get_training_set,get_test_set
import argparse
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='train.log', level=logging.DEBUG, format=LOG_FORMAT)

def parse_training():
    parse = argparse.ArgumentParser(description='pytorch Super Res Example')
    parse.add_argument('--upscale_factor',type=int,required=True,help='super resolution upscale factor')
    parse.add_argument('--batch_size',type=int,default=32,help='training batch size')
    parse.add_argument('--test_bacth_size',type=int,default=10,help='test bacth size')
    parse.add_argument('--nEpochs',type=int,default=30,help='number of epochs')
    parse.add_argument('--lr',type=float,default=0.01,help='learning rate,default=0.01')
    parse.add_argument('--cuda',default=0,help='use cuda?')
    parse.add_argument('--threads',type=int,default=4,help='number of threads for datalooader to use')
    parse.add_argument('--seed',type=int,default=1234,help='random seed to use')

    return parse.parse_args()
def train(model,train_loader,test_loader,opt,device):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=opt.lr)

    best_test_psnr = None

    for i in range(opt.nEpochs):
        epoch_loss =  0
        for iteration,(X,y) in enumerate(train_loader):
            optimizer.zero_grad()
            X,y = X.to(device),y.to(device)
            y_ = model(X)
            loss = criterion(y_,y)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

            message = '====> Epoch [{}] ({} / {}), Loss : {:.4f}'.format(i,iteration,len(train_loader),loss.item())
            print(message);logging.info(message)
        message = '====> Epoch [{}] , Avg Loss : {:.4f}'.format(i, epoch_loss / len(train_loader))
        print(message);logging.info(message)

        test_psnr = test(model,test_loader,device)

        checkpoint_path = 'checkpoint_scale{}'.format(opt.upscale_factor)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        if best_test_psnr == None or test_psnr > best_test_psnr:
            best_test_psnr = test_psnr
            model_out_path = os.path.join(checkpoint_path,'model_epoch_{}.pth'.format(i))
            torch.save(model,model_out_path)
            message = 'checkpoint saved to {}'.format(model_out_path)
            print(message);logging.info(message)

def test(model,test_loader,device):
    avg_psnr = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for (X,y) in test_loader:
            X,y = X.to(device),y.to(device)

            pred = model(X)
            mse = criterion(pred,y)
            psnr = 10 * math.log10 ( 1 / mse.item())
            avg_psnr += psnr
    print("=====> Avg PSNR: {:.4f} dB".format(avg_psnr / len(test_loader)))

    return avg_psnr / len(test_loader)

def main():
    opt = parse_training()
    torch.manual_seed(opt.seed)
    device = ('cuda' if opt.cuda else 'cpu')

    train_set = get_training_set(opt.upscale_factor)
    train_loader = DataLoader(train_set,batch_size=opt.batch_size,shuffle=True,num_workers=opt.threads)

    test_set = get_test_set(opt.upscale_factor)
    test_loader = DataLoader(test_set, batch_size=opt.test_bacth_size, shuffle=False, num_workers=opt.threads)

    print('====> Build Model...')
    model = SuperResolutionNet(opt.upscale_factor).to(device)
    train(model,train_loader,test_loader,opt,device)

if __name__ == '__main__':
    main()