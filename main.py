#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import multiprocessing
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import HydroNet
from torch.utils.data import random_split
from model import DGCNN,DGCNN_VAE
import numpy as np
from torch.utils.data import DataLoader
from util import IOStream
from util import createConfusionMatrix
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    train_ds = HydroNet(num_points=args.num_points, survey_list=['hampton'], resolution = [1])
    test_ds = HydroNet(num_points=args.num_points, partition = 'test', survey_list=['hampton'], resolution = [1])
    val_ds = HydroNet(num_points=args.num_points, partition = 'val', survey_list=['hampton'], resolution = [1])
    
    train_dataset, _ = random_split(train_ds, [0.8, 0.2])
    _, val_dataset = random_split(val_ds, [0.8, 0.2])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_ds, args.test_batch_size, shuffle=False, drop_last=True)

    device = torch.device("cuda" if args.cuda else "cpu")
    

    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    elif args.model == 'dgcnnvae':
        model = DGCNN_VAE(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-6)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = MSELoss()
    
    best_val_loss = 1e16

    for epoch in range(args.epochs):
        print("Epoch " + str(epoch))
        ####################
        # Train
        ####################
        train_loss = 0.0
        train_sqloss = 0.0
        train_mseloss = 0.0
        train_kldloss = 0.0
        
        count = 0.0
        model.train()
        train_pred = []
  

        for data, label in train_loader:
            
            data, label = data.to(device, dtype=torch.float), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            rec,mu,var = model(data)
            mse_loss = criterion(rec, data) 
            kld_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1), dim = 0)
            loss = mse_loss + 0.7*kld_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            count += batch_size
            train_loss += loss.item() * batch_size
            train_sqloss += loss.item()**2 * batch_size
            train_mseloss += mse_loss.item() * batch_size
            train_kldloss += kld_loss.item() *batch_size
            train_pred.append(loss.item() < 100)
        
        train_avg = train_loss*1.0/count
        train_var = train_sqloss*1.0/count - train_avg**2
        train_mse_avg = train_mseloss*1.0/count
        train_kld_avg = train_kldloss*1.0/count
        
        writer.add_scalar('Mean Loss/train', train_avg, epoch)
        writer.add_scalar('Mean MSE Loss/train', train_mse_avg, epoch)
        writer.add_scalar('Mean KLD Loss/train', train_kld_avg, epoch)
        writer.add_scalar('Var Loss/train', train_var, epoch)
        scheduler.step()
        ####################
        # Test
        ####################
        test_loss = 0.0
        test_sqloss = 0.0
        count = 0.0
        
        model.eval()
        test_pred = []
        for data, label in test_loader:
            data, label = data.to(device, dtype=torch.float), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            rec,_,_ = model(data)
            loss = criterion(rec[:,:,-1], data[:,:,-1]) 
            count += 1
            test_loss += loss.item() * batch_size
            test_sqloss += loss.item()**2 * batch_size
            test_pred.append(loss.item() < 100)
        test_avg = test_loss*1.0/count
        writer.add_scalar('Mean Loss/test', test_avg, epoch)
        
        ####################
        # Validation
        ####################
        val_loss = 0.0
        val_sqloss = 0.0
        count = 0.0
        model.eval()
        val_pred = []
        
        for data, label in val_loader:
            data, label = data.to(device, dtype=torch.float), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            rec,_,_ = model(data)
            loss = criterion(rec[:,:,-1], data[:,:,-1]) 
            count += 1
            val_loss += loss.item() * batch_size
            val_sqloss += loss.item()**2 * batch_size
            val_pred.append(loss.item() < 100)
        val_avg = val_loss*1.0/count
        val_var = val_sqloss*1.0/count - val_avg**2
        writer.add_scalar('Mean Loss/val', val_avg, epoch)
        writer.add_scalar('Var Loss/val', val_var, epoch)
        
        
        y_pred = np.concatenate((np.array(train_pred),np.array(val_pred),np.array(test_pred)))
        y_true = np.concatenate((np.ones(np.array(train_pred).shape),np.ones(np.array(val_pred).shape),np.zeros(np.array(test_pred).shape)))
    
        
        #writer.add_figure("Confusion matrix", createConfusionMatrix(y_true,y_pred), epoch)
        
        if val_avg < best_val_loss:
            io.cprint("saving model")
            torch.save(model.state_dict(), 'checkpoints/%s/models/model_%s.t7' % (args.exp_name,str(epoch))) 
            best_val_loss = val_avg



if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnnvae', metavar='N',
                        choices=['dgcnnvae', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='t_batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001, 0.01 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=256, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=16, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    #else:
    #    test(args, io)
