#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import multiprocessing
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import HydroNet
from model import get_graph_feature
from torch.utils.data import random_split
from model import DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import IOStream
from util import createConfusionMatrix
import sklearn.metrics as metrics
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
    ds = HydroNet(num_points=args.num_points, survey_list=['hampton'], resolution = [1])
    train_dataset, val_dataset = random_split(ds, [0.75, 0.25])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(HydroNet(num_points=args.num_points, partition = 'test', survey_list=['hampton'], resolution = [1]),
                              args.test_batch_size, shuffle=False, drop_last=True)

    device = torch.device("cuda" if args.cuda else "cpu")
    

    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    
    

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = MSELoss()

    for epoch in range(args.epochs):
        print("Epoch " + str(epoch))
        ####################
        # Train
        ####################
        train_loss = 0.0
        train_sqloss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        #train_true = []
        for data, label in train_loader:
            
            data, label = data.to(device, dtype=torch.float), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            true_feats = get_graph_feature(data)
            loss = criterion(logits[:,:,0,:], true_feats[:,:,0,:]) 
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            count += batch_size
            train_loss += loss.item() * batch_size
            train_sqloss += loss.item()**2 * batch_size
            train_pred.append(loss.item() < 0.5)
        train_avg = train_loss*1.0/count
        train_var = train_sqloss*1.0/count - train_avg**2
        writer.add_scalar('Mean Loss/train', train_avg, epoch)
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
            logits = model(data)
            true_feats = get_graph_feature(data)
            loss = criterion(logits[:,:,0,:], true_feats[:,:,0,:]) 
            count += 1
            test_loss += loss.item() * batch_size
            test_sqloss += loss.item()**2 * batch_size
            test_pred.append(loss.item() < 0.5)
        test_avg = test_loss*1.0/count
        test_var = test_sqloss*1.0/count - test_avg**2
        writer.add_scalar('Mean Loss/test', test_avg, epoch)
        writer.add_scalar('Var Loss/test', test_var, epoch)
        
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
            logits = model(data)
            true_feats = get_graph_feature(data)
            loss = criterion(logits[:,:,0,:], true_feats[:,:,0,:]) 
            count += 1
            val_loss += loss.item() * batch_size
            val_sqloss += loss.item()**2 * batch_size
            val_pred.append(loss.item() < 0.5)
        val_avg = val_loss*1.0/count
        val_var = val_sqloss*1.0/count - val_avg**2
        writer.add_scalar('Mean Loss/val', val_avg, epoch)
        writer.add_scalar('Var Loss/val', val_var, epoch)
        
        
        y_pred = np.concatenate((np.array(train_pred),np.array(val_pred),np.array(test_pred)))
        y_true = np.concatenate((np.ones(np.array(train_pred).shape),np.ones(np.array(val_pred).shape),np.ones(np.array(test_pred).shape)))
    
        
        writer.add_figure("Confusion matrix", createConfusionMatrix(y_true,y_pred), epoch)
        
        if epoch%10 == 0:
            io.cprint("saving model")
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name) 



if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
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
    parser.add_argument('--emb_dims', type=int, default=8, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
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
