import open3d as o3d
import numpy as np
import pyarrow.dataset as ds
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import MSELoss
import sklearn.metrics as metrics

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class DGCNN(nn.Module):
    def __init__(self,args):
        super(DGCNN, self).__init__()
        self.arg = args
        self.k = args.k
        self.emb_dims = args.emb_dims
        self.batch_size = args.batch_size
        self.num_points = args.num_points
        
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        #encode
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(128, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.dp3 = nn.Dropout(p=0.5)
        
        self.linear4 = nn.Linear(32,self.emb_dims)
        self.bn9 = nn.BatchNorm1d(self.emb_dims)
        self.dp4 = nn.Dropout(p=0.5)
        
        self.linear5 = nn.Linear(self.emb_dims, 32)
        self.bn10 = nn.BatchNorm1d(32)
        self.dp5 = nn.Dropout(p=0.5)
        
        self.linear6 = nn.Linear(32, 128)
        self.bn11 = nn.BatchNorm1d(128)
        self.dp6 = nn.Dropout(p=0.5)
        
        self.linear7 = nn.Linear(128, 512)
        self.bn12 = nn.BatchNorm1d(512)
        self.dp7 = nn.Dropout(p=0.5)
        
        self.linear8 = nn.Linear(512, 3 * (self.num_points + 1))
        self.bn13 = nn.BatchNorm1d(3 * (self.num_points + 1))
        self.dp8 = nn.Dropout(p=0.5)
           

    def forward(self, x):
        batch_size = x.size(0)   
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]   
        
  
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
    
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        
        x = F.leaky_relu(self.bn8(self.linear3(x)), negative_slope=0.2)
        x = self.dp3(x)
        
        x= F.leaky_relu(self.bn9(self.linear4(x)), negative_slope=0.2)
        x = self.dp4(x)
        
        x = F.leaky_relu(self.bn10(self.linear5(x)), negative_slope=0.2)
        x = self.dp5(x)
        
        x = F.leaky_relu(self.bn11(self.linear6(x)), negative_slope=0.2)
        x = self.dp6(x)
        
        x = F.leaky_relu(self.bn12(self.linear7(x)), negative_slope=0.2)
        x = self.dp7(x)
        
        x = F.leaky_relu(self.bn13(self.linear8(x)), negative_slope=0.2)
        x = self.dp8(x)
            
        return x.view(batch_size,3,-1)
 

