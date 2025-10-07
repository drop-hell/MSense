# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 13:20:23 2018

@author: gk
"""
from re import I, S
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, BatchNorm1d
import math     #Informer embedding、attention
import scipy.sparse as sp
from scipy.sparse import linalg
import scipy.stats 

"""
x-> [batch_num,in_channels,num_nodes,tem_size],
"""

class TATT(nn.Module):
    def __init__(self,c_in,num_nodes,tem_size):
        super(TATT,self).__init__()
        self.conv1=Conv2d(c_in, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(num_nodes, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.w=nn.Parameter(torch.rand(num_nodes,c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b=nn.Parameter(torch.zeros(tem_size,tem_size), requires_grad=True)
        
        self.v=nn.Parameter(torch.rand(tem_size,tem_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.v)
        
    def forward(self,seq):
        c1 = seq.permute(0,1,3,2)#b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze(1)#b,l,n
        
        c2 = seq.permute(0,2,1,3)#b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze(1)#b,c,l
     
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1,self.w),f2)+self.b)
        logits = torch.matmul(self.v,logits)
        ##normalization
        a,_ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits,-1)
        return coefs
    
class SATT(nn.Module):
    def __init__(self,c_in,num_nodes,tem_size):
        super(SATT,self).__init__()
        self.conv1=Conv2d(c_in, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(tem_size, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.w=nn.Parameter(torch.rand(tem_size,c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b=nn.Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        
        self.v=nn.Parameter(torch.rand(num_nodes,num_nodes), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.v)
        
    def forward(self,seq):
        c1 = seq
        f1 = self.conv1(c1).squeeze(1)#b,n,l
        
        c2 = seq.permute(0,3,1,2)#b,c,n,l->b,l,n,c
        f2 = self.conv2(c2).squeeze(1)#b,c,n
     
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1,self.w),f2)+self.b)
        logits = torch.matmul(self.v,logits)
        ##normalization
        a,_ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits,-1)
        return coefs

class cheby_conv_ds(nn.Module):
    def __init__(self,device,c_in,c_out,K):
        super(cheby_conv_ds,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.K=K
        self.device = device
        
    def forward(self,x,adj,ds):
        nSample, feat_in,nNode, length  = x.shape
        Ls = []
        L0 = torch.eye(nNode).to(self.device)
        L1 = adj
    
        L = ds*adj
        I = ds*torch.eye(nNode).to(self.device)
        Ls.append(I)
        Ls.append(L)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            L3 =ds*L2
            Ls.append(L3)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out  

    
###ASTGCN_block
class ST_BLOCK_0(nn.Module):
    def __init__(self,device,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_0,self).__init__()
        
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.TATT=TATT(c_in,num_nodes,tem_size)
        self.SATT=SATT(c_in,num_nodes,tem_size)
        self.dynamic_gcn=cheby_conv_ds(device,c_in,c_out,K)
        self.K=K
        
        self.time_conv=Conv2d(c_out, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        #self.bn=BatchNorm2d(c_out)
        self.bn=LayerNorm([c_out,num_nodes,tem_size])
        
    def forward(self,x,supports):
        x_input=self.conv1(x)
        T_coef=self.TATT(x)
        T_coef=T_coef.transpose(-1,-2)
        x_TAt=torch.einsum('bcnl,blq->bcnq',x,T_coef)
        S_coef=self.SATT(x)#B x N x N
        
        spatial_gcn=self.dynamic_gcn(x_TAt,supports,S_coef)
        spatial_gcn=torch.relu(spatial_gcn)
        time_conv_output=self.time_conv(spatial_gcn)
        out=self.bn(torch.relu(time_conv_output+x_input))
        
        return  out,S_coef,T_coef    
     


###1
###DGCN_Mask&&DGCN_Res
class T_cheby_conv(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(T_cheby_conv,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        Lap = Lap.transpose(-1,-2)
        #print(Lap)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 

class ST_BLOCK_1(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_1,self).__init__()
        
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.TATT_1=TATT_1(c_out,num_nodes,tem_size)
        self.dynamic_gcn=T_cheby_conv(c_out,2*c_out,K,Kt)
        self.K=K
        self.time_conv=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        #self.bn=BatchNorm2d(c_out)
        self.c_out=c_out
        self.bn=LayerNorm([c_out,num_nodes,tem_size])
    def forward(self,x,supports):
        x_input=self.conv1(x)
        x_1=self.time_conv(x)
        x_1=F.leaky_relu(x_1)
        x_1=F.dropout(x_1,0.5,self.training)
        x_1=self.dynamic_gcn(x_1,supports)
        filter,gate=torch.split(x_1,[self.c_out,self.c_out],1)
        x_1=torch.sigmoid(gate)*F.leaky_relu(filter)
        x_1=F.dropout(x_1,0.5,self.training)
        T_coef=self.TATT_1(x_1)
        T_coef=T_coef.transpose(-1,-2)
        x_1=torch.einsum('bcnl,blq->bcnq',x_1,T_coef)
        out=self.bn(F.leaky_relu(x_1)+x_input)
        return out,supports,T_coef
        
    
###2    
##DGCN_R  
class T_cheby_conv_ds(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(T_cheby_conv_ds,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).repeat(nSample,1,1).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 


    
class SATT_2(nn.Module):
    def __init__(self,c_in,num_nodes):
        super(SATT_2,self).__init__()
        self.conv1=Conv2d(c_in, c_in, kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(c_in, c_in, kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False)
        self.bn=LayerNorm([num_nodes,num_nodes,12])
        self.c_in=c_in
    def forward(self,seq):
        shape = seq.shape
        f1 = self.conv1(seq).view(shape[0],self.c_in//4,4,shape[2],shape[3]).permute(0,3,1,4,2).contiguous()
        f2 = self.conv2(seq).view(shape[0],self.c_in//4,4,shape[2],shape[3]).permute(0,1,3,4,2).contiguous()
        
        logits = torch.einsum('bnclm,bcqlm->bnqlm',f1,f2)
        logits=logits.permute(0,3,1,2,4).contiguous()
        logits = torch.sigmoid(logits)
        logits = torch.mean(logits,-1)
        return logits
  

class TATT_1(nn.Module):
    def __init__(self,c_in,num_nodes,tem_size):
        super(TATT_1,self).__init__()
        self.conv1=Conv2d(c_in, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(num_nodes, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.w=nn.Parameter(torch.rand(num_nodes,c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b=nn.Parameter(torch.zeros(tem_size,tem_size), requires_grad=True)
        
        self.v=nn.Parameter(torch.rand(tem_size,tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn=BatchNorm1d(tem_size)
        
    def forward(self,seq):
        c1 = seq.permute(0,1,3,2)#b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()#b,l,n
        
        c2 = seq.permute(0,2,1,3)#b,c,n,l->b,n,c,l
        
        f2 = self.conv2(c2).squeeze()#b,c,n
         
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1,self.w),f2)+self.b)
        logits = torch.matmul(self.v,logits)                                   
        logits = logits.permute(0,2,1).contiguous()
        logits=self.bn(logits).permute(0,2,1).contiguous()
        coefs = torch.softmax(logits,-1)
        return coefs   


class ST_BLOCK_2_r(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_2_r,self).__init__()
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.TATT_1=TATT_1(c_out,num_nodes,tem_size)
        
        self.SATT_2=SATT_2(c_out,num_nodes)
        self.dynamic_gcn=T_cheby_conv_ds(c_out,2*c_out,K,Kt)
        self.LSTM=nn.LSTM(num_nodes,num_nodes,batch_first=True)#b*n,l,c
        self.K=K
        self.tem_size=tem_size
        self.time_conv=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.bn=BatchNorm2d(c_out)
        self.c_out=c_out
      
        
        
    def forward(self,x,supports):
        x_input=self.conv1(x)
        x_1=self.time_conv(x)
        x_1=F.leaky_relu(x_1)
        S_coef=self.SATT_2(x_1)
        shape=S_coef.shape
        h = Variable(torch.zeros((1,shape[0]*shape[2],shape[3]))).cuda()
        c=Variable(torch.zeros((1,shape[0]*shape[2],shape[3]))).cuda()
        hidden=(h,c)
        S_coef=S_coef.permute(0,2,1,3).contiguous().view(shape[0]*shape[2],shape[1],shape[3])
        S_coef=F.dropout(S_coef,0.5,self.training) 
        _,hidden=self.LSTM(S_coef,hidden)
        adj_out=hidden[0].squeeze().view(shape[0],shape[2],shape[3]).contiguous()
        adj_out1=(adj_out)*supports
        x_1=F.dropout(x_1,0.5,self.training)
        x_1=self.dynamic_gcn(x_1,adj_out1)
        filter,gate=torch.split(x_1,[self.c_out,self.c_out],1)
        x_1=torch.sigmoid(gate)*F.leaky_relu(filter)
        x_1=F.dropout(x_1,0.5,self.training)
        T_coef=self.TATT_1(x_1)
        T_coef=T_coef.transpose(-1,-2)
        x_1=torch.einsum('bcnl,blq->bcnq',x_1,T_coef)
        out=self.bn(F.leaky_relu(x_1)+x_input)
        return out,adj_out,T_coef



###Gated-STGCN(IJCAI)
class cheby_conv(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,device,c_in,c_out,K,Kt):
        super(cheby_conv,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.K=K
        self.device = device

    def forward(self, x, adj):
        nSample, feat_in, nNode, length = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).to(self.device)
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 

class ST_BLOCK_4(nn.Module):
    def __init__(self,device,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_4,self).__init__()
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.gcn=cheby_conv(device, c_out//2,c_out,K,1)
        self.conv2=Conv2d(c_out, c_out*2, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.c_out=c_out
        self.conv_1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
       

    def forward(self,x,supports):
        x_input1=self.conv_1(x)
        x1=self.conv1(x)
        filter1,gate1=torch.split(x1,[self.c_out//2,self.c_out//2],1)
        x1=(filter1)*torch.sigmoid(gate1)
        x2=self.gcn(x1,supports)
        x2=torch.relu(x2)
        #x_input2=self.conv_2(x2)
        x3=self.conv2(x2)
        filter2,gate2=torch.split(x3,[self.c_out,self.c_out],1)
        x=(filter2+x_input1)*torch.sigmoid(gate2)
        return x

###GRCN(ICLR)
class gcn_conv_hop(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ] - input of one single time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : gcn_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,device,c_in,c_out,K,Kt):
        super(gcn_conv_hop,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv1d(c_in_new, c_out, kernel_size=1,
                          stride=1, bias=True)
        self.K=K
        self.device=device
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode  = x.shape
        
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).to(self.device)
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcn,knq->bckq', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode)
        out = self.conv1(x)
        return out 



class ST_BLOCK_5(nn.Module):
    def __init__(self,device, c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_5,self).__init__()
        self.gcn_conv=gcn_conv_hop(device, c_out+c_in,c_out*4,K,1)
        self.c_out=c_out
        self.tem_size=tem_size
        self.device = device
        
        
    def forward(self,x,supports):
        shape = x.shape
        h = Variable(torch.zeros((shape[0],self.c_out,shape[2]))).to(self.device)
        c = Variable(torch.zeros((shape[0],self.c_out,shape[2]))).to(self.device)
        out=[]
        
        for k in range(self.tem_size):
            input1=x[:,:,:,k]
            tem1=torch.cat((input1,h),1)
            fea1=self.gcn_conv(tem1,supports)
            i,j,f,o = torch.split(fea1, [self.c_out, self.c_out, self.c_out, self.c_out], 1)
            new_c=c*torch.sigmoid(f)+torch.sigmoid(i)*torch.tanh(j)
            new_h=torch.tanh(new_c)*(torch.sigmoid(o))
            c=new_c
            h=new_h
            out.append(new_h)
        x=torch.stack(out,-1)
        return x 

    
###OTSGGCN(ITSM)
class cheby_conv1(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(cheby_conv1,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 

class ST_BLOCK_6(nn.Module):
    def __init__(self,device,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_6,self).__init__()
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.gcn=cheby_conv(device,c_out,2*c_out,K,1)
        
        self.c_out=c_out
        self.conv_1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        
    def forward(self,x,supports):
        x_input1=self.conv_1(x)
        x1=self.conv1(x)   
        x2=self.gcn(x1,supports)
        filter,gate=torch.split(x2,[self.c_out,self.c_out],1)
        x=(filter+x_input1)*torch.sigmoid(gate)
        return x


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.kaiming_normal_(self.W.data, mode='fan_in', nonlinearity='leaky_relu')
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.kaiming_normal_(self.a.data, mode='fan_in', nonlinearity='leaky_relu')

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):

        h = torch.matmul(inp, self.W)  # inp.shape: (B, N, in_features), h.shape: (B, N, out_features)
        N = h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features),
                             h.repeat(1, N, 1)], dim=-1).view(-1, N, N, 2 * self.out_features)

        # [B, N, N, 1] => [B, N, N] 
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# gwnet
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        """
        x in shape [batch_size, # of nodes, seq_len]
        adj in shape [# of nodes, # of nodes]
        return in shape [batch_size, # of nodes, seq_len]
        """
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        return x


class multi_gat(nn.Module):
    def __init__(self, c_in, seq_len, dropout, support_len=3):
        super(multi_gat, self).__init__()
        self.gat = GAT(nfeat=seq_len*c_in, nhid=32, nclass=seq_len*c_in, dropout=dropout, nheads=support_len, alpha=0.2)

    def forward(self, x, support):
        '''
        x in shape [batch, 32, # of nodes, seq_len]
        return in shape [batch, 32, # of nodes, seq_len]
        '''
        supports = torch.stack(support)
        agg_sup = torch.sum(supports, dim=0)
        s = x.shape
        input_x = x.permute(0, 2, 1, 3).reshape(s[0], s[2], -1).contiguous()
        output_m = self.gat(input_x, agg_sup).reshape(s[0], s[2], s[1], s[3])
        y = output_m.permute(0, 2, 1, 3).contiguous()
        return y
    

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        A = A.transpose(-1, -2)            
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()                      

    
class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class multi_gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(multi_gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        '''
        x in shape [batch, 32, # of nodes, seq_len]
        '''
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)           #'ncvl,vw->ncwl'
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
    

class GCNLayer(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.2):
        super(GCNLayer, self).__init__()

        self.fc = nn.Conv2d(c_in, c_out, kernel_size=1, bias=True) # 留空间结构，适合处理四维输入，参数共享，效率更高。
        self.time_dilated_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 3), dilation=(1, 2), padding=(0, 2) )
                                            #  kernel_size=(1, 3), dilation=(1, 2), padding=(0, 2))
        
        # # 节点维度的扩张卷积，使用双向卷积
        self.node_dilated_conv_forward = nn.Conv2d(c_in, c_out, kernel_size=(3, 1),dilation=(2, 1), padding=(2, 0))
        
        self.node_dilated_conv_backward = nn.Conv2d(c_in, c_out, kernel_size=(3, 1), dilation=(2, 1), padding=(2, 0))
        self.combine = nn.Conv2d(2 * c_out, c_out, kernel_size=1)
        
        self.nrom = nn.BatchNorm2d(c_out)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, input, adj):
        """
        x: [batch_size, channels, numnodes, len]
        adj: [numnodes, numnodes]
        """
        
        # 使用 einsum 进行 A * X
        # x = torch.einsum('ncvl,vw->ncwl', (input, adj))  # [batch_size, channels, numnode, len]
        x = torch.einsum('bcvl,vw->bcwl', (input, adj))  # [batch_size, channels, numnodes, len]
        # 线性变换
        x = self.fc(x)  # [batch_size, c_out, numnode, len]

        time_conv = self.time_dilated_conv(input)
        # 节点维度卷积
        node_conv_forward = self.node_dilated_conv_forward(input)  # 前向卷积 (Batchsize, out_channels, numnodes, len)
        reversed_x = torch.flip(input, dims=[-2]) 
        node_conv_backward = self.node_dilated_conv_backward(reversed_x)  # 反向卷积 (Batchsize, out_channels, numnodes, len)
        node_conv = node_conv_forward+node_conv_backward
        nt_conv = torch.cat([time_conv,node_conv], dim=1)  #  (Batchsize, 2 * out_channels, numnodes, len)
        x = x + self.combine(nt_conv)
        x = self.nrom(x)
        
        x = self.relu(x)
        
        x = self.dropout(x)
        
        return x



class EnhancedGatingNetwork(nn.Module):
    def __init__(self, input_dim, support_len, dataft, hidden_dim=32, num_heads=3):
        super(EnhancedGatingNetwork, self).__init__()
        self.num_heads = num_heads
        # print(f"adj gate type depend on data: {dataft}")
        
        # print(f"gate with {hidden_dim} dims {num_heads} heads ")

        self.act = nn.LeakyReLU() if dataft == 'CPU' else nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                self.act,
                nn.Linear(hidden_dim, support_len)
            ) for _ in range(num_heads)
        ])
        self._initialize_weights()

    def _initialize_weights(self):
        for head in self.heads:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    # 使用 Kaiming 正态分布初始化
                    # nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    def forward(self, x):
        # 每个头独立生成权重
        gating_weights = [F.softmax(head(x), dim=-1) for head in self.heads]  
        # gating_weights = [self.softmax(head(x)) for head in self.heads]  
        gating_weights = torch.stack(gating_weights, dim=1)  # [batch_size, num_heads, support_len]

        # 合并多头权重，通过平均
        final_weights = gating_weights.mean(dim=1)  # [batch_size, support_len]
        # final_weights = gating_weights.mean(dim=[0,1])  # [support_len]
        
        return torch.nan_to_num(final_weights, nan=0.0)
    
class Fuseadj(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len, dataft='CPU', adjid=None, sigma=0.0):
        super(Fuseadj, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.support_len = support_len
        self.adjid=adjid
        self.sigma=sigma
        self.gating_network = EnhancedGatingNetwork(c_in, support_len, dataft)
        
        initial_alpha= 1.0
        initial_threshold=0.5
        # print(f"sig alpha: {initial_alpha} down: { initial_threshold}")
        self.alphas = nn.Parameter(torch.ones(support_len) * initial_alpha)
        self.thresholds = nn.Parameter(torch.ones(support_len) * initial_threshold)
        # self.upholds = nn.Parameter(torch.ones(support_len) * initial_uphold)
        
        # self.thresholds = nn.Parameter(torch.empty(num_matrices).uniform_(0.1, 0.5))
        # self.upholds = nn.Parameter(torch.empty(num_matrices).uniform_(0.7, 0.9))
        # 用于记录和输出权重的相关参数
        self.count = 0
        self.weight_records = []
        

    def output_weights(self, adj_weights):
        self.count += 1

        average_weights = adj_weights.mean(dim=0).detach().cpu().numpy()
        # average_weights = adj_weights.detach().cpu().numpy()
        self.weight_records.append(average_weights)
        
        # 达到指定次数后，计算平均权重并输出
        if self.count == 38 :  # 每38次输出一次
            average_weights = torch.mean(torch.tensor(self.weight_records), dim=0)
            print([f"{weight:.3f}" for weight in average_weights])
            self.weight_records = []  # 重置记录
            self.count = 0  # 重置计数
   
    def forward(self, x, support):
        """
        x: [batch_size, channels, numnodes, len]
        support: [support_len, numnodes, numnodes]
        """
        gating_input = torch.cat([
            x.mean(dim=[2, 3]), 
            # x.std(dim=[2, 3])
        ], dim=-1)
        # 通过门控网络生成权重
        gating_weights = self.gating_network(gating_input)  # [batch_size, support_len]
        # 对 support 中的每个邻接矩阵加权求和，得到每个样本的融合邻接矩阵
        support = torch.stack(support).to(x.device)  # [support_len, numnodes, numnodes]
        adjusted_matrices = []
       
        for i, adj_matrix in enumerate(support):
            scaled_matrix = 1 / (1 + torch.exp(-self.alphas[i] * adj_matrix))
            scaled_matrix = torch.nan_to_num(scaled_matrix, nan=0.0)
            sparse_matrix = torch.where(scaled_matrix >= self.thresholds[i], scaled_matrix, torch.zeros_like(scaled_matrix))
            sparse_matrix = torch.nan_to_num(sparse_matrix, nan=0.0)
            if (self.adjid != None) and (i == (self.adjid)) :
                A = sparse_matrix
                mean_A = torch.mean(A)
                # var_A = torch.var(A)
                # sigma_A = torch.sqrt(var_A)
                # sigma_A = torch.maximum(sigma_A, torch.tensor(1e-6)) 
                # print(mean_A.item(), sigma_A.item())

                noise = torch.normal(mean_A, self.sigma, size=A.shape).to(sparse_matrix.device)
                sparse_matrix = sparse_matrix + noise
            adjusted_matrices.append(sparse_matrix)
        adjusted_matrices = torch.stack(adjusted_matrices, dim=0)
        
        # A_fused = torch.einsum('s,sij->ij', gating_weights, adjusted_matrices) # [ numnodes, numnodes]
        A_fused = torch.einsum('bs,sij->bij', gating_weights, adjusted_matrices)  # [batch_size, numnodes, numnodes]
        A_fused = A_fused.mean(dim=0).squeeze()

        self.output_weights(gating_weights)
        
        return  A_fused


class GraphSAGELayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        """
        自定义的GraphSAGE层。

        Args:
            in_feats (int): 输入特征维度。
            out_feats (int): 输出特征维度。
            activation (callable, optional): 激活函数。默认使用ReLU。
        """
        super(GraphSAGELayer, self).__init__()
        self.W_self = nn.Linear(in_feats, out_feats, bias=False)
        self.W_neigh = nn.Linear(in_feats, out_feats, bias=False)

    def forward(self, X, A, degree):
        """
        前向传播。

        Args:
            X (torch.Tensor): 输入特征，形状为 (Batch, C, N, L)。
            A (torch.Tensor): 邻接矩阵，形状为 (N, N)。
            degree (torch.Tensor): 每个节点的度，形状为 (N,)。

        Returns:
            torch.Tensor: 输出特征，形状为 (Batch, out_feats, N, L)。
        """
        # X: Batch x C x N x L -> Batch x N x L x C
        X = X.permute(0, 2, 3, 1)  # Batch x N x L x C

        # 聚合邻居特征:X @ A
        mean_neighbors = torch.einsum('bnlc, nm -> bmlc', X, A)# 结果: Batch x N x L x C

        # 均值聚合
        mean_neighbors = mean_neighbors / degree.view(1, -1, 1, 1)  # Batch x N x L x C

        h_self = self.W_self(X)       # Batch x N x L x out_feats
        h_neigh = self.W_neigh(mean_neighbors)  # Batch x N x L x out_feats

        # 结合自身特征和邻居特征
        h = h_self + h_neigh  # Batch x N x L x out_feats

        # h = F.relu(h)

        h = h.permute(0, 3, 1, 2)  # Batch x out_feats x N x L

        return h
class MoE_SAG(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len,vmindex_in_cities, hidden_size = 128):
        super(MoE_SAG, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dropout = dropout
        self.support_len = support_len
        self.num_cities = len(vmindex_in_cities)
        # 定义多个GCN专家，每个专家对应一个邻接矩阵
        # print(f"split into {self.num_cities} SAG of dim {hidden_size}")

        self.convs =  nn.Conv2d(c_in, hidden_size, kernel_size=1)
        
        self.experts = nn.ModuleList([
            GraphSAGELayer(hidden_size, hidden_size) for _ in range(self.num_cities)
        ])
        self.negative_slope = 0.01

        self.convt = nn.ConvTranspose2d(hidden_size, c_out, kernel_size=1)
        self.vmindex_in_cities = vmindex_in_cities

    def leaky_relu(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope)

    def inverse_leaky_relu(self, y):
        y_cleaned = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.where(y_cleaned > 0, y_cleaned, y_cleaned / self.negative_slope)
    
    def forward(self, input, A_fuse):
        """
        Args:
            X (torch.Tensor): 输入特征，形状为 (Batch, C, N, L)。
            support (torch.Tensor): 总的邻接矩阵，形状为 (N, N)。

        Returns:
            torch.Tensor: 输出特征，形状为 (Batch, out_feats, Numnodes, L)。
        """
        batch_size, C, N, L = input.size()

        output = torch.zeros(batch_size, self.c_out, N, L, device=input.device)
        # expert_outputs = []

        for i in range(self.num_cities):
            city_indices = self.vmindex_in_cities[i]  # List[int]
            city_data = input[:, :, city_indices, :]      # Batch x C x N_city x L
            A_city = A_fuse[city_indices][:, city_indices] # N_city x N_city
            degree = A_city.sum(dim=1)                   # N_city

            x = self.convs(city_data)
            x = self.leaky_relu(x)    
            x = self.experts[i](x, A_city, degree)  # Batch x out_feats x N_city x L
            x = self.inverse_leaky_relu(x)       
            x = self.convt(x)

            output[:, :, city_indices, :] = x  # Batch x out_feats x Numnodes x L

        return output
    
from typing import List, Tuple
class PMoE_SAG(nn.Module):
    vmindex_in_cities_flat: torch.Tensor
    group_offsets: torch.Tensor
    def __init__(self, c_in, c_out, dropout, support_len,vmindex_in_cities, hidden_size = 128):
        super(PMoE_SAG, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dropout = dropout
        self.support_len = support_len
        self.num_cities = len(vmindex_in_cities)
        # 定义多个GCN专家，每个专家对应一个邻接矩阵
        # print(f"split into {self.num_cities} SAG of dim {hidden_size}")
        # self.vmindex_in_cities = vmindex_in_cities
        vmindex_in_cities_flat = []
        group_offsets = [0]
        for group in vmindex_in_cities:
            vmindex_in_cities_flat.extend(group)
            group_offsets.append(len(vmindex_in_cities_flat))
        self.register_buffer(
            'vmindex_in_cities_flat',
            torch.tensor(vmindex_in_cities_flat, dtype=torch.long)
        )
        self.register_buffer(
            'group_offsets',
            torch.tensor(group_offsets, dtype=torch.long)
        )


        self.convs =  nn.Conv2d(c_in, hidden_size, kernel_size=1)
        
        self.experts = nn.ModuleList([
            GraphSAGELayer(hidden_size, hidden_size) for _ in range(self.num_cities)
        ])
        self.negative_slope = 0.01

        self.convt = nn.ConvTranspose2d(hidden_size, c_out, kernel_size=1)
        

    def leaky_relu(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope)

    def inverse_leaky_relu(self, y):
        y_cleaned = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.where(y_cleaned > 0, y_cleaned, y_cleaned / self.negative_slope)
    
    def process_city(self, i: int, city_data, A_city, city_indices) :
        """
        独立处理单个城市的计算逻辑
        """
        degree = A_city.sum(dim=1)  # N_city
        x = self.convs(city_data)
        x = self.leaky_relu(x)
        # x = self.experts[i](x, A_city, degree)  # 使用专家网络
        out = x
        for idx, expert_layer in enumerate(self.experts):
            if idx == i:
                out = expert_layer(x, A_city, degree)
        
        x = self.inverse_leaky_relu(out)
        x = self.convt(x)
        return (city_indices, x)  # 返回城市索引和输出
    
    def forward(self, input, A_fuse):
        """
        Args:
            X (torch.Tensor): 输入特征，形状为 (Batch, C, N, L)。
            support (torch.Tensor): 总的邻接矩阵，形状为 (N, N)。

        Returns:
            torch.Tensor: 输出特征，形状为 (Batch, out_feats, Numnodes, L)。
        """
        batch_size, C, N, L = input.size()

        output = torch.zeros(batch_size, 32, N, L, device=input.device)
        
        futures = torch.jit.annotate(List[torch.jit.Future[Tuple[torch.Tensor, torch.Tensor]]], [])
        
        
        for group_idx in range(self.num_cities):
            start = self.group_offsets[group_idx].item()
            end = self.group_offsets[group_idx + 1].item()
            city_indices = self.vmindex_in_cities_flat[start:end]
            
            # 确保 city_indices 是一个 Tensor
            # 如果一个组内只有一个城市，保持维度
            if city_indices.numel() == 1:
                city_indices = city_indices.unsqueeze(0)
            
            # 提取城市数据
            city_data = input[:, :, city_indices, :]  # Batch x C x N_city x L
            A_city = A_fuse[city_indices][:, city_indices]  # N_city x N_city
            futures.append(torch.jit.fork(self.process_city,
                                          group_idx,city_data, A_city,city_indices)
                                          )

        for future in futures:
            city_indices, city_output = torch.jit.wait(future)
            output[:, :, city_indices, :] = city_output  # 合并每个城市的输出

        return output

class nconv_batch(nn.Module):
    def __init__(self):
        super(nconv_batch, self).__init__()

    def forward(self,x, A):
        A=A.transpose(-1,-2)
        x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        return x.contiguous()
    
class linear_time(nn.Module):
    def __init__(self,c_in,c_out,Kt):
        super(linear_time,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class multi_gcn_time(nn.Module):
    def __init__(self,c_in,c_out,Kt,dropout,support_len=3,order=2):
        super(multi_gcn_time,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear_time(c_in,c_out,Kt)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class SATT_pool(nn.Module):
    def __init__(self,c_in,num_nodes):
        super(SATT_pool,self).__init__()
        self.conv1=Conv2d(c_in, c_in, kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(c_in, c_in, kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False)
        self.c_in=c_in
    def forward(self,seq):
        shape = seq.shape
        f1 = self.conv1(seq).view(shape[0],self.c_in//4,4,shape[2],shape[3]).permute(0,3,1,4,2).contiguous()#通道数减少到原来的1/4
        f2 = self.conv2(seq).view(shape[0],self.c_in//4,4,shape[2],shape[3]).permute(0,1,3,4,2).contiguous()#并产生新的通道，通道维度分为四个子张量
        
        logits = torch.einsum('bnclm,bcqlm->bnqlm',f1,f2)
        
        logits=logits.permute(0,3,1,2,4).contiguous()
        logits = F.softmax(logits,2)
        logits = torch.mean(logits,-1)
        return logits

class SATT_h_gcn(nn.Module):
    def __init__(self,c_in,tem_size):
        super(SATT_h_gcn,self).__init__()
        self.conv1=Conv2d(c_in, c_in//8, kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(c_in, c_in//8, kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=False)
        self.c_in=c_in
    def forward(self,seq,a):
        shape = seq.shape
        f1 = self.conv1(seq).squeeze().permute(0,2,1).contiguous()
        f2 = self.conv2(seq).squeeze().contiguous()
        
        logits = torch.matmul(f1,f2)
        
        logits=F.softmax(logits,-1)
        
        return logits

class multi_gcn_batch(nn.Module):
    def __init__(self,c_in,c_out,Kt,dropout,support_len=3,order=2):
        super(multi_gcn_batch,self).__init__()
        self.nconv = nconv_batch()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear_time(c_in,c_out,Kt)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:            
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gate(nn.Module): #
    def __init__(self,c_in):
        super(gate,self).__init__()
        self.conv1=Conv2d(c_in, c_in//2, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
    def forward(self,seq,seq_cluster):
        
        out=torch.cat((seq,(seq_cluster)),1)
        
        return out
           
    
class Transmit(nn.Module): 
    def __init__(self,c_in,tem_size,transmit,num_nodes,cluster_nodes):
        super(Transmit,self).__init__()
        self.conv1=Conv2d(c_in, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(tem_size, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.w=nn.Parameter(torch.rand(tem_size,c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b=nn.Parameter(torch.zeros(num_nodes,cluster_nodes), requires_grad=True)
        self.c_in=c_in
        self.transmit=transmit

    def forward(self,seq,seq_cluster):
        
        c1 = seq
        f1 = self.conv1(c1).squeeze(1)#b,n,l
        
        c2 = seq_cluster.permute(0,3,1,2)#b,c,n,l->b,l,n,c
        f2 = self.conv2(c2).squeeze(1)#b,c,n
        logits=torch.sigmoid(torch.matmul(torch.matmul(f1,self.w),f2)+self.b)#w*f1*f2+b
        a = torch.mean(logits, 1, True)
        logits = logits - a
        logits = torch.sigmoid(logits)
        
        coefs = (logits)*self.transmit  
        return coefs    

class T_cheby_conv_ds_1(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(T_cheby_conv_ds_1,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, Kt),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).repeat(nSample,1,1).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 
    
class dynamic_adj(nn.Module):
    def __init__(self,c_in,num_nodes):
        super(dynamic_adj,self).__init__()
        
        self.SATT=SATT_pool(c_in,num_nodes)
        self.LSTM=nn.LSTM(num_nodes,num_nodes,batch_first=True)#b*n,l,c
    def forward(self,x):
        S_coef=self.SATT(x)        
        shape=S_coef.shape
        h = Variable(torch.zeros((1,shape[0]*shape[2],shape[3]))).cuda()
        c=Variable(torch.zeros((1,shape[0]*shape[2],shape[3]))).cuda()
        hidden=(h,c)
        S_coef=S_coef.permute(0,2,1,3).contiguous().view(shape[0]*shape[2],shape[1],shape[3])
        S_coef=F.dropout(S_coef,0.5,self.training) #2020/3/28/22:17,试验下效果
        _,hidden=self.LSTM(S_coef,hidden)
        adj_out=hidden[0].squeeze().view(shape[0],shape[2],shape[3]).contiguous()
        
        return adj_out
    
    
class GCNPool_dynamic(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,
                 Kt,dropout,pool_nodes,support_len=3,order=2):
        super(GCNPool_dynamic,self).__init__()
        self.dropout=dropout
        self.time_conv=Conv2d(c_in, 2*c_out, kernel_size=(1, Kt),padding=(0,0),
                          stride=(1,1), bias=True,dilation=2)
        
        self.multigcn=multi_gcn_time(c_out,2*c_out,Kt,dropout,support_len,order)
        self.multigcn1=multi_gcn_batch(c_out,2*c_out,Kt,dropout,support_len,order)
        self.dynamic_gcn=T_cheby_conv_ds_1(c_out,2*c_out,order+1,Kt)
        self.num_nodes=num_nodes
        self.tem_size=tem_size
        self.TAT=TATT_1(c_out,num_nodes,tem_size)
        self.c_out=c_out
        #self.gate=gate1(c_out)
        self.bn=BatchNorm2d(c_out)
        
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.SATT=SATT_pool(c_out,num_nodes)
        self.LSTM=nn.LSTM(num_nodes,num_nodes,batch_first=True)#b*n,l,c
        
    
    def forward(self,x,support):
        residual = self.conv1(x)
        
        x=self.time_conv(x)
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*torch.sigmoid(x2)
        
        
        x=self.multigcn(x,support) 
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*(torch.sigmoid(x2)) 
        
          
        T_coef=self.TAT(x)
        T_coef=T_coef.transpose(-1,-2)
        x=torch.einsum('bcnl,blq->bcnq',x,T_coef)       
        out=self.bn(x+residual[:, :, :, -x.size(3):])
        #out=torch.sigmoid(x)
        return out



           
class GCNPool(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,
                 Kt,dropout,pool_nodes,support_len=3,order=2):
        super(GCNPool,self).__init__()
        self.time_conv=Conv2d(c_in, 2*c_out, kernel_size=(1, Kt),padding=(0,0),
                          stride=(1,1), bias=True,dilation=2)#
        
        self.multigcn=multi_gcn_time(c_out,2*c_out,Kt,dropout,support_len,order)
        
        self.num_nodes=num_nodes
        self.tem_size=tem_size
        self.TAT=TATT_1(c_out,num_nodes,tem_size)
        self.c_out=c_out
       
        self.bn=BatchNorm2d(c_out)
        
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        
        
        
    
    def forward(self,x,support):
        residual = self.conv1(x)
        
        x=self.time_conv(x)
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*torch.sigmoid(x2)           
        
        
        x=self.multigcn(x,support)       
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*(torch.sigmoid(x2)) 
       
        
        T_coef=self.TAT(x)
        T_coef=T_coef.transpose(-1,-2)
        x=torch.einsum('bcnl,blq->bcnq',x,T_coef)
        out=self.bn(x+residual[:, :, :, -x.size(3):])
        return out
        

# Informer   
## Informer Decoder
class DecoderLayerI(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayerI, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class DecoderI(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(DecoderI, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
## Informer Encoder
class ConvLayerI(nn.Module):
    def __init__(self, c_in):
        super(ConvLayerI, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x

class EncoderLayerI(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayerI, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
       
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn

class EncoderI(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(EncoderI, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns    
## Informer Attention
### from math import sqrt
### from utils.masking import TriangularCausalMask, ProbMask
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
## Informer Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        # self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:,:,3]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,2])
        weekday_x = self.weekday_embed(x[:,:,1])
        day_x = self.day_embed(x[:,:,0])

        return hour_x + weekday_x + day_x + minute_x
        #return minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        # freq_map = {'h':3, 't':4, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq  
                                                    ) if embed_type!='timeF' else TimeFeatureEmbedding(
                                                    d_model=d_model, embed_type=embed_type, freq=freq)
    
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
       
        a = self.value_embedding(x)
        b = self.position_embedding(x)
        c = self.temporal_embedding(x_mark)
        x = a + b + c
       
        return self.dropout(x)

# Autoformer
class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
class EncoderLayerA(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayerA, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class EncoderA(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(EncoderA, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
class DecoderLayerA(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayerA, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class DecoderA(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(DecoderA, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
    
## Autoformer DataEmbeding  ,Token\Pos\Temp Embed is same with Informer
class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

## Autoformer Attention
class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

# N-BEATS
class NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self, input_size, theta_size, basis_function, layers, layer_size):
        """
        N-BEATS block.

        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features=input_size, out_features=layer_size)] +
                                      [nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x):
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input) # [64,1,194,24]
        return self.basis_function(basis_parameters)

class GenericBasis(nn.Module):
    """
    Generic basis function.
    """
    def __init__(self, backcast_size, forecast_size):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta):
        #a = theta 
        #b, c = theta[:,: ,: , :self.backcast_size], theta[:, :, :, -self.forecast_size:] #
        return theta[:,: ,: , :self.backcast_size], theta[:, :, :, -self.forecast_size:]

# TimesNet
class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, d_ff, top_k, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res
# DCRNN
    

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))
def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)

class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str, device):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type
        self.device = device

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=self.device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=self.device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]
    
class DCGRUCell(torch.nn.Module):
    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, device, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        self.device = device

        supports = []
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
            supports.append(calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support, self.device))

        self._fc_params = LayerParams(self, 'fc', self.device)
        self._gconv_params = LayerParams(self, 'gconv', self.device)

    @staticmethod
    def _build_sparse_matrix(L, device):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L

    def forward(self, inputs, hx):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = torch.sigmoid(fn(inputs, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])
    
# NHITS 
class IdentityBasis(nn.Module):
    def __init__(self, backcast_size, forecast_size):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta):

        backcast = theta[:, :,: , :self.backcast_size]
        forecast = theta[:, :,: , -self.forecast_size:]
        a = self.backcast_size
        b = self.forecast_size
        return backcast, forecast

class NHITSBlock(nn.Module):
    def __init__(self, input_size, theta_size, basis_function, layers=3, layer_size=512, pool_kernel_size=2):
        super(NHITSBlock, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, layer_size)] +
                                    [nn.Linear(layer_size, layer_size) for _ in range(layers - 1)])
        self.basis_parameters = nn.Linear(layer_size, theta_size)
        self.basis_function = basis_function

        self.pooling_layer = nn.MaxPool2d(kernel_size=(1, pool_kernel_size), stride=(1, pool_kernel_size))
    def forward(self, x):
        x = self.pooling_layer(x)  
        # x = x.reshape(x.size(0), -1)  
        for layer in self.layers:
            x = torch.relu(layer(x))
        theta = self.basis_parameters(x)
        return self.basis_function(theta)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.ones(1)) 

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
    

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x 输入维度为 (seqlen, Batch, d_model)
        x = self.transformer(x)  # 输出: (seqlen, Batch, d_model)
        return x

class SpatialGCN(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.1, use_einsum=True):
        super(SpatialGCN, self).__init__()
        self.use_einsum = use_einsum

        # 两者都是处理第二维度，
        self.fc = nn.Conv1d(c_in, c_out, kernel_size=1, bias=True) # 留空间结构，适合处理三维输入，参数共享，效率更高。
       
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        """
        x: [batch_size, numnodes, len]
        adj: [numnodes, numnodes]
        """
       
        adj = adj + torch.eye(adj.size(0)).to(adj.device)  # 添加自环
        D = torch.diag(torch.pow(adj.sum(1), -0.5))
        adj_normalized = torch.matmul(torch.matmul(D, adj), D)  # 对称归一化
        # 使用 einsum 进行 A * X
        x = torch.einsum('nvl,vw->nwl', (x, adj_normalized))  # [batch_size, numnode, len]
        # 线性变换
        x = x.permute(0, 2, 1) # [batch_size, len, numnode]
        x = self.fc(x)  # # [batch_size, d_model, numnode]
        x = x.permute(0, 2, 1)# [batch_size, numnode, d_model]
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
class SelfAttentionModule(nn.Module):
    def __init__(self, d_model):
        super(SelfAttentionModule, self).__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        return output
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.attention_heads = nn.ModuleList([SelfAttentionModule(d_model) for _ in range(num_heads)])
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # 多头注意力层输出为各个自注意力模块的输出相加
        attention_outputs = [head(x) for head in self.attention_heads]
        multi_head_output = sum(attention_outputs)
        
        # 残差连接
        x = x + multi_head_output
        x = self.norm1(x)
        
        # 输出后续标准化和Softmax
        x = torch.softmax(x, dim=-1)
        x = self.norm2(x)
        
        return x    
# 时间嵌入模块
class TVOM(nn.Module):
    def __init__(self, period=96, d_model=128):
        super(TVOM, self).__init__()
        self.period = period
        self.d_model = d_model

        # 使用傅里叶级数生成周期编码
        position = np.arange(self.period).reshape(-1, 1)
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        time_embedding = np.zeros((self.period, self.d_model))
        time_embedding[:, 0::2] = np.sin(position * div_term)
        time_embedding[:, 1::2] = np.cos(position * div_term)
        self.time_embedding = nn.Parameter(torch.FloatTensor(time_embedding), requires_grad=False)

    def forward(self, x):
        # x 输入维度为 (Batch, seqlen, d_model)
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.d_model, "Embedding dimension mismatch"
        
        if seq_len > self.period:
            raise ValueError(f"Sequence length {seq_len} exceeds period {self.period}")
        
        # 复制时间编码以匹配输入序列长度
        time_encoding = self.time_embedding[:seq_len].unsqueeze(0).repeat(batch_size, 1, 1)
        return x + time_encoding.to(x.device)

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1)

    def forward(self, x, adj):
        # Applying graph convolution by filtering each node's neighborhood
        x = torch.einsum('bcn,nm->bcm', x, adj)  # Apply adjacency matrix
        return F.relu(self.conv(x))  # Pass through Conv2D

class GCGRNCell(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(GCGRNCell, self).__init__()
        self.gcn_r = GraphConvLayer(in_features + hidden_size, hidden_size)
        self.gcn_u = GraphConvLayer(in_features + hidden_size, hidden_size)
        self.gcn_C = GraphConvLayer(in_features + hidden_size, hidden_size)

    def forward(self, x, H_prev, adj):

        x_H = torch.cat([x, H_prev], dim=1)  # Concatenate input and previous hidden state
        r = torch.sigmoid(self.gcn_r(x_H, adj))  # Reset gate (batch_size, hidden_size, num_nodes)
        u = torch.sigmoid(self.gcn_u(x_H, adj))  # Update gate (batch_size, hidden_size, num_nodes)
        C = torch.tanh(self.gcn_C(torch.cat([x, r * H_prev], dim=1), adj))  # Candidate memory content
        H = u * H_prev + (1 - u) * C  # Final memory at current time step
        return H
    

class NodeEmbeddingModule(nn.Module):
    def __init__(self, in_channels, node_feature_size=12, k_depth=3, ra=32, channels = [48, 24, 12]):
        super(NodeEmbeddingModule, self).__init__()

        self.k_depth = k_depth

        
        # 卷积层
        self.func1 = nn.Sequential(
            nn.Conv2d(in_channels, node_feature_size, kernel_size=1),
            nn.Sigmoid(),
        )
        
        # 池化层
        self.func2 = nn.Sequential(
            nn.Conv2d(in_channels, node_feature_size, kernel_size=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=1)
        )

        self.seqc_conv = nn.ModuleList()
        self.time_conv = nn.ModuleList()
        node_feature_size = node_feature_size*2
        for k in range(k_depth):
            self.seqc_conv.append(nn.Conv2d(in_channels, (node_feature_size if k==0 else channels[k-1]), kernel_size=1))
            
            # 添加卷积来实现时间步之间的递归传递
            self.time_conv.append(nn.Conv2d(node_feature_size*3 if k==0 else channels[k-1]*3, channels[k], kernel_size=(1, 3), padding=(0, 1)))

    def forward(self, x, adj_matrix):
        """
        x: 输入特征，形状为 (Batchsize, channels, numnodes, len)
        adj_matrix: 邻接矩阵，形状为 (numnodes, numnodes)
        """
        # 初始特征，形状为 (Batchsize, node_feature_size, numnodes, len)
        Batchsize, node_feature_size, numnodes, len = x.shape
        # x.mean(dim=[2,3], keepdim=True).repeat(1,1,self.numnode ,self.seqlen)
        h1 = self.func1(x.mean(dim=2, keepdim=True).repeat(1,1,numnodes ,1))
        h2 = self.func2(x)
        h = torch.cat([h1, h2], dim=1)  # 卷积和池化结果拼接

        for k in range(self.k_depth):
            # 邻居聚合：通过邻接矩阵进行邻居特征计算
            h_neighbors = torch.einsum('bcnl,nm->bcml', h, adj_matrix)  # 邻居聚合，输出: (Batchsize, channels, numnodes, len)
            s = self.seqc_conv[k](x)
            h_concat = torch.cat((h, h_neighbors, s), dim=1)  # 拼接，输出: (Batchsize, channels * 3, numnodes, len)
            # 使用卷积对时间维度进行传递
            h_now = self.time_conv[k](h_concat)  # 时间卷积，输出: (Batchsize, aggregation_size, numnodes, len)
            # print(k,s.shape, h.shape, h_concat.shape, h_now.shape)
            h = h_now

        return h  # 最终输出: (Batchsize, aggregation_size, numnodes, len)
class DCConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2, filter_size=3, dilation_factors=[1, 2]):
        super(DCConvLayer, self).__init__()
        
        # 时间维度的扩张卷积
        
        # self.time_dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, filter_size), 
        #                                    dilation=(1, dilation_factors[0]), padding=(0, dilation_factors[0]))
        self.time_dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), dilation=(1, dilation_factors[0]))
        # # 节点维度的扩张卷积，使用双向卷积
        # self.node_dilated_conv_forward = nn.Conv2d(in_channels, out_channels, kernel_size=(filter_size, 1), 
        #                                            dilation=(dilation_factors[1], 1), padding=(dilation_factors[1], 0))
        self.node_dilated_conv_forward = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), dilation=(dilation_factors[1], 1))
        
        # self.node_dilated_conv_backward = nn.Conv2d(in_channels, out_channels, kernel_size=(filter_size, 1), 
        #                                             dilation=(dilation_factors[1], 1), padding=(dilation_factors[1], 0))
        self.combine = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 时间维度卷积
        time_conv = self.time_dilated_conv(x)  #  (Batchsize, out_channels, numnodes, len)

        # 节点维度卷积
        node_conv_forward = self.node_dilated_conv_forward(x)  # 前向卷积 (Batchsize, out_channels, numnodes, len)
        # reversed_x = torch.flip(x, dims=[-2]) 
        # node_conv_backward = self.node_dilated_conv_backward(reversed_x)  # 反向卷积 (Batchsize, out_channels, numnodes, len)
        # node_conv = node_conv_forward+node_conv_backward
        node_conv = node_conv_forward
        x = torch.cat((time_conv, node_conv), dim=1)  #  (Batchsize, 2 * out_channels, numnodes, len)
        x = self.combine(x) #  (Batchsize, out_channels, numnodes, len)
        x = self.bn(x) 
        x = self.relu(x)  
        x = self.dropout(x) 

        return x
class MDCConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2, dilation_factors=[1, 2], filter_size=3, residual_layers = 2):
        super(MDCConv, self).__init__()
        self.trans = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dilated_convs = nn.ModuleList([
            DCConvLayer(in_channels if i == 0 else out_channels, out_channels, dropout, filter_size, dilation_factors) 
            for i in range(residual_layers)
        ])
        

    def forward(self, x):
        residual =  self.trans(x) 
        for conv in self.dilated_convs:
            x = conv(x)  # (Batchsize, 32, numnodes, len)
       
        x = F.relu(x + residual)
        return x  # (Batchsize, 32, numnodes, len)




class staticNodeEmbeddingModule(nn.Module):
    def __init__(self, in_channels, node_feature_size=12, k_depth=3, ra=32, channels = [48, 24, 12]):
        super(staticNodeEmbeddingModule, self).__init__()

        self.k_depth = k_depth

        
        # 卷积层
        self.func1 = nn.Sequential(
            nn.Conv2d(in_channels, node_feature_size, kernel_size=1),
            nn.Sigmoid(),
        )
        
        # 池化层
        self.func2 = nn.Sequential(
            nn.Conv2d(in_channels, node_feature_size, kernel_size=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=1)
        )

        self.seqc_conv = nn.ModuleList()
        self.time_conv = nn.ModuleList()
        node_feature_size = node_feature_size*2
        for k in range(k_depth):
            self.seqc_conv.append(nn.Conv2d(in_channels, (node_feature_size if k==0 else channels[k-1]), kernel_size=1))
            
            # 添加卷积来实现时间步之间的递归传递
            self.time_conv.append(nn.Conv2d(node_feature_size*3 if k==0 else channels[k-1]*3, channels[k], kernel_size=(1, 3), padding=(0, 1)))

    def forward(self, x, adj_matrix):
        """
        x: 输入特征，形状为 (Batchsize, channels, numnodes, len)
        adj_matrix: 邻接矩阵，形状为 (numnodes, numnodes)
        """
        # 初始特征，形状为 (Batchsize, node_feature_size, numnodes, len)
        Batchsize, node_feature_size, numnodes, len = x.shape
        # x.mean(dim=[2,3], keepdim=True).repeat(1,1,self.numnode ,self.seqlen)
        h1 = self.func1(x.mean(dim=2, keepdim=True).repeat(1,1,numnodes ,1))
        h2 = self.func2(x)
        h = torch.cat([h1, h2], dim=1)  # 卷积和池化结果拼接

        # for k in range(self.k_depth):
        for k, (conv1, conv2) in enumerate(zip(self.seqc_conv, self.time_conv)):
            # 邻居聚合：通过邻接矩阵进行邻居特征计算
            h_neighbors = torch.einsum('bcnl,nm->bcml', h, adj_matrix)  # 邻居聚合，输出: (Batchsize, channels, numnodes, len)
            s = conv1(x)
            h_concat = torch.cat((h, h_neighbors, s), dim=1)  # 拼接，输出: (Batchsize, channels * 3, numnodes, len)
            # 使用卷积对时间维度进行传递
            h_now = conv2(h_concat)  # 时间卷积，输出: (Batchsize, aggregation_size, numnodes, len)
            # print(k,s.shape, h.shape, h_concat.shape, h_now.shape)
            h = h_now

        return h  # 最终输出: (Batchsize, aggregation_size, numnodes, len)
    
class staticDCConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2, filter_size=3, dilation_factors=[1, 2]):
        super(staticDCConvLayer, self).__init__()
        
        # 时间维度的扩张卷积
        
        # self.time_dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, filter_size), 
        #                                    dilation=(1, dilation_factors[0]), padding=(0, dilation_factors[0]))
        self.time_dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), dilation=(1, dilation_factors[0]))
        # # 节点维度的扩张卷积，使用双向卷积
        # self.node_dilated_conv_forward = nn.Conv2d(in_channels, out_channels, kernel_size=(filter_size, 1), 
        #                                            dilation=(dilation_factors[1], 1), padding=(dilation_factors[1], 0))
        self.node_dilated_conv_forward = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), dilation=(dilation_factors[1], 1))
        
        # self.node_dilated_conv_backward = nn.Conv2d(in_channels, out_channels, kernel_size=(filter_size, 1), 
        #                                             dilation=(dilation_factors[1], 1), padding=(dilation_factors[1], 0))
        self.combine = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 时间维度卷积
        time_conv = self.time_dilated_conv(x)  #  (Batchsize, out_channels, numnodes, len)

        # 节点维度卷积
        node_conv_forward = self.node_dilated_conv_forward(x)  # 前向卷积 (Batchsize, out_channels, numnodes, len)
        # reversed_x = torch.flip(x, dims=[-2]) 
        # node_conv_backward = self.node_dilated_conv_backward(reversed_x)  # 反向卷积 (Batchsize, out_channels, numnodes, len)
        # node_conv = node_conv_forward+node_conv_backward
        node_conv = node_conv_forward
        x = torch.cat((time_conv, node_conv), dim=1)  #  (Batchsize, 2 * out_channels, numnodes, len)
        x = self.combine(x) #  (Batchsize, out_channels, numnodes, len)
        x = self.bn(x) 
        x = self.relu(x)  
        x = self.dropout(x) 

        return x
class staticMDCConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2, dilation_factors=[1, 2], filter_size=3, residual_layers = 2):
        super(staticMDCConv, self).__init__()
        self.trans = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dilated_convs = nn.ModuleList([
            staticDCConvLayer(in_channels if i == 0 else out_channels, out_channels, dropout, filter_size, dilation_factors) 
            for i in range(residual_layers)
        ])
        

    def forward(self, x):
        residual =  self.trans(x) 
        for conv in self.dilated_convs:
            x = conv(x)  # (Batchsize, 32, numnodes, len)
       
        x = F.relu(x + residual)
        return x  # (Batchsize, 32, numnodes, len)


class staticFuseadj(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len, dataft='CPU', adjid=-1, sigma=0.0):
        super(staticFuseadj, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.support_len = support_len
        self.adjid=adjid
        self.sigma=sigma
        self.gating_network = EnhancedGatingNetwork(c_in, support_len, dataft)
        
        initial_alpha= 1.0
        initial_threshold=0.5
        # print(f"sig alpha: {initial_alpha} down: { initial_threshold}")
        self.alphas = nn.Parameter(torch.ones(support_len) * initial_alpha)
        self.thresholds = nn.Parameter(torch.ones(support_len) * initial_threshold)
        # self.upholds = nn.Parameter(torch.ones(support_len) * initial_uphold)
        
        # self.thresholds = nn.Parameter(torch.empty(num_matrices).uniform_(0.1, 0.5))
        # self.upholds = nn.Parameter(torch.empty(num_matrices).uniform_(0.7, 0.9))
        # 用于记录和输出权重的相关参数
        self.count = 0
        self.weight_records = []
        

    def output_weights(self, adj_weights):
        self.count += 1

        average_weights = adj_weights.mean(dim=0).detach().cpu().numpy()
        # average_weights = adj_weights.detach().cpu().numpy()
        self.weight_records.append(average_weights)
        
        # 达到指定次数后，计算平均权重并输出
        if self.count == 38 :  # 每38次输出一次
            average_weights = torch.mean(torch.tensor(self.weight_records), dim=0)
            print([f"{weight:.3f}" for weight in average_weights])
            self.weight_records = []  # 重置记录
            self.count = 0  # 重置计数
   
    def forward(self, x, support):
        """
        x: [batch_size, channels, numnodes, len]
        support: [support_len, numnodes, numnodes]
        """
        gating_input = torch.cat([
            x.mean(dim=[2, 3]), 
            # x.std(dim=[2, 3])
        ], dim=-1)
        # 通过门控网络生成权重
        gating_weights = self.gating_network(gating_input)  # [batch_size, support_len]
        # 对 support 中的每个邻接矩阵加权求和，得到每个样本的融合邻接矩阵
        
        adjusted_matrices = []
       
        for i, adj_matrix in enumerate(support):
            scaled_matrix = 1 / (1 + torch.exp(-self.alphas[i] * adj_matrix))
            scaled_matrix = torch.nan_to_num(scaled_matrix, nan=0.0)
            sparse_matrix = torch.where(scaled_matrix >= self.thresholds[i], scaled_matrix, torch.zeros_like(scaled_matrix))
            sparse_matrix = torch.nan_to_num(sparse_matrix, nan=0.0)
            if (self.adjid != -1 ) and (i == (self.adjid)) :
                A = sparse_matrix
                mean_A = torch.mean(A)
                # var_A = torch.var(A)
                # sigma_A = torch.sqrt(var_A)
                # sigma_A = torch.maximum(sigma_A, torch.tensor(1e-6)) 
                # print(mean_A.item(), sigma_A.item())

                noise = torch.normal(mean_A, self.sigma, size=A.shape).to(sparse_matrix.device)
                sparse_matrix = sparse_matrix + noise
            adjusted_matrices.append(sparse_matrix)
        adjusted_matrices = torch.stack(adjusted_matrices, dim=0)
        
        # A_fused = torch.einsum('s,sij->ij', gating_weights, adjusted_matrices) # [ numnodes, numnodes]
        A_fused = torch.einsum('bs,sij->bij', gating_weights, adjusted_matrices)  # [batch_size, numnodes, numnodes]
        A_fused = A_fused.mean(dim=0).squeeze()

        # self.output_weights(gating_weights)
        
        return  A_fused

class staticMoE_SAG(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len,vmindex_in_cities, hidden_size = 128):
        super(staticMoE_SAG, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dropout = dropout
        self.support_len = support_len
        self.num_cities = len(vmindex_in_cities)
        # 定义多个GCN专家，每个专家对应一个邻接矩阵
        # print(f"split into {self.num_cities} SAG of dim {hidden_size}")

        self.convs =  nn.Conv2d(c_in, hidden_size, kernel_size=1)
        
        self.experts = nn.ModuleList([
            GraphSAGELayer(hidden_size, hidden_size) for _ in range(self.num_cities)
        ])
        self.negative_slope = 0.01

        self.convt = nn.ConvTranspose2d(hidden_size, c_out, kernel_size=1)
        self.vmindex_in_cities = vmindex_in_cities

    def leaky_relu(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope)

    def inverse_leaky_relu(self, y):
        y_cleaned = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.where(y_cleaned > 0, y_cleaned, y_cleaned / self.negative_slope)
    
    def forward(self, input, A_fuse):
        """
        Args:
            X (torch.Tensor): 输入特征，形状为 (Batch, C, N, L)。
            support (torch.Tensor): 总的邻接矩阵，形状为 (N, N)。

        Returns:
            torch.Tensor: 输出特征，形状为 (Batch, out_feats, Numnodes, L)。
        """
        batch_size, C, N, L = input.size()

        output = torch.zeros(batch_size, self.c_out, N, L, device=input.device)
        # expert_outputs = []

        for idx, expert in enumerate(self.experts):
            city_indices = self.vmindex_in_cities[idx]  # List[int]
            city_data = input[:, :, city_indices, :]      # Batch x C x N_city x L
            A_city = A_fuse[city_indices][:, city_indices] # N_city x N_city
            degree = A_city.sum(dim=1)                   # N_city

            x = self.convs(city_data)
            x = self.leaky_relu(x)    
            x = expert(x, A_city, degree)  # Batch x out_feats x N_city x L
            x = self.inverse_leaky_relu(x)       
            x = self.convt(x)

            output[:, :, city_indices, :] = x  # Batch x out_feats x Numnodes x L

        return output