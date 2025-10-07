import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


import numpy as np
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter,LayerNorm,InstanceNorm2d

from util import RNC_loss, NT_Xent_loss, NT_Logistic_loss, Margin_Triplet_loss
from scipy.linalg import eigh
from utils import multi_gat
from utils import ST_BLOCK_0 #ASTGCN
from utils import ST_BLOCK_1 #DGCN_Mask/DGCN_Res
from utils import ST_BLOCK_2_r #DGCN_recent

from utils import ST_BLOCK_4 #Gated-STGCN
from utils import ST_BLOCK_5 #GRCN
from utils import ST_BLOCK_6 #OTSGGCN
from utils import multi_gcn#gwnet
from utils import GCNPool #H_GCN
from utils import Transmit
from utils import gate
from utils import GCNPool_dynamic
# from utils import GCNPool_h
from utils import T_cheby_conv_ds_1
from utils import dynamic_adj
from utils import SATT_h_gcn
# gloabl and whole 
from utils import Swish, Mish
# Informer
from utils import TriangularCausalMask, ProbMask                    # masking
from utils import EncoderI, EncoderLayerI, ConvLayerI               # encoder
from utils import DecoderI, DecoderLayerI                           # decoder
from utils import FullAttention, ProbAttention, AttentionLayer      # attention
from utils import DataEmbedding                                     # embedding
# Autoformer
from utils import DataEmbedding, DataEmbedding_wo_pos               # embedding
from utils import AutoCorrelation, AutoCorrelationLayer             # attention
from utils import EncoderA, DecoderA, EncoderLayerA, DecoderLayerA, my_Layernorm, series_decomp #De/Encoder
# N-BEATS
from utils import NBeatsBlock, GenericBasis
# TimesNet
from utils import TimesBlock
# DCRNN
from utils import DCGRUCell
# NHITS
from utils import IdentityBasis, NHITSBlock
# GCformer
from utils import TransformerEncoder, SpatialGCN, MultiHeadAttentionLayer, TVOM
from utils import  GCNLayer
from utils import MoE_SAG, Fuseadj, PMoE_SAG, staticMoE_SAG, staticFuseadj
# GCGRN
from utils import GCGRNCell
# GTCN
from utils import NodeEmbeddingModule, MDCConv, staticNodeEmbeddingModule, staticMDCConv
class ASTGCN_Recent(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(ASTGCN_Recent,self).__init__()
        self.block1=ST_BLOCK_0(device, in_dim,dilation_channels,num_nodes,length,K,Kt)
        self.block2=ST_BLOCK_0(device, dilation_channels,dilation_channels,num_nodes,length,K,Kt)
        self.final_conv=Conv2d(length,out_dim,kernel_size=(1, dilation_channels),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        
    def forward(self,input):
        x=self.bn(input)

        A=self.supports[0]
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)

        adj=A 
        x,_,_ = self.block1(x,adj)
        x,d_adj,t_adj = self.block2(x,adj)
        x = x.permute(0,3,2,1)                  
        x = self.final_conv(x)#b,l,n,1
        return x

    
class DGCN_recent(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3): 
        super(DGCN_recent,self).__init__()
        tem_size=length
        self.block1=ST_BLOCK_2_r(in_dim,dilation_channels,num_nodes,length,K,Kt)
        self.block2=ST_BLOCK_2_r(dilation_channels,dilation_channels,num_nodes,length,K,Kt)
        
        self.conv1=Conv2d(dilation_channels,1,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True)
        
        self.supports=supports
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        
    def forward(self,input):
        x=input
    
        A=self.h+self.supports[0]               # Modify the adjacency matrix A by adding a learnable parameter h
        d=1/(torch.sum(A,-1)+0.0001)            # The reciprocal of the sum of each row of A, with a small ε added for numerical stability.
        D=torch.diag_embed(d)                   # Create a diagonal matrix D where the diagonal elements are values from the vector d
        A=torch.matmul(D,A)                     # Multiply the adjacency matrix A with D to normalize it. This operation will make A a row normalized adjacency matrix.
        A1=F.dropout(A,0.5,self.training)       # Dropout with probability 0.5 on the normalized adjacency matrix A. It is a regularization technique in which a fraction of the values are randomly set to zero during training to prevent overfitting.
              
        x,_,_=self.block1(x,A1)
        x,d_adj,t_adj=self.block2(x,A1)
    
        x=self.conv1(x).permute(0,3,2,1).contiguous()#b,c,n,l 
        return x,d_adj,A


class LSTM(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.2, seqlen=12, in_dim=1,prelen=12,d_model=32,supports=None, vmindex_in_cities=None):
        super(LSTM,self).__init__()
        self.lstm=nn.LSTM(in_dim, d_model,batch_first=True)#b*n,l,c
        self.c_out=d_model

        self.device = device
        
        self.supports = supports
        self.supports_len = 0
        self.vmindex_in_cities = vmindex_in_cities
        if supports is not None:
            self.supports_len += len(supports)
        if supports is None:
            self.supports = []
        self.gconv = Fuseadj(in_dim, d_model, dropout, self.supports_len)
        self.gcn = GCNLayer(in_dim, d_model, dropout)
        self.scale = nn.Parameter(torch.tensor(0.1))
        if vmindex_in_cities is not None: 
            self.gcity = MoE_SAG(d_model, d_model, dropout, self.supports_len, vmindex_in_cities)
        
        self.conv1=Conv2d(d_model, prelen, kernel_size=(1, seqlen), padding=(0, 0),stride=(1, 1), bias=True)
        self.convc=Conv2d(d_model, prelen, kernel_size=(1, seqlen), padding=(0, 0),stride=(1, 1), bias=True)
        
    def forward(self,input):
        x=input
        if self.supports_len > 1 :
            A1 = self.gconv(x, self.supports)# [batch_size,d_model, numnode, seqlen]
        else:
            A1 =  torch.stack(self.supports).squeeze()
        d = torch.sum(A1, dim=-1) + 1e-5  
        D = torch.diag_embed(1.0 / d)
        A1 = torch.matmul(D, A1)

        city_features = None
        if self.vmindex_in_cities is not None: 
            city_features = self.gcity(x, A1)
        
        shape = x.shape
        h = Variable(torch.zeros((1,shape[0]*shape[2],self.c_out))).to(self.device)#hide LSTM
        c = Variable(torch.zeros((1,shape[0]*shape[2],self.c_out))).to(self.device)#cell
        hidden=(h,c)
        
        x=x.permute(0,2,3,1).contiguous().view(shape[0]*shape[2],shape[3],shape[1])  
        x,hidden=self.lstm(x,hidden)                                                 
        x=x.view(shape[0],shape[2],shape[3],self.c_out).permute(0,3,1,2).contiguous()
        x=self.conv1(x)#b,c,n,l 
        if city_features is not None:
            features = self.convc(city_features)
        else:
            features = 0
        output = x + self.scale * features
              

        return output


class GRU(nn.Module): # Gated Recurrent Unit (GRU) model
    def __init__(self,device, num_nodes, dropout=0.2, supports=None,seqlen=12, in_dim=1,prelen=12,d_model=32,vmindex_in_cities=None):
        super(GRU,self).__init__()
        self.gru=nn.GRU(in_dim,d_model,batch_first=True)#b*n,l,c
        self.c_out=d_model
        tem_size=seqlen
        self.tem_size=tem_size
        
        self.device = device

        self.supports = supports
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
        if supports is None:
            self.supports = []
        self.gconv = Fuseadj(in_dim, d_model, dropout, self.supports_len)
        self.gcn = GCNLayer(in_dim, d_model, dropout)
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.vmindex_in_cities = vmindex_in_cities
        if vmindex_in_cities is not None: 
            self.gcity = MoE_SAG(d_model, d_model, dropout, self.supports_len, vmindex_in_cities)
        

        self.conv1=Conv2d(d_model, prelen, kernel_size=(1, seqlen), padding=(0, 0),stride=(1, 1), bias=True)
        self.convc=Conv2d(d_model, prelen, kernel_size=(1, seqlen), padding=(0, 0),stride=(1, 1), bias=True)
       
        
    def forward(self,input):
        x=input
        if self.supports_len > 1 :
            A1 = self.gconv(x, self.supports)# [batch_size,d_model, numnode, seqlen]
            d = torch.sum(A1, dim=-1) + 1e-5  # 避免除零
            D = torch.diag_embed(1.0 / d)
            A1 = torch.matmul(D, A1)
        city_features = None
        if self.vmindex_in_cities is not None: 
            city_features = self.gcity(x, A1)

        shape = x.shape
        h =Variable(torch.zeros((1,shape[0]*shape[2],self.c_out))).to(self.device)   
        hidden=h
        
        x=x.permute(0,2,3,1).contiguous().view(shape[0]*shape[2],shape[3],shape[1])  
        x,hidden=self.gru(x,hidden)                                                  
        x=x.view(shape[0],shape[2],shape[3],self.c_out).permute(0,3,1,2).contiguous()
        x=self.conv1(x)#b,c,n,l

        if city_features is not None:
            features = self.convc(city_features)
            spatial_features = self.conv2(features) # batch_size, prelen, num_nodes,1
            x = x + self.scale * spatial_features

        return x


class Gated_STGCN(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(Gated_STGCN,self).__init__()
        tem_size=length
        self.block1=ST_BLOCK_4(device,in_dim,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_4(device,dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block3=ST_BLOCK_4(device,dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        
        
        self.conv1=Conv2d(dilation_channels,out_dim,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
    def forward(self,input):
        x=self.bn(input)

        A=self.supports[0]
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)

        adj=A  
        x=self.block1(x,adj)
        x=self.block2(x,adj)
        x=self.block3(x,adj)
        x=self.conv1(x)#b,12,n,1
        return x 


class GRCN(nn.Module):      
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(GRCN,self).__init__()
       
        self.block1=ST_BLOCK_5(device, in_dim,dilation_channels,num_nodes,length,K,Kt)
        self.block2=ST_BLOCK_5(device,dilation_channels,dilation_channels,num_nodes,length,K,Kt)
        
        self.tem_size=length
        
        self.conv1=Conv2d(dilation_channels,out_dim,kernel_size=(1,length),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)

    def forward(self,input):
        x=self.bn(input)
       

        A=self.supports[0]
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)

        adj= A #F.dropout(A,0.5)
        x=self.block1(x,adj)
        x=self.block2(x,adj)
        x=self.conv1(x)
        return x


class OGCRNN(nn.Module):   #Operator Gated Convolutional Recurrent Neural Network 算子门控卷积递归神经网络
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(OGCRNN,self).__init__()
       
        self.block1=ST_BLOCK_5(device, in_dim,dilation_channels,num_nodes,length,K,Kt)
        
        self.tem_size=length
        
        self.conv1=Conv2d(dilation_channels,out_dim,kernel_size=(1,length),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)

    def forward(self,input):
        x=self.bn(input)
        mask=(self.supports[0]!=0).float() 
        A=self.h*mask     
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        x=self.block1(x,A)
        
        x=self.conv1(x)
        return x


class OTSGGCN(nn.Module): #Operator Temporal Shift Graph Convolutional Network 算子时移图卷积网络
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3, vmindex_in_cities= None):
        super(OTSGGCN,self).__init__()
        tem_size=length
        self.num_nodes=num_nodes
        self.block1=ST_BLOCK_6(device,in_dim,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_6(device,dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block3=ST_BLOCK_6(device,dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        
        self.conv1=Conv2d(dilation_channels, out_dim, kernel_size=(1, tem_size), padding=(0, 0),
                          stride=(1, 1), bias=True)
        
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.h=Parameter(torch.ones(num_nodes,num_nodes), requires_grad=True)
        #nn.init.uniform_(self.h, a=0, b=0.0001)
        self.device = device

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
        
        if supports is None:
            self.supports = []

        self.gconv = Fuseadj(in_dim, dilation_channels,dropout, self.supports_len)
        if vmindex_in_cities is not None: 
            self.gcity = MoE_SAG(dilation_channels, dilation_channels, dropout, self.supports_len, vmindex_in_cities)

        self.scale = nn.Parameter(torch.tensor(0.1))
        self.gate = nn.Sequential(
            nn.BatchNorm2d(dilation_channels),
            nn.Sigmoid(),
            # nn.Tanh()
        )

    def forward(self,input):
        x=input#self.bn(input)
        if self.supports_len > 1 :
            spatial_features, A_fused= self.gconv(x, self.supports)
            spatial_features = self.gate(spatial_features)

            d = torch.sum(A_fused, dim=-1) + 1e-5  # 避免除零
            D = torch.diag_embed(1.0 / d)
            A1 = torch.matmul(D, A_fused)
            spatial_features = self.gcity(spatial_features, A1)
        else :
            # mask=(self.supports[0]!=0).float() 
            # A=self.h*mask                      
            
            # d=1/(torch.sum(A,-1)+0.0001)
            # D=torch.diag_embed(d)
            # A=torch.matmul(D,A)
            # A1=torch.eye(self.num_nodes).cuda(str(self.device))-A  
            A1 = torch.stack(self.supports).squeeze()
            d = torch.sum(A1, dim=-1) + 1e-5  
            D = torch.diag_embed(1.0 / d)
            A1 = torch.matmul(D, A1)
  
        x=self.block1(x,A1)# b dilation_channels num_nodes len
        x=self.block2(x,A1)
        x=self.block3(x,A1)

        if self.supports_len > 1 :
            x = x + self.scale*spatial_features
        x=self.conv1(x)#b,12,n,1
        return x 
    


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, seqlen=12,
                 in_dim=1, prelen=12, vmindex_in_cities=None, residual_channels=32,dilation_channels=32,
                 skip_channels=128, end_channels=256, kernel_size=2, blocks=4, layers=2, gat=False, addaptadj=True):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gat = gat
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList() 
        self.gate_convs = nn.ModuleList()  
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.conv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
        
        if supports is None:
            self.supports = []
            
        self.gconv = Fuseadj(in_dim, dilation_channels, dropout, self.supports_len)
        self.gcn = GCNLayer(in_dim, dilation_channels, dropout)
        self.scale = nn.Parameter(torch.tensor(0.1))

        if vmindex_in_cities is not None: 
            self.gcity = MoE_SAG(dilation_channels,dilation_channels, dropout, self.supports_len, vmindex_in_cities)
        

        self.conv1=Conv2d(dilation_channels, prelen, kernel_size=(1, seqlen), padding=(0, 0),stride=(1, 1), bias=True)
        self.gate = nn.Sequential(
            nn.BatchNorm2d(dilation_channels),
            nn.Sigmoid(),
            # nn.Tanh()
        )

        self.remain_len = seqlen + 1
        kernel_size = (seqlen // 12) +1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions    
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
               
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))#批处理规范化层
                if self.gat:
                    self.remain_len = self.remain_len - new_dilation * (kernel_size - 1)
                    self.conv.append(multi_gat(dilation_channels, self.remain_len,
                                                dropout, support_len=self.supports_len))
                new_dilation *= 2 
                receptive_field += additional_scope 
                additional_scope *= 2
                if not self.gat:
                    # b c n len,len 逐渐变成 1
                    self.conv.append(multi_gcn(dilation_channels, residual_channels,
                                                dropout, self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels, kernel_size=(1, 1), bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=prelen, kernel_size=(1, 1), bias=True)

        self.receptive_field = receptive_field
        
    def forward(self, input):# batch_size Channel Numnode Len

        if self.supports_len > 1 :
            spatial_features, A_fused = self.gconv(input, self.supports)# [batch_size,d_model, numnode, seqlen]
            spatial_features = self.gate(spatial_features)
            d = torch.sum(A_fused, dim=-1) + 1e-5  
            D = torch.diag_embed(1.0 / d)
            A1 = torch.matmul(D, A_fused)

            spatial_features = self.gcity(spatial_features, A1)

        in_len = input.size(3)                                                  
        if in_len < self.receptive_field:                                       
            x = nn.functional.pad(input, (self.receptive_field-in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)                                                  
        skip = 0

            
        new_supports = self.supports

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]
            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x                                                
            # dilated convolution
            filter = self.filter_convs[i](residual)                    
            filter = torch.tanh(filter)                                 
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)                                  
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)                                   
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip    
            x = self.conv[i](x, new_supports)   
            x = x + residual[:, :, :, -x.size(3):]                      
            x = self.bn[i](x)                                          

        x = F.relu(skip)                                                
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)          
        if self.supports_len > 1 :
            spatial_features = self.conv1(spatial_features) 
            x = x + self.scale * spatial_features                         
        return x


class H_GCN_wh(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(H_GCN_wh, self).__init__()
        self.dropout = dropout
        self.num_nodes=num_nodes
        
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        
        self.supports = supports
        
        self.supports_len = 0
        
        if supports is not None:
            self.supports_len += len(supports)
        
        if supports is None:
            self.supports = []
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        
        self.supports_len += 1

        Kt1=2
        self.block1=GCNPool(dilation_channels,dilation_channels,num_nodes,length-6,3,dropout,num_nodes,
                            self.supports_len)
        self.block2=GCNPool(dilation_channels,dilation_channels,num_nodes,length-9,2,dropout,num_nodes,
                            self.supports_len)
        
        self.skip_conv1=Conv2d(dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.skip_conv2=Conv2d(dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,3),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
        self.bn=BatchNorm2d(in_dim,affine=False)
        

    def forward(self, input):   
        x=self.bn(input)            
        shape=x.shape
        
        if self.supports is not None:
            A1 = torch.stack(self.supports).squeeze()
            d = torch.sum(A1, dim=-1) + 1e-5  
            D = torch.diag_embed(1.0 / d)
            A1 = torch.matmul(D, A1)          
            new_supports = self.supports + [A1]                      
            
        skip=0
        x = self.start_conv(x)
        
        #1                                      
        x=self.block1(x,new_supports)           
        
        s1=self.skip_conv1(x)
        skip=s1+skip
        
        #2
        x=self.block2(x,new_supports)           
                                                
        s2=self.skip_conv2(x)                   
        skip = skip[:, :, :,  -s2.size(3):]
        skip = s2 + skip
                
        #output
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


class H_GCN_wdf(nn.Module):
    def __init__(self,device, num_nodes, cluster_nodes,dropout=0.3, supports=None,supports_cluster=None,transmit=None,length=12, 
                 in_dim=1,in_dim_cluster=3,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(H_GCN_wdf, self).__init__()
        self.dropout = dropout
        self.num_nodes=num_nodes
        self.transmit=transmit
        self.cluster_nodes=cluster_nodes
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.start_conv_cluster = nn.Conv2d(in_channels=in_dim_cluster,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports
        self.supports_cluster = supports_cluster
        
        self.supports_len = 0
        self.supports_len_cluster = 0
        if supports is not None:
            self.supports_len += len(supports)
            self.supports_len_cluster+=len(supports_cluster)

        
        if supports is None:
            self.supports = []
            self.supports_cluster = []
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.h_cluster=Parameter(torch.zeros(cluster_nodes,cluster_nodes), requires_grad=True)
        nn.init.uniform_(self.h_cluster, a=0, b=0.0001)
        self.supports_len +=1
        self.supports_len_cluster +=1
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)  
        self.nodevec1_c = nn.Parameter(torch.randn(cluster_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2_c = nn.Parameter(torch.randn(10,cluster_nodes).to(device), requires_grad=True).to(device)  
        
       
        self.block1=GCNPool(2*dilation_channels,dilation_channels,num_nodes,length-6,3,dropout,num_nodes,
                            self.supports_len)
        self.block2=GCNPool(2*dilation_channels,dilation_channels,num_nodes,length-9,2,dropout,num_nodes,
                            self.supports_len)
        
        self.block_cluster1=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-6,3,dropout,cluster_nodes,
                            self.supports_len)
        self.block_cluster2=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-9,2,dropout,cluster_nodes,
                            self.supports_len)
       
        self.skip_conv1=Conv2d(2*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.skip_conv2=Conv2d(2*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,3),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
        
        
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.conv_cluster1=Conv2d(dilation_channels,out_dim,kernel_size=(1,3),
                          stride=(1,1), bias=True)
        self.bn_cluster=BatchNorm2d(in_dim_cluster,affine=False)
        self.gate1=gate(2*dilation_channels)
        self.gate2=gate(2*dilation_channels)
        self.gate3=gate(2*dilation_channels)
        
        
       

    def forward(self, input, input_cluster):
        x=self.bn(input)
        shape=x.shape
        input_c=input_cluster
        x_cluster=self.bn_cluster(input_c)
        if self.supports is not None:
            #nodes
            A=F.relu(torch.mm(self.nodevec1, self.nodevec2))
            d=1/(torch.sum(A,-1))
            D=torch.diag_embed(d)
            A=torch.matmul(D,A)
            
            new_supports = self.supports + [A]
            # 'region
            A_cluster=F.relu(torch.mm(self.nodevec1_c, self.nodevec2_c))
            d_c=1/(torch.sum(A_cluster,-1))
            D_c=torch.diag_embed(d_c)
            A_cluster=torch.matmul(D_c,A_cluster)
            
            new_supports_cluster = self.supports_cluster + [A_cluster]
        
        #network
        transmit=self.transmit              
        x = self.start_conv(x)
        x_cluster = self.start_conv_cluster(x_cluster)
        
        x_1=(torch.einsum('mn,bcnl->bcml',transmit,x_cluster))   
        
        x=self.gate1(x,x_1)
        
       
        skip=0
        skip_c=0
        #1
        x_cluster=self.block_cluster1(x_cluster,new_supports_cluster) 
        x=self.block1(x,new_supports)   
        
        x_2=(torch.einsum('mn,bcnl->bcml',transmit,x_cluster)) 
        
        x=self.gate2(x,x_2) 
        
        
        s1=self.skip_conv1(x)
        skip=s1+skip 
        
       
        #2       
        x_cluster=self.block_cluster2(x_cluster,new_supports_cluster)
        x=self.block2(x,new_supports) 
        
        x_3=(torch.einsum('mn,bcnl->bcml',transmit,x_cluster)) 
        
        x=self.gate3(x,x_3)
           
        
        s2=self.skip_conv2(x)      
        skip = skip[:, :, :,  -s2.size(3):]
        skip = s2 + skip        
        
       
        
        #output
        x = F.relu(skip)      
        x = F.relu(self.end_conv_1(x))            
        x = self.end_conv_2(x)              
        return x 
    
    
class H_GCN(nn.Module):
    def __init__(self,device, num_nodes, cluster_nodes,dropout=0.3, supports=None,supports_cluster=None,transmit=None,length=12, 
                 in_dim=1,in_dim_cluster=3,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(H_GCN, self).__init__()
        self.dropout = dropout
        self.num_nodes=num_nodes
        self.transmit=transmit
        self.cluster_nodes=cluster_nodes
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.start_conv_cluster = nn.Conv2d(in_channels=in_dim_cluster,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports
        self.supports_cluster = supports_cluster
        
        self.supports_len = 0
        self.supports_len_cluster = 0
        if supports is not None:
            self.supports_len += len(supports)
            self.supports_len_cluster+=len(supports_cluster)

        
        if supports is None:
            self.supports = []
            self.supports_cluster = []
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.h_cluster=Parameter(torch.zeros(cluster_nodes,cluster_nodes), requires_grad=True)
        nn.init.uniform_(self.h_cluster, a=0, b=0.0001)
        self.supports_len +=1
        self.supports_len_cluster +=1
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)  
        self.nodevec1_c = nn.Parameter(torch.randn(cluster_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2_c = nn.Parameter(torch.randn(10,cluster_nodes).to(device), requires_grad=True).to(device)  
        
        
        self.block1=GCNPool(2*dilation_channels,dilation_channels,num_nodes,length-6,3,dropout,num_nodes,
                            self.supports_len)
        self.block2=GCNPool(2*dilation_channels,dilation_channels,num_nodes,length-9,2,dropout,num_nodes,
                            self.supports_len)
        
        self.block_cluster1=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-6,3,dropout,cluster_nodes,
                            self.supports_len)
        self.block_cluster2=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-9,2,dropout,cluster_nodes,
                            self.supports_len)
        
        self.skip_conv1=Conv2d(2*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.skip_conv2=Conv2d(2*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,3),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.conv_cluster1=Conv2d(dilation_channels,out_dim,kernel_size=(1,3),
                          stride=(1,1), bias=True)
        self.bn_cluster=BatchNorm2d(in_dim_cluster,affine=False)
        self.gate1=gate(2*dilation_channels)
        self.gate2=gate(2*dilation_channels)
        self.gate3=gate(2*dilation_channels)
        
        self.transmit1=Transmit(dilation_channels,length,transmit,num_nodes,cluster_nodes)
        self.transmit2=Transmit(dilation_channels,length-6,transmit,num_nodes,cluster_nodes)
        self.transmit3=Transmit(dilation_channels,length-9,transmit,num_nodes,cluster_nodes)
       

    def forward(self, input, input_cluster):
        x=self.bn(input)
        shape=x.shape
        input_c=input_cluster
        x_cluster=self.bn_cluster(input_c)
        if self.supports is not None:
            # nodes
            A=F.relu(torch.mm(self.nodevec1, self.nodevec2))
            d=1/(torch.sum(A,-1))
            D=torch.diag_embed(d)
            A=torch.matmul(D,A)
            
            new_supports = self.supports + [A]
            # 'region
            A_cluster=F.relu(torch.mm(self.nodevec1_c, self.nodevec2_c))
            d_c=1/(torch.sum(A_cluster,-1))
            D_c=torch.diag_embed(d_c)
            A_cluster=torch.matmul(D_c,A_cluster)
            
            new_supports_cluster = self.supports_cluster + [A_cluster]
        
        #network
        transmit=self.transmit              
        x = self.start_conv(x)
        x_cluster = self.start_conv_cluster(x_cluster)
        transmit1 = self.transmit1(x,x_cluster)
        x_1=(torch.einsum('bmn,bcnl->bcml',transmit1,x_cluster))        
        x=self.gate1(x,x_1)
        skip=0
        skip_c=0
        #1
        x_cluster=self.block_cluster1(x_cluster,new_supports_cluster) 
        x=self.block1(x,new_supports)   
        transmit2 = self.transmit2(x,x_cluster)
        x_2=(torch.einsum('bmn,bcnl->bcml',transmit2,x_cluster))
        x=self.gate2(x,x_2)
        s1=self.skip_conv1(x)
        skip=s1+skip
        #2       
        x_cluster=self.block_cluster2(x_cluster,new_supports_cluster)
        x=self.block2(x,new_supports) 
        transmit3 = self.transmit3(x,x_cluster)                         
        x_3=(torch.einsum('bmn,bcnl->bcml',transmit3,x_cluster))
        x=self.gate3(x,x_3)

        s2=self.skip_conv2(x)      
        skip = skip[:, :, :,  -s2.size(3):] 
        skip = s2 + skip
        
        #output
        x = F.relu(skip)      
        x = F.relu(self.end_conv_1(x))            
        x = self.end_conv_2(x)              
        # return x, transmit3, A
        return x


# Informer d_model=dimension of multi-head attention’s output
class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, out_len,
                factor=3, d_model=128, n_heads=8, e_layers=2, d_layers=1, d_ff=256, # 512 1024
                dropout=0.05, attn='prob', embed='fixed', freq='t', activation='gelu', #timeF doesn't fit data
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = EncoderI(
            [
                EncoderLayerI(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayerI(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = DecoderI(
            [
                DecoderLayerI(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
       
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]

# Autoformer

class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, enc_in, dec_in, c_out, out_len,label_len,
                factor=3, d_model=128, n_heads=8, e_layers=2, d_layers=1, d_ff=512, #512 1024
                dropout=0.05, embed='fixed', freq='t', activation='gelu', 
                output_attention = False, 
                moving_avg=25,#Autoformer
                device=torch.device('cuda:0')):
        super(Autoformer, self).__init__()
        self.label_len = label_len
        self.pred_len = out_len
        self.output_attention = output_attention

        # Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq, dropout)

        # Encoder
        self.encoder = EncoderA(
            [
                EncoderLayerA(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = DecoderA(
            [
                DecoderLayerA(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc) 
        trend_init = torch.cat([trend_init[:, :-self.label_len, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, :-self.label_len, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec 
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

class N_BEATS(nn.Module):
    def __init__(self, input_size, output_size, stacks = 10, layers = 2, layer_size = 256,
                device=torch.device('cuda:0')):
        super(N_BEATS, self).__init__()
        self.blocks = nn.ModuleList([NBeatsBlock(input_size=input_size,
                                                theta_size=input_size + output_size,
                                                basis_function=GenericBasis(backcast_size=input_size,
                                                                            forecast_size=output_size),
                                                layers=layers,
                                                layer_size=layer_size)
                                    for _ in range(stacks)])
    def forward(self, input):
        x = input 
        residuals = x.flip(dims=(3,))
        # input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, :, :, -1:]# [64,1,194,1]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            #residuals = (residuals - backcast) * input_mask
            residuals = residuals - backcast
            forecast = forecast + block_forecast
        return forecast.transpose(1, 3)
class  TimesNet(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, pred_len, label_len,
                factor=3, d_model=32, n_heads=8, e_layers=5, d_layers=1, d_ff=128,  # 32 5 128
                dropout=0.2, embed='fixed', freq='15min', top_k = 3, num_kernels = 12): # 3 12
        super(TimesNet, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        print(f"dff:{d_ff}, topk:{top_k}, numkernels:{num_kernels}")
        self.model = nn.ModuleList([TimesBlock(seq_len, pred_len, d_model, d_ff, top_k, num_kernels)
                                    for _ in range(e_layers)])
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.predict_linear = nn.Linear(
            self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

# DCRNN
class EncoderModel(nn.ModuleList):
    def __init__(self,device, num_nodes, dropout=0.3, adj_mx=None, in_dim=1,seq_len=1):
        super(EncoderModel,self).__init__()
        self.max_diffusion_step = 2
        self.cl_decay_steps =  2000 #1000
        self.filter_type = 'dual_random_walk' #'laplacian'
        self.num_nodes = num_nodes
        self.num_rnn_layers = 1               # 1
        self.rnn_units = 128
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.device = device

        self.input_dim = in_dim
        self.seq_len = seq_len # for the encoder
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes, self.device,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module):
    def __init__(self,device, num_nodes, dropout = 0.3, adj_mx = None, out_dim = 1,horizon = 12):
        super(DecoderModel,self).__init__()
        self.max_diffusion_step = 2
        self.cl_decay_steps =  2000 #1000
        self.filter_type = 'dual_random_walk' #'laplacian'
        self.num_nodes = num_nodes
        self.num_rnn_layers = 1               # 1
        self.rnn_units = 128
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.device = device

        self.output_dim = out_dim
        self.horizon = horizon
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes, self.device,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)
class DCRNN(nn.Module):
    def __init__(self,device, num_nodes, dropout = 0.3, supports=None, in_dim = 1, seq_len = 12, horizon = 12):
        super(DCRNN,self).__init__()
        adj_mx=supports[0].cpu().numpy()
        self.num_nodes = num_nodes
        self.encoder_model = EncoderModel( device, num_nodes, dropout, adj_mx, in_dim, seq_len)
        self.decoder_model = DecoderModel( device, num_nodes, dropout, adj_mx, in_dim, horizon)
        self.cl_decay_steps = 2000 #1000
        self.use_curriculum_learning = True 
        self.device = device
    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning and (batches_seen != None):
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        inputs = inputs.permute(2, 0, 1) # [batch_size,num_nodes, seq_len] to [seq_len, batch_size, num_nodes]
        labels = labels.permute(2, 0, 1)
        encoder_hidden_state = self.encoder(inputs)
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        outputs = outputs.permute(1,2,0)

        return outputs

# NHITS
class NHITS(nn.Module):
    def __init__(self, input_size, output_size, num_blocks=3, layers=3, layer_size=512,  
                 pool_kernel_sizes=[2, 2, 1],n_freq_downsample=[4, 2, 1],
                 device=torch.device('cuda:0')):
        super(NHITS, self).__init__()
        self.output_size = output_size // n_freq_downsample[-1]
        self.blocks = nn.ModuleList([NHITSBlock(input_size=input_size // pool_kernel_sizes[i],
                                                theta_size=input_size + output_size // n_freq_downsample[i],
                                                basis_function=IdentityBasis(backcast_size=input_size,
                                                                             forecast_size=output_size // n_freq_downsample[i]),
                                                layers=layers,
                                                layer_size=layer_size,
                                                pool_kernel_size=pool_kernel_sizes[i])
                                     for i in range(num_blocks)])

    def forward(self, x):
        
        residuals = x.flip(dims=(-1,))
        forecast = x[:, :, :, -1:].repeat(1, 1, 1, self.output_size)
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast# b 1 n len
            if block_forecast.size(-1) != self.output_size:
                batch_size, channels, num_nodes, len = block_forecast.shape
                tensor_reshaped = block_forecast.view(-1, num_nodes, len)
                tensor_interpolated = F.interpolate(tensor_reshaped, size=self.output_size, mode='linear', align_corners=False)
                block_forecast = tensor_interpolated.view(batch_size, channels, num_nodes, self.output_size)

            forecast = forecast + block_forecast
        return forecast

class DeepAR(nn.Module):
    def __init__(self, seq_len, pred_len, num_nodes, batch_size=32, lstm_hidden_dim=32, lstm_layers=3,lstm_dropout=0.1,
                 device=torch.device('cuda:0')):
        super(DeepAR, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.lstm_layers = lstm_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.device = device
        self.lstm = nn.LSTM(input_size=num_nodes ,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers,
                            bias=True,
                            batch_first=True,  # Change to True to handle input as (batch, seq, feature)
                            dropout=lstm_dropout)

        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.0)

        self.relu = nn.ReLU()
        self.distribution_mu = nn.Linear(lstm_hidden_dim * lstm_layers, num_nodes)
        self.distribution_presigma = nn.Linear(lstm_hidden_dim * lstm_layers, num_nodes)
        self.distribution_sigma = nn.Softplus()

    def forward(self, input):
        # input [b, 1, n, len]
        x = input.squeeze(dim=1)

        hidden, cell = self.init_hidden(), self.init_cell()

        mu_list = []
        sigma_list = []
        x = x.transpose(1,2)
        for t in range(self.seq_len):
            x_t = x[:, t, :].unsqueeze(1)  # Get the t-th time step

            output, (hidden, cell) = self.lstm(x_t, (hidden, cell))# hidden=[layer,batch,hdim]

            hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
            mu_t = self.distribution_mu(hidden_permute)
            pre_sigma_t = self.distribution_presigma(hidden_permute)
            sigma_t = self.distribution_sigma(pre_sigma_t)

            mu_list.append(mu_t)
            sigma_list.append(sigma_t)

        last_mu = mu_list[-1]
        last_sigma = sigma_list[-1]

        predictions = []
        flag = 1
        if flag == 0:
            print("666")
            for t in range(self.pred_len):
                # Generate prediction based on mu and sigma
                last_mu = last_mu.clamp(min=1e-9)  
                last_sigma = last_sigma.clamp(min=1e-9)  
                eps = torch.randn_like(last_sigma)  # Sampling from a standard normal distribution
                next_pred = last_mu + last_sigma * eps  
                next_pred = next_pred.unsqueeze(1)  

                output, (hidden, cell) = self.lstm(next_pred, (hidden, cell))

                hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
                last_mu = self.distribution_mu(hidden_permute)
                last_sigma = self.distribution_presigma(hidden_permute)
                predictions.append(next_pred.squeeze(1))
        else :
            for t in range(self.pred_len):
                # Generate prediction based on mu and sigma
                last_mu = last_mu.clamp(min=1e-9) 
                last_sigma = last_sigma.clamp(min=1e-9)  

                next_pred = last_mu.unsqueeze(1)  

                output, (hidden, cell) = self.lstm(next_pred, (hidden, cell))

                hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
                last_mu = self.distribution_mu(hidden_permute)
                last_sigma = self.distribution_presigma(hidden_permute)
                predictions.append(next_pred.squeeze(1))
        predictions = torch.stack(predictions, dim=1)
        return predictions.transpose(1,2).unsqueeze(dim=1)

    def init_hidden(self):
        return torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_dim, device=self.device)

    def init_cell(self):
        return torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_dim, device=self.device)


class TemporalCNN(nn.Module):
    def __init__(self, in_dim=1, d_model=32, numnode=2060):
        super(TemporalCNN, self).__init__()
        
        self.numnode = numnode
        
        self.pad_numnodes = (64 - (numnode % 64)) % 64  
        padded_numnode = self.numnode + self.pad_numnodes
        
        
        channel_list = [1, 8, 16, 32]
        stride_list = [8, 8, 33]  
        
        layers = [
            nn.ConstantPad2d((0, 0, 0, self.pad_numnodes), value=0) 
        ]
        
     
        for in_c, out_c, stride in zip(channel_list[:-1], channel_list[1:], stride_list):
            layers += [
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=(stride, 3),  
                    stride=(stride, 1),        
                    padding=(0, 1)             
                ),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            ]
        
        
        self.temporal_cnn = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.temporal_cnn(x)
    
class GCformer(nn.Module):
    def __init__(self, device, in_dim, seqlen,  prelen, supports, numnode,dilation_channels , dropout,vmindex_in_cities=None,
                 d_model=32, nhead=8, num_layers=2, num_heads=3,period=96): 
        super(GCformer, self).__init__()
        self.device = device
        self.numnode = numnode
        self.pad_numnodes = (64 - (numnode % 64)) % 64
        
        
        self.temporal_cnn = TemporalCNN(in_dim,d_model,numnode)
        
        self.tvom = TVOM(period, d_model)
        
        self.transformer = TransformerEncoder(d_model, nhead, num_layers)
        self.multi_head_attention = MultiHeadAttentionLayer(d_model, num_heads)  
        
        self.fc = nn.Linear(d_model, prelen)

        self.supports = supports
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
        self.vmindex_in_cities = vmindex_in_cities
        if supports is None:
            self.supports = []
        self.gconv = Fuseadj(in_dim, d_model, dropout, self.supports_len)
        self.gcn = GCNLayer(in_dim, d_model, dropout)

        if vmindex_in_cities is not None: 
            self.gcity = MoE_SAG(d_model, d_model, dropout, self.supports_len, vmindex_in_cities)
        

        self.scale = nn.Parameter(torch.tensor(0.1))
        self.conv1=Conv2d(d_model, prelen, kernel_size=(1, seqlen), padding=(0, 0),stride=(1, 1), bias=True)

    def forward(self, input):
        """
        :param inputs: Batchsize, 1, numnodes, len

        :return
        - Output: Batchsize, prelen, numnodes, 1
        """
       
        
        if self.supports_len > 1 :
            A1 = self.gconv(input, self.supports)# [batch_size,d_model, numnode, seqlen]
        else:
            A1 =  torch.stack(self.supports).squeeze()
           
        d = torch.sum(A1, dim=-1) + 1e-5  
        D = torch.diag_embed(1.0 / d)
        A1 = torch.matmul(D, A1)  
        city_features = None
        if self.vmindex_in_cities is not None: 
            city_features = self.gcity(x, A1)
        x = input  
        x = self.temporal_cnn(x)  # [batch_size, 1 , numnodes, seqlen] -> [batch_size, d_model, 1, seqlen]
        x = torch.relu(x)
        x = x.squeeze() # [batch_size, d_model, seqlen]
        
        
        x = x.permute(0, 2, 1)  # [batch_size, seqlen, d_model]
        x = self.tvom(x)  # (batch_size, seqlen, d_model)
        

        x = x.permute(1, 0, 2)  # 调整为 (seqlen, batch_size, d_model)
        x = self.transformer(x)  # 输出: (seqlen, batch_size, d_model)
        x = x.permute(1, 0, 2)  # 调整回 (batch_size, seqlen, d_model)

       
        x = self.multi_head_attention(x)  # (batch_size, seqlen, d_model)
       
        x = x[:, -1, :]  # [batch_size, d_model]
        x = self.fc(x) # [batch_size, prelen]
        x = x.unsqueeze(2).unsqueeze(3).repeat(1, 1, self.numnode, 1)# [batch_size, prelen, numnode, 1] 
       
    
        #combined = torch.cat([x, spatial_features], dim=-1) # (Batch, numnode,2d_model)
        spatial_features = self.gcn(input, A1) # [batch_size,d_model, numnode, seqlen]]
        spatial_features = self.conv1(spatial_features) # [batch_size,prelen, numnode, 1]
        
        output = x + self.scale*spatial_features
        
        return output

# 14256
# 14730

class GCGRN(nn.Module):
    def __init__(self, device, in_dim, seqlen,  prelen, supports, numnode, d_model, dropout, vmindex_in_cities=None, 
                 dilation_channels=32):
        super(GCGRN, self).__init__()
        print("GCGRN  dilation_channels:", dilation_channels)
        self.seqlen = seqlen
        self.prelen = prelen
        self.gcgrn_cell = GCGRNCell(in_dim, dilation_channels)
        self.fc = nn.Linear(dilation_channels, in_dim)  # Output channel size
        self.hidden_size = dilation_channels
        self.supports = supports
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
        if supports is None:
            self.supports = []
        # self.gconv = SimplifiedMoE_GCN(in_dim, d_model, dropout, self.supports_len)
        self.gconv = Fuseadj(in_dim, d_model, dropout, self.supports_len)
        self.gcn = GCNLayer(in_dim, d_model, dropout)
        self.scale = nn.Parameter(torch.tensor(0.1))

        self.vmindex_in_cities = vmindex_in_cities
        if vmindex_in_cities is not None: 
            self.gcity = MoE_SAG(in_dim, d_model, dropout, self.supports_len, vmindex_in_cities)
        

        self.convc=Conv2d(d_model, prelen, kernel_size=(1, seqlen), padding=(0, 0),stride=(1, 1), bias=True)
        self.conv1=Conv2d(dilation_channels, prelen, kernel_size=(1, seqlen), padding=(0, 0),stride=(1, 1), bias=True)
        self.gate = nn.Sequential(
            nn.BatchNorm2d(d_model),
            nn.Sigmoid(),
            
            # nn.LeakyReLU(negative_slope=0.1)
            # nn.Tanh(),
            
        )
    def forward(self, x):
        """
        :param inputs: Batchsize, 1, numnodes, len

        :return
        - Output: Batchsize, prelen, numnodes, 1
        """
        city_features = None
        if self.supports_len > 1 :
            A1 = self.gconv(x, self.supports)
        else:
            A1 =  torch.stack(self.supports).squeeze()
        d = torch.sum(A1, dim=-1) + 1e-5  
        D = torch.diag_embed(1.0 / d)
        A1 = torch.matmul(D, A1)
        
        if self.vmindex_in_cities is not None: 
            city_features = self.gcity(x, A1)
        
        batch_size, in_features, numnode, seq_len = x.size()
        H = torch.zeros(batch_size,  self.hidden_size, numnode).to(x.device)
        
        for t in range(seq_len):
            H = self.gcgrn_cell(x[:, :, :, t], H, A1) 


        if city_features is None:
            
            output = H.unsqueeze(3).repeat(1, 1, 1, self.seqlen) # batch_size, hidden_size, num_nodes,seqlen
            output = self.conv1(output) # batch_size, prelen, num_nodes,1
        else:
            output = H.unsqueeze(3).repeat(1, 1, 1, self.seqlen) # batch_size, hidden_size, num_nodes,seqlen
            output = self.conv1(output) # batch_size, prelen, num_nodes,1
           
            features = self.convc(city_features)
            output = output + self.scale * features
        
        return output


class GTCN(nn.Module):
    # __init__(self, device, in_dim, seqlen,  prelen, supports, numnode, d_model, dropout, vmindex_in_cities=None,
    def __init__(self, device, in_dim, seqlen,  prelen, supports, numnode, d_model, dropout, vmindex_in_cities=None, dataft='LD', adjid=None, sigma=0.0,
                 k_depth=3, node_feature_size=12, dilation_factors=[1, 2]):
        super(GTCN, self).__init__()
        self.numnode = numnode
        self.seqlen = seqlen
        self.node_embedding = NodeEmbeddingModule(in_dim, node_feature_size, k_depth, d_model)
        
        self.residual_blocks = nn.ModuleList([
            MDCConv(node_feature_size if i ==0 else d_model, d_model, dropout, dilation_factors=dilation_factors) for i in range(4)
        ])
        
        self.supports = supports
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
        if supports is None:
            self.supports = []
        self.gconv = Fuseadj(in_dim, d_model, dropout, self.supports_len, dataft, adjid, sigma)

        self.scale = nn.Parameter(torch.tensor(0.1))

        self.vmindex_in_cities = vmindex_in_cities
        if vmindex_in_cities is not None: 
            self.gcity = MoE_SAG(in_dim, d_model, dropout, self.supports_len, vmindex_in_cities)
            # self.gcity = PMoE_SAG(in_dim, d_model, dropout, self.supports_len, vmindex_in_cities)
            # self.gcity = torch.jit.script(self.gcity)

        self.convc=Conv2d(d_model, prelen, kernel_size=(1, seqlen), padding=(0, 0),stride=(1, 1), bias=True)
        self.conv1=Conv2d(d_model, prelen, kernel_size=(1, seqlen), padding=(0, 0),stride=(1, 1), bias=True)

    def forward(self, x):

        if self.supports_len > 1 :
            A1 = self.gconv(x, self.supports)
        else:
            A1 = torch.stack(self.supports).squeeze()
        
        d = torch.sum(A1, dim=-1) + 1e-5  
        D = torch.diag_embed(1.0 / d)
        A1 = torch.matmul(D, A1)
        
        city_features = None
        if self.vmindex_in_cities is not None: 
            city_features = self.gcity(x, A1)

        x = self.node_embedding(x, A1)# (Batchsize, 12, numnodes, len)
        for block in self.residual_blocks:
            x = block(x)  #(Batchsize, 32, numnodes, len)

        output = self.conv1(x)# batch_size, prelen, num_nodes,1

        if city_features is not None:
            features = self.convc(city_features)
        else:
            features = 0
        output = output + self.scale * features
        return output

class staticGTCN(nn.Module):
    # __init__(self, device, in_dim, seqlen,  prelen, supports, numnode, d_model, dropout, vmindex_in_cities=None,
    def __init__(self, device, in_dim, seqlen,  prelen, supports, numnode, d_model, dropout, vmindex_in_cities=None, dataft='LD', adjid=-1, sigma=0.0,
                 k_depth=3, node_feature_size=12, dilation_factors=[1, 2]):
        super(staticGTCN, self).__init__()
        self.numnode = numnode
        self.seqlen = seqlen
        self.node_embedding = staticNodeEmbeddingModule(in_dim, node_feature_size, k_depth, d_model)
        
        self.residual_blocks = nn.ModuleList([
            staticMDCConv(node_feature_size if i ==0 else d_model, d_model, dropout, dilation_factors=dilation_factors) for i in range(4)
        ])
        

        
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
        if self.supports_len > 1 :
            self.supports = torch.stack(supports).to(device)
        else :
            self.supports = torch.stack(supports).squeeze().to(device)

        self.gconv = staticFuseadj(in_dim, d_model, dropout, self.supports_len, dataft, adjid, sigma)

        self.scale = nn.Parameter(torch.tensor(0.1))

        self.vmindex_in_cities = vmindex_in_cities
        if vmindex_in_cities is not None: 
            self.gcity = staticMoE_SAG(in_dim, d_model, dropout, self.supports_len, vmindex_in_cities)

        self.convc=Conv2d(d_model, prelen, kernel_size=(1, seqlen), padding=(0, 0),stride=(1, 1), bias=True)
        self.conv1=Conv2d(d_model, prelen, kernel_size=(1, seqlen), padding=(0, 0),stride=(1, 1), bias=True)

    def forward(self, x):
        if self.supports_len > 1 :
            A1 = self.gconv(x, self.supports)
        else:
            A1 = self.supports
        
        d = torch.sum(A1, dim=-1) + 1e-5  
        D = torch.diag_embed(1.0 / d)
        A1 = torch.matmul(D, A1)
        
        city_features = None
        if self.vmindex_in_cities is not None: 
            city_features = self.gcity(x, A1)

        x = self.node_embedding(x, A1)# (Batchsize, 12, numnodes, len)
        for block in self.residual_blocks:
            x = block(x)  #(Batchsize, 32, numnodes, len)

        output = self.conv1(x)# batch_size, prelen, num_nodes,1

        if city_features is not None:
            features = self.convc(city_features)
        else:
            features = 0
        output = output + self.scale * features
       
        return output
