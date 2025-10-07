import torch
import torch.optim as optim
from model import *
import util
from adabelief_pytorch import AdaBelief


class trainer1():
    def __init__(self, scaler, in_dim, seq_length, pred_len, num_nodes, nhid, dropout,
                 lrate, wdecay, device, supports, vmindex_in_cities=None,scheduler = 'Static'):
        self.model = gwnet(device, num_nodes, dropout, supports, seq_length, in_dim, pred_len, vmindex_in_cities,
                           residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
       
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input) 
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        nrmse = util.masked_nrmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2

    def eval(self, input, real_val):
        self.model.eval()
       
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        
        predict = self.scaler.inverse_transform(output) if self.scaler is not None else output
        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        nrmse = util.masked_nrmse(predict, real, 0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2
    
    
class trainer2():
    def __init__(self, in_dim, seq_length, pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports, vmindex_in_cities=None,scheduler = 'Static' ):
        self.model = ASTGCN_Recent(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=pred_len, length=seq_length,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val, dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2    
    
class trainer3():
    def __init__(self, in_dim, seq_length, pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports, vmindex_in_cities=None,scheduler = 'Static' ):
        self.model = GRCN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=pred_len, length=seq_length,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
       
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2     
    
class trainer4():
    def __init__(self, in_dim, seq_length, pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports, vmindex_in_cities=None,scheduler = 'Static' ):
        self.model = Gated_STGCN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=pred_len, length=seq_length,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2  
    

    
class trainer5():
    def __init__(self, in_dim, seq_length, pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports, vmindex_in_cities=None,scheduler = 'Static' ):
        self.model = H_GCN_wh(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=pred_len, length=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2      

class trainer6():
    def __init__(self, in_dim,in_dim_cluster, seq_length, num_nodes, cluster_nodes, nhid , dropout, lrate, wdecay, device, supports,supports_cluster,transmit ):
        self.model = H_GCN_wdf(device, num_nodes,cluster_nodes, dropout, supports=supports, supports_cluster=supports_cluster,
                           in_dim=in_dim,in_dim_cluster=in_dim_cluster, out_dim=seq_length, transmit=transmit,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.00015)
        self.loss = util.masked_mae
        
        self.clip = 5
        self.supports=supports
        self.num_nodes=num_nodes

    def train(self, input, input_cluster, real_val,real_val_cluster):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input,input_cluster)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = output
        
        loss = self.loss(predict, real,0.0)
        (loss).backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2

    def eval(self, input, input_cluster, real_val,real_val_cluster):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input,input_cluster)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2
    
    
class trainer7():
    def __init__(self, in_dim,in_dim_cluster, seq_length, num_nodes, cluster_nodes, nhid , dropout, lrate, wdecay, device, supports,supports_cluster,transmit ):
        self.model = H_GCN(device, num_nodes,cluster_nodes, dropout, supports=supports, supports_cluster=supports_cluster,
                           in_dim=in_dim,in_dim_cluster=in_dim_cluster, out_dim=seq_length, transmit=transmit,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.00015)
        self.loss = util.masked_mae
        
        self.clip = 5
        self.supports=supports
        self.num_nodes=num_nodes

    def train(self, input, input_cluster, real_val,real_val_cluster):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input,input_cluster)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = output
        
        loss = self.loss(predict, real,0.0)
        (loss).backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2

    def eval(self, input, input_cluster, real_val,real_val_cluster):
        self.model.eval()
        output = self.model(input,input_cluster)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2
    
class trainer8():
    def __init__(self, in_dim, seq_length, pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports, vmindex_in_cities=None,scheduler = 'Static' ):
        self.model = OGCRNN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=pred_len, length=seq_length,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2
    
class trainer9():
    def __init__(self, in_dim, seq_length,pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports , vmindex_in_cities=None,scheduler = 'Static'):
        self.model = OTSGGCN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=pred_len, length=seq_length,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16, vmindex_in_cities=vmindex_in_cities)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=0.00015)
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2
    
    
class trainer10():
    def __init__(self, in_dim, seq_length, pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports, vmindex_in_cities=None,scheduler = 'Static' ):
        self.model = LSTM(device, num_nodes, dropout, seq_length,in_dim, pred_len, nhid,supports, vmindex_in_cities )
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output.transpose(1,3)
       
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(),mae,mape,nrmse,r2

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(),mae,mape,nrmse,r2 
    
    
class trainer11():
    def __init__(self, in_dim, seq_length, pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports, vmindex_in_cities=None,scheduler = 'Static'):
        self.model = GRU(device, num_nodes, dropout, supports, seq_length, in_dim, pred_len, nhid, vmindex_in_cities )
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)  # real in shape [batch_size, 1, num_nodes, pre_len]
        predict = output

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        mse = util.masked_mse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(),mae,mape,mse,r2

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        mse = util.masked_mse(predict, real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, mse, r2

class trainer12():
    def __init__(self, num_nodes, seq_length, pred_len, nhid , dropout, lrate, wdecay, device, supports, vmindex_in_cities=None,scheduler = 'Static' ):
        self.model = Informer(num_nodes,num_nodes,num_nodes, out_len=pred_len, dropout=dropout, device=device)

        self.model.to(device)
        self.device=device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        self.loss = util.masked_mae
        self.pred_len=pred_len
        self.clip = 5
   
    #btrainx, x_mark, dec_inp, y_mark
    def train(self, input, input_mark, real_val, dec_inp,real_val_mark):
        self.model.train()
        self.optimizer.zero_grad()
        
        
        output = self.model(input, input_mark, dec_inp, real_val_mark)#x_enc, x_mark_enc, x_dec, x_mark_dec,
        #output = [batch_size,pre_len,num_nodes]
        #real = torch.unsqueeze(real_val, dim=1)  # real in shape [batch_size, 1, num_nodes, pre_len]
        predict = output
        real_val=real_val[:,-self.pred_len:,:].to(self.device)
        loss = self.loss(predict, real_val, 0.0)# real val= [batch_size,pre_len,num_nodes]
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real_val,0.0).item()
        mape = util.masked_mape(predict,real_val,0.0).item()
        nrmse = util.masked_nrmse(predict,real_val,0.0).item()
        r2 = util.masked_r_squared(predict,real_val, 0.0).item()
        return loss.item(),mae,mape,nrmse,r2

    def eval(self, input, input_mark, real_val, dec_inp,real_val_mark):
        self.model.eval()
        output = self.model(input, input_mark, dec_inp, real_val_mark)
        
        predict = output
        real_val=real_val[:,-self.pred_len:,:].to(self.device)
        loss = self.loss(predict, real_val,0.0)
        mae = util.masked_mae(predict,real_val,0.0).item()
        mape = util.masked_mape(predict,real_val,0.0).item()
        nrmse = util.masked_nrmse(predict, real_val,0.0).item()
        r2 = util.masked_r_squared(predict,real_val, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2
    
class trainer13():
    def __init__(self, num_nodes, seq_length, pred_len, label_len, dropout, lrate, wdecay, device, supports, vmindex_in_cities=None,scheduler = 'Static' ):
        self.model = Autoformer(num_nodes,num_nodes,num_nodes, out_len=pred_len, label_len=label_len, dropout=dropout, device=device)
        self.model.to(device)
        self.device=device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        self.loss = util.masked_mae
        self.pred_len=pred_len
        self.clip = 5
    #btrainx, x_mark, dec_inp, y_mark
    def train(self, input, input_mark, real_val, dec_inp,real_val_mark):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, input_mark, dec_inp, real_val_mark)#x_enc, x_mark_enc, x_dec, x_mark_dec,
       
        predict = output
        real_val=real_val[:,-self.pred_len:,:].to(self.device)
        loss = self.loss(predict, real_val, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real_val,0.0).item()
        mape = util.masked_mape(predict,real_val,0.0).item()
        nrmse = util.masked_nrmse(predict,real_val,0.0).item()
        r2 = util.masked_r_squared(predict,real_val, 0.0).item()
        return loss.item(),mae,mape,nrmse,r2

    def eval(self, input, input_mark, real_val, dec_inp,real_val_mark):
        self.model.eval()
        output = self.model(input, input_mark, dec_inp, real_val_mark)
        predict = output
        real_val=real_val[:,-self.pred_len:,:].to(self.device)
        loss = self.loss(predict, real_val,0.0)
        mae = util.masked_mae(predict,real_val,0.0).item()
        mape = util.masked_mape(predict,real_val,0.0).item()
        nrmse = util.masked_nrmse(predict,real_val,0.0).item()
        r2 = util.masked_r_squared(predict,real_val, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2

class trainer14():
    def __init__(self, iterations, seq_length, pred_len, nhid , dropout, lrate, wdecay, device, supports, vmindex_in_cities=None,scheduler = 'Static' ):
        self.model = N_BEATS(seq_length, pred_len)
        self.model.to(device)
        self.learning_rate = lrate #
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        self.loss = util.masked_mae
        
        self.clip = 5
        self.lr_decay_step = iterations // 3
        if self.lr_decay_step == 0:
            self.lr_decay_step = 1

    def train(self, input, real_val, iter):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output.transpose(1, 3)
        predict = output
        
        real = torch.unsqueeze(real_val,dim=1)
        loss = self.loss(predict, real_val, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate * 0.5 ** (iter // self.lr_decay_step)
        
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(),mae,mape,nrmse,r2

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1, 3)
        predict = output

        real = torch.unsqueeze(real_val,dim=1)
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2
    
class trainer15():
    def __init__(self, num_nodes, seq_length, pred_len, label_len, dropout, lrate, wdecay, device, supports, vmindex_in_cities=None,scheduler = 'Static' ):
        self.model = TimesNet(num_nodes,num_nodes,num_nodes, seq_len=seq_length, pred_len=pred_len, label_len=label_len, dropout=dropout)
        self.model.to(device)
        self.device=device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        self.loss = util.masked_mae
        self.pred_len=pred_len
        self.clip = 5
    #btrainx, x_mark, dec_inp, y_mark
    def train(self, input, input_mark, real_val, dec_inp,real_val_mark):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, input_mark, dec_inp, real_val_mark)#x_enc, x_mark_enc, x_dec, x_mark_dec,
       
        predict = output
        real_val=real_val[:,-self.pred_len:,:].to(self.device)
        loss = self.loss(predict, real_val, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real_val,0.0).item()
        mape = util.masked_mape(predict,real_val,0.0).item()
        mse = util.masked_mse(predict,real_val,0.0).item()
        r2 = util.masked_r_squared(predict,real_val, 0.0).item()
        return loss.item(),mae,mape,mse,r2

    def eval(self, input, input_mark, real_val, dec_inp,real_val_mark):
        self.model.eval()
        output = self.model(input, input_mark, dec_inp, real_val_mark)
        predict = output
        real_val=real_val[:,-self.pred_len:,:].to(self.device)
        loss = self.loss(predict, real_val,0.0)
        mae = util.masked_mae(predict,real_val,0.0).item()
        mape = util.masked_mape(predict,real_val,0.0).item()
        mse = util.masked_mse(predict, real_val,0.0).item()
        r2 = util.masked_r_squared(predict,real_val, 0.0).item()
        return loss.item(), mae, mape, mse, r2

class trainer16():
    def __init__(self, in_dim, seq_length, pre_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports, vmindex_in_cities=None,scheduler = 'Static' ):
        self.model = DCRNN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, seq_len=seq_length, horizon=pre_len)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input[:, 0, :, :],real_val)
        
        real=real_val
        predict = output
        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input[:, 0, :, :],real_val)
        output = torch.unsqueeze(output,dim=1)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        real = real.transpose(1,3)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2
    
class trainer17():
    def __init__(self, seq_length, pred_len, nhid , dropout, lrate, wdecay, device, supports, vmindex_in_cities=None,scheduler = 'Static' ):
        self.model = NHITS(seq_length, pred_len)
        self.model.to(device)
        self.learning_rate = lrate #
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        real_val = torch.unsqueeze(real_val, dim=1)  # real in shape [batch_size, 1, num_nodes, pre_len]
        predict = output
        loss = self.loss(predict, real_val, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real_val,0.0).item()
        mape = util.masked_mape(predict,real_val,0.0).item()
        mse = util.masked_mse(predict,real_val,0.0).item()
        r2 = util.masked_r_squared(predict,real_val, 0.0).item()
        return loss.item(),mae,mape,mse,r2

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        mse = util.masked_mse(predict, real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, mse, r2
class trainer18():
    def __init__(self, batch_size, seq_length, pred_len, num_nodes, nhid , dropout, lrate, wdecay, device, supports, vmindex_in_cities=None,scheduler = 'Static' ):
        self.model = DeepAR(seq_length, pred_len, num_nodes, batch_size=batch_size, lstm_hidden_dim=nhid, 
                            lstm_dropout=dropout, device=device)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        
        output = self.model(input)
        real = torch.unsqueeze(real_val,dim=1)
        predict = output
        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(),mae,mape,nrmse,r2

    def eval(self, input, real_val):
        self.model.eval()
        
        output = self.model(input)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(),mae,mape,nrmse,r2 

class trainer19():
    def __init__(self, in_dim, seq_length,pred_len, num_nodes, nhid, dropout, lrate, wdecay, device, supports, vmindex_in_cities=None,scheduler = 'Static'):
        self.model = GCformer(device, in_dim, seq_length, pred_len, supports, num_nodes, nhid*4, dropout, vmindex_in_cities)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2
class trainer20():
    def __init__(self, in_dim, seq_length,pred_len, num_nodes, nhid, dropout, lrate, wdecay, device, supports , vmindex_in_cities=None,scheduler = 'Static'):
        self.model = GCGRN(device, in_dim, seq_length, pred_len, supports, num_nodes, nhid, dropout, vmindex_in_cities)
        self.model.to(device)

        self.optimizer = optim.Adam([
            {'params': [param for name, param in self.model.named_parameters() if "gconv" not in name], 'lr': lrate},
            {'params': [param for name, param in self.model.named_parameters() if "gconv" in name], 'lr': lrate*5 }
        ], weight_decay=wdecay)
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.345) 
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        
        self.loss = util.masked_mae
        
        self.clip = 5
        

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)

        loss.backward() 
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2

class trainer21():
    def __init__(self, in_dim, seq_length,pred_len, num_nodes, nhid, dropout, lrate, wdecay, device, supports , vmindex_in_cities=None,scheduler = 'Static', dataft='LD', adjid=None, sigma=0.0):
        self.model = GTCN(device, in_dim, seq_length, pred_len, supports, num_nodes, nhid, dropout, vmindex_in_cities, dataft, adjid, sigma)
        self.model.to(device)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.optimizer = optim.Adam([
            {'params': [param for name, param in self.model.named_parameters() if "gconv" not in name], 'lr': lrate},
            {'params': [param for name, param in self.model.named_parameters() if "gconv" in name], 'lr': lrate*5 } # *5 is default
        ], weight_decay=wdecay)
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.345) 
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        loss.backward()
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nnrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
       

        return loss.item(), mae, mape, nnrmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2


from model import staticGTCN
class trainer22():
    def __init__(self, in_dim, seq_length,pred_len, num_nodes, nhid, dropout, lrate, wdecay, device, supports , vmindex_in_cities=None,scheduler = 'Static', dataft='LD', adjid=-1, sigma=0.0):
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        self.model = staticGTCN(device, in_dim, seq_length, pred_len, supports, num_nodes, nhid, dropout, vmindex_in_cities, dataft, adjid, sigma)
        self.model.to(device)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.optimizer = optim.Adam([
            {'params': [param for name, param in self.model.named_parameters() if "gconv" not in name], 'lr': lrate},
            {'params': [param for name, param in self.model.named_parameters() if "gconv" in name], 'lr': lrate*5 } # *5 is default
        ], weight_decay=wdecay)
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.345) 
        if scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.0002)
        elif scheduler == 'Reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                       mode='min', threshold=0.0002, patience=5, factor=0.1, min_lr=0.0002)
        
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nnrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nnrmse, r2

    def eval(self, input, real_val):
        self.model.eval()
        scripted_model = torch.jit.script(self.model)
        output = scripted_model(input)
        # output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        nrmse = util.masked_nrmse(predict,real,0.0).item()
        r2 = util.masked_r_squared(predict,real, 0.0).item()
        return loss.item(), mae, mape, nrmse, r2