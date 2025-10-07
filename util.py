import pickle
import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import requests
import json
import math
from tslearn.metrics import dtw
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import List             #informer
from pandas.tseries import offsets  # informer
from pandas.tseries.frequencies import to_offset # informer


# informer
class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"

class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5

class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5

class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5

class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5

class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5

class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5

class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5

class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.week - 1) / 52.0 - 0.5

def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth],
        # offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            # DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)

def time_features(dates, timeenc=0, freq='t'):
    """
    > `time_features` takes in a `dates` dataframe with a 'dates' column and extracts the date down to `freq` where freq can be any of the following if `timeenc` is 0: 
    > * m - [month]
    > * w - [month]
    > * d - [month, day, weekday]
    > * b - [month, day, weekday]
    > * h - [month, day, weekday, hour]
    > * t - [month, day, weekday, hour, *minute]
    > 
    > If `timeenc` is 1, a similar, but different list of `freq` values are supported (all encoded between [-0.5 and 0.5]): 
    > * Q - [month]
    > * M - [month]
    > * W - [Day of month, week of year]
    > * D - [Day of week, day of month, day of year]
    > * B - [Day of week, day of month, day of year]
    > * H - [Hour of day, day of week, day of month, day of year]
    > * T - [Minute of hour*, hour of day, day of week, day of month, day of year]
    > * S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]

    *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.
    """

    if timeenc == 0:
        dates['month'] = dates.date.apply(lambda row:row.month,1)
        dates['day'] = dates.date.apply(lambda row:row.day,1)
        dates['weekday'] = dates.date.apply(lambda row:row.weekday(),1)
        dates['hour'] = dates.date.apply(lambda row:row.hour,1)
        dates['minute'] = dates.date.apply(lambda row:row.minute,1)
        dates['minute'] = dates.minute.map(lambda x:x//15)
        freq_map = {
            'y':[],'m':['month'],'w':['month'],'d':['month','day','weekday'],
            'b':['month','day','weekday'],
            'h':['day','weekday','hour'],# 'h':['month','day','weekday','hour']
            't':['day','weekday','hour','minute'],# 't':[ 'month','day','weekday','hour','minute'],
        }
        return dates[freq_map[freq.lower()]].values
    if timeenc == 1:
        dates = pd.to_datetime(dates.date.values)
        see = np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)]).transpose(1,0)
        return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)]).transpose(1,0)
    
class EnsSDateset(Dataset):
    def __init__(self, root_path='./data/', flag='train', interval=1,
                 data_type='LD', data='site', size=[12, 12, 0],
                 scale=False, inverse=False,  freq='15min', site=None,model='gwnet',target_freq='15min'):
        self.interval = interval
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.label_len = size[2]
        self.model = model
        self.data_type = data_type
        self.data = data

        # init
        self.flag =flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.inverse = inverse

        self.freq = freq
        self.target_freq=target_freq
        self.root_path = root_path
        self.site_path = self.root_path + self.data + '/' + self.freq + '/'
        data_parser = {
            'LD': 'load_rate.csv',
            'BW': 'up_bw.csv',
        }
        self.data_path = data_parser[data_type]
        self.site = site
        self.__read_data__()
    def freq_to_minutes(self, freq):
        if 'min' in freq:
            return int(freq.replace('min', ''))
        elif 'T' in freq:
            return int(freq.replace('T', ''))
        elif 'h' in freq:
            return int(freq.replace('h', '')) * 60
        elif 'H' in freq:
            return int(freq.replace('H', '')) * 60
        elif 'd' in freq:
            return int(freq.replace('d', '')) * 1440  
        elif 'D' in freq:
            return int(freq.replace('D', '')) * 1440
        else:
            raise ValueError(f"Unsupported frequency format: {freq}")
    def __read_data__(self):
        bw_raw = pd.read_csv(os.path.join(self.site_path, self.data_path))
        bw_raw = bw_raw.fillna(method='ffill', limit=len(bw_raw)).fillna(method='bfill', limit=len(bw_raw))
        bw_raw['date'] = pd.to_datetime(bw_raw['date'])
        bw_raw.set_index('date', inplace=True)
        if self.data != 'site':
            if self.data_type == 'LD':
                ins = pd.read_csv(self.root_path + 'e_vm_instance.csv', usecols=['uuid', 'ens_region_id', 'cores'])
                bb = bw_raw.T              
                bb.reset_index(inplace=True)
                ld = pd.merge(bb, ins, how='left', left_on='index', right_on='uuid')
                
                ld.iloc[:, 1:2881] = ld.iloc[:, 1:2881].multiply(ld.loc[:, 'cores'], axis="index")
                bw_raw = ld.iloc[:, 1:2881].T
               
            else:
                bw_raw = bw_raw / 1e6  # bps -> Mbps
        
        original_freq_minutes = self.freq_to_minutes(self.freq)
        target_freq_minutes = self.freq_to_minutes(self.target_freq)
        bw_raw.index = pd.to_datetime(bw_raw.index)
        if target_freq_minutes != original_freq_minutes:
            if target_freq_minutes < original_freq_minutes:
                
                
                if target_freq_minutes < 60:
                    resample_rule = f'{target_freq_minutes}T'  
                else:
                    resample_rule = f'{target_freq_minutes // 60}H'  
                
                bw_raw = bw_raw.resample(resample_rule).interpolate(method='spline', order=3)
                print(f"Upsampled data from {self.freq} to {self.target_freq} using spline interpolation.")
            else:
                if target_freq_minutes % 1440 != 0 and 1440 % target_freq_minutes != 0:
                    raise ValueError("Target frequency should be a divisor or multiple of 1440 minutes (24 hours) to preserve periodicity.")

                resample_rule = self.target_freq
                
                bw_raw = bw_raw.resample(resample_rule).mean()
                print(f"Downsampled data from {self.freq} to {self.target_freq} using mean aggregation.")
        else:
            bw_raw = bw_raw
            print(f"No resampling needed. Data remains at {self.freq} frequency.")
        


        if self.site is not None:
            vml = pd.read_csv(self.site_path + 'vmlist.csv')
            ins = pd.read_csv(self.root_path + 'e_vm_instance.csv', usecols=['uuid', 'ens_region_id', 'cores'])
            vmr = pd.merge(vml, ins, how='left', left_on='vm', right_on='uuid')
            idx = (vmr['ens_region_id'].values == self.site)                    
            bw_raw = bw_raw.loc[:, idx].copy()                                  
        
        bw_stamp=pd.DataFrame(bw_raw.index, columns=['date'])
        self.length = len(bw_raw)
        
        border1s = [0, int(0.6*self.length), int(0.8*self.length)]              
        border2s = [int(0.6*self.length), int(0.8*self.length), self.length]    

        border1 = border1s[self.set_type]                                       
        border2 = border2s[self.set_type]
        bw = bw_raw.values

        bw_stamp = bw_stamp[border1:border2]# transformer
        bw_stamp = pd.to_datetime(bw_stamp['date']).astype(np.int64)
        self.data_stamp = bw_stamp.values.astype(np.int64)
        if self.scale:
            self.scaler = StandardScaler()
            b_u_tr = bw[border1s[0]:border2s[0]]
            self.scaler.fit(b_u_tr)
            bu_data = self.scaler.transform(bw)
        else:
            bu_data = bw
        self.data_x = bu_data[border1:border2][:, :, np.newaxis]
        if self.inverse:
            self.data_y = bw.values[border1:border2][:, :, np.newaxis]
        else:
            self.data_y = bu_data[border1:border2][:, :, np.newaxis]
        
        if  border2 - border1 < self.seq_len + self.pred_len:
            
            padding_length = self.seq_len + self.pred_len - border2 + border1
            print("padding ", padding_length," pos  for ",self.flag, " dataset")
            
            last_stamp = self.data_stamp[-1] 
            padded_stamps = np.full(padding_length, last_stamp)  
            self.data_stamp = np.concatenate((self.data_stamp, padded_stamps))  

            last_x = self.data_x[-1, :, :]  
            padded_x = np.tile(last_x, (padding_length, 1, 1))  
            self.data_x = np.concatenate((self.data_x, padded_x), axis=0)  

            last_y = self.data_y[-1, :, :]
            padded_y = np.tile(last_y, (padding_length, 1, 1))  
            self.data_y = np.concatenate((self.data_y, padded_y), axis=0) 
        
        
    def __getitem__(self, index):
        lo = index
        hi = lo + self.seq_len

        if self.model=='Informer' or self.model == 'Autoformer' or self.model == 'TimesNet':
            train_data = self.data_x[lo: hi, :, :]
            target_data = self.data_y[hi - self.label_len: hi + self.pred_len, :, :]
        else:
            train_data = self.data_x[lo: hi, :, :]
            target_data = self.data_y[hi:hi + self.pred_len, :, :]
        x = torch.from_numpy(train_data).double()
        y = torch.from_numpy(target_data).double()
        x_mark = self.data_stamp[lo:hi]
        y_mark = self.data_stamp[hi - self.label_len: hi + self.pred_len]
        return x, y, x_mark, y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    # def inverse_transform(self, data):
    #     return self.scaler.inverse_transform(data)


class Dataloader(object):
    def __init__(self, xs, ys, xm, ym, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0            
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            
            xm_date = pd.DataFrame(np.vstack([pd.to_datetime(row, unit='ns') for row in xm]),
                       columns=[f'datetime_col_{i}' for i in range(xm.shape[1])])
            ym_date = pd.DataFrame(np.vstack([pd.to_datetime(row, unit='ns') for row in ym]),
                       columns=[f'datetime_col_{i}' for i in range(ym.shape[1])])
            
            last_xm_values = xm_date.iloc[-1].values
            last_ym_values = ym_date.iloc[-1].values
            padding_interval_seconds = 15 * 60
            padded_xm_values = np.arange(last_xm_values[-1] + padding_interval_seconds,
                                            last_xm_values[-1] + (num_padding + 1) * padding_interval_seconds,
                                            padding_interval_seconds)
            padded_ym_values = np.arange(last_ym_values[-1] + padding_interval_seconds,
                                            last_ym_values[-1] + (num_padding + 1) * padding_interval_seconds,
                                            padding_interval_seconds)
            padded_xm = pd.DataFrame(padded_xm_values.reshape(-1, 1), columns=['datetime_col_0'])
            padded_ym = pd.DataFrame(padded_ym_values.reshape(-1, 1), columns=['datetime_col_0'])
            newx_dates = pd.concat([xm_date, padded_xm], ignore_index=True)    
            newy_dates = pd.concat([ym_date, padded_ym], ignore_index=True)    
            xm = newx_dates
            ym = newy_dates
           
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        

        format_xm = []
        for row in range(xm.shape[0]):
            fuk = [xm.iloc[row,col] for col in range(xm.shape[1])]
            fuk = pd.DataFrame(pd.to_datetime(fuk),columns=['date'])
            fuk = time_features(fuk)
            format_xm.append(fuk)
        self.xm = format_xm
        
        format_ym = []
        for row in range(ym.shape[0]):
            fuk = [ym.iloc[row,col] for col in range(ym.shape[1])]
            fuk = pd.DataFrame(pd.to_datetime(fuk),columns=['date'])
            fuk = time_features(fuk)
            format_ym.append(fuk)
        self.ym = format_ym
        

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        xm, ym = self.xm[permutation], self.ym[permutation]
        self.xs = xs
        self.ys = ys
        self.xm = xm
        self.ym = ym
    def len(self):
        return self.num_batch
    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                xm_i = self.xm[start_ind: end_ind]
                ym_i = self.ym[start_ind: end_ind]
                yield (x_i, y_i, xm_i, ym_i)
                self.current_ind += 1

        return _wrapper()

class Standardscaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


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
    return L.astype(np.float32).todense()               


def load_pickle(pickle_file):                           
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)               
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def load_dataset(root_path, data_type, data_, batch_size,
                 seq_len, pred_len, scaler_flag=False, site=None,lable_len=0,model='gwnet',target_freq='15min'):
    data = {}
    for category in ['train', 'val', 'test']:
        dataset = EnsSDateset(root_path=root_path, data_type=data_type, data=data_,
                                flag=category, size=[seq_len, pred_len, lable_len], site=site,model=model,target_freq=target_freq)
        dataloader = DataLoader(dataset, batch_size=64)
        for i, (i_x, i_y, i_x_mark, i_y_mark) in enumerate(dataloader):
            if i == 0:
                a_x, a_y, a_x_mark, a_y_mark  = i_x, i_y, i_x_mark, i_y_mark
            a_x = torch.cat((a_x, i_x), dim=0)
            a_y = torch.cat((a_y, i_y), dim=0)
            a_x_mark = torch.cat((a_x_mark, i_x_mark), dim=0)# transformer
            a_y_mark = torch.cat((a_y_mark, i_y_mark), dim=0)# transformer
        data['x_' + category] = a_x.numpy()
        data['y_' + category] = a_y.numpy()
        data['x_mark_' + category] = a_x_mark.numpy()# transformer
        data['y_mark_' + category] = a_y_mark.numpy()# transformer
    # Data format
    if scaler_flag:
        scaler = Standardscaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
        for category in ['train', 'val', 'test']:
            data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['scaler'] = scaler
    data['train_loader'] = Dataloader(data['x_train'], data['y_train'], data['x_mark_train'], data['y_mark_train'], batch_size)
    data['val_loader'] = Dataloader(data['x_val'], data['y_val'], data['x_mark_val'], data['y_mark_val'], batch_size)
    data['test_loader'] = Dataloader(data['x_test'], data['y_test'], data['x_mark_test'], data['y_mark_test'], batch_size)

    return data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2                   
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_nrmse(preds, labels, null_val=np.nan, norm_type="range"):
    if np.isnan(null_val):                      
        mask = ~torch.isnan(labels) 
    else:
        mask = (labels != null_val)  
    mask = mask.float()
    mask /= torch.mean(mask)  
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)  
   
    mse = torch.square(preds - labels) * mask
    mse = torch.where(torch.isnan(mse), torch.zeros_like(mse), mse)
    mse = torch.mean(mse)
    
    rmse = torch.sqrt(mse)
    
 
    if norm_type == "std":
        norm = torch.std(labels)
    elif norm_type == "range":
        norm = torch.max(labels) - torch.min(labels)
    else:
        raise ValueError("Unsupported normalization type. Use 'std' or 'range'.")
    nrmse = rmse / (norm + 1e-7)  
    
    return nrmse


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)              
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):                      
        mask = ~torch.isnan(labels)             
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)                    
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    eps = 1e-7
    loss = torch.abs(preds - labels) / (torch.abs(labels) + torch.abs(preds) + eps)
    loss = loss * mask                          
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 2                 

def masked_r_squared(preds, labels, null_val=np.nan):
    mean_labels = torch.mean(labels)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
        mask_labels = ~torch.isnan(mean_labels)
    else:
        mask = (labels != null_val)
        mask_labels = (mean_labels != null_val)
    mask, mask_labels = mask.float(), mask_labels.float()
    mask /= torch.mean(mask)
    mask_labels /= torch.mean(mask_labels)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mask_labels = torch.where(torch.isnan(mask_labels), torch.zeros_like(mask_labels), mask_labels)
    ssr = torch.sum((labels - preds) ** 2 * mask)  
    sst = torch.sum((labels - mean_labels) ** 2 * mask_labels)  
    r_squared = 1.0 - ssr / sst
    return r_squared


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    smape = masked_mape(pred, real, 0.0).item()
    nrmse = masked_nrmse(pred, real, 0.0).item()
    r2 = masked_r_squared(pred, real, 0.0).item()
    return mae, smape, nrmse, r2


places = {
    "hefei": 1,"beijing": 2,"chongqing": 3,"fuzhou": 4,"lanzhou": 5,
    "guangzhou": 6,"nanning": 7,"guiyang": 8,"haikou": 9,"shijiazhuang": 10,
    "haerbin": 11,"zhengzhou": 12,"wuhan": 13,"changsha": 14,"nanjing": 15,
    "nanchang": 16,"changchun": 17,"shenyang":18,"hohhot": 19,"yinchuan": 20,
    "xining": 21,"xian": 22,"jinan": 23,"shanghai": 24,"taiyuan": 25,
    "chengdu": 26,"tianjin": 27,"urumqi": 28,"lhasa": 29,"kunming": 30,
    "hangzhou": 31
}
city_mapping = {
    "合肥": "hefei","北京": "beijing","重庆": "chongqing","福州": "fuzhou","兰州": "lanzhou",
    "广州": "guangzhou","南宁": "nanning","贵阳": "guiyang","海口": "haikou","石家庄": "shijiazhuang",
    "哈尔滨": "haerbin","郑州": "zhengzhou","武汉": "wuhan","长沙": "changsha","南京": "nanjing",
    "南昌": "nanchang","长春": "changchun","沈阳": "shenyang","呼和浩特": "hohhot","银川": "yinchuan",
    "西宁": "xining","西安": "xian","济南": "jinan","上海": "shanghai","太原": "taiyuan",
    "成都": "chengdu","天津": "tianjin","乌鲁木齐": "urumqi","拉萨": "lhasa","昆明": "kunming",
    "杭州": "hangzhou"
}
def vm_apt(root_path, data, freq, dis_threshold=4e5, rtt_threshold=30,
                temp_threshold=40):
    site_path = root_path + 'site/' + freq + '/'
    data_path = root_path + data + '/' + freq + '/'
    
    B_dis = np.load(site_path+'remain/dis_adj_'+str(dis_threshold)
                    + '.npy')
    B_tem = np.load(site_path+'remain/tem_adj_'+str(temp_threshold) 
                    + '.npy')
    B_rtt = np.load(site_path+'remain/rtt_adj_'+str(rtt_threshold)
                    + '.npy')
    

    df0 = pd.read_csv(site_path + 'vmlist.csv')
    df0 = pd.merge(df0, df0['ens_region_id'].str.split('-', expand=True), left_index=True, right_index=True)
    df0.rename(columns={0: 'city', 1: 'ISP', 2: 'num'}, inplace=True)
    site_lst = df0['ens_region_id'].tolist()
    site_to_index = {site: idx for idx, site in enumerate(site_lst)}
     
    df = pd.read_csv(data_path + "vmlist.csv")
    ins = pd.read_csv(root_path + 'e_vm_instance.csv', usecols=['uuid', 'cores', 'ens_region_id'])
    df_merged = pd.merge(df, ins, left_on='vm', right_on='uuid', how='left')
    df_merged.drop(['vm', 'uuid'], axis=1, inplace=True)
    # df_merged['site'] = df_merged['ens_region_id'].str.split('-').str[0]
    df_merged['site'] = df_merged['ens_region_id']
    
    
    df_merged['site_index'] = df_merged['site'].map(site_to_index)
    
    if df_merged['site_index'].isnull().any():
        unmapped_sites = df_merged[df_merged['site_index'].isnull()]['site'].unique()
        print("warning unmapped sites: ", unmapped_sites)
    else:
        print("Dis、Tem、Rtt all sites mapped\n")

    
    site_indices = df_merged['site_index'].astype(int).values

    dis_vm = B_dis[site_indices[:, None], site_indices[None, :]]
    tem_vm = B_tem[site_indices[:, None], site_indices[None, :]]
    rtt_vm = B_rtt[site_indices[:, None], site_indices[None, :]]
    return dis_vm, tem_vm, rtt_vm

def cityft_adj(root_path, site, freq):

    root_path_ = root_path + site + '/' + freq + '/'
    file_path = root_path +'Municipal_202006_data.xlsx'  
    print(file_path)

    df = pd.read_csv(root_path_ + "vmlist.csv")
    ins = pd.read_csv(root_path + 'e_vm_instance.csv', usecols=['uuid', 'cores', 'ens_region_id'])
    df1 = pd.merge(df, ins, left_on='vm', right_on='uuid', how='left')
    df1.drop(['vm', 'uuid'], axis=1, inplace=True)
    df1['city'] = df1['ens_region_id'].str.split('-').str[0]
    df1['is_in_places'] = df1['city'].isin(places.keys())

    df1['city_index'] = df1['city'].map(places) - 1
    if df1['city_index'].isnull().any():
        print("Warning, There are unmapped cities")
    else:
        print("All cities covered.")

    num_vms = df1.shape[0]
    adj = np.full((num_vms, num_vms), 0.0)
    in_places_mask = df1['is_in_places'].values
    in_places_indices = np.where(in_places_mask)[0]

    print("City in indices num: ",len(in_places_indices))

    city_indices = df1.loc[in_places_indices, 'city_index'].astype(int).values


    data = pd.read_excel(file_path, header=1) 
    data.columns = ['city_name'] + [f'feature_{i}' for i in range(1, data.shape[1])]
    data['city_pinyin'] = data['city_name'].map(city_mapping)
    data = data.dropna(subset=['city_pinyin']).reset_index(drop=True)
    data['order'] = data['city_pinyin'].map(places)
    data_sorted = data.sort_values(by='order')
    data_final = data_sorted.drop(columns=['city_name', 'city_pinyin', 'order']).reset_index(drop=True)
    data_final.fillna(0, inplace=True)

    scaler_std = StandardScaler()
    data_standardized = scaler_std.fit_transform(data_final)
    normalized_data = normalize(data_standardized, norm='l2')

    corr_matrix = pd.DataFrame(normalized_data).corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)] 
    # print(len(to_drop))
    reduced_data = pd.DataFrame(normalized_data).drop(columns=to_drop).values

    reduced_data = np.nan_to_num(reduced_data, nan=0.0)
    similarity_matrix = cosine_similarity(reduced_data)

    sub_similarity = similarity_matrix[np.ix_(city_indices, city_indices)]
    adj[np.ix_(in_places_indices, in_places_indices)] = sub_similarity
    
    return adj

def ntl_adj(root_path, site, freq):
    root_path_ = root_path + site + '/' + freq + '/'
    similarity_matrix_path = root_path + "VNL_swin_L_22k_m512_o32_75Q0.368.npy"
    print(similarity_matrix_path)
    df = pd.read_csv(root_path_ + "vmlist.csv")
    ins = pd.read_csv(root_path + 'e_vm_instance.csv', usecols=['uuid', 'cores', 'ens_region_id'])
    df1 = pd.merge(df, ins, left_on='vm', right_on='uuid', how='left')
    df1.drop(['vm', 'uuid'], axis=1, inplace=True)
    df1['city'] = df1['ens_region_id'].str.split('-').str[0]
    df1['is_in_places'] = df1['city'].isin(places.keys())
    
    df1['city_index'] = df1['city'].map(places) - 1
    if df1['city_index'].isnull().any():
        print("Warning, There are unmapped cities")
    else:
        print("All cities covered.")
    
    
    
    num_vms = df1.shape[0]
    
    adj = np.full((num_vms, num_vms), 0.0)
    
    
    in_places_mask = df1['is_in_places'].values
    in_places_indices = np.where(in_places_mask)[0]
    
   
    print("City in indices num: ",len(in_places_indices))
   
    city_indices = df1.loc[in_places_indices, 'city_index'].astype(int).values
 
    similarity_matrix = np.load(similarity_matrix_path)

    sub_similarity = similarity_matrix[np.ix_(city_indices, city_indices)]
   
    adj[np.ix_(in_places_indices, in_places_indices)] = sub_similarity
    
    return adj
def GAIA_adj(root_path, site, freq):
    root_path_ = root_path + site + '/' + freq + '/'
    similarity_matrix_path = root_path + "GAIA_swin_L_22k_m512_o32_75Q0.202.npy"
    print(similarity_matrix_path)
    df = pd.read_csv(root_path_ + "vmlist.csv")
    ins = pd.read_csv(root_path + 'e_vm_instance.csv', usecols=['uuid', 'cores', 'ens_region_id'])
    df1 = pd.merge(df, ins, left_on='vm', right_on='uuid', how='left')
    df1.drop(['vm', 'uuid'], axis=1, inplace=True)
    df1['city'] = df1['ens_region_id'].str.split('-').str[0]
    df1['is_in_places'] = df1['city'].isin(places.keys())
    
    df1['city_index'] = df1['city'].map(places) - 1
    if df1['city_index'].isnull().any():
        print("Warning, There are unmapped cities")
    else:
        print("All cities covered.")
    
    
   
    num_vms = df1.shape[0]
    
    adj = np.full((num_vms, num_vms), 0.0)
    
  
    in_places_mask = df1['is_in_places'].values
    in_places_indices = np.where(in_places_mask)[0]
    

    print("City in indices num: ",len(in_places_indices))
    
    city_indices = df1.loc[in_places_indices, 'city_index'].astype(int).values
    
    similarity_matrix = np.load(similarity_matrix_path)
    
    sub_similarity = similarity_matrix[np.ix_(city_indices, city_indices)]
   
    adj[np.ix_(in_places_indices, in_places_indices)] = sub_similarity
    
    return adj
def GHSL_adj(root_path, site, freq):
    root_path_ = root_path + site + '/' + freq + '/'
    similarity_matrix_path = root_path + "GHSL_swin_L_22k_m512_o32_75Q0.149.npy"
    print(similarity_matrix_path)
    df = pd.read_csv(root_path_ + "vmlist.csv")
    ins = pd.read_csv(root_path + 'e_vm_instance.csv', usecols=['uuid', 'cores', 'ens_region_id'])
    df1 = pd.merge(df, ins, left_on='vm', right_on='uuid', how='left')
    df1.drop(['vm', 'uuid'], axis=1, inplace=True)
    df1['city'] = df1['ens_region_id'].str.split('-').str[0]
   
    df1['is_in_places'] = df1['city'].isin(places.keys())
    
   
    df1['city_index'] = df1['city'].map(places) - 1
    if df1['city_index'].isnull().any():
        print("Warning, There are unmapped cities")
    else:
        print("All cities covered.")
    
   
    
    num_vms = df1.shape[0]
    
 
    adj = np.full((num_vms, num_vms), 0.0)
    
 
    in_places_mask = df1['is_in_places'].values
    in_places_indices = np.where(in_places_mask)[0]
    
  
    print("City in indices num: ",len(in_places_indices))
    
    
    city_indices = df1.loc[in_places_indices, 'city_index'].astype(int).values
   
    similarity_matrix = np.load(similarity_matrix_path)
    
    sub_similarity = similarity_matrix[np.ix_(city_indices, city_indices)]
   
    adj[np.ix_(in_places_indices, in_places_indices)] = sub_similarity
    
    return adj
def phy_adj(root_path, site, freq):
    root_path_ = root_path + site + '/' + freq + '/'
    df = pd.read_csv(root_path_ + "vmlist.csv")
    ins = pd.read_csv(root_path + 'e_vm_instance.csv', usecols=['uuid', 'cores', 'memory', 'storage'])
    df1 = pd.merge(df, ins, left_on='vm', right_on='uuid', how='left')
    df1.drop(['vm', 'uuid'], axis=1, inplace=True)
    df1['memory'] = df1['memory'] / 1024
    df1['storage'] = df1['storage'] / 1024
    
    # scl = OneHotEncoder()
    # df_a = scl.fit_transform(df1.values)
    # adj = cosine_similarity(df_a)
    print("Physics Data of vm got for Similarity Matrix")
    data = df1.values 
    scaler_std = StandardScaler()
    data_standardized = scaler_std.fit_transform(data)
    data_normalized = normalize(data_standardized, norm='l2')
    data_normalized = np.nan_to_num(data_normalized, nan=0.0)
    similarity_matrix = cosine_similarity(data_normalized)
    
    
    return similarity_matrix


def log_adj(root_path, site, freq):
    root_path_ = root_path + site + '/' + freq + '/'
    df = pd.read_csv(root_path_ + "vmlist.csv")
    ins = pd.read_csv(root_path + 'e_vm_instance.csv', usecols=['uuid', 'ali_uid', 'nc_name',
                      'ens_region_id', 'image_id'])
    df1 = pd.merge(df, ins, left_on='vm', right_on='uuid', how='left')
    split_cols = df1['ens_region_id'].str.split('-', expand=True)
    split_cols.columns = ['city', 'ISP', 'num']
    split_cols['num'] = split_cols['num'].fillna('0')
    df1 = pd.concat([df1, split_cols], axis=1)
    df1.drop(['vm', 'uuid', 'ens_region_id'], axis=1, inplace=True)
    print("Logic  Data of vm got for Similarity Matrix")

    enc = OneHotEncoder()
    data_coded = enc.fit_transform(df1.values).toarray()
   
    data_centered = data_coded - np.mean(data_coded, axis=0)
    data_normalized = normalize(data_centered , norm='l2')
    data_normalized = np.nan_to_num(data_normalized, nan=0.0)
    similarity_matrix = cosine_similarity(data_normalized)
    
    
    return similarity_matrix


def normalize_adjacency(adj):
    adj = np.nan_to_num(adj, nan=0.0)
    adj = adj + np.eye(adj.shape[0])  
    degree = np.sum(adj, axis=1)
    degree[degree <= 0] = 1e-8  #
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    return D_inv_sqrt @ adj @ D_inv_sqrt


def local_adj_(root_path, site, freq, adjtype):
    lst = []
    if adjtype == 'all':
        lst.append(ntl_adj(root_path, site, freq))# 470
        lst.append(GAIA_adj(root_path, site, freq))# 470
        lst.append(GHSL_adj(root_path, site, freq))# 470
        
        lst.append(phy_adj(root_path, site, freq))# 8866
        lst.append(log_adj(root_path, site, freq))# 1398331
        lst.append(cityft_adj(root_path, site, freq))# 471

        dis_vm, tem_vm, rtt_vm = vm_apt(root_path, site, freq)
        lst.append(dis_vm)# 39
        lst.append(tem_vm)# 47
        lst.append(rtt_vm)# 2


    elif adjtype == 'pic':
        lst.append(ntl_adj(root_path, site, freq))
        lst.append(GAIA_adj(root_path, site, freq))
        lst.append(GHSL_adj(root_path, site, freq))
    elif adjtype == 'site':
        lst.append(phy_adj(root_path, site, freq))
        lst.append(log_adj(root_path, site, freq))
        
    elif adjtype == 'city':
        lst.append(cityft_adj(root_path, site, freq))
        dis_vm, tem_vm, rtt_vm = vm_apt(root_path, site, freq)
        lst.append(dis_vm)
        lst.append(tem_vm)
        lst.append(rtt_vm)
    elif adjtype == 'identity':
        adj_mx = log_adj(root_path, site, freq)
        lst.append(np.diag(np.ones(adj_mx.shape[0])).astype(np.float32))
    else :
        print("not get a local adj mx.")
    # for i in range(len(lst)):
    #     np.save(f'./adjmx{i}.npy', lst[i])

    if adjtype == 'identity':
        normalized_lst = lst
    else:
        normalized_lst = []
        i = 0
        for adj in lst:
           
            i += 1

        
            normalized_adj = normalize_adjacency(adj)

            normalized_lst.append(normalized_adj)
    
    return normalized_lst


def site_index(root_path, data, freq, site):
    ins = pd.read_csv(root_path + 'e_vm_instance.csv', usecols=['uuid', 'ens_region_id'])
    root_path_ = root_path + data + '/' + freq + '/'
    vmlist = pd.read_csv(root_path_ + 'vmlist.csv')

    need_vm = ins[ins['ens_region_id'] == site]['uuid']
    is_in_vmlist=vmlist['vm'].isin(need_vm)
    indices = np.where(is_in_vmlist)[0]
    indices_list = indices.tolist()
    return indices_list

def cities_index(root_path, data, freq):
    root_path_ = root_path + data + '/' + freq + '/'
    vmlist = pd.read_csv(root_path_ + "vmlist.csv")
    ins = pd.read_csv(root_path + 'e_vm_instance.csv', usecols=['uuid', 'ens_region_id'])
    
    ins['city'] = ins['ens_region_id'].str.split('-').str[0]
    
    cities_list = []
    for i, city in enumerate(places.keys()):
        vm_in1city = ins[ins['city'] == city]['uuid']
        indata_vm_in1city = vmlist['vm'].isin(vm_in1city)
        indices_vm_in1city = np.where(indata_vm_in1city)[0]
        cities_list.append(indices_vm_in1city.tolist())
    
    return cities_list

