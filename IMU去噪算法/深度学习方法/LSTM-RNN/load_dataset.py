# pytorch创建自己的数据集
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class Accleration_Dataset():
    def __init__(self, directory, max_time=300, num_samples = 100000, seq_len = 2000):
        '''
        directory: 数据文件所在的目录
        max_time: 时间序列数据的最大时间限制
        num_samples: 每个时间序列的样本数
        seq_len: 每个样本的长度
        '''
        self.directory = directory
        self.max_time = max_time
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.file_list = [f for f in os.listdir(directory) if f.endswith('.csv')]
        self.data = self._load_data()
        pass

    def _load_data(self):
        '''
        '''
        data = []
        for file in self.file_list:
            file_path = os.path.join(self.directory, file)
            print(file_path)
            df = pd.read_csv(file_path)
            time = df['Time (s)'].values
            acceleration = df['Acceleration x (m/s^2)'].values
    
            # 去除偏置
            acceleration = acceleration - acceleration.mean()

            mask = time <= self.max_time
            acceleration = acceleration[mask]

            acceleration = acceleration[:self.num_samples]
            print(len(acceleration))

            for i in range(0, len(acceleration), self.seq_len):
                batch = acceleration[i:i+self.seq_len if i+self.seq_len < len(acceleration) else len(acceleration)]
                if len(batch) == self.seq_len:
                    data.append(batch)

        return torch.tensor(data, dtype=torch.float32).unsqueeze(-1) # shape(bs,seq_len,1)