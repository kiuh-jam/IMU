"""
提取时间序列信号的特征
"""
from scipy.stats import skew, kurtosis
import numpy as np

class Feature:
    def __init__(self, data):
        '''
        data: 时间序列数据 (N,)
        '''
        self.data = data
    
    def max(self):
        '''
        数据的最大值
        '''
        return np.max(self.data)

    def min(self):
        '''
        数据的最小值
        '''
        return np.min(self.data)

    def mean(self):
        '''
        数据的平均值
        '''
        return np.mean(self.data)

    def std(self):
        '''
        数据的标准差
        '''
        return np.std(self.data)

    def zero_crossing_rate(self)->float:
        '''
        数据的过零率
        '''
        ZCR = np.where(np.diff(np.sign(signal)))[0]
        return len(ZCR)/len(signal)

    def magnitude(self):
        '''
        计算模
        '''
        return np.linalg.norm(self.data)
    
    def skewness(self):
        '''
        计算偏度
        '''
        return skew(self.data)
    
    def kurtosis(self):
        '''
        计算峰度
        '''
        return kurtosis(self.data)
    
    def RMS(self):
        '''
        计算均方根
        '''
        return np.sqrt(np.mean(np.square(self.data)))
    
    def SMA(self):
        '''
        计算信号幅度面积
        '''
        return np.mean(np.abs(self.data)).sum()
    
    def FFT(self):
        '''
        计算傅里叶变换
        '''
        return np.fft.fft(self.data)

class Sensor_Feature(Feature):
    def __init__(self, data, frequence=400):
        '''
        data: 时间序列数据 (N,3)
        frequence: 采样频率
        '''
        self.data = data
        self.x = data[:,0]
        self.y = data[:,1]
        self.z = data[:,2]
        self.frequence = frequence
        
    def Sensor_max(self):
        '''
        数据的最大值
        '''
        return self.max(self.x), self.max(self.y), self.max(self.z)

    def Sensor_min(self):
        '''
        数据的最小值
        '''
        return self.min(self.x), self.min(self.y), self.min(self.z)

    def Sensor_mean(self):
        '''
        数据的平均值
        '''
        return self.mean(self.x), self.mean(self.y), self.mean(self.z)

    def Sensor_std(self):
        '''
        数据的标准差
        '''
        return self.std(self.x), self.std(self.y), self.std(self.z)

    def Sensor_zero_crossing_rate(self):
        '''
        数据的过零率
        '''
        return self.zero_crossing_rate(self.x), self.zero_crossing_rate(self.y), self.zero_crossing_rate(self.z)
            
    def Sensor_skewness(self):
        '''
        计算偏度
        '''
        return self.skewness(self.x), self.skewness(self.y), self.skewness(self.z)

    def Sensor_kurtosis(self):
        '''
        计算峰度
        '''
        return self.kurtosis(self.x), self.kurtosis(self.y), self.kurtosis(self.z)

    def Sensor_RMS(self):
        '''
        计算均方根
        '''
        return self.RMS(self.x), self.RMS(self.y), self.RMS(self.z)

    def Sensor_SMA(self):
        '''
        计算信号幅度面积
        '''
        return self.SMA(self.x), self.SMA(self.y), self.SMA(self.z)

    def Sensor_FFT(self):
        '''
        计算傅里叶变换
        '''
        return self.FFT(self.x), self.FFT(self.y), self.FFT(self.z)

    def Sensor_magnitude_max(self):
        '''
        计算模的最大值
        '''
        return np.linalg.norm(self.data, axis=1).max()

    def Sensor_magnitude_min(self):
        '''
        计算模的最小值
        '''
        return np.linalg.norm(self.data, axis=1).min()

    def Sensor_variance(self):
        '''
        计算方差
        '''
        return np.var(self.data, axis=0)