# 根据论文实现算法EMD-WT
# 《High-G Calibration Denoising Method for High-G MEMS Accelerometer Based on EMD and Wavelet Threshold》
# 该算法的基本思想是先利用EMD将信号分解为多个IMU分量 对特定的高频IMF分量使用WT去噪 低频保持不变 随后重建信号
import EMD
import numpy as np
import WT
import pandas as pd
import matplotlib.pyplot as plt
# 设置支持中文的字体，SimHei 是常见的中文字体
plt.rcParams['font.family'] = ['SimHei']  # 或者 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
def EMD_WT(time_series_measurement, threshold=None, mode = 'soft', level = 4, wavelet = 'db4'):
    '''
    '''
    N = len(time_series_measurement)
    #经验模态分解
    IMD, res = EMD.EMD(time_series_measurement)
    
    # 获得需要进行小波去噪的阈值
    IMD_var = []
    for i in range(len(IMD)):
        IMD_var.append(np.mean(np.square(IMD[i])))
    ks = int(min(IMD_var))+1
    kkk = min(IMD_var)
    # 小波去噪
    wave_IMF = []
    for i in range(ks):
        wave_IMF.append(WT.simple_WT(IMD[i], threshold=threshold, wavelet=wavelet, mode=mode, level=level))
    
    # 重构信号
    time_series_predict = np.zeros(N)
    for i in range(len(wave_IMF)):
        time_series_predict = [k+j for k,j in zip(time_series_predict,wave_IMF[i])]
    for i in range(len(wave_IMF),len(IMD)):
        time_series_predict = [k+j for k,j in zip(time_series_predict, IMD[i])]
    time_series_predict = [k+j for k,j in zip(time_series_predict, res)]

    return time_series_predict, ks, kkk

if __name__ == '__main__':
    file_path = r'C:\Users\13106\Desktop\2024\编程\传感器信号去噪\深度学习方法\LSTM-RNN\data\Accelerometer.csv'
    time_series = pd.read_csv(file_path)
    data = time_series['Acceleration x (m/s^2)'].to_numpy()
    returndata,_,_ = EMD_WT(data)
    print(np.std(data))
    print(np.std(returndata))
    # plt.plot(data,color='red',label='输入信号')
    # plt.plot(returndata,color='blue',linestyle='--',label='去噪信号')
    # plt.legend()
    # plt.show()


