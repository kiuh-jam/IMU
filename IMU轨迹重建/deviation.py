"""
# 计算Allan方差
get_allen(y, sampling_rate=400):
    输入：
    y: 加速度或角速度数据
    sampling_rate: 采样率，默认为400Hz
    输出：
    Sigma: 不同τ下的阿伦方差值
    Tau: 不同τ值

# 绘制Allan标准差曲线
plot_allan(Sigmas, Taus):
    输入：
    Sigmas: 包含x,y,z三轴Sigma值的列表 [sigma_x, sigma_y, sigma_z]
    Taus: 包含x,y,z三轴Tau值的列表 [tau_x, tau_y, tau_z]
    输出：
    绘制三轴Allan标准差曲线

# 计算加速度的标准差和均值
accelerometer_deviation(acc_data):
    输入：
    acc_data: 包含x,y,z加速度的二维数组，shape=(N,3)
    输出：
    acc_std: 加速度的标准差
    acc_mean: 加速度的均值

# 计算角速度的标准差和均值
gyroscope_deviation(gyro_data):
    输入：
    gyro_data: 包含x,y,z角速度的二维数组，shape=(N,3)
    输出：
    gyro_std: 角速度的标准差
    gyro_mean: 角速度的均值
"""

import numpy as np
import matplotlib.pyplot as plt

def get_allen(y, sampling_rate=400):
    tau0 = 1/sampling_rate
    N = len(y)
    NL = N 
    Tau  = [] # 保存不同的tau
    Sigma = [] # 保存不同tao下的阿伦方差值
    Err = []
    for k in np.arange(1, 1000):
        sigma_k = np.sqrt(1/(2*(NL-1)) * np.sum(np.power(y[1:NL]-y[0:NL-1], 2))) #Allan的时域表达式
        Sigma.append(sigma_k)
        tau_k = 2 ** (k-1) * tau0 #将取样时间加倍， tau2 = 2tau1
        Tau.append(tau_k)
        err = 1 / np.sqrt(2* (NL-1))
        Err.append(err)
        NL = np.floor(NL/2) 
        NL = int(NL) 
        if NL < 3: 
            break
        y = 1/2 * (y[0:2*NL:2] + y[1:2*NL:2]) # 对应的序列长度减半
    return  Sigma, Tau

def plot_allan(Sigmas, Taus):
    """
    绘制三轴Allan标准差曲线
    Sigmas: 包含x,y,z三轴Sigma值的列表 [sigma_x, sigma_y, sigma_z]
    Taus: 包含x,y,z三轴Tau值的列表 [tau_x, tau_y, tau_z]
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # X轴
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.plot(Taus[0], Sigmas[0], 'r-')
    ax1.set_ylabel('X轴Allan标准差')
    ax1.grid(True)
    
    # Y轴
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.plot(Taus[1], Sigmas[1], 'g-')
    ax2.set_ylabel('Y轴Allan标准差')
    ax2.grid(True)
    
    # Z轴
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.plot(Taus[2], Sigmas[2], 'b-')
    ax3.set_xlabel('Tau (对数坐标)')
    ax3.set_ylabel('Z轴Allan标准差')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

# 计算加速度的标准差和均值
def accelerometer_deviation(acc_data):
    """
    计算加速度的标准差和均值
    acc_data: 包含x,y,z加速度的二维数组，shape=(N,3)
    """
    acc_data = np.array(acc_data)
    # 计算标准差
    acc_std = np.std(acc_data, axis=0)
    # 计算均值
    acc_mean = np.mean(acc_data, axis=0)
    return acc_std, acc_mean

# 计算角速度的标准差和均值
def gyroscope_deviation(gyro_data):
    """
    计算角速度的标准差和均值
    gyro_data: 包含x,y,z角速度的二维数组，shape=(N,3)
    """
    gyro_data = np.array(gyro_data)
    # 计算标准差
    gyro_std = np.std(gyro_data, axis=0)
    # 计算均值
    gyro_mean = np.mean(gyro_data, axis=0)
    return gyro_std, gyro_mean

