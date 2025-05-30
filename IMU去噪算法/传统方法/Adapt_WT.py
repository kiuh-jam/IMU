"""
自适应阈值函数的小波去噪算法
基于论文《基于多源异构信号的运动轨迹还原与动力学特征研究》
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt

# 自适应阈值函数
def adaptive_threshold_function(D_i_j, lambda_i, i, L):
    """
    自适应阈值函数（ATF）
    :param D_i_j: 第i层的第j个高频小波系数
    :param lambda_i: 第i层的阈值
    :param i: 当前分解层数
    :param L: 总分解层数
    :return: 修正后的高频小波系数
    """
    beta_i = i / L  # 调节因子
    alpha_i_j = 2 / np.pi * np.arctan((beta_i * (np.abs(D_i_j) - lambda_i)) / lambda_i)  # 自适应因子

    if np.abs(D_i_j) <= lambda_i:
        D_hat_i_j = 0
    else:
        D_hat_i_j = alpha_i_j * D_i_j + (1 - alpha_i_j) * np.sign(D_i_j) * (np.abs(D_i_j) - lambda_i)

    return D_hat_i_j

# 进行小波分解
def wavelet_decomposition(signal, wavelet = 'db4', levels = 3):
    '''
    signal: 输入信号
    wavelet: 小波基 默认为db4
    levels: 分解层数 默认为3
    '''
    coeffs = pywt.wavedec(signal, wavelet, level = levels)
    return coeffs

# 进行阈值处理
def threshold_coeffs(coeffs, lambda_i, levels, L):
    '''
    coeffs: 分解后的系数列表
    lambda_i: 阈值列表
    levels: 分解层数
    L: 总分解层数
    '''
    new_coeffs = [coeffs[0]]  # 近似系数不变
    for i in range(1, levels + 1):
        # 使用列表推导式处理细节系数，并保持数组类型
        new_coeffs.append(np.array([adaptive_threshold_function(c, lambda_i[i - 1], i, L) for c in coeffs[i]]))
    return new_coeffs

# 进行小波重构
def wavelet_reconstruction(coeffs, wavelet = 'db4'):
    '''
    coeffs: 分解后的系数列表
    wavelet: 小波基 默认为db4
    '''
    return pywt.waverec(coeffs, wavelet)

def Wavelet(time, x, y, z, wavelet = 'db4', levels = 3):
    '''
    time: 时间序列 (N,)
    x: 加速度数据或者是角速度数据 (N,)
    y: 加速度数据或者是角速度数据 (N,)
    z: 加速度数据或者是角速度数据 (N,)
    wavelet: 小波基 默认为db4
    levels: 分解层数 默认为3
    '''
    wavelet = wavelet
    levels = levels
    
    # 计算阈值
    sigma = npstd([x,y,z])
    lambda_i = [sigma * np.sqrt(2 * np.log(len(signal))) for signal in [x, y, z]]

    # 对每个轴的信号进行小波分解和重构
    signals = [x, y, z]
    reconstructed_signals = []

    for signal in signals:
        coeffs = wavelet_decomposition(signal, wavelet, levels)
        new_coeffs = threshold_coeffs(coeffs, lambda_i, levels, levels)
        reconstructed_signal = wavelet_reconstruction(new_coeffs, wavelet)
        reconstructed_signals.append(reconstructed_signal)
    
    signal_length = len(time)
    signals_ = []
    for signal_ in reconstructed_signals:
        signals_.append(signal_[:signal_length])

    reconstructed_signals = signals_

    reconstructed_signals_ = [
        [reconstructed_signals[0][i], reconstructed_signals[1][i], reconstructed_signals[2][i]]
        for i in range(signal_length)
    ]
    
    