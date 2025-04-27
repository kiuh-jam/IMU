import numpy as np
import pywt
import pandas as pd
import matplotlib.pyplot as plt

# 计算噪声标准差 σ
def estimate_noise_standard_deviation(coeffs):
    # coeffs 是小波变换的系数，这里我们只用细节系数的中位数来估计噪声标准差
    detail_coeffs = np.concatenate([coeffs[i] for i in range(1, len(coeffs))], axis=0)  # 获取所有细节系数
    median = np.median(np.abs(detail_coeffs))
    sigma = median / 0.6745  # 计算噪声标准差
    return sigma

# 计算阈值 b
def calculate_threshold(sigma, N):
    return sigma * np.sqrt(2 * np.log(N))

def simple_WT(time_series_measurement,level: int, wavelet='db3', threshold=None, mode='soft'):
    '''
    最基本的小波去噪
    wavelet: 使用何种小波基函数
    level: 分解层数 int
    threshold: 小波阈值 可以根据噪声强度调整 float
    mode: 软间隔还是硬间隔
    '''
    coeffs = pywt.wavedec(time_series_measurement, wavelet, level=level)

    threshold = calculate_threshold(estimate_noise_standard_deviation(coeffs), len(time_series_measurement)) if threshold == None else threshold

    coeffs_thresholded = [pywt.threshold(c, threshold, mode=mode) for c in coeffs]
    
    denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)

    return denoised_signal
