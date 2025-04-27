import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IMU import CsvImuFileLoader
import deviation
import Trajectory_drawing as TD
import Quaternion_inertial_navigation as QIN
from ellipsoid_fit import ellipsoid_fit, ellipsoid_plot, data_regularize
import ellipsoid
import os
import change_point
import sys
sys.path.append(r'C:\Users\13106\Desktop\code\IMU\MyIMUCode\IMU去噪算法\深度学习方法\GRU')
sys.path.append(r'C:\Users\13106\Desktop\code\IMU\MyIMUCode\IMU去噪算法\传统方法')
import EKF.EKF as EKF
import EMD_WT.WT as WT
import LSTM_GRU
import load_dataset
import torch

def process_imu_folder(folder_path):
    """
    处理包含多个IMU CSV文件的文件夹
    folder_path: 包含Accelerometer*.csv和Gyroscope*.csv文件的文件夹路径
    """
    acc_mean = []
    gyro_mean = []
    acc_std = []
    gyro_std = []
    temperatures = []  # 存储温度值
    
    # 获取所有加速度计和陀螺仪文件
    acc_files = [f for f in os.listdir(folder_path) if f.startswith('Accelerometer') and f.endswith('.csv')]
    gyro_files = [f for f in os.listdir(folder_path) if f.startswith('Gyroscope') and f.endswith('.csv')]
    
    # 提取温度值
    def extract_temperature(filename):
        # 处理两种格式: Accelerometer-10.csv (-10度) 和 Accelerometer10.csv (10度)
        temp_str = filename.split('.')[0].replace('Accelerometer', '').replace('Gyroscope', '')
        return int(temp_str)  # 转换为整数
    
    # 创建文件温度映射
    acc_file_map = {extract_temperature(f): f for f in acc_files}
    gyro_file_map = {extract_temperature(f): f for f in gyro_files}
    
    # 处理匹配的文件对
    for temp in sorted(set(acc_file_map.keys()) & set(gyro_file_map.keys())):
        acc_path = os.path.join(folder_path, acc_file_map[temp])
        gyro_path = os.path.join(folder_path, gyro_file_map[temp])
        
        # 处理单个IMU数据文件对
        acc_m, acc_s, gyro_m, gyro_s = process_imu_files(acc_path, gyro_path)
        acc_mean.append(acc_m)
        acc_std.append(acc_s)
        gyro_mean.append(gyro_m)
        gyro_std.append(gyro_s)
        temperatures.append(temp)
    
    return temperatures, acc_mean, acc_std, gyro_mean, gyro_std

def plot_imu_statistics(temperatures, acc_mean, acc_std, gyro_mean, gyro_std):
    """
    绘制IMU数据统计图(分图显示)
    """
    # 加速度统计图
    plt.figure(figsize=(12, 8))
    colors = ['r', 'g', 'b']  # 红: X轴, 绿: Y轴, 蓝: Z轴
    # 加速度标准差子图
    plt.subplot(2, 1, 1)
    for i, axis in enumerate(['X轴', 'Y轴', 'Z轴']):
        plt.plot(temperatures, [std[i] for std in acc_std], 'o-', 
                color=colors[i], label=f'{axis}加速度标准差')
    plt.title('三轴加速度统计 vs 温度')
    plt.ylabel('标准差')
    plt.legend()
    plt.grid(True)
    
    # 加速度均值子图
    plt.subplot(2, 1, 2)
    for i, axis in enumerate(['X轴', 'Y轴', 'Z轴']):
        plt.plot(temperatures, [mean[i] for mean in acc_mean], 'o-', 
                color=colors[i], label=f'{axis}加速度均值')
    plt.xlabel('温度(℃)')
    plt.ylabel('均值')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 角速度统计图
    plt.figure(figsize=(12, 8))
    
    # 角速度标准差子图
    plt.subplot(2, 1, 1)
    for i, axis in enumerate(['X轴', 'Y轴', 'Z轴']):
        plt.plot(temperatures, [std[i] for std in gyro_std], 'o-', 
                color=colors[i], label=f'{axis}角速度标准差')
    plt.title('三轴角速度统计 vs 温度')
    plt.ylabel('标准差')
    plt.legend()
    plt.grid(True)
    
    # 角速度均值子图
    plt.subplot(2, 1, 2)
    for i, axis in enumerate(['X轴', 'Y轴', 'Z轴']):
        plt.plot(temperatures, [mean[i] for mean in gyro_mean], 'o-', 
                color=colors[i], label=f'{axis}角速度均值')
    plt.xlabel('温度(℃)')
    plt.ylabel('均值')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def process_imu_files(acc_path, gyro_path):
    """
    处理单个IMU数据文件对
    """
    print(f"\n处理文件: {acc_path} 和 {gyro_path}")
    
    # 加载数据
    CSV_IMU = CsvImuFileLoader(acc_path, gyro_path)
    acc_data, gyro_data, time_data = CSV_IMU._get_all_data()
    acc_data = np.array(acc_data)
    gyro_data = np.array(gyro_data)
    time_data = np.array(time_data)
    
    # 计算统计量
    acc_std, acc_mean = deviation.accelerometer_deviation(acc_data)
    gyro_std, gyro_mean = deviation.gyroscope_deviation(gyro_data)

    return acc_mean, acc_std, gyro_mean, gyro_std

def process_signal_segments(data, A, process_even_func, process_odd_func, save_path=None):
    """
    处理信号分段并重新合并
    :param data: 原始信号数据
    :param A: 分段点索引列表
    :param process_even_func: 处理偶数索引段的函数
    :param process_odd_func: 处理奇数索引段的函数 
    :param save_path: 结果保存路径(可选)
    :return: 处理后的完整信号
    """
    process_data = np.zeros_like(data)

    for i in range(0,len(A)-1,2):
        segment = data[A[i]:A[i+1]]
        processed_segment = process_even_func(segment) # 处理运动数据
        process_data[A[i]:A[i+1]] = processed_segment[0:A[i+1]-A[i]]

    for i in range(1,len(A)-1,2):
        segment = data[A[i]:A[i+1]]
        processed_segment = process_odd_func(segment) # 处理静止数据
        # 这里是因为使用了LSTM-GRU模型来处理静止数据，模型需要输入200个数据点，所以需要在前面填充200个数据点 可自行修改
        processed_segment = np.concatenate([segment[0:200], processed_segment]) 
        process_data[A[i]:A[i+1]] = processed_segment

    process_data[0:A[0]] = np.concatenate([process_data[0:200], process_odd_func(data[0:A[0]])])

    if save_path is not None:
        np.save(save_path, process_data)

    return process_data

def example_odd_processor(segments):
    # 将segments分成10份处理 这里是因为使用了LSTM-GRU模型来处理静止数据，输入数据过大所以分成10份 可自行修改
    segment_length = len(segments)
    chunk_size = segment_length // 10 # 数据分成10块
    processed_chunks = []

    for i in range(0, segment_length, chunk_size):
        chunk = segments[i:i+chunk_size]
        # 处理每个chunk
        data_x = torch.tensor(np.array(chunk), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        model = LSTM_GRU.LSTM_GRU(data_x, [1]*20, step=20)
        # 加载已经训练好的模型参数
        model.load_state_dict(torch.load(r'C:\Users\13106\Desktop\code\IMU\MyIMUCode\IMU去噪算法\深度学习方法\GRU\model.pth'))
        model.eval()
        model.to(device='cpu')
        out_x = model.forward((torch.zeros(1, data_x.shape[0],model.h_size_list[0], dtype=torch.float32), 
                             torch.zeros(1, data_x.shape[0],model.h_size_list[0], dtype=torch.float32)))
        processed_chunks.append(out_x.view(-1).detach().numpy())

    # 拼接处理后的chunks
    return np.concatenate(processed_chunks)

# def example_odd_processor(segments):
#     # 这里替换为您的实际处理函数
#     data_x = torch.tensor(np.array(segments), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
#     model = LSTM_GRU.LSTM_GRU(data_x, [1]*20, step=20)
#     model.load_state_dict(torch.load(r'C:\Users\13106\Desktop\code\IMU\MyIMUCode\IMU去噪算法\深度学习方法\GRU\model.pth'))
#     model.eval()
#     model.to(device='cpu')
#     out_x = model.forward((torch.zeros(1, data_x.shape[0],model.h_size_list[0], dtype=torch.float32), 
#                           torch.zeros(1, data_x.shape[0],model.h_size_list[0], dtype=torch.float32)))
#     return out_x.view(-1).detach().numpy()

def example_even_processor_EKF(segments):
    # 这里替换为您的实际处理函数
    segments = np.array(segments, dtype=np.float32)
    return EKF.EKF_func(segments)

def example_even_processor(segments):
    # 这里替换为您的实际处理函数
    segments = np.array(segments, dtype=np.float32)
    return WT.simple_WT(segments,4)  