import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IMU import CsvImuFileLoader
import deviation
import Trajectory_drawing as TD
import Quaternion_inertial_navigation as QIN
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
import signal_segement

if __name__ == '__main__':
    #数字字母的轨迹重建——初步成功
    acc_filepath = r'C:\Users\13106\Desktop\code\IMU\IMUData\数字字母\Linear Acceleration.csv'
    gyro_filepath = r'C:\Users\13106\Desktop\code\IMU\IMUData\数字字母\Gyroscope.csv'
    CSV_IMU = CsvImuFileLoader(acc_filepath, gyro_filepath, interpolate=True)
    acc_data, gyro_data, time_data = CSV_IMU._get_all_data()
    # TD.plot_accelerations(acc_data)
    # result = change_point.change_point_detection(acc_data[2000:10000,2], 50)
    # print(result)
    # exit()
    gyro_data = np.array(gyro_data)
    time_data = np.array(time_data)
    acc_data_x = np.array(acc_data)[:,0]
    acc_data_y = np.array(acc_data)[:,1]
    acc_data_z = np.array(acc_data)[:,2]
    # A = [4250, 10560, 14800, 19150, 29890, 34100, 37890, 44940] # 数字字母 1 s 3 C
    # process_data_x = signal_segement.process_signal_segments(acc_data_x, A, example_even_processor, example_odd_processor,save_path='processed_data_x_linear_shuzi.npy')
    # process_data_y = signal_segement.process_signal_segments(acc_data_y, A, example_even_processor, example_odd_processor,save_path='processed_data_y_linear_shuzi.npy')
    # process_data_z = signal_segement.process_signal_segments(acc_data_z, A, example_even_processor, example_odd_processor,save_path='processed_data_z_linear_shuzi.npy')
    process_data_x = WT.simple_WT(np.load('processed_data_x_linear_shuzi.npy'),4)
    process_data_y = WT.simple_WT(np.load('processed_data_y_linear_shuzi.npy'),4)
    process_data_z = WT.simple_WT(np.load('processed_data_z_linear_shuzi.npy'),4)

    acc_data_LSTM_GRU = []
    for i in range(len(process_data_x)):
        acc_data_LSTM_GRU.append([process_data_x[i], process_data_y[i], process_data_z[i]])
    acc_data_LSTM_GRU = np.array(acc_data_LSTM_GRU)
    # TD.plot_accelerations(acc_data[:,:])
    # TD.plot_accelerations(acc_data_LSTM_GRU[:,:])
    acc_data = acc_data_LSTM_GRU
    a=4250
    b=10560
    acc_data = acc_data[a:b,:]
    gyro_data = gyro_data[a:b,:]
    time_data = time_data[a:b]
    IMS = QIN.INS(acc_data, gyro_data, time_data)
    IMS.G = [0,0,0] # 重力加速度
    IMS.update_quaternion_rk4()
    IMS.get_euler_angles_matrix()
    v , s = IMS.ZVC()
    # TD.plot_v(v)
    # TD.plot_gyro(gyro_data)
    TD.plot_trajectory(np.array(s))