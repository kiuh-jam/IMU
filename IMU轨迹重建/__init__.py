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
    acc_filepath = r'C:\Users\13106\Desktop\code\IMU\IMUData\传感器数据采集\静止数据\已提交\4\X\Accelerometer.csv'
    gyro_filepath = r'C:\Users\13106\Desktop\code\IMU\IMUData\传感器数据采集\静止数据\已提交\4\X\Gyroscope.csv'
    mag_filepath = r'C:\Users\13106\Desktop\code\IMU\IMUData\传感器数据采集\静止数据\已提交\4\X\Magnetometer.csv'
    CSV_IMU = CsvImuFileLoader(acc_filepath, gyro_filepath, mag_filepath)
    acc_data, gyro_data, mag_data, _, time_data = CSV_IMU._get_all_data()
    acc_datax = np.array(acc_data)
    gyro_data = np.array(gyro_data)
    time_data = np.array(time_data)
    mag_data = np.array(mag_data)
    ins = QIN.INS(acc_datax, gyro_data, mag_data, time_data)
    ins.update_quaternion_rk4()
    ins.get_euler_angles_matrix()
    v, p = ins.ins_PoseEstimation()
    print(len(p))
    TD.plot_trajectory(p)
    