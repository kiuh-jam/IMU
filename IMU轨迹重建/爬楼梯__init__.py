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
sys.path.append(r'C:\Users\13106\Desktop\code\IMU\MyIMUCode\github\IMU去噪算法\深度学习方法\GRU')
sys.path.append(r'C:\Users\13106\Desktop\code\IMU\MyIMUCode\github\IMU去噪算法\传统方法')
import EKF.EKF as EKF
import EMD_WT.WT as WT
import LSTM_GRU
import load_dataset
from feature import Sensor_Feature as SF
import torch


if __name__ == '__main__':

    acc_filepath = r'C:\Users\13106\Desktop\code\IMU\IMUData\爬楼梯\2025.5.19\Accelerometer.csv'
    gyro_filepath = r'C:\Users\13106\Desktop\code\IMU\IMUData\爬楼梯\2025.5.19\Gyroscope.csv'
    mag_filepath = r'C:\Users\13106\Desktop\code\IMU\IMUData\爬楼梯\2025.5.19\Magnetometer.csv'
    linear_acc_filepath = r'C:\Users\13106\Desktop\code\IMU\IMUData\爬楼梯\2025.5.19\Linear Acceleration.csv'
    CSV_IMU = CsvImuFileLoader(acc_filepath, gyro_filepath, mag_filepath, linear_acc_filepath, interpolate=True)
    acc_data, gyro_data, mag_data, time_data, linear_acc_data = CSV_IMU.acc_all_data, CSV_IMU.gyro_all_data, CSV_IMU.mag_all_data, CSV_IMU.time_all_data, CSV_IMU.linear_acc_all_data
    a = 0
    b = -1
    acc_data = np.array(acc_data[a:b,:])
    # acc_data = np.array(np.zeros_like(acc_data[a:b,:]))
    gyro_data = np.array(gyro_data[a:b,:])
    mag_data = np.array(mag_data[a:b,:])
    time_data = np.array(time_data[a:b])
    linear_acc_data = np.array(linear_acc_data[a:b,:])
    INS = QIN.INS(acc_data, gyro_data, mag_data, time_data, linear_acc_data,if_linear_acc=True)
    v,p=INS.ZUPT_ZARU_HDR_Compass_EKF_INS_1(if_ZARU=True, if_HDR=True, if_Compass=False)
    TD.plot_trajectory(p)
    TD.plot_trajectory_xy(p)