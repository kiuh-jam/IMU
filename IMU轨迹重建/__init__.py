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
    CSV_IMU = CsvImuFileLoader(acc_filepath, gyro_filepath)
    acc_data, gyro_data, time_data = CSV_IMU._get_all_data()
    acc_datax = np.array(acc_data)
    gyro_data = np.array(gyro_data)
    time_data = np.array(time_data)
    acc_std, acc_mean = deviation.accelerometer_deviation(acc_data)
    print(f"加速度均值: {acc_mean}")
    # result = change_point.change_point_detection(acc_datax[:10000,0], 100)
    # print(result)
    # exit()
    #[1510, 2060, 3410, 4155, 5720, 6045, 7415, 8075, 9500, 10000]
    # [1705, 1865, 3740, 3890, 4125, 4315, 5655, 6020, 7315, 8030, 8245, 9395, 10000]
    acc_filepath = r'C:\Users\13106\Desktop\code\IMU\IMUData\传感器数据采集\手写数字字母\测试'
    dataset = load_dataset.Accleration_Dataset(acc_filepath)
    data_x = dataset.data_x
    # print(data.shape)
    model = LSTM_GRU.LSTM_GRU(data_x, [1]*20, step=20)
    print(type(data_x))
    print(data_x.shape)
    # model.train((torch.zeros(1, data.shape[0],model.h_size_list[0]), torch.zeros(1, data.shape[0],model.h_size_list[0])), 50, 0.01)
    # model.loss_plt(save_path)
    # 加载权重（严格匹配模式）
    exit()
    model.load_state_dict(torch.load(r'C:\Users\13106\Desktop\code\IMU\MyIMUCode\IMU去噪算法\深度学习方法\GRU\model.pth'))
    model.eval()
    model.to(device='cpu')
    out_x = model.forward((torch.zeros(1, data_x.shape[0],model.h_size_list[0]), torch.zeros(1, data_x.shape[0],model.h_size_list[0])))
    plt.figure(figsize=(12,6))
    plt.subplot(2, 1, 1)
    plt.plot(data_x.view(-1).numpy(),color='red',label='输入信号')
    plt.title('输入信号')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(out_x.view(-1).detach().numpy(),color='blue',label='输出信号')
    plt.title('输出信号')
    plt.legend()
    plt.show()
    data_y = dataset.data_y
    model_y = LSTM_GRU.LSTM_GRU(data_y, [1]*20, step=20)
    model_y.load_state_dict(torch.load(r'C:\Users\13106\Desktop\code\IMU\MyIMUCode\IMU去噪算法\深度学习方法\GRU\model.pth'))
    model_y.eval()
    model_y.to(device='cpu')
    out_y = model_y.forward((torch.zeros(1, data_y.shape[0],model_y.h_size_list[0]), torch.zeros(1, data_y.shape[0],model_y.h_size_list[0])))
    data_z = dataset.data_z
    model_z = LSTM_GRU.LSTM_GRU(data_z, [1]*20, step=20)
    model_z.load_state_dict(torch.load(r'C:\Users\13106\Desktop\code\IMU\MyIMUCode\IMU去噪算法\深度学习方法\GRU\model.pth'))
    model_z.eval()
    model_z.to(device='cpu')
    out_z = model_z.forward((torch.zeros(1, data_z.shape[0],model_z.h_size_list[0]), torch.zeros(1, data_z.shape[0],model_z.h_size_list[0])))
    
    acc_data_LSTM_GRU = []
    out_x = out_x.view(-1).detach().numpy()
    out_y = out_y.view(-1).detach().numpy()
    out_z = out_z.view(-1).detach().numpy()
    for i in range(len(out_x)):
        acc_data_LSTM_GRU.append([out_x[i], out_y[i], out_z[i]])
    acc_data_LSTM_GRU = np.array(acc_data_LSTM_GRU)
    acc_data = acc_data_LSTM_GRU
    a=3410
    b=4315
    # result = change_point.change_point_detection(acc_datax[:10000,0], 100)
    # result = change_point.change_point_detection(acc_data[:10000,0], 100)
    # print(result)
    acc_data = acc_data[a:b,:]
    gyro_data = gyro_data[a:b,:]
    time_data = time_data[a:b]
    acc_std, acc_mean = deviation.accelerometer_deviation(acc_data)
    gyro_std, gyro_mean = deviation.gyroscope_deviation(gyro_data)
    print(f"加速度标准差: {acc_std}")
    print(f"加速度均值: {acc_mean}")
    print(f"角速度标准差: {gyro_std}")
    print(f"角速度均值: {gyro_mean}")
    TD.plot_accelerations(acc_data)
    TD.plot_gyro(gyro_data)
    IMS = QIN.INS(acc_data, gyro_data, time_data)
    acc_x_mean = np.mean(np.array(acc_data)[:,0])
    acc_y_mean = np.mean(np.array(acc_data)[:,1])
    acc_z_mean = np.mean(np.array(acc_data)[:,2])
    IMS.G = [0,0,0] # 重力加速度
    IMS.update_quaternion_rk4()
    IMS.get_euler_angles_matrix()
    v , s = IMS.ins_PoseEstimation()
    TD.plot_v(v)
    TD.plot_trajectory(s)