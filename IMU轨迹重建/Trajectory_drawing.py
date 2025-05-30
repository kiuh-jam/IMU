"""
plot_trajectory(positions) 绘制三维轨迹图，positions是包含x,y,z坐标的数组 shape=(N,3)
plot_trajectory_xy(positions) 绘制XY平面轨迹图，positions是包含x,y坐标的数组 shape=(N,2)
plot_x_trajectory(positions) 绘制X轴轨迹图，positions是包含x坐标的数组 shape=(N,1)
plot_y_trajectory(positions) 绘制Y轴轨迹图，positions是包含y坐标的数组 shape=(N,1)
plot_z_trajectory(positions) 绘制Z轴轨迹图，positions是包含z坐标的数组 shape=(N,1)
plot_x_v(velocity) 绘制X轴速度图，velocity是包含x速度的数组 shape=(N,1)
plot_y_v(velocity) 绘制Y轴速度图，velocity是包含y速度的数组 shape=(N,1)
plot_z_v(velocity) 绘制Z轴速度图，velocity是包含z速度的数组 shape=(N,1)
plot_acceleration(acc_data) 绘制加速度图，acc_data是包含加速度的数组 shape=(N,3)
def plot_gyro(gyro_data) 绘制陀螺仪数据图，gyro_data是包含陀螺仪数据的数组 shape=(N,3)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设positions是包含x,y,z坐标的数组 shape=(N,3)
# 设置支持中文的字体，SimHei 是常见的中文字体
plt.rcParams['font.family'] = ['SimHei']  # 或者 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_trajectory(positions):
    positions = np.array(positions)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取x,y,z坐标
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # 绘制轨迹
    ax.plot(x, y, z, 'b-', linewidth=2, label='轨迹')
    
    # 标记起点和终点
    ax.scatter(x[0], y[0], z[0], c='g', marker='o', s=100, label='起点')
    ax.scatter(x[-1], y[-1], z[-1], c='r', marker='^', s=100, label='终点')
    
    # 设置坐标轴标签
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.set_title('3D轨迹图')
    ax.legend()
    
    # # 调节刻度
    # ax.set_zticks(np.linspace(-0.5, 0.5, 5))
    # ax.set_yticks(np.linspace(-0.5, 0.5, 5))

    plt.tight_layout()
    plt.show()

def plot_trajectory_xy(positions):
    positions = np.array(positions)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # 提取x,y坐标
    x = positions[:, 0]
    y = positions[:, 1]

    # 绘制XY平面轨迹
    ax.plot(x, y, 'b-', linewidth=2, label='轨迹')
    
    # 标记起点和终点
    ax.scatter(x[0], y[0], c='g', marker='o', s=100, label='起点')
    ax.scatter(x[-1], y[-1], c='r', marker='^', s=100, label='终点')
    
    # 设置坐标轴标签
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_title('XY平面轨迹图')
    ax.legend()
    

    plt.tight_layout()
    plt.show()

def plot_x_trajectory(positions):
    positions = np.array(positions)
    plt.figure()
    plt.plot(positions[:, 0], 'r-', linewidth=2)
    plt.xlabel('时间步长')
    plt.ylabel('X坐标')
    plt.title('X轴轨迹')
    plt.grid(True)
    plt.show()

def plot_y_trajectory(positions):
    positions = np.array(positions)
    plt.figure()
    plt.plot(positions[:, 1], 'g-', linewidth=2)
    plt.xlabel('时间步长')
    plt.ylabel('Y坐标')
    plt.title('Y轴轨迹')
    plt.grid(True)
    plt.show()

def plot_z_trajectory(positions):
    positions = np.array(positions)
    plt.figure()
    plt.plot(positions[:, 2], 'b-', linewidth=2)
    plt.xlabel('时间步长')
    plt.ylabel('Z坐标')
    plt.title('Z轴轨迹')
    plt.grid(True)
    plt.show()

def plot_x_v(velocity):
    velocity = np.array(velocity)
    plt.figure()
    plt.plot(velocity[:, 0], 'r-', linewidth=2)
    plt.xlabel('时间步长')
    plt.ylabel('X坐标')
    plt.title('X轴速度')
    plt.grid(True)
    plt.show()

def plot_y_v(velocity):
    velocity = np.array(velocity)
    plt.figure()
    plt.plot(velocity[:, 1], 'g-', linewidth=2)
    plt.xlabel('时间步长')
    plt.ylabel('Y坐标')
    plt.title('Y轴速度')
    plt.grid(True)
    plt.show()

def plot_z_v(velocity):
    velocity = np.array(velocity)
    plt.figure()
    plt.plot(velocity[:, 2], 'b-', linewidth=2)
    plt.xlabel('时间步长')
    plt.ylabel('Z坐标')
    plt.title('Z轴速度')
    plt.grid(True)
    plt.show()

def plot_v(velocity):

    """
    绘制三轴加速度数据
    acc_data: 包含x,y,z加速度的二维数组，shape=(N,3)
    """
    acc_data = np.array(velocity)
    plt.figure(figsize=(12, 8))
    
    # X轴加速度
    plt.subplot(3, 1, 1)
    plt.plot(acc_data[:, 0], 'r-', linewidth=1)
    plt.ylabel('X速度 (m/s)')
    plt.title('三轴速度数据')
    plt.grid(True)
    
    # Y轴加速度
    plt.subplot(3, 1, 2)
    plt.plot(acc_data[:, 1], 'g-', linewidth=1)
    plt.ylabel('Y速度 (m/s)')
    plt.grid(True)
    
    # Z轴加速度
    plt.subplot(3, 1, 3)
    plt.plot(acc_data[:, 2], 'b-', linewidth=1)
    plt.ylabel('Z速度 (m/s)')
    plt.xlabel('时间步长')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_accelerations(acc_data):
    """
    绘制三轴加速度数据
    acc_data: 包含x,y,z加速度的二维数组，shape=(N,3)
    """
    acc_data = np.array(acc_data)
    plt.figure(figsize=(12, 8))
    
    # X轴加速度
    plt.subplot(3, 1, 1)
    plt.plot(acc_data[:, 0], 'r-', linewidth=1)
    plt.ylabel('X加速度 (m/s^2)')
    plt.title('三轴加速度数据')
    plt.grid(True)
    
    # Y轴加速度
    plt.subplot(3, 1, 2)
    plt.plot(acc_data[:, 1], 'g-', linewidth=1)
    plt.ylabel('Y加速度 (m/s^2)')
    plt.grid(True)
    
    # Z轴加速度
    plt.subplot(3, 1, 3)
    plt.plot(acc_data[:, 2], 'b-', linewidth=1)
    plt.ylabel('Z加速度 (m/s^2)')
    plt.xlabel('时间步长')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_gyro(gyro_data):
    """
    绘制三轴加速度数据
    gyro_data: 包含x,y,z角速度的二维数组，shape=(N,3)
    """
    gyro_data = np.array(gyro_data)
    plt.figure(figsize=(12, 8))
    
    # X轴加速度
    plt.subplot(3, 1, 1)
    plt.plot(gyro_data[:, 0], 'r-', linewidth=1)
    plt.ylabel('X角速度 (m/s^2)')
    plt.title('三轴角速度数据')
    plt.grid(True)
    
    # Y轴加速度
    plt.subplot(3, 1, 2)
    plt.plot(gyro_data[:, 1], 'g-', linewidth=1)
    plt.ylabel('Y角速度 (m/s^2)')
    plt.grid(True)
    
    # Z轴加速度
    plt.subplot(3, 1, 3)
    plt.plot(gyro_data[:, 2], 'b-', linewidth=1)
    plt.ylabel('Z角速度 (m/s^2)')
    plt.xlabel('时间步长')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def xyz_position(position):
    """
    绘制三轴加速度数据
    acc_data: 包含x,y,z加速度的二维数组，shape=(N,3)
    """
    acc_data = np.array(position)
    plt.figure(figsize=(12, 8))
    
    # X轴加速度
    plt.subplot(3, 1, 1)
    plt.plot(acc_data[:, 0], 'r-', linewidth=1)
    plt.ylabel('X (m)')
    plt.title('三轴位置数据')
    plt.grid(True)
    
    # Y轴加速度
    plt.subplot(3, 1, 2)
    plt.plot(acc_data[:, 1], 'g-', linewidth=1)
    plt.ylabel('Y (m)')
    plt.grid(True)
    
    # Z轴加速度
    plt.subplot(3, 1, 3)
    plt.plot(acc_data[:, 2], 'b-', linewidth=1)
    plt.ylabel('Z (m)')
    plt.xlabel('时间步长')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
