"""
使用惯性导航算法进行轨迹还原
INS
    包括姿态估计、位置估计、速度估计、轨迹重建
    包括普通惯性导航算法和基于零速更新的惯性导航算法 算法均使用零速补偿
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.spatial.transform import Rotation as R
import time

class INS:
    '''
    惯性导航算法
    包括姿态估计、位置估计、速度估计、轨迹重建
    '''
    def __init__(self, 
                acc_data, 
                gyro_data, 
                time,
                start_time = 400,
                G = [0,0,9.7926], 
                initial_position=np.zeros(3), 
                initial_velocity=np.zeros(3), 
                ):
        '''
        acc_data: 加速度数据 二维数组 每行为一个时刻的加速度值 每列为一个轴的加速度值
        gyro_data: 角速度数据 二维数组 每行为一个时刻的角速度值 每列为一个轴的角速度值
        time: 时间数据 一维数组 每行为一个时刻的时间值 单位为s
        initial_position: 初始位置 三维数组 默认为[0,0,0]
        initial_velocity: 初始速度 三维数组 默认为[0,0,0]
        acc_mean: 加速度数据的均值 三维数组
        G: 重力加速度 默认为9.7926m/s^2
        '''
        self.acc_data = acc_data
        self.gyro_data = gyro_data
        self.time = time
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity
        self.G = G # 重力加速度
        self.start_acc_mean = np.mean(np.array(acc_data[0:start_time,:]),axis = 0) # 初始加速度均值
        self.initial_quaternion = self.estimate_initial_quaternion_from_acc(self.start_acc_mean) 
        self.Q = [] # 存储每个时刻的四元数
        self.C = [] # 存储每个时刻的旋转矩阵

    def estimate_initial_quaternion_from_acc(self, acc_mean):
        '''
        根据初始加速度数据估计初始四元数
        acc_mean: 初始加速度数据的均值 三维数组
        返回值: 初始四元数 三维数组 [w, x, y, z]
        '''
        ax, ay, az = acc_mean
        pitch = np.arctan2(ay, np.sqrt(ax**2 + az**2))
        roll = np.arctan2(-ax, az)
        yaw = 0

        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)

        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = cy * sp * cr + sy * cp * sr
        qz = sy * cp * cr - cy * sp * sr

        return np.array([qw , qx, qy, qz])  # 注意格式：[w, x, y, z]

    def update_quaternion_rk4(self):
        '''
        使用Runge-Kutta方法更新四元数
        使用陀螺仪更新法更新四元数
        '''
        def quat_derivative(q,omega):
            '''
            四元数微分方程
            q: 四元数 三维数组 [x,y,z,w]
            omega: 角速度 三维数组 [wx,wy,wz]
            '''
            mat = np.array([[0,-omega[0],-omega[1],-omega[2]],
                            [omega[0],0,omega[2],-omega[1]],
                            [omega[1],-omega[2],0,omega[0]],
                            [omega[2],omega[1],-omega[0],0]])
            return 0.5*np.dot(mat,q)
        dt = []
        self.Q.append(self.initial_quaternion)
        for i in range(1,len(self.time)):
            dt.append(self.time[i]-self.time[i-1])
        for i in range(0,len(self.time)-1):
            k1 = (dt[i]/2)*quat_derivative(self.Q[i],self.gyro_data[i])
            k2 = (dt[i]/2)*quat_derivative(self.Q[i]+k1/2,(self.gyro_data[i]+self.gyro_data[i+1])/2)
            k3 = (dt[i]/2)*quat_derivative(self.Q[i]+k2/2,(self.gyro_data[i]+self.gyro_data[i+1])/2)
            k4 = (dt[i]/2)*quat_derivative(self.Q[i]+k3,self.gyro_data[i+1])
            new_Q = self.Q[i] + (k1+2*k2+2*k3+k4)/6
            new_Q = new_Q/np.linalg.norm(new_Q)
            self.Q.append(new_Q)
        self.dt = dt
        return

    def get_euler_angles_matrix(self):
        '''
        获取欧拉角 和 旋转矩阵
        '''
        if not hasattr(self, 'Q') or len(self.Q) == 0:
            raise ValueError("四元数列表为空，请先调用update_quaternion_rk4方法更新四元数列表")
        self.euler_angles = []
        for q in self.Q:
            q = np.array([q[1],q[2],q[3],q[0]])
            r = R.from_quat(q)
            self.C.append(r.as_matrix())
            self.euler_angles.append(r.as_euler('xyz', degrees=True))
        return

    def ins_PoseEstimation(self):
        '''
        姿态估计
        返回速度值和位置信息
        '''
        if not hasattr(self, 'Q') or len(self.Q) == 0:
            raise ValueError("四元数列表为空，请先调用update_quaternion_rk4方法更新四元数列表")
        if not hasattr(self, 'C') or len(self.C) == 0:
            raise ValueError("旋转矩阵列表为空，请先调用get_euler_angles_matrix方法获取旋转矩阵列表")
        acc_e = [] # 地理坐标系下的加速度
        acc_norms = [] # 存储加速度幅值
        for i in range(0,len(self.time)):
            acc = np.dot(self.C[i],self.acc_data[i]) # 将加速度从载体坐标系转换到地理坐标系
            acc_e.append(acc) # 存储每时刻地理加速度的值
            acc_norms.append(np.linalg.norm(acc)) # 加速度归一化
        
        velocity_e = [] # 地理坐标系下的速度
        velocity_e.append(self.initial_velocity)
        for i in range(1,len(self.time)):
            velocity_e.append(velocity_e[i-1]+(acc_e[i-1]+acc_e[i])*self.dt[i-1]/2) # 梯形积分公式更新速度值
        
        # 末尾速度修正
        velocity_ec = []
        velocity_ec.append(velocity_e[0])
        for i in range(1,len(self.time)):
            velocity_ec.append(velocity_e[i]-velocity_e[-1]*i/(len(self.time)-1)) # 末尾速度修正

        velocity_e = velocity_ec
        position_e = [] # 地理坐标系下的位置
        position_e.append(self.initial_position)
        for i in range(1,len(self.time)):
            position_e.append(position_e[i-1]+(velocity_e[i-1]+velocity_e[i])*self.dt[i-1]/2) # 梯形积分公式更新位置值
        return velocity_e, position_e
        
    def is_ZVC(self, acc, acc_ZVC = 0.0289):
        '''
        判断是否为使用零速度补偿
        当加速度幅值小于acc_ZVC时认为是使用零速度补偿
        '''
        if np.linalg.norm(acc) < acc_ZVC:
            return True
        else:
            return False
        
    def ZVC(self):
        '''
        零速度补偿法
        '''
        if not hasattr(self, 'Q') or len(self.Q) == 0:
            raise ValueError("四元数列表为空，请先调用update_quaternion_rk4方法更新四元数列表")
        if not hasattr(self, 'C') or len(self.C) == 0:
            raise ValueError("旋转矩阵列表为空，请先调用get_euler_angles_matrix方法获取旋转矩阵列表")
        acc_e = [] # 地理坐标系下的加速度
        for i in range(0,len(self.time)):
            acc_e.append(np.dot(self.C[i],self.acc_data[i])-self.G)
        # 零速度补偿 加速度
        velocity_e = [] # 地理坐标系下的速度
        velocity_e.append(self.initial_velocity)
        for i in range(1,len(self.time)):
            if self.is_ZVC(acc_e[i]): # 如果使用零速度补偿
                velocity_e.append(np.zeros(3))
            else:
                velocity_e.append(velocity_e[i-1]+(acc_e[i-1]+acc_e[i])*self.dt[i-1]/2)
        # 末尾速度修正
        velocity_ec = []
        velocity_ec.append(velocity_e[0])
        for i in range(1,len(self.time)):
            velocity_ec.append(velocity_e[i]-velocity_e[-1]*i/(len(self.time)-1))
        velocity_e = velocity_ec
        position_e = [] # 地理坐标系下的位置
        position_e.append(self.initial_position)
        for i in range(1,len(self.time)):
            position_e.append(position_e[i-1]+(velocity_e[i-1]+velocity_e[i])*self.dt[i-1]/2)
        return velocity_e, position_e
