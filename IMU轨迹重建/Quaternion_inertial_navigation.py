"""
使用惯性导航算法进行轨迹还原
INS
    包括姿态估计、位置估计、速度估计、轨迹重建
    包括普通惯性导航算法和基于零速更新的惯性导航算法 算法均使用零速补偿
    根据论文《Indoor Pedestrian Navigation using an INS/EKF framework for Yaw Drift Reduction and a Foot-mounted IMU》完成零速更新
    具体看函数ZUPT_ZARU_HDR_Compass_EKF_INS_1() 注意的是函数中有些地方的实现和论文中对应的数学式子有些不同 估计是论文错了 代码的结果是没问题的
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.spatial.transform import Rotation as R
import time
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import Trajectory_drawing as TD
import sys
sys.path.append(r'C:\Users\13106\Desktop\code\IMU\MyIMUCode\github\IMU去噪算法\传统方法')
import EMD_WT.WT as WT
import EMD_WT.EMD_WT as EMD_WT
import signal_segement

class INS:
    '''
    惯性导航算法
    包括姿态估计、位置估计、速度估计、轨迹重建
    '''
    def __init__(self, 
                acc_data, 
                gyro_data, 
                mag_data,
                time,
                linear_acc_data=None,
                start_time = 200,
                G = [0,0,9.81], 
                initial_position=np.zeros(3), 
                initial_velocity=np.zeros(3), 
                if_linear_acc = False,
                ):
        '''
        acc_data: 加速度数据 二维数组 每行为一个时刻的加速度值 每列为一个轴的加速度值 (N,3)
        gyro_data: 角速度数据 二维数组 每行为一个时刻的角速度值 每列为一个轴的角速度值 (N,3)
        mag_data: 磁力数据 二维数组 每行为一个时刻的磁力值 每列为一个轴的磁力值 (N,3)
        time: 时间数据 一维数组 每行为一个时刻的时间值 单位为s
        initial_position: 初始位置 三维数组 默认为[0,0,0]
        initial_velocity: 初始速度 三维数组 默认为[0,0,0]
        acc_mean: 加速度数据的均值 三维数组
        G: 重力加速度 默认为9.7926m/s^2
        if_linear_acc: 是否使用线性加速度数据进行惯性导航 默认为False
        '''
        self.acc_data = acc_data  # 加速度数据 (N,3)
        self.linear_acc_data = linear_acc_data # 线性加速度数据 (N,3) 即加速度数据减去重力加速度
        self.gyro_data = gyro_data # 角速度数据 (N,3)
        self.mag_data = mag_data # 磁力计数据 (N,3)
        self.time = time # 时间 (N, )
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity
        self.if_linear_acc = if_linear_acc
        self.g = 9.81
        self.G = [0,0,self.g] # 重力加速度
        self.linear_acc_data_mean = np.mean(np.array(linear_acc_data[0:start_time,:]),axis = 0) if self.if_linear_acc else None # 初始线性加速度均值
        self.start_acc_mean = np.mean(np.array(acc_data[0:start_time,:]),axis = 0) # 初始加速度均值
        self.gyro_data_mean = np.mean(np.array(gyro_data[0:start_time,:]),axis = 0) # 初始角速度均值 
        _,_,_,self.initial_quaternion,_ = self.estimate_initial_quaternion_from_linear_acc() if self.if_linear_acc else [None,None,None,None,None] # 初始线性加速度均值
        _,self.initial_quaternion = self.estimate_initial_quaternion_from_acc() if self.initial_quaternion is None else [None,self.initial_quaternion] # 初始四元数
        self.Q = [] # 存储每个时刻的四元数
        self.C = [] # 存储每个时刻的旋转矩阵
    
    def estimate_initial_quaternion_from_acc(self):
        '''
        根据初始加速度数据估计初始方向角
        acc_mean: 初始加速度数据的均值 三维数组
        返回值: 方向角
        '''
        ax, ay, az = self.start_acc_mean
        g = np.sqrt(ax**2 + ay**2 + az**2)
        pitch = -np.arcsin(ax/g)
        roll = np.arctan(ay/az)
        yaw = 0
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)

        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = cy * sp * cr + sy * cp * sr
        qz = sy * cp * cr - cy * sp * sr
        C = np.array([
            [np.cos(pitch)*np.cos(yaw), (np.sin(roll)*np.sin(pitch)*np.cos(yaw))-(np.cos(roll)*np.sin(yaw)), (np.cos(roll)*np.sin(pitch)*np.cos(yaw))+(np.sin(roll)*np.sin(yaw))],
            [np.cos(pitch)*np.sin(yaw), (np.sin(roll)*np.sin(pitch)*np.sin(yaw))+(np.cos(roll)*np.cos(yaw)), (np.cos(roll)*np.sin(pitch)*np.sin(yaw))-(np.sin(roll)*np.cos(yaw))],
            [-np.sin(pitch), np.sin(roll)*np.cos(pitch), np.cos(roll)*np.cos(pitch)]
            ])
        return C,np.array([qw, qx, qy, qz])

    def estimate_initial_quaternion_from_linear_acc(self):
        '''
        根据初始线性加速度数据估计初始四元数
        acc_mean: 初始加速度数据的均值 三维数组
        返回值: 初始四元数 三维数组 [w, x, y, z]
        '''
        ax, ay, az = self.linear_acc_data_mean 
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
        C = np.array([
            [np.cos(pitch)*np.cos(yaw), (np.sin(roll)*np.sin(pitch)*np.cos(yaw))-(np.cos(roll)*np.sin(yaw)), (np.cos(roll)*np.sin(pitch)*np.cos(yaw))+(np.sin(roll)*np.sin(yaw))],
            [np.cos(pitch)*np.sin(yaw), (np.sin(roll)*np.sin(pitch)*np.sin(yaw))+(np.cos(roll)*np.cos(yaw)), (np.cos(roll)*np.sin(pitch)*np.sin(yaw))-(np.sin(roll)*np.cos(yaw))],
            [-np.sin(pitch), np.sin(roll)*np.cos(pitch), np.cos(roll)*np.cos(pitch)]
            ])

        return pitch,roll,yaw,np.array([qw , qx, qy, qz]), C  # 注意格式：[w, x, y, z]

    def update_quaternion_rk4(self):
        '''
        使用Runge-Kutta方法更新四元数
        使用陀螺仪更新法更新四元数
        '''
        def quat_derivative(q,omega):
            '''
            四元数微分方程
            q: 四元数 三维数组 [w,x,y,z]
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
            new_Q = new_Q/np.linalg.norm(new_Q) # 会产生归一化误差
            self.Q.append(new_Q)
        self.dt = dt
        return

    def bortz_update_quaternion_rk4(self):
        '''
        使用等效旋转向量更新四元数
        避免四元数的归一化误差
        先求解等效旋转矢量Bortz方程的RK4解 再利用等效旋转矢量更新四元数
        '''

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
        if self.if_linear_acc == True and not hasattr(self, 'linear_acc_data'):
            raise ValueError("使用线性加速度数据进行惯性导航，请先传入线性加速度数据 self.linear_acc_data")
        if not hasattr(self, 'Q') or len(self.Q) == 0:
            raise ValueError("四元数列表为空，请先调用update_quaternion_rk4方法更新四元数列表")
        if not hasattr(self, 'C') or len(self.C) == 0:
            raise ValueError("旋转矩阵列表为空，请先调用get_euler_angles_matrix方法获取旋转矩阵列表")
        acc_e = [] # 地理坐标系下的加速度
        acc_norms = [] # 存储加速度幅值
        for i in range(0,len(self.time)):
            if not self.if_linear_acc: 
                acc = np.dot(self.C[i],self.acc_data[i])-self.G # 将加速度从载体坐标系转换到地理坐标系
            else:
                acc = np.dot(self.C[i],self.linear_acc_data[i]) # 将加速度从载体坐标系转换到地理坐标系
            acc_e.append(acc) # 存储每时刻地理加速度的值
            acc_norms.append(np.linalg.norm(acc)) # 加速度归一化
        
        velocity_e = [] # 地理坐标系下的速度
        velocity_e.append(self.initial_velocity)
        for i in range(1,len(self.time)):
            velocity_e.append(velocity_e[i-1]+(acc_e[i-1]+acc_e[i])*self.dt[i-1]/2) # 梯形积分公式更新速度值
        
        # # 末尾速度修正
        # velocity_ec = []
        # velocity_ec.append(velocity_e[0])
        # for i in range(1,len(self.time)):
        #     velocity_ec.append(velocity_e[i]-velocity_e[-1]*i/(len(self.time)-1)) # 末尾速度修正
        # velocity_e = velocity_ec
        
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
            if not self.if_linear_acc: 
                acc_e.append(np.dot(self.C[i],self.acc_data[i])-self.G) # 将加速度从载体坐标系转换到地理坐标系
            else:
                acc_e.append(np.dot(self.C[i],self.linear_acc_data[i])) # 将加速度从载体坐标系转换到地理坐标系
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
    
    def angle_diff_between_quaternions(q1, q2):
        # 三轴角度差
        # 旋转向量
        # 计算两个四元数之间的相对旋转
        r = R.from_quat(q2) * R.from_quat(q1).inv()
        # 将相对旋转转换为欧拉角
        euler_angles = r.as_euler('xyz', degrees=True)
        # 计算角度差的总和
        angle_diff = np.sqrt(np.sum(np.square(euler_angles)))
        return angle_diff # 返回角度差

    def ZUPT_Detection_1(self,tha_min=9,tha_max=11, s=15, tha=3, thw_max=0.5,window_size=399)->list:
        '''
        进行零速检测
        tha_min: 最小加速度幅值
        tha_max: 最大加速度幅值
        s: 滑动窗口大小
        tha: 局部加速度方差阈值
        thw_max: 最大角速度幅值
        window: 中值滤波滑动窗口
        返回：一维列表，反映每一个时刻是否为零速 是为1 否为0
        '''
        acc_norm_detection = []
        gyro_norm_detection = []
        for i in range(0,len(self.acc_data)):
            if np.linalg.norm(self.acc_data[i]) < tha_max and np.linalg.norm(self.acc_data[i]) > tha_min:
                acc_norm_detection.append(1)
            else:
                acc_norm_detection.append(0)
            if np.linalg.norm(self.gyro_data[i]) < thw_max:
                gyro_norm_detection.append(1)
            else:
                gyro_norm_detection.append(0)

        local_acc_var = []
        for i in range(s,len(self.acc_data)-s):
            acc_var = np.var(self.acc_data[i-s:i+s])
            if acc_var > tha:
                local_acc_var.append(1)
            else:
                local_acc_var.append(0)
        local_acc_var = [local_acc_var[0]]*s + local_acc_var + [local_acc_var[-1]]*s 

        if len(local_acc_var) != len(acc_norm_detection):
            raise ValueError("acc_norm_detection和local_acc_var长度不一致 程序出错")
        is_ZUPT = []
        for i in range(0,len(local_acc_var)):
            if local_acc_var[i] == 1 and acc_norm_detection[i] == 1 and gyro_norm_detection[i] == 1:
                is_ZUPT.append(1)
            else:
                is_ZUPT.append(0)
        
        # 假设a是加速度数据，b是零速点标记(0或1的列表)
        is_ZUPT = medfilt(is_ZUPT, kernel_size=window_size)
        b = is_ZUPT
        a = self.acc_data[:,0]  # 假设加速度数据是一维数组，每个元素对应一个时间点的加速度值
        plt.figure(figsize=(12, 6))
        plt.plot(a, label='加速度数据')  # 绘制加速度曲线

        # # 在零速点位置标记红点
        # zero_vel_indices = [i for i, val in enumerate(b) if val == 1]
        # plt.scatter(zero_vel_indices, [a[i] for i in zero_vel_indices], 
        #         color='red', label='零速点', zorder=5)

        # plt.xlabel('时间/采样点')
        # plt.ylabel('加速度值')
        # plt.title('加速度数据与零速点标记')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
         # 提取区间
        stationary_intervals = []
        moving_intervals = []
        in_stationary = in_moving = False
        start_idx = 0

        for i, val in enumerate(is_ZUPT):
            if val == 1 and not in_stationary:
                in_stationary = True
                start_idx = i
            elif val == 0 and in_stationary:
                if i - start_idx > 50:
                    stationary_intervals.append((start_idx, i - 1))
                in_stationary = False

            if val == 0 and not in_moving:
                in_moving = True
                start_idx = i
            elif val == 1 and in_moving:
                if i - start_idx > 50:
                    moving_intervals.append((start_idx, i - 1))
                in_moving = False

        if in_stationary and len(is_ZUPT) - start_idx > 50:
            stationary_intervals.append((start_idx, len(is_ZUPT) - 1))
        if in_moving and len(is_ZUPT) - start_idx > 50:
            moving_intervals.append((start_idx, len(is_ZUPT) - 1))
        
        return is_ZUPT, stationary_intervals, moving_intervals
    
    def ZUPT_Detection_2(self,  acc_stationary_threshold_H=11, acc_stationary_threshold_L=9, gyro_stationary_threshold=0.6, window_size=10):
        """
        Detect stationary periods based on accelerometer and gyroscope magnitudes.
        
        Parameters:
        - acc_mag: Acceleration magnitudes (numpy array).
        - gyro_mag: Gyroscope magnitudes (numpy array).
        - acc_stationary_threshold_H: High threshold for accelerometer.
        - acc_stationary_threshold_L: Low threshold for accelerometer.
        - gyro_stationary_threshold: Threshold for gyroscope.
        - window_size: Size of the window for cleaning false stance detection.

        Returns:
        - stationary: 1D numpy array with 1 for stationary and 0 for non-stationary.
        """
        acc_mag=np.linalg.norm(self.acc_data, axis=1)
        gyro_mag=np.linalg.norm(self.gyro_data, axis=1)
        # Detect acceleration based on thresholds
        stationary_acc_H = (acc_mag < acc_stationary_threshold_H)
        stationary_acc_L = (acc_mag > acc_stationary_threshold_L)
        stationary_acc = np.logical_and(stationary_acc_H, stationary_acc_L)  # C1

        # Detect gyro stationary periods
        stationary_gyro = (gyro_mag < gyro_stationary_threshold)  # C2

        # Combine both to get stationary periods
        stationary = np.logical_and(stationary_acc, stationary_gyro)

        # Convert boolean values (True/False) to (1/0)
        stationary = stationary.astype(int)

        # Clean false stance detections using a sliding window approach
        for k in range(len(stationary) - window_size + 1):
            if np.all(stationary[k:k + window_size] == 1):
                stationary[k:k + window_size] = 1
            elif np.all(stationary[k:k + window_size] == 0):
                stationary[k:k + window_size] = 0

        # Extract non-zero-velocity intervals longer than 50
        moving_intervals = []
        in_motion = False
        start_idx = 0

        for i, val in enumerate(stationary):
            if val == 0 and not in_motion:
                in_motion = True
                start_idx = i
            elif val == 1 and in_motion:
                if i - start_idx > 50:
                    moving_intervals.append((start_idx, i - 1))
                in_motion = False
        if in_motion and len(stationary) - start_idx > 50:
            moving_intervals.append((start_idx, len(stationary) - 1))
        
        b = stationary
        a = self.acc_data[:,0]  # 假设加速度数据是一维数组，每个元素对应一个时间点的加速度值
        plt.figure(figsize=(12, 6))
        plt.plot(a, label='加速度数据')  # 绘制加速度曲线

        # 在零速点位置标记红点
        zero_vel_indices = [i for i, val in enumerate(b) if val == 1]
        plt.scatter(zero_vel_indices, [a[i] for i in zero_vel_indices], 
                color='red', label='零速点', zorder=5)

        plt.xlabel('时间/采样点')
        plt.ylabel('加速度值')
        plt.title('加速度数据与零速点标记')
        plt.legend()
        plt.grid(True)
        plt.show()
        return stationary, moving_intervals

    def skew_symmetric_matrix(self, v):
        '''
        计算向量的反对称矩阵
        v: 三维向量
        '''
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])

    def ZUPT_ZARU_HDR_Compass_EKF_INS_Matrix_Initialization(self, if_ZARU, if_HDR, if_Compass, para, dt_rate, Md):
        '''
        初始化矩阵
        if_ZARU: 是否使用零角速度修正
        if_HDR: 是否使用航向角修正
        if_Compass: 是否使用磁力计校正
        para: 是否使用参数来
        dt_rate: 是否使用dt_rate
        '''
        print("")

    def ZUPT_ZARU_HDR_Compass_EKF_INS_1(self, if_ZARU, if_HDR, if_Compass, initial_quaternion_from_acc=True, if_acc = True, para = False, Md = -2, dt_rate = False, tha_min=9,tha_max=11, s=15, tha=3, thw_max=0.6,window_size=7):

        '''
        融合ZUPT_ZARU_HDR_Compass_EKF的惯性导航算法
        if_ZARU: 是否使用零角速度修正
        if_HDR: 是否使用航向角修正
        if_Compass: 是否使用磁力计校正
        # 已经完成了 代码只需要优化一下初始值的选取就可以了
        '''
        # 默认进行零速度检测
        ## 加载初始加速度偏置
        if not if_acc:
            acc_bias = self.linear_acc_data_mean # 将加速度偏置设置为初始加速度均值 大概1秒
        else:
            acc_bias = [0,0,0]
        is_ZUPT, stationary, _= self.ZUPT_Detection_1() # 进行零速检测
        if len(is_ZUPT)!= len(self.time):
            raise ValueError("is_ZUPT和time长度不一致 程序出错")

        # # 使用小波变换对moving_intervals进行去噪
        # for i in range(0,len(moving_intervals)):
        #     start, end = moving_intervals[i]
        #     for j in range(0,3):
        #         self.acc_data[start:end,j] = WT.simple_WT(self.acc_data[start:end,j],level=4)[:end-start]


        # 角速度偏置
        gyro_bias = self.gyro_data_mean

        Md = Md

        # 根据不同情况初始化矩阵
        ## 初始化矩阵Q 非零值对应的是加速度计和角速度的方差
        Q = np.diag([1e-2,1e-2,1e-2,0,0,0,0,0,0,1e-1,1e-1,1e-1,0,0,0]) # shape(15,15)
        ## 初始化矩阵P
        P = np.diag([0,0,0,1e-2,1e-2,1e-2,0,0,0,0,0,0,1e-2,1e-2,1e-2]) # shape(15,15)
        # P = np.zeros((15,15))
        ## 初始化方向旋转矩阵
        # pitch,roll,yaw,q0 = self.estimate_initial_quaternion_from_acc()
        # q0 = np.array([q0[1],q0[2],q0[3],q0[0]])
        # C0 = R.from_quat(q0).as_matrix()
        if initial_quaternion_from_acc:
            C0,_ = self.estimate_initial_quaternion_from_acc()
        else:
            _,_,_,_,C0 = self.estimate_initial_quaternion_from_linear_acc()
        self.C.append(C0)
        yaw = 0
        ## 是否使用零角速度检测
        if if_ZARU and not if_HDR: # 使用零角速度修正
            R0 = np.diag([0.1,0.1,0.1,0.01,0.01,0.01])
            H_block1 = np.block([np.zeros((3, 3)), np.eye(3), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))])
            H_block2 = np.block([np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))])
            H = np.vstack([H_block1, H_block2])
        else: # 不使用零角速度修正
            R0 = np.diag([0.01,0.01,0.01])**2
            H = np.block([np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))])

        if if_HDR and not if_ZARU: # 使用航向角修正
            R0 = np.diag([0.1,0.01,0.01,0.01])
            H_row1 = np.block([np.array([[0, 0, 1]]), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3))])  # 第1行
            H_block2 = np.block([np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))])
            H = np.vstack([H_row1, H_block2])
            Yaw = []
            Yaw.append(yaw)
            step_samples = []
        if if_ZARU and if_HDR: # 使用零角速度修正和航向角修正
            R0 = np.diag([0.1,0.1,0.1,0.1,0.01,0.01,0.01])
            H_row1 = np.block([np.array([[0, 0, 1.5]]), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3))])  # 第1行
            H_block2 = np.block([np.zeros((3, 3)), np.eye(3), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))])            # 第2-4行
            H_block3 = np.block([np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))])            # 第5-7行
            H = np.vstack([H_row1, H_block2, H_block3]) # shape(7,15)
            Yaw = []
            Yaw.append(yaw)
            step_samples = []
        if if_ZARU and if_HDR and if_Compass: # 使用零角速度修正和航向角修正和磁力计校正
            R0 = np.diag([0.1,0.1,0.1,0.1,0.01,0.01,0.01])
            H_row1 = np.block([np.array([[0, 0, 1]]), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3))])  # 第1行
            H_block2 = np.block([np.zeros((3, 3)), np.eye(3), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))])            # 第2-4行
            H_block3 = np.block([np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))])            # 第5-7行
            H = np.vstack([H_row1, H_block2, H_block3]) # shape(7,15)
            Yaw = []
            Yaw.append(yaw)
            step_samples = []
            if para == False:
                raise ValueError("请输入磁力计校正参数")
        print(f"H shape: {H.shape}")
        print(f"P shape: {P.shape}") 
        print(f"R0 shape: {R0.shape}")

        ## 初始化误差状态向量
        d_x_k = np.zeros(15) # shape(15,1) [d_psi,d_gyro_bias,d_r,d_v,d_acc_bias]

        ## 初始化加速度偏置和角速度偏置
        d_x_k[3:6] = gyro_bias
        d_x_k[12:15] = acc_bias

        ## 初始化角速度和加速度
        gyro_b = [] # b坐标系向下的角速度
        acc_b = [] # b坐标系下的加速度
        acc_e = [] # e坐标系下的加速度
        v_k_k_1 = [] # e坐标系下的速度 但未经过EKF更新
        velocity_e = [] # e坐标系下的速度 经过EKF更新
        position_e = [] # e坐标系下的位置
        gyro_b.append(self.gyro_data[0]-d_x_k[3:6]) # 初始角速度
        if not if_acc: # 使用线性加速度
            acc_b.append(self.linear_acc_data_mean-d_x_k[12:15]) # 初始加速度
        else: # 不使用线性加速度
            acc_b.append(self.acc_data[0]-d_x_k[12:15]) # 初始加速度
        if not if_acc: # 使用线性加速度
            acc_e.append(np.dot(C0,acc_b[0])) # 初始加速度
        else: # 不使用线性加速度
            acc_e.append(np.dot(C0,acc_b[0])-self.G) # 初始加速度
        v_k_k_1.append(self.initial_velocity) # 初始速度
        velocity_e.append(self.initial_velocity) # 初始速度
        position_e.append(self.initial_position) # 初始位置
        
        m = []

        def calculate_delta_psi(current_sample, step_samples, heading_history,th_psi = 0.0003):
            # 计算当前时刻与前 step_samples 个时刻的航向角差
            if len(step_samples) < 3:
                return 0
            ks = step_samples[-1]
            ks_1 = step_samples[-2]
            ks_2 = step_samples[-3]
            psi_k = heading_history[current_sample]
            avg_psi = (heading_history[current_sample-(ks-ks_1)]+heading_history[current_sample-(ks-ks_2)])/2
            if np.abs(psi_k-avg_psi)<=th_psi:
                return psi_k-avg_psi
            else:
                return 0

        for k in range(1,len(self.time)):
            # 时间间隔
            if not dt_rate:
                dt = self.time[k]-self.time[k-1]
            else:
                dt = (self.time[k]-self.time[k-1])*(1/dt_rate)
            
            # first phase 去除偏置
            ## 去除角速度偏置
            gyro_b.append(self.gyro_data[k]-d_x_k[3:6])
            ## 去除加速度偏置
            if not if_acc:
                acc_b.append(self.linear_acc_data[k]-d_x_k[12:15])
            else: 
                acc_b.append(self.acc_data[k]-d_x_k[12:15])

            # second phase 更新传感器方向 使用pade approximation of the exp function
            C_k_k_1 = self.C[k-1]@(2*np.eye(3)+self.skew_symmetric_matrix(gyro_b[k])*dt)@np.linalg.inv(2*np.eye(3)-self.skew_symmetric_matrix(gyro_b[k])*dt)

            # third phase 将加速度转移到e坐标系上
            if not if_acc:
                acc_s = np.dot(0.5*(self.C[k-1]+C_k_k_1),acc_b[k])
                acc_e.append(np.dot(0.5*(self.C[k-1]+C_k_k_1),acc_b[k]))
            else:
                acc_s = np.dot(0.5*(self.C[k-1]+C_k_k_1),acc_b[k])
                acc_e.append(np.dot(0.5*(self.C[k-1]+C_k_k_1),acc_b[k])-self.G)

            # fourth phase 计算速度
            v_k_k_1.append(velocity_e[k-1]+(acc_e[k-1]+acc_e[k])*dt/2) # 二次积分
            # v_k_k_1.append(velocity_e[k-1]+acc_e[k]*dt) # 一次积分
            # fifth phase 计算位置
            r_k_k_1 = (position_e[k-1]+(velocity_e[k-1]+v_k_k_1[k])*dt/2) # 二次积分
            # r_k_k_1 = (position_e[k-1]+v_k_k_1[k]*dt) # 一次积分

            # 状态转移矩阵
            F1 = np.block([np.eye(3), dt*C_k_k_1, np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))])
            F2 = np.block([np.zeros((3,3)), np.eye(3), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))])
            F3 = np.block([np.zeros((3,3)), np.zeros((3,3)), np.eye(3), dt*np.eye(3), np.zeros((3,3))])
            F4 = np.block([-1*dt*self.skew_symmetric_matrix(acc_s), np.zeros((3,3)), np.zeros((3,3)), np.eye(3), dt*C_k_k_1])
            F5 = np.block([np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.eye(3)])
            F = np.vstack([F1,F2,F3,F4,F5]) # shape(15,15)

            P = F@P@F.T + Q # shape(15,15)

            if if_HDR: # 使用航向角修正
                euler = R.from_matrix(C_k_k_1).as_euler('xyz', degrees=True) # shape(3,1)
                roll = euler[0]
                pitch = euler[1]
                yaw = euler[2]
                Yaw.append(yaw)
                d_psi = calculate_delta_psi(k, step_samples, Yaw)
                print(d_psi)
            if if_HDR and if_Compass: # 使用航向角修正和磁力计校正
                euler = R.from_matrix(C_k_k_1).as_euler('xyz', degrees=True) # shape(3,1)
                roll = euler[0]
                pitch = euler[1]
                yaw = euler[2]
                Yaw.append(yaw)
                R_y = np.array([np.cos(pitch), 0, -np.sin(pitch), 0, 1, 0, -np.sin(pitch), 0, np.cos(pitch)]).reshape(3,3)
                R_x = np.array([1, 0, 0, 0, np.cos(roll), -np.sin(roll), 0, np.sin(roll), np.cos(roll)]).reshape(3,3)
                B_n = R_y@R_x@self.mag_data[k]
                psi_compass = -np.arctan2(B_n[2],B_n[1])-Md
                d_psi = (1-para)*calculate_delta_psi(k, step_samples, Yaw)+para*(Yaw[k]-psi_compass)
            
            if is_ZUPT[k] == 1: # 零速
                m = []
                for i in range(3):
                    m.append(v_k_k_1[k][i])
                if if_ZARU and not if_HDR: # 使用零角速度修正
                    m = []
                    for i in range(3):
                        m.append(gyro_b[k][i]) # mk = gyro_b-[0,0,0]
                    for i in range(3):
                        m.append(v_k_k_1[k][i]) # mk = v_k_k_1-[0,0,0]
                if if_HDR and not if_ZARU: # 使用航向角修正
                    m = []
                    step_samples.append(k)
                    m.append(d_psi)
                    for i in range(3):
                        m.append(v_k_k_1[k][i]) # mk = v_k_k_1-[0,0,0]
                if if_ZARU and if_HDR: # 使用零角速度修正和航向角修正
                    m = []
                    step_samples.append(k)
                    m.append(d_psi)
                    for i in range(3):
                        m.append(gyro_b[k][i]) # mk = gyro_b-[0,0,0]
                    for i in range(3):
                        m.append(v_k_k_1[k][i]) # mk = v_k_k_1-[0,0,0]
                if if_ZARU and if_HDR and if_Compass: # 使用航向角修正和磁力计校正
                    m = []
                    step_samples.append(k)
                    m.append(d_psi)
                    for i in range(3):
                        m.append(gyro_b[k][i]) # mk = gyro_b-[0,0,0]
                    for i in range(3):
                        m.append(v_k_k_1[k][i]) # mk = v_k_k_1-[0,0,0]
                m = np.array(m)
                K = P@H.T@np.linalg.inv(H@P@H.T+R0)
                P = (np.eye(15)-K@H)@P@(np.eye(15)-K@H).T + K@R0@K.T # shape(15,15)
                d_x_k = K@m # shape(15,1)

                # 更新
                position_e.append(r_k_k_1-d_x_k[6:9]) # 位置
                velocity_e.append(v_k_k_1[k]-d_x_k[9:12]) # 速度
                self.C.append((2*np.eye(3)-self.skew_symmetric_matrix(d_x_k[0:3]))@np.linalg.inv(2*np.eye(3)+self.skew_symmetric_matrix(d_x_k[0:3]))@C_k_k_1) # 方向
                # 将d_x_k除了角速度和加速度偏置全部设置为0 因为误差已经被补偿了
                d_x_k[6:9] = [0,0,0]
                d_x_k[9:12] = [0,0,0]
                d_x_k[0:3] = [0,0,0]
                d_x_k = np.zeros(15)
            else: # 非零速
                position_e.append(r_k_k_1) # 位置
                velocity_e.append(v_k_k_1[k]) # 速度
                self.C.append(C_k_k_1) # 方向
        TD.plot_accelerations(acc_e)
        TD.plot_v(velocity_e)
        TD.xyz_position(np.array(position_e)[:,:])
        return velocity_e, np.array(position_e)

