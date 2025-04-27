from filterpy.kalman import ExtendedKalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置支持中文的字体
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class EKF_Acceleration:
    def __init__(self, acc_data):
        """
        初始化EKF滤波器
        acc_data: x轴加速度数据(1D数组)
        """
        self.acc_data = acc_data
        self.ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)  # 状态: [位置,速度], 观测: 加速度
        
        # 初始化状态和协方差矩阵
        self.ekf.x = np.array([0., 0.])  # 初始状态 [位置,速度]
        self.ekf.P = np.diag([100, 100])  # 初始协方差矩阵
        
        # 过程噪声矩阵
        self.ekf.Q = np.diag([0.1, 0.1])
        
        # 观测噪声矩阵
        self.ekf.R = np.array([[1.]])
        
        # 观测函数
        self.ekf.H = np.array([[0, 1]])  # 观测到的是速度
        
        # 状态转移函数(非线性)
        def f(x, dt):
            return np.array([x[0] + x[1]*dt, x[1]])
        
        self.ekf.f = f
        
        # 观测函数(非线性)
        def h(x):
            return np.array([x[1]])
        
        self.ekf.h = h
        
        # 状态转移雅可比矩阵
        def F(x, dt):
            return np.array([[1, dt],
                            [0, 1]])
        
        self.ekf.F = F
        
        # 观测雅可比矩阵
        def H(x):
            return np.array([[0, 1]])
        
        self.ekf.HJacobian = H
    
    def process(self, dt=0.01):
        """
        处理加速度数据
        dt: 采样时间间隔(秒)
        """
        positions = []
        velocities = []
        
        for z in self.acc_data:
            # 更新状态转移矩阵F中的时间步长
            self.ekf.F = np.array([[1, dt],
                                  [0, 1]])
            self.ekf.predict()  # 移除dt参数
            self.ekf.update(np.array([z]), self.ekf.HJacobian, self.ekf.h)
            
            positions.append(self.ekf.x[0])
            velocities.append(self.ekf.x[1])
        
        return np.array(positions), np.array(velocities)

def EKF_func(acc_data):
    """
    初始化EKF滤波器
    acc_data: x轴加速度数据(1D数组)
    """
    acc_data = np.array(acc_data)
    ekf = EKF_Acceleration(acc_data)
    positions, velocities = ekf.process(dt=0.01)  # 假设采样率为100Hz
    return velocities
# 使用示例
if __name__ == '__main__':
    # 加载加速度数据(替换为您的实际数据路径)
    file_path = r'C:\Users\13106\Desktop\code\IMU\MyIMUCode\IMU去噪算法\深度学习方法\LSTM-RNN\data\Accelerometer.csv'
    data = pd.read_csv(file_path)
    acc_x = data['Acceleration x (m/s^2)'].values
    
    # 创建并运行EKF
    ekf = EKF_Acceleration(acc_x)
    positions, velocities = ekf.process(dt=0.01)  # 假设采样率为100Hz
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    plt.plot(acc_x, 'r-', label='原始加速度')
    plt.plot(velocities, 'b-', label='EKF估计速度')
    plt.xlabel('时间步长')
    plt.ylabel('值')
    plt.title('EKF处理x轴加速度数据')
    plt.legend()
    plt.grid(True)
    plt.show()