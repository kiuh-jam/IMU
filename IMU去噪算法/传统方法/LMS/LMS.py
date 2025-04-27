# 最小化均方误差自适应滤波算法，这是一种基于梯度下降法
# 的自适应滤波算法，目标是通过最小化输入信号和期望信号之间的
# 误差平方来调整滤波器的参数，该算法用于噪声抑制、回声消除等

# 滤波器n时刻的输出 与n时刻的输入信号有关的同时还与n时刻之前的M-1个时刻的输入有关

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置支持中文的字体，SimHei 是常见的中文字体
plt.rcParams['font.family'] = ['SimHei']  # 或者 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def LMS(time_series_measurement: np.ndarray,time_series_real: np.ndarray,miu: float,M: int,max_iter=1):
    '''
    time_series_measurement是测量的时间序列数据: array
    time_series_real是真实的时间序列数据: array
    miu是更新步长
    M是滤波器的阶数
    返回均方差、滤波器的参数和去噪之后的信号
    '''    
    #时间序列数据的长度
    time_series_length = len(time_series_measurement)

    if time_series_length < M:
        print("信号长度小于滤波器阶数")
        return
    if time_series_length!=len(time_series_real):
        print("测量信号和期望信号长度不一致")
        return

    MSE = []

    #初始化滤波器参数
    # w = np.zeros(M)
    w = np.random.rand(M)

    #滤波器输出
    y = np.zeros(time_series_length)

    #误差
    e = np.zeros(time_series_length)

    for n in range(M,time_series_length):
        
        #当前的输入信号
        xx = time_series_measurement[n-M:n][::-1]
        
        for i in range(max_iter):

            #计算滤波器输出
            y[n] = np.dot(w,xx)

            #计算误差
            e[n] = time_series_real[n] - y[n]

            #更新滤波器权重
            w = w + miu*e[n]*xx

    MSE.append(np.mean((time_series_real-y)**2))
        

    return MSE,w,y

# data = pd.read_csv(r'C:\Users\13106\Desktop\2024\传感器数据采集\已提交\4\X\Accelerometer.csv')
# x = np.array(data['Acceleration x (m/s^2)'].values)
# d = np.zeros(len(x))
# MSE,w,y = LMS(x,d,0.01,32,95)



# 设置随机种子以保证结果可重复
np.random.seed(0)

# 参数设置
N = 1000  # 信号长度
M = 32    # 滤波器长度
mu = 0.01 # 步长

# 创建目标信号 (没有噪声)
d = np.sin(2 * np.pi * 0.05 * np.arange(N))

# 添加噪声
noise = 0.5 * np.random.randn(N)
x = d + noise  # 输入信号：目标信号 + 噪声
MSElist = []
y= []
for i in range(100):
    i+=1
    y.append(i)
    MSE,_,_ = LMS(x,d,mu,32,i)
    MSElist.append(MSE)
plt.figure(figsize=(12,6))
plt.plot(y,MSElist,label='MSE')
plt.xlabel('iter')
plt.ylabel('MSE')


MSE,w,y = LMS(x,d,mu,32,100)

# 绘制结果
plt.figure(figsize=(12, 6))

# 目标信号
plt.subplot(3, 1, 1)
plt.plot(d, label='目标信号 d(n)')
plt.title('目标信号')
plt.legend()

# 带噪声信号
plt.subplot(3, 1, 2)
plt.plot(x, label='带噪声信号 x(n)')
plt.title('带噪声信号')
plt.legend()

# 去噪信号
plt.subplot(3, 1, 3)
plt.plot(y, label='去噪信号 y(n)')
plt.title('去噪信号')
plt.legend()

plt.tight_layout()
plt.show()

print(MSE[-1])