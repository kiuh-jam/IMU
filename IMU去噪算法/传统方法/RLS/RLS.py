#递归最小二乘滤波算法
import numpy as np
import matplotlib.pyplot as plt

# 设置支持中文的字体，SimHei 是常见的中文字体
plt.rcParams['font.family'] = ['SimHei']  # 或者 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def rls(time_series_measurement: np.ndarray,time_series_target: np.ndarray,M: int,lambda_para=0.99,p_init=None):
    '''
    N = time_series_length
    time_series_measurement: ndarray, 输入信号矩阵
    time_series_target: ndarray, 目标信号
    M: int, 滤波器系数
    lambda_para: float, 遗忘因子
    p_init: ndarray, 初始化的协方差矩阵, 默认设为单位阵

    return
    w: ndarray 滤波器参数(M,)
    MSE: ndarray 每次迭代的均方差
    '''
    time_series_length = len(time_series_target)
    
    if time_series_length < M:
        print("信号长度小于滤波器阶数")
        return
    if time_series_length!=len(time_series_measurement):
        print("测量信号和期望信号长度不一致")
        return
    if lambda_para > 1 or lambda_para <= 0:
        print("遗忘参数设置错误")
        return
    
    #初始化权重和协方差矩阵
    # w = np.zeros((M,1))
    w = np.random.rand(M, 1)

    if p_init is None:
        p = np.eye(M)*1000 #未提供处置则使用单位矩阵乘以较大的值来初始化
    else:
        p = p_init        

    e = np.zeros(time_series_length)

    MSE = []

    y = np.zeros(time_series_length)

    #对每一个时刻进行计算
    for n in range(time_series_length-M):
        x_n = np.array(time_series_measurement[n:n+M][::-1]).reshape(M, 1)
        d_n = time_series_target[n]
        y[n] = np.dot(x_n.T, w)
        e_n = d_n - y[n]
        g = np.dot(p, x_n)
        k = g/(lambda_para+np.dot(x_n.T, g))
        w = w + e_n*k
        p = lambda_para*(p - np.dot(k, np.dot(x_n.T, p)))
        e[n] = e_n
        MSE.append(np.mean((time_series_target-y)**2))
    return y,w,MSE,e


if __name__ == "__main__":
    # 设置随机种子以保证结果可重复
    np.random.seed(0)

    # 参数设置
    N = 1000  # 信号长度
    M = 32    # 滤波器长度

    # 创建目标信号 (没有噪声)
    d = np.sin(2 * np.pi * 0.05 * np.arange(N))

    # 添加噪声
    noise = 0.5 * np.random.randn(N)
    x = d + noise  # 输入信号：目标信号 + 噪声

    y,w,MSE,e = rls(x,d,32)

    # 绘制结果
    plt.figure(figsize=(12, 6))

    # 目标信号
    plt.subplot(3, 1, 1)
    plt.plot(d, label='目标信号')
    plt.title('目标信号')
    plt.legend()

    # 带噪声信号
    plt.subplot(3, 1, 2)
    plt.plot(x, label='带噪声信号')
    plt.title('带噪声信号')
    plt.legend()

    # 去噪信号
    plt.subplot(3, 1, 3)
    plt.plot(y, label='去噪信号')
    plt.title('去噪信号')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(MSE[-1])


        