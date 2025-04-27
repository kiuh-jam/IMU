# 卡尔曼滤波
from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from scipy.stats import norm, shapiro
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import normaltest
import pandas as pd
import matplotlib.pyplot as plt

# 根据论文Research on the Compensation in MEMS Gyroscope Random Drift 
# Based on Time- Series Analysis and Kalman Filtering进行

# data acquistion 
# 获取数据

class ARMA_Kalman_Filter:
    # ARMA结合卡尔曼滤波
    def __init__(self, time_series_measurement):
        '''
        输入是传感器时间序列数据
        '''
        self.time_series_measurement = time_series_measurement
        self.time_series_length = len(time_series_measurement)
    
    # time_series_preprocessing
    def zero_mean(self):
        '''
        zero mean
        '''
        print("zero mean action")
        mean_value = np.mean(self.time_series_measurement)
        self.time_series_measurement = self.time_series_measurement-mean_value
        print("zero mean over")

    def smooth(self, lags=40):
        '''
        去除趋势项使得时间序列平滑
        '''
        print("smooth action")
        plt.figure(figsize=(10,6))
        plot_acf(self.time_series_measurement, lags=lags)
        plt.title("ACF of Original Data")
        plt.show()
        choice = int(input("请输入是否差分 1表示进行差分"))

        while choice == 1:
            self.time_series_measurement = np.diff(self.time_series_measurement)
            
            print("差分完毕")
            diff_data = self.time_series_measurement
            # 绘制差分后的ACF图
            plt.figure(figsize=(10, 6))
            plot_acf(diff_data, lags=lags)
            plt.title("ACF of First Order Differenced Data")
            plt.show()

            choice = int(input("请输入是否差分 1表示进行差分"))
        print("smooth over")

    def stationary_test(self, stationary_value=0.05):
        '''
        平稳性检验
        '''
        print("stationary_test action")
        result = adfuller(self.time_series_measurement)
        while result[1] > stationary_value:
            print("diff action")
            self.time_series_measurement = np.diff(self.time_series_measurement)
            result = adfuller(self.time_series_measurement)
        print("stationary_test over")

    def normality_test(self):
        print("normality_test action")
        
        # 使用 D'Agostino's K-squared 检验（normaltest）进行正态性检验
        stat, p_value = normaltest(self.time_series_measurement)
        print(f"D'Agostino's K-squared 检验统计量={stat}, p值={p_value}")
    
        if p_value<0.05:
            print("数据与正态分布存在显著差异")
        else:
            print("通过正态分布检验")
        # 可视化
        plt.figure(figsize=(10,6))
        sns.histplot(self.time_series_measurement, color='skyblue', kde=False)
        
        # 拟合并绘制正态分布曲线
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, np.mean(self.time_series_measurement), np.std(self.time_series_measurement))
        plt.plot(x, p, 'k', linewidth=2)
        
        # 添加标题和标签
        plt.title('Histogram and Normal Distribution Fit', fontsize=15)
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        
        # 添加网格线
        plt.grid(True)
        
        # 显示图表
        plt.show()
        
        print("normality_test over")

    
    def AIC_BIC(self, lags=20):
        plt.figure(figsize=(10,6))

        plt.subplot(121)
        plot_acf(self.time_series_measurement, lags=lags, ax=plt.gca(), title='ACF')
        plt.subplot(122)
        plot_pacf(self.time_series_measurement, lags=lags, ax=plt.gca(), title='PACF')
        plt.show()

        self.p = int(input("请输入自回归模型可能的最大阶数p"))
        self.q = int(input("请输入滑动平均模型可能的最大阶数q"))

    def fit_ARMA(self,p,q):
        model = ARIMA(self.time_series_measurement, order=(p,self.d ,q))
        fit_model = model.fit(method='innovations_mle')
        return fit_model.aic,fit_model.bic

    def ARMA_model(self,d=0):
        self.d = d
        result = []
        for p in range(self.p):
            for q in range(self.q):
                aic, bic = self.fit_ARMA(p,q)
                result.append((p,q,aic,bic))

        results_df = pd.DataFrame(result, columns=['p', 'q', 'AIC', 'BIC'])
        best_model_aic = results_df.loc[results_df['AIC'].idxmin()]
        best_model_bic = results_df.loc[results_df['BIC'].idxmin()]

        if best_model_aic['AIC'] < best_model_bic['AIC']:
            best_model = best_model_aic
        else:
            best_model = best_model_bic
        
        self.p = best_model['p']
        self.q = best_model['q']
        model = ARIMA(self.time_series_measurement, order=(self.p,self.d,self.q))
        ARMA_Model = model.fit(method='innovations_mle')
        print(ARMA_Model.summary())

    def KF(self, initial_x, dim_x, dim_z, P=None, F=None, R=None, Q=None, H=None):
        '''
        进行卡尔曼滤波
        Q = gamma*Q*gamma
        '''
        print("KF_action")
        self.dim_x = dim_x # 输入状态矩阵的维度 state
        self.dim_z = dim_z # 观测矩阵的维度
        self.P = np.eye(self.dim_x) # uncertainty covariance  
        self.F = np.eye(self.dim_x) if F is None else F # 状态转移矩阵
        self.Q = np.eye(self.dim_x) if Q is None else Q # 过程噪声矩阵
        self.R = np.eye(dim_z) if R is None else R
        self.H = np.zeros((self.dim_z,self.dim_x)) if H is None else H
        
        # 创建卡尔曼滤波器实例
        kf_model = KalmanFilter(self.dim_x, self.dim_z)
        kf_model.x = initial_x
        kf_model.F = self.F
        kf_model.P = self.P
        kf_model.Q = self.Q
        kf_model.R = self.R
        kf_model.H = self.H

        self.time_series_kf = list()
        for z in range(self.time_series_length):
            self.time_series_kf.append(kf_model.x)
            kf_model.predict()
            kf_model.update(self.time_series_measurement[z])

        print("KF_over")

if __name__ == '__main__':
    # 设置支持中文的字体，SimHei 是常见的中文字体
    plt.rcParams['font.family'] = ['SimHei']  # 或者 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    file_path = r'C:\Users\13106\Desktop\2024\编程\传感器信号去噪\深度学习方法\LSTM-RNN\data\Accelerometer.csv'
    time_series = pd.read_csv(file_path)
    time_series_measurement = time_series['Acceleration x (m/s^2)'].to_numpy()
    model = ARMA_Kalman_Filter(time_series_measurement)
    # ensure data is zero mean
    model.zero_mean()
    # # # make smooth test
    # # model.smooth()
    # # 平稳性检验
    # model.stationary_test()
    # # 正态分布检验
    # model.normality_test()
    # # AIC BIC
    # model.AIC_BIC()
    # model.p=3
    # model.q=2
    # # ARMA
    # model.ARMA_model()
    # print(model.p)
    # print(model.q)
    initial_x = np.array([[0,0]]).T # 列向量
    dim_x = 2
    dim_z = 1
    p = np.array([[100,0],[0,100]])
    F = np.array([[-0.2485,1],[-0.0483,0]])
    T = np.array([[1,0],[0.4417,0]])
    QQ = np.array([[1,0],[0,1]])
    H = np.array([[1,0]])
    Q = np.dot(T,QQ)
    Q = np.dot(Q,T)
    model.KF(initial_x=initial_x,dim_x=dim_x,dim_z=dim_z,P=p,F=F,H=H,Q=Q)
    time_series_predict = []
    for i in range(len(model.time_series_kf)):
        time_series_predict.append(-0.1712*model.time_series_kf[i][0]-0.0532*model.time_series_kf[i][1])
    print(np.std(time_series_measurement))
    print(np.std(time_series_predict))
    # plt.figure(figsize=(12,6))
    # plt.plot(time_series_measurement-time_series_measurement.mean(),color='red',label='输入信号')
    # plt.plot(time_series_predict,color='blue',label='去噪信号',linestyle='--')
    # plt.legend()
    # plt.show()
    pass
