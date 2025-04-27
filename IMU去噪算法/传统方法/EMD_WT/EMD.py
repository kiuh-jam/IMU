# 经验模态分解算法
# 算法分析过程：
# 1.求极值点：通过Find Peaks算法获取信号序列的全部极值点
# 2.拟合包络曲线 通过信号序列的极大值和极小值组，经过三次样条插值法获得两条光滑的波峰/波谷拟合曲线，
# 即信号的上包络线与下包络线
# 3.均值包络线 将两条极值曲线平均获得平均包络线
# 4.中间信号 原始信号减均值包络线，得到中间信号
# 5.判断本征模函数（IMF） IMF需要符合两个条件： 
# 1）在整个数据段内，极值点的个数和过零点的个数必须相等或相差最多不能超过一个。 
# 2）在任意时刻，由局部极大值点形成的上包络线和由局部极小值点形成的下包络线的平均值为零，
# 即上、下包络线相对于时间轴局部对称。
# IMF 1 获得的第一个满足IMF条件的中间信号即为原始信号的第一个本征模函数分量IMF 1 
# （由原数据减去包络平均后的新数据，若还存在负的局部极大值和正的局部极小值，
# 说明这还不是一个本征模函数，需要继续进行“筛选”。）
# 使用上述方法得到第一个IMF后，用原始信号减IMF1，作为新的原始信号，
# 再通过上述的筛选分析，可以得到IMF2，以此类推，完成EMD分解。

from scipy.signal import argrelextrema
import scipy as spi
import numpy as np
import matplotlib.pyplot as plt

def Find_peaks(data):
    '''
    data是输入的时间序列
    输出该序列的极值点
    '''
    max_peaks = argrelextrema(data,np.greater) #极大值点 返回存储极值位置的array数组
    min_peaks = argrelextrema(data,np.less) #极小值点
    
    return max_peaks,min_peaks

def Envelope_function_fit(data):
    '''
    输入时间序列，拟合包络函数
    EMD步骤的核心，也是分解出本征函数IMFs的前提
    返回用原信号减去平均包络线的新信号，也就是中间信号
    '''
    index = list(range(len(data)))

    # 获取极值点
    max_peaks_array,min_peaks_array = Find_peaks(data)
    max_peaks = list(max_peaks_array[0])
    min_peaks = list(min_peaks_array[0])

    # 将极值点拟合成曲线
    ipo3_max = spi.interpolate.splrep(max_peaks,data[max_peaks],k=3) # 对极大值点进行三次样条
    iy3_max = spi.interpolate.splev(index,ipo3_max)

    ipo3_min = spi.interpolate.splrep(min_peaks,data[min_peaks],k=3)
    iy3_min = spi.interpolate.splev(index,ipo3_min)

    # 计算平均包络线
    iy3_mean = (iy3_max+iy3_min)/2

    # # 绘制图像 这个很卡，不要运行
    # plt.figure(figsize = (18,6))
    # plt.plot(data, label='Original')
    # plt.plot(iy3_max, label='Maximun Peaks')
    # plt.plot(iy3_min, label='Minimun Peaks')
    # plt.plot(iy3_mean, label='Mean')
    # plt.legend()
    # plt.xlabel('time (s)')
    # plt.ylabel('microvolts (uV)')
    # plt.title("Cubic Spline Interpolation")

    return data - iy3_mean

def hasPeaks(data):
    max_peaks_array,min_peaks_array = Find_peaks(data)
    max_peaks = list(max_peaks_array[0])
    min_peaks = list(min_peaks_array[0])

    if len(max_peaks) > 3 and len(min_peaks) > 3:
        return True
    else:
        return False

# 判断IMFs
def isIMFs(data_mean):
    '''
    输入信号 判断其是否为IMFs
    '''
    max_peaks_array,min_peaks_array = Find_peaks(data_mean)
    max_peaks = list(max_peaks_array[0])
    min_peaks = list(min_peaks_array[0])

    if min(data_mean[max_peaks]) < 0 or max(data_mean[min_peaks]) > 0:
        return False
    else:
        return True

def getIMFs(data):
    iter=0
    while(not isIMFs(data) and iter <= 10000):
        data = Envelope_function_fit(data)
        iter+=1
        if iter == 10000:    
            print('getIMFswrong')
    # while(not isIMFs(data)):
    #     data = Envelope_function_fit(data)
    return data

def EMD(data):
    IMFs = []
    iter = 0
    while hasPeaks(data):
        data_imf = getIMFs(data)
        IMFs.append(data_imf)
        data = data-data_imf
        iter+=1
        if iter == 100000:
            print('wrong')
    # while hasPeaks(data):
    #     data_imf = getIMFs(data)
    #     IMFs.append(data_imf)
    #     data = data-data_imf
    res = data
    return IMFs, res
