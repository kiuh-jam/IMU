"""
ruptures python 包涉及的算法出自于论文《Selective review of offline change point detection methods》

进行突变点检测(change point detection)

change_point_detection(data, pen)
    data: 输入数据，一维数组
    pen: 惩罚参数，用于控制检测的灵敏度
    return: 检测到的变化点索引
"""

import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

def change_point_detection(data, pen):
    """
    使用ruptures库进行基于Pelt算法的动态阈值检测
    :param data: 输入数据，一维数组
    :param pen: 惩罚参数，用于控制检测的灵敏度
    :return: 检测到的变化点索引
    建议传入的数据不要太大 否则运算时间会很长
    """
    algo = rpt.Pelt(model="rbf").fit(data)
    result = algo.predict(pen=pen)
    rpt.display(data, result)
    plt.show()
    return result