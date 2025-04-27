import torch
import load_dataset
import LSTM_GRU
import matplotlib.pyplot as plt
import numpy as np

# 设置支持中文的字体，SimHei 是常见的中文字体
plt.rcParams['font.family'] = ['SimHei']  # 或者 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
directory = r'C:\Users\13106\Desktop\code\IMU\IMUData\传感器数据采集\手写数字字母\测试'
save_path = r'C:\Users\13106\Desktop\code\IMU\IMUData\传感器数据采集\手写数字字母\测试'
dataset = load_dataset.Accleration_Dataset(directory)
data = dataset.data_x
# print(data.shape)
model = LSTM_GRU.LSTM_GRU(data, [1]*20, step=20)
model.train1((torch.zeros(1, data.shape[0],model.h_size_list[0]), torch.zeros(1, data.shape[0],model.h_size_list[0])), 50, 0.01)
model.loss_plt(save_path)
# 加载权重（严格匹配模式）

# model.load_state_dict(torch.load(r'C:\Users\13106\Desktop\2024\编程\传感器信号去噪\深度学习方法\GRU\model.pth'))
# model.eval()
# model.to(device='cpu')
# out = model.forward((torch.zeros(1, data.shape[0],model.h_size_list[0]), torch.zeros(1, data.shape[0],model.h_size_list[0])))
# print(out.shape)
# x=torch.cat(model.o,dim=1)
# x=x.view(-1).numpy()
# y=out.view(-1).detach().numpy()-out.view(-1).detach().numpy().mean()
# z=x-y
# print(np.std(x))
# print(np.std(y))
# plt.figure(figsize=(12,6))
# plt.subplot(3, 1, 1)
# plt.plot(x,color='red',label='输入信号')
# plt.title('输入信号')
# plt.legend()
# plt.subplot(3, 1, 2)
# plt.plot(y,color='blue',label='输出信号')
# plt.title('输出信号')
# plt.legend()
# plt.subplot(3, 1, 3)
# plt.plot(x,color='red',label='输入信号')
# plt.plot(y,color='blue',label='输出信号')
# plt.title('去噪信号')
# plt.legend()
# plt.show()