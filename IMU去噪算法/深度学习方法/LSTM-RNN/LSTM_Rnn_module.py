import torch
import torch.nn as nn
import pandas as pd
import csv
import matplotlib
import matplotlib.pyplot as plt

class LSTM_Rnn(nn.Module):
    def __init__(self, data, h_size_list,step=20):
        '''
        data 输入数据 shape(bs,seq_len,i_size) 表示批次为bs,数据长度为seq_len(时间节点长度),输入节点个数为i_size
        比如[[[1],[2],[3],[4],[5],[6],[7]],[[2],[3],[4],[5],[6],[7],[8]]]]
        step 表示每次训练的数据长度 也就是一次看前step个时间点的数据后输出
        h_size_list是一个列表 表示每一层的隐藏层的节点个数 len(h_size_list) = step 每个step都有一个lstm来接受一个时间点的输入
        '''
        super(LSTM_Rnn,self).__init__()
        self.h_size_list = h_size_list
        # input是输入用于训练的数据  o是输出用于验证的数据
        self.step = step
        self.input, self.o = self.prepare_data(data) #获取输入数据和验证数据 input[N-step个(bs,step,i_size)] o[N-step个(bs,1,i_size)]
        self.lstm_list = [] # 存放LSTM单元
        for i in range(self.step):
            self.lstm_list.append(nn.LSTM(data.shape[2], h_size_list[i], batch_first = True)) # 存放step个单元 每个单元的输入是i_size 输出是h_size_list[i]
        self.lstm = nn.ModuleList(self.lstm_list)
        self.fc = nn.Linear(h_size_list[-1], 1)

    def forward(self, initial_state):
        '''
        前向传播
        initial_state是一个初始值的元组 (h,c)
        '''
        h_n, c_n = initial_state
        out = []
        x = self.input
        step = self.step
        for i in range(len(x)): # len(x) = N-step
            for t in range(step):
                lstmmodule = self.lstm_list[t]
                y, (h_n, c_n) = lstmmodule(x[i][:,t,:].unsqueeze(-1), (h_n, c_n)) #每个时间点的数据都要经过step个lstm单元
            out.append(self.fc(h_n).squeeze(0)) #获取最后一个单元的输出 shape(bs,1)
        return torch.cat(out, dim=1)
    
    def verify(self,initial_state):
        h_n, c_n = initial_state
        out = []
        x = self.input
        step = self.step
        for i in range(len(x)): # len(x) = N-step
            for t in range(step):
                lstmmodule = self.lstm_list[t]
                y, (h_n, c_n) = lstmmodule(x[i][:,t,:].unsqueeze(-1), (h_n, c_n)) #每个时间点的数据都要经过step个lstm单元
            out.append(self.fc(h_n).squeeze(0)) #获取最后一个单元的输出 shape(bs,1)
        return torch.cat(out, dim=1)
    
    def train1(self, initial_state, epoah, lr):
        '''
        epoah 表示训练次数
        lr 表示学习率
        '''
        self.loss_data = []
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss() #损失函数 采用均方误差
        min_loss = float('inf')
        for i in range(epoah):
            optimizer.zero_grad()
            print("forward")
            out = self.forward(initial_state)
            print("loss")
            loss = loss_fn(out, torch.cat(self.o,dim=1))
            print("backward")
            loss.backward()
            print("step")
            optimizer.step()
            print('epoah:', i, 'loss:', loss.item())
            self.loss_data.append(loss.item())

            # 保存模型
            if loss.item() < min_loss:
                min_loss = loss.item()
                torch.save(self.state_dict(), 'model.pth')
                best_output = out.flatten().tolist()  # 记录当前最小损失时的输出
        output_file_path = "./output.csv"
        # 保存最小损失时的输出到CSV文件
        with open(output_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(best_output)

        print(f"Best output saved to {output_file_path}")
        print(f"Best model saved with loss: {min_loss}")

    def prepare_data(self,data):
        '''
        data是原始的输入数据 shape(bs,seq_len,i_size) 表示批次为bs,数据长度为seq_len,输入节点个数为i_size
        输出x和y
        x为训练数据 是一个张量列表 每个张量的shape为(bs, step, i_size) (seq_len-step)个张量
        y为输出数据的验证集 是一个张量列表，每个列表的shape为(bs, 1, i_size)
        '''
        x, y =[], []
        for i in range(data.shape[1] - self.step):
            x.append(data[:,i:i+self.step,:])
            y.append(data[:,i+self.step,:])
        return x, y

    def loss_plt(self,save_path):
        '''
        绘制loss曲线
        '''
        plt.figure(figsize=(10,6))
        plt.plot(self.loss_data,label='traning loss')
        plt.title('Traning loss curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)
        plt.show()
