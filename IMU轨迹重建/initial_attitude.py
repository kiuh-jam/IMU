class initial_attitude:
    '''
    计算初始姿态
    '''
    def __init__(self, acc:np.ndarray, gyro:np.ndarray, time:np.ndarray, dt:float, mag:np.ndarray = None):
        '''
        acc: 加速度数据 (N,3) 默认为(N,(ax,ay,az))
        gyro: 角速度数据 (N,3) 默认为(N,(wx,wy,wz))
        mag: 磁力计数据 (N,3) 默认为(N,(mx,my,mz))
        time: 时间数据 (N,) 默认为(N,) 单位为s
        通常N是信号开始的一段时间内的采样点数
        dt: 采样间隔
        '''
        self.acc = acc
        self.gyro = gyro
        self.mag = mag
        self.time = time
        self.dt = dt
    
    def calculate_g(self)->float:
        '''
        计算重力加速度 G
        首先计算加速度的平均值
        随后计算重力加速度
        '''
        acc_ave = np.mean(self.acc, axis=0) # 计算加速度的平均值
        g_ = np.linalg.norm(acc_ave) # 计算重力加速度
        return acc_ave[0], acc_ave[1], acc_ave[2], g_
        
    def weighted_average(self, values):
        '''
        计算加权平均
        '''
        numerical_integral = np.trapz(values, self.time, axis=0) # 计算数值积分（使用梯形法则）
        time_width = self.time[-1] - self.time[0] # 计算时间宽度
        weighted_avg = numerical_integral / time_width # 计算加权平均
        return weighted_avg

    def calculate_initial_attitude(self):
        '''
        计算初始姿态 使用磁力计数据和加速度数据来计算横滚角、俯仰角和航向角
        返回：
        phi(float): 横滚角，单位为弧度
        theta(float): 俯仰角，单位为弧度
        psi(float): 航向角，单位为弧度
        '''
        # 计算加速度的平均值
        ax, ay, az, g = self.calculate_g()

        # 计算横滚角 phi 俯仰角 theta
        phi = np.arctan2(ay, az)
        theta = np.arcsin(-ax / g)

        psi = 0

        if self.mag is None:
            print("磁力计数据为空，无法计算航向角，返回0")
            return phi, theta, psi
        else:
            # 计算磁力计数据的平均值
            mag_ave = np.mean(self.mag, axis=0)
            # 三轴磁力计的平均值
            mx = mag_ave[0]
            my = mag_ave[1]
            mz = mag_ave[2]

            # 计算航向角 psi
            psi = np.arctan2(my * np.cos(phi) - mz * np.sin(phi),
                            mx * np.cos(theta) + my * np.sin(theta) + mz * np.cos(phi) * np.sin(theta))

            return phi, theta, psi

