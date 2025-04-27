import csv

filepath = r'C:\Users\13106\Desktop\code\IMU\IMUData\传感器数据采集\已提交\4\X\Accelerometer.csv'
g_filepath = r'C:\Users\13106\Desktop\code\IMU\IMUData\传感器数据采集\已提交\4\X\Gyroscope.csv'
def _init_reader(self, acc_filepath:str, gyro_filepath:ste) -> [csv.reader,csv.reader]:
        '''
        初始化csv文件读取器
        '''
        acc_file = open(acc_filepath, 'r', newline='')
        gyro_file = open(gyro_filepath, 'r', newline='')
        if self.has_header: # 跳过标题行
            acc_file.readline()
            gyro_file.readline()
        self.acc_pos = acc_file.tell() # 当前位置
        self.gyro_pos = gyro_file.tell() # 当前位置
        acc_initial_time = acc_file.readline().split(',')[self.time_col] # 初始时间
        gyro_initial_time = gyro_file.readline().split(',')[self.time_col] # 初始时间
        while acc_initial_time != gyro_initial_time: # 初始时间不同步
            if acc_initial_time < gyro_initial_time: # 加速度数据时间早于角速度数据时间
                self.acc_pos = acc_file.tell() # 保存当前位置
                acc_initial_time = acc_file.readline().split(',')[self.time_col] # 读取下一行
            elif gyro_initial_time < acc_initial_time: # 角速度数据时间早于加速度数据时间
                self.gyro_pos = gyro_file.tell() # 保存当前位置
                gyro_initial_time = gyro_file.readline().split(',')[self.time_col] # 读取下一行
        acc_file.seek(self.acc_pos) # 移动文件指针到初始位置
        gyro_file.seek(self.gyro_pos) # 移动文件指针到初始位置
        acc_reader = csv.reader(acc_file) # 创建csv文件读取器
        gyro_reader = csv.reader(gyro_file) # 创建csv文件读取器
        print(acc_file.readline().split(',')[self.time_col]) # 读取第一行
        print(gyro_file.readline().split(',')[self.time_col])
        return [acc_reader, gyro_reader]
_init_reader(1,filepath,g_filepath)