"""
class IMU：IMU数据结构 存储每时刻的IMU测量数据和时间信息 包括加速度和角速度以及该时刻的时间 IMU的采样间隔
class CsvImuFileLoader：从csv文件中提取IMU数据 包括加速度、角速度和磁力计文件路径 采样率 时间索引 加速度索引 角速度索引 以及是否插值

"""
import pandas as pd
import numpy as np
import csv
from typing import Tuple, Optional, TextIO
import Trajectory_drawing as TD
import Quaternion_inertial_navigation as QIN
import deviation
from scipy.interpolate import interp1d

class IMU:
    '''
    IMU数据结构
    存储IMU测量数据和时间信息
    '''
    def __init__(self, time:int = 0.0, 
                 dacc:np.ndarray = np.zeros(3), 
                 dgyro:np.ndarray = np.zeros(3), 
                 dmag:np.ndarray = np.zeros(3),
                 linear_acc:np.ndarray = np.zeros(3),
                 dt:float = 0.0):
        self.time = time # 时间(s)
        self.dgyro = dgyro # 当前时刻角速度值
        self.dacc = dacc # 当前时刻加速度值
        self.dmag = dmag # 当前时刻磁力值
        self.linear_acc = linear_acc # 当前时刻线性加速度值
        self.dt = dt # 采样间隔
        pass

class CsvImuFileLoader:
    '''
    从文件中提取IMU数据
    '''
    def __init__(self, 
                 acc_filepath:str, 
                 gyro_filepath:str, 
                 mag_filepath:str = None,
                 linear_acc_filepath:str = None,
                 rate:int = 400, 
                 has_header:bool = True,
                 time_col:int = 0,
                 acc_col:list = [1,2,3],
                 gyro_col:list = [1,2,3],
                 mag_col:list = [1,2,3],
                 linear_acc_col:list = [1,2,3],
                 interpolate:bool = False,
                 interp_kind:str = 'cubic'):
        '''
        acc_filepath: 加速度数据文件路径
        gyro_filepath: 角速度文件路径
        mag_filepath: 磁力文件路径
        linear_acc_filepath: 线性加速度文件路径
        rate: 采样率(Hz) 默认400Hz
        time_col: csv文件的时间索引 默认为第一列为时间 即0
        acc_col: csv文件加速度索引 默认为[x,y,z] = [1,2,3]
        gyro_col: csv文件角速度索引 默认为[x,y,z] = [1,2,3]
        mag_col: csv文件磁力索引 默认为[x,y,z] = [1,2,3]
        linear_acc_col: csv文件线性加速度索引 默认为[x,y,z] = [1,2,3]
        interpolate: 是否插值 默认不插值
        interp_kind: 插值方式 默认三次样条插值 可选['linear', 'quadratic', 'cubic']
        '''
        self.acc_filepath = acc_filepath
        self.gyro_filepath = gyro_filepath
        self.mag_filepath = mag_filepath
        self.linear_acc_filepath = linear_acc_filepath
        self.rate = rate
        self.dt = 1.0/float(rate)
        self.time_col = time_col
        self.acc_col = acc_col
        self.gyro_col = gyro_col
        self.mag_col = mag_col
        self.linear_acc_col = linear_acc_col
        self.has_header = has_header
        self.interpolate = interpolate # 是否插值 如果加速度时间和角速度时间大部分同步 则可不插值 否则插值处理
        self.interp_kind = interp_kind # 插值方式

        if not self.interpolate: # 不插值
            # 初始化文件读取器 并同步时间
            self.accel_reader, self.gyro_reader, self.mag_reader, self.linear_acc_reader = self._init_reader(
                acc_filepath, gyro_filepath, mag_filepath, linear_acc_filepath)

            # 预读取第一行数据
            self.next_accel = self._read_next(self.accel_reader)
            self.next_gyro = self._read_next(self.gyro_reader)
            self.next_mag = self._read_next(self.mag_reader) if mag_filepath else None
            self.next_linear_acc = self._read_next(self.linear_acc_reader) if linear_acc_filepath else None
           
            # 初始化IMU数据
            self.current_IMU = IMU()
            self.prev_IMU = IMU()

        else:
            # 使用插值
            self.acc_all_data, self.gyro_all_data, self.mag_all_data, self.linear_acc_all_data, self.time_all_data = self._interpolate_all_data()
            self.interp_index = 0
        pass

    def _init_reader(self, acc_filepath, gyro_filepath, mag_filepath, linear_acc_filepath) -> Tuple[csv.reader, csv.reader, Optional[csv.reader], Optional[csv.reader]]:
        '''
        初始化csv文件读取器
        acc_filepath: 加速度数据文件路径
        gyro_filepath: 角速度文件路径
        mag_filepath: 磁力数据文件路径
        linear_acc_filepath: 线性加速度文件路径
        return: acc_reader, gyro_reader, mag_reader, linear_acc_reader
        分别为加速度、角速度、磁力计和线性加速度的csv文件读取器
        读取器会跳过标题行 并同步时间
        时间同步方式为: 读取第一行数据 比较时间 时间不同步则跳过时间早的行 直到时间同步
        时间同步后 移动文件指针到时间同步处的位置
        '''
        acc_file = open(acc_filepath, 'r', newline='')
        gyro_file = open(gyro_filepath, 'r', newline='')
        mag_file = open(mag_filepath, 'r', newline='') if mag_filepath else None
        linear_acc_file = open(linear_acc_filepath, 'r', newline='') if linear_acc_filepath else None

        if self.has_header: # 跳过标题行
            acc_file.readline()
            gyro_file.readline()
            if mag_file:
                mag_file.readline()
            if linear_acc_file:
                linear_acc_file.readline()
                
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
                
        acc_file.seek(self.acc_pos) # 移动加速度文件指针到时间同步的位置
        gyro_file.seek(self.gyro_pos) # 移动角速度文件指针到时间同步的位置
        acc_reader = csv.reader(acc_file) # 创建csv文件读取器
        gyro_reader = csv.reader(gyro_file) # 创建csv文件读取器
        mag_reader = csv.reader(mag_file) if mag_file else None
        linear_acc_reader = csv.reader(linear_acc_file) if linear_acc_file else None
        return acc_reader, gyro_reader, mag_reader, linear_acc_reader

    def _read_next(self, reader:csv.reader) -> Optional[Tuple[float, list]]:
        '''
        读取该行数据 移动文件指针到下一行 返回(time, data_row)
        '''
        try:
            row = next(reader) # 读取该行数据 返回row(time,x,y,z) 并将文件指针移动到下一行
            time = float(row[self.time_col]) # 读取时间
            return (time, row)
        except StopIteration:
            return None

    def get_next_imu(self) -> Optional[IMU]:
        '''
        读取下一个IMU数据
        如果数据结束(文件指针移动到文件末尾) 返回None
        '''
        if self.next_accel is None or self.next_gyro is None:
            return None
        acc_time, acc_row = self.next_accel
        gyro_time, gyro_row = self.next_gyro
        mag_time, mag_row = self.next_mag if self.next_mag else (acc_time, ['0', '0', '0'])
        linear_acc_time, linear_acc_row = self.next_linear_acc if self.next_linear_acc else (acc_time, ['0', '0', '0'])
        
        if format(acc_time,'.4f') != format(gyro_time,'.4f'):
            print(f"加速度计和陀螺仪时间不同步: 加速度时间={acc_time}, 陀螺仪时间={gyro_time}")
            acc_time = self.dt + self.prev_IMU.time # 时间不同步就转为上一时间 + 时间间隔(采样率分之一)
            gyro_time = self.dt + self.prev_IMU.time
            
        self.prev_IMU = self.current_IMU
        self.current_IMU = IMU(
            time=acc_time, 
            dacc=np.array([float(acc_row[i]) for i in self.acc_col]), 
            dgyro=np.array([float(gyro_row[i]) for i in self.gyro_col]), 
            dmag=np.array([float(mag_row[i]) for i in self.mag_col]) if self.next_mag else np.zeros(3),
            linear_acc=np.array([float(linear_acc_row[i]) for i in self.linear_acc_col]) if self.next_linear_acc else np.zeros(3),
            dt=self.dt
        )
        
        self.next_accel = self._read_next(self.accel_reader)
        self.next_gyro = self._read_next(self.gyro_reader)
        self.next_mag = self._read_next(self.mag_reader) if self.mag_reader else None
        self.next_linear_acc = self._read_next(self.linear_acc_reader) if self.linear_acc_reader else None
        return self.current_IMU

    def _interpolate_all_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        acc_df = pd.read_csv(self.acc_filepath, header=0 if self.has_header else None)
        gyro_df = pd.read_csv(self.gyro_filepath, header=0 if self.has_header else None)
        mag_df = pd.read_csv(self.mag_filepath, header=0 if self.has_header else None) if self.mag_filepath else None
        linear_acc_df = pd.read_csv(self.linear_acc_filepath, header=0 if self.has_header else None) if self.linear_acc_filepath else None

        acc_time = acc_df.iloc[:, self.time_col].values
        gyro_time = gyro_df.iloc[:, self.time_col].values
        mag_time = mag_df.iloc[:, self.time_col].values if mag_df is not None else acc_time
        linear_acc_time = linear_acc_df.iloc[:, self.time_col].values if linear_acc_df is not None else acc_time

        all_times = np.union1d(np.union1d(np.union1d(acc_time, gyro_time), mag_time), linear_acc_time)

        acc_interp = interp1d(acc_time, acc_df.iloc[:, self.acc_col].values, axis=0,
                            kind=self.interp_kind, fill_value="extrapolate")
        acc_data = acc_interp(all_times)

        gyro_interp = interp1d(gyro_time, gyro_df.iloc[:, self.gyro_col].values, axis=0,
                            kind=self.interp_kind, fill_value="extrapolate")
        gyro_data = gyro_interp(all_times)

        if mag_df is not None:
            mag_interp = interp1d(mag_time, mag_df.iloc[:, self.mag_col].values, axis=0,
                                kind=self.interp_kind, fill_value="extrapolate")
            mag_data = mag_interp(all_times)
        else:
            mag_data = np.zeros((len(all_times), 3))

        if linear_acc_df is not None:
            linear_acc_interp = interp1d(linear_acc_time, linear_acc_df.iloc[:, self.linear_acc_col].values, axis=0,
                                      kind=self.interp_kind, fill_value="extrapolate")
            linear_acc_data = linear_acc_interp(all_times)
        else:
            linear_acc_data = np.zeros((len(all_times), 3))

        return acc_data, gyro_data, mag_data, linear_acc_data, all_times

    def _get_next_interpolated(self) -> Optional[IMU]:
        if self.interp_index >= len(self.time_all_data):
            return None
        imu = IMU(
            time=self.time_all_data[self.interp_index],
            dacc=self.acc_all_data[self.interp_index],
            dgyro=self.gyro_all_data[self.interp_index],
            dmag=self.mag_all_data[self.interp_index],
            linear_acc=self.linear_acc_all_data[self.interp_index],
            dt=self.dt
        )
        self.interp_index += 1
        return imu

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__close__()
        return False

    def __close__(self):
        '''
        关闭文件
        '''
        if hasattr(self, 'accel_reader'):
            # 获取csv.reader内部的文件对象
            acc_file = getattr(self.accel_reader, '_file', None)
            if acc_file is not None:
                acc_file.close()
                
        if hasattr(self, 'gyro_reader'):
            # 获取csv.reader内部的文件对象
            gyro_file = getattr(self.gyro_reader, '_file', None)
            if gyro_file is not None:
                gyro_file.close()
                
        if hasattr(self, 'mag_reader') and self.mag_reader is not None:
            mag_file = getattr(self.mag_reader, '_file', None)
            if mag_file is not None:
                mag_file.close()
                
        if hasattr(self, 'linear_acc_reader') and self.linear_acc_reader is not None:
            linear_acc_file = getattr(self.linear_acc_reader, '_file', None)
            if linear_acc_file is not None:
                linear_acc_file.close()
                
        self.next_accel = None
        self.next_gyro = None
        self.next_mag = None
        self.next_linear_acc = None
        pass
    
    def _get_all_data(self, start=800, limitlen=120000):
        '''
        读取所有数据
        start: 开始读取的索引
        limitlen: 读取的最大长度
        return: acc, gyro, mag, linear_acc, time
        '''
        print("该函数有bug 未修复")
        if self.interpolate:
            acc = self.acc_all_data[start:limitlen]
            gyro = self.gyro_all_data[start:limitlen]
            mag = self.mag_all_data[start:limitlen]
            linear_acc = self.linear_acc_all_data[start:limitlen]
            time = self.time_all_data[start:limitlen]
            return acc, gyro, mag, linear_acc, time
        else:
            self.acc_all_data = []
            self.gyro_all_data = []
            self.mag_all_data = []
            self.linear_acc_all_data = []
            self.time_all_data = []
            while True:
                imu = self.get_next_imu()
                if imu is None:
                    break
                self.acc_all_data.append(imu.dacc)
                self.gyro_all_data.append(imu.dgyro)
                self.mag_all_data.append(imu.dmag)
                self.linear_acc_all_data.append(imu.linear_acc)
                self.time_all_data.append(imu.time)
            return (np.array(self.acc_all_data)[start:limitlen],
                    np.array(self.gyro_all_data)[start:limitlen],
                    np.array(self.mag_all_data)[start:limitlen],
                    np.array(self.linear_acc_all_data)[start:limitlen],
                    np.array(self.time_all_data)[start:limitlen])

        




