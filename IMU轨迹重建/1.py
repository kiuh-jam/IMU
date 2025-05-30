import numpy as np
import meshplot
from meshzoo import poisson_surface_reconstruction

# 假设points是一个包含点云数据的NumPy数组
points = np.random.rand(1000, 3)

# 使用Poisson表面重建算法生成Mesh网格
mesh = poisson_surface_reconstruction(points)

# 绘制生成的Mesh网格
meshplot.plot(mesh)