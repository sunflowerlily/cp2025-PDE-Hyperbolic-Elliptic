"""学生模板：ParallelPlateCapacitor
文件：parallel_plate_capacitor_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def solve_laplace_jacobi(xgrid, ygrid, w, d, tol=1e-5):
    """
    使用Jacobi迭代法求解拉普拉斯方程
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        tol (float): 收敛容差
    
    返回:
        tuple: (potential_array, iterations, convergence_history)
    
    物理背景: 求解平行板电容器内部的电势分布，满足拉普拉斯方程 \(\nabla^2 V = 0\)。
    数值方法: 使用Jacobi迭代法，通过反复迭代更新每个网格点的电势值，直至收敛。
    
    实现步骤:
    1. 初始化电势网格，设置边界条件（极板电势）。
    2. 循环迭代，每次迭代根据周围点的电势更新当前点的电势。
    3. 记录每次迭代的最大变化量，用于收敛历史分析。
    4. 检查收敛条件，如果最大变化量小于容差，则停止迭代。
    5. 返回最终的电势分布、迭代次数和收敛历史。
    """
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def solve_laplace_sor(xgrid, ygrid, w, d, omega=1.25, Niter=1000, tol=1e-5):
    """
    实现SOR算法求解平行板电容器的电势分布
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        omega (float): 松弛因子
        Niter (int): 最大迭代次数
        tol (float): 收敛容差
    返回:
        tuple: (电势分布数组, 迭代次数, 收敛历史)
    
    物理背景: 求解平行板电容器内部的电势分布，满足拉普拉斯方程 \(\nabla^2 V = 0\)。
    数值方法: 使用逐次超松弛（SOR）迭代法，通过引入松弛因子加速收敛。
    
    实现步骤:
    1. 初始化电势网格，设置边界条件（极板电势）。
    2. 循环迭代，每次迭代根据周围点和松弛因子更新当前点的电势。
    3. 记录每次迭代的最大变化量，用于收敛历史分析。
    4. 检查收敛条件，如果最大变化量小于容差，则停止迭代。
    5. 返回最终的电势分布、迭代次数和收敛历史。
    """
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def plot_results(x, y, u, method_name):
    """
    绘制三维电势分布、等势线和电场线
    
    参数:
        x (array): X坐标数组
        y (array): Y坐标数组
        u (array): 电势分布数组
        method_name (str): 方法名称
    
    实现步骤:
    1. 创建包含两个子图的图形。
    2. 在第一个子图中绘制三维线框图显示电势分布以及在z方向的投影等势线。
    3. 在第二个子图中绘制等势线和电场线流线图。
    4. 设置图表标题、标签和显示(注意不要出现乱码)。
    """
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

if __name__ == "__main__":
    # 示例用法（学生可以取消注释并在此处测试他们的实现）
    pass