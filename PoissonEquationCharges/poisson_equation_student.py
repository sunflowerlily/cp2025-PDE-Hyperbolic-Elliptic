#!/usr/bin/env python3
"""
学生模板：求解正负电荷构成的泊松方程
文件：poisson_equation_student.py
重要：函数名称必须与参考答案一致！
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def solve_poisson_equation(M: int = 100, target: float = 1e-6, max_iterations: int = 10000) -> Tuple[np.ndarray, int, bool]:
    """
    使用松弛迭代法求解二维泊松方程
    
    参数:
        M (int): 每边的网格点数，默认100
        target (float): 收敛精度，默认1e-6
        max_iterations (int): 最大迭代次数，默认10000
    
    返回:
        tuple: (phi, iterations, converged)
            phi (np.ndarray): 电势分布数组，形状为(M+1, M+1)
            iterations (int): 实际迭代次数
            converged (bool): 是否收敛
    
    物理背景:
        求解泊松方程 ∇²φ = -ρ/ε₀，其中：
        - φ 是电势
        - ρ 是电荷密度分布
        - 边界条件：四周电势为0
        - 正电荷位于 (60:80, 20:40)，密度 +1 C/m²
        - 负电荷位于 (20:40, 60:80)，密度 -1 C/m²
    
    数值方法:
        使用有限差分法离散化，迭代公式：
        φᵢⱼ = 0.25 * (φᵢ₊₁ⱼ + φᵢ₋₁ⱼ + φᵢⱼ₊₁ + φᵢⱼ₋₁ + h²ρᵢⱼ)
    
    实现步骤:
    1. 初始化电势数组和电荷密度数组
    2. 设置边界条件（四周为0）
    3. 设置电荷分布
    4. 松弛迭代直到收敛
    5. 返回结果
    """
    # TODO: 设置网格间距
    h = 1.0
    
    # TODO: 初始化电势数组，形状为(M+1, M+1)
    # 提示：使用 np.zeros() 创建数组
    
    # TODO: 创建电荷密度数组
    # 提示：同样使用 np.zeros() 创建
    
    # TODO: 设置电荷分布
    # 正电荷：rho[60:80, 20:40] = 1.0
    # 负电荷：rho[20:40, 60:80] = -1.0
    
    # TODO: 初始化迭代变量
    # delta = 1.0  # 用于存储最大变化量
    # iterations = 0  # 迭代计数器
    # converged = False  # 收敛标志
    
    # TODO: 创建前一步的电势数组副本
    # 提示：使用 np.copy()
    
    # TODO: 主迭代循环
    # while delta > target and iterations < max_iterations:
    #     # 1. 使用有限差分公式更新内部网格点
    #     #    phi[1:-1, 1:-1] = 0.25 * (...)
    #     # 2. 计算最大变化量
    #     #    delta = np.max(np.abs(phi - phi_prev))
    #     # 3. 更新前一步解
    #     #    phi_prev = np.copy(phi)
    #     # 4. 增加迭代计数
    #     #    iterations += 1
    
    # TODO: 检查是否收敛
    # converged = (delta <= target)
    
    # TODO: 返回结果
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def visualize_solution(phi: np.ndarray, M: int = 100) -> None:
    """
    可视化电势分布
    
    参数:
        phi (np.ndarray): 电势分布数组
        M (int): 网格大小
    
    功能:
        - 使用 plt.imshow() 显示电势分布
        - 添加颜色条和标签
        - 标注电荷位置
    """
    # TODO: 创建图形
    # plt.figure(figsize=(10, 8))
    
    # TODO: 绘制电势分布
    # 提示：使用 plt.imshow(phi, extent=[0, M, 0, M], origin='lower', cmap='RdBu_r')
    
    # TODO: 添加颜色条
    # 提示：plt.colorbar() 和 set_label()
    
    # TODO: 标注电荷位置
    # 可以使用 plt.fill_between() 或 plt.rectangle()
    
    # TODO: 添加标题和标签
    # plt.xlabel(), plt.ylabel(), plt.title()
    
    # TODO: 显示图形
    # plt.show()
    
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def analyze_solution(phi: np.ndarray, iterations: int, converged: bool) -> None:
    """
    分析解的统计信息
    
    参数:
        phi (np.ndarray): 电势分布数组
        iterations (int): 迭代次数
        converged (bool): 收敛状态
    
    功能:
        打印解的基本统计信息，如最大值、最小值、迭代次数等
    """
    # TODO: 打印基本信息
    # print(f"迭代次数: {iterations}")
    # print(f"是否收敛: {converged}")
    # print(f"最大电势: {np.max(phi):.6f} V")
    # print(f"最小电势: {np.min(phi):.6f} V")
    
    # TODO: 找到极值位置
    # 提示：使用 np.unravel_index() 和 np.argmax(), np.argmin()
    
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

if __name__ == "__main__":
    # 测试代码区域
    print("开始求解二维泊松方程...")
    
    # 设置参数
    M = 100
    target = 1e-6
    max_iter = 10000
    
    # TODO: 调用求解函数
    # phi, iterations, converged = solve_poisson_equation(M, target, max_iter)
    
    # TODO: 分析结果
    # analyze_solution(phi, iterations, converged)
    
    # TODO: 可视化结果
    # visualize_solution(phi, M)
    
    print("请实现上述函数以完成项目！")