"""
学生模板：波动方程FTCS解
文件：wave_equation_ftcs_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt

def u_t(x, C=1, d=0.1, sigma=0.3, L=1):
    """
    计算初始速度剖面 psi(x)。

    参数:
        x (np.ndarray): 位置数组。
        C (float): 振幅常数。
        d (float): 指数项的偏移量。
        sigma (float): 指数项的宽度。
        L (float): 弦的长度。
    返回:
        np.ndarray: 初始速度剖面。
    """
    # TODO: 实现初始速度剖面函数
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

def solve_wave_equation_ftcs(parameters):
    """
    使用FTCS有限差分法求解一维波动方程。
    
    参数:
        parameters (dict): 包含以下参数的字典：
            - 'a': 波速 (m/s)。
            - 'L': 弦的长度 (m)。
            - 'd': 初始速度剖面的偏移量 (m)。
            - 'C': 初始速度剖面的振幅常数 (m/s)。
            - 'sigma': 初始速度剖面的宽度 (m)。
            - 'dx': 空间步长 (m)。
            - 'dt': 时间步长 (s)。
            - 'total_time': 总模拟时间 (s)。
    返回:
        tuple: 包含以下内容的元组：
            - np.ndarray: 解数组 u(x, t)。
            - np.ndarray: 空间数组 x。
            - np.ndarray: 时间数组 t。
    
    物理背景: 描述弦振动的波动方程，初始条件为弦静止，给定初始速度剖面。
    数值方法: 使用有限差分法中的FTCS (Forward-Time Central-Space) 方案。
    
    实现步骤:
    1. 从 parameters 字典中获取所有必要的物理和数值参数。
    2. 初始化空间网格 x 和时间网格 t。
    3. 创建一个零数组 u 来存储解，其维度为 (x.size, t.size)。
    4. 计算稳定性条件 c = (a * dt / dx)^2。如果 c >= 1，打印警告信息。
    5. 应用初始条件：u(x, 0) = 0。
    6. 计算第一个时间步 u(x, 1) 的值，使用初始速度 u_t(x, 0) 和给定的公式。
    7. 使用FTCS方案迭代计算后续时间步的解。
    8. 返回解数组 u、空间数组 x 和时间数组 t。
    """
    # TODO: 验证输入参数
    # TODO: 初始化变量
    # TODO: 计算稳定性条件
    # TODO: 应用初始条件
    # TODO: 实现FTCS主算法
    # TODO: 返回结果
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")


if __name__ == "__main__":
    # Demonstration and testing
    params = {
        'a': 100,
        'L': 1,
        'd': 0.1,
        'C': 1,
        'sigma': 0.3,
        'dx': 0.01,
        'dt': 5e-5,
        'total_time': 0.1
    }
    u_sol, x_sol, t_sol = solve_wave_equation_ftcs(params)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, xlim=(0, params['L']), ylim=(u_sol.min() * 1.1, u_sol.max() * 1.1))
    line, = ax.plot([], [], 'g-', lw=2)
    ax.set_title("1D Wave Equation (FTCS)")
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Displacement")

    def update(frame):
        line.set_data(x_sol, u_sol[:, frame])
        return line,

    ani = FuncAnimation(fig, update, frames=t_sol.size, interval=1, blit=True)
    plt.show()