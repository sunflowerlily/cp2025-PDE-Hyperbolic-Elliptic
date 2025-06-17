"""
学生模板：松弛迭代法解常微分方程
文件：relaxation_method_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt

def solve_ode(h, g, max_iter=10000, tol=1e-6):
    """
    实现松弛迭代法求解常微分方程 d²x/dt² = -g
    边界条件：x(0) = x(10) = 0（抛体运动问题）
    
    参数:
        h (float): 时间步长
        g (float): 重力加速度
        max_iter (int): 最大迭代次数
        tol (float): 收敛容差
    
    返回:
        tuple: (时间数组, 解数组)
    
    物理背景: 质量为1kg的球从高度x=0抛出，10秒后回到x=0
    数值方法: 松弛迭代法，迭代公式 x(t) = 0.5*h²*g + 0.5*[x(t+h)+x(t-h)]
    
    实现步骤:
    1. 初始化时间数组和解数组
    2. 应用松弛迭代公式直到收敛
    3. 返回时间和解数组
    """
    # 初始化时间数组
    t = np.arange(0, 10 + h, h)
    
    # 初始化解数组，边界条件已满足：x[0] = x[-1] = 0
    x = np.zeros(t.size)
    
    # TODO: 实现松弛迭代算法
    # 提示：
    # 1. 设置初始变化量 delta = 1.0
    # 2. 当 delta > tol 时继续迭代
    # 3. 对内部点应用公式：x_new[1:-1] = 0.5 * (h*h*g + x[2:] + x[:-2])
    # 4. 计算最大变化量：delta = np.max(np.abs(x_new - x))
    # 5. 更新解：x = x_new
    
    raise NotImplementedError(f"请在 {__file__} 中实现此函数")

if __name__ == "__main__":
    # 测试参数
    h = 10 / 100  # 时间步长
    g = 9.8       # 重力加速度
    
    # 调用求解函数
    t, x = solve_ode(h, g)
    
    # 绘制结果
    plt.plot(t, x)
    plt.xlabel('时间 (s)')
    plt.ylabel('高度 (m)')
    plt.title('抛体运动轨迹 (松弛迭代法)')
    plt.grid()
    plt.show()