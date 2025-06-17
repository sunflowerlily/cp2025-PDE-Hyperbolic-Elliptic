"""
Module: WaveEquationFTCS Solution
File: wave_equation_ftcs_solution.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def u_t(x, C=1, d=0.1, sigma=0.3, L=1):
    """
    Calculates the initial velocity profile psi(x).
    Args:
        x (np.ndarray): Position array.
        C (float): Amplitude constant.
        d (float): Offset for the exponential term.
        sigma (float): Width of the exponential term.
        L (float): Length of the string.
    Returns:
        np.ndarray: Initial velocity profile.
    """
    return C * x * (L - x) / L / L * np.exp(-(x - d)**2 / (2 * sigma**2))

def solve_wave_equation_ftcs(parameters):
    """
    Solves the 1D wave equation using the FTCS finite difference method.

    Args:
        parameters (dict): A dictionary containing the following parameters:
            - 'a': Wave speed (m/s).
            - 'L': Length of the string (m).
            - 'd': Offset for the initial velocity profile (m).
            - 'C': Amplitude constant for the initial velocity profile (m/s).
            - 'sigma': Width of the initial velocity profile (m).
            - 'dx': Spatial step size (m).
            - 'dt': Time step size (s).
            - 'total_time': Total simulation time (s).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The solution array u(x, t).
            - np.ndarray: The spatial array x.
            - np.ndarray: The time array t.
    """
    a = parameters.get('a', 100)
    L = parameters.get('L', 1)
    d = parameters.get('d', 0.1)
    C = parameters.get('C', 1)
    sigma = parameters.get('sigma', 0.3)
    dx = parameters.get('dx', 0.01)
    dt = parameters.get('dt', 5e-5)
    total_time = parameters.get('total_time', 0.1)

    x = np.arange(0, L + dx, dx)
    t = np.arange(0, total_time + dt, dt)
    u = np.zeros((x.size, t.size), float)

    # Stability condition check (c < 1)
    c_val = (a * dt / dx)**2
    if c_val >= 1:
        print(f"Warning: Stability condition c = {c_val} >= 1. Solution may be unstable.")

    # Initial conditions: u(x, 0) = 0 (string at rest)
    # u(x, 1) calculation using initial velocity u_t(x, 0)
    # u_i,1 = c/2 * (u_i+1,0 + u_i-1,0) + (1-c) * u_i,0 + u_t(x,0) * dt
    # Since u_i,0 = 0, this simplifies to:
    # u_i,1 = u_t(x,0) * dt
    # The provided formula in the markdown is:
    # u_i,1 = c/2 * (u_i+1,0 + u_i-1,0) + (1-c) * u_i,0 + u_t(x,0) * dt
    # This formula is for a general case where u_i,0 might not be zero.
    # Given u(x,0) = 0, the terms with u_i,0 become zero.
    # So, u[1:-1, 1] = u_t(x[1:-1]) * dt should be sufficient if u_i,0 is strictly 0.
    # However, the provided markdown code uses:
    # u[1:-1,1] = c/2*(u[2:,0]+u[:-2,0])+(1-c)*u[1:-1,0]+u_t(x[1:-1])*dt
    # Let's stick to the provided code's implementation for u[1:-1,1] for consistency.
    # Note: u[2:,0], u[:-2,0], u[1:-1,0] are all zeros due to np.zeros initialization.
    # So, u[1:-1,1] effectively becomes u_t(x[1:-1]) * dt.
    u[1:-1, 1] = u_t(x[1:-1], C, d, sigma, L) * dt

    # FTCS scheme for subsequent time steps
    # u_i,j+1 = c * (u_i+1,j + u_i-1,j) + 2 * (1-c) * u_i,j - u_i,j-1
    for j in range(1, t.size - 1):
        u[1:-1, j + 1] = c_val * (u[2:, j] + u[:-2, j]) + 2 * (1 - c_val) * u[1:-1, j] - u[1:-1, j - 1]

    return u, x, t

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