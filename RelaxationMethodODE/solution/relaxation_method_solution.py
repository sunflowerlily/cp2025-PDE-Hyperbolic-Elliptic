"""Module: Relaxation Method Solution
File: relaxation_method_solution.py
"""
import numpy as np
import matplotlib.pyplot as plt

def solve_ode(h, g, max_iter=10000, tol=1e-6):
    """
    Solve projectile motion ODE using relaxation method
    d²x/dt² = -g with boundary conditions x(0) = x(10) = 0
    
    Args:
        h (float): Time step size
        g (float): Gravitational acceleration
        max_iter (int): Maximum iterations
        tol (float): Convergence tolerance
    Returns:
        tuple: (time array, solution array)
    """
    # Initialize time array
    t = np.arange(0, 10 + h, h)
    
    # Initialize solution array
    x = np.zeros(t.size)
    
    # Apply relaxation iteration
    delta = 1.0
    iteration = 0
    
    while delta > tol and iteration < max_iter:
        x_new = np.copy(x)
        
        x_new[1:-1] = 0.5 * (h * h * g + x[2:] + x[:-2])
        
        # Calculate maximum change
        delta = np.max(np.abs(x_new - x))
        
        # Update solution
        x = x_new
        iteration += 1
    
    return t, x

if __name__ == "__main__":
    # Problem parameters
    h = 10.0 / 100  # Time step
    g = 9.8          # Gravitational acceleration
    
    # Solve the ODE
    t, x = solve_ode(h, g)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(t, x, 'b-', linewidth=2, label='Projectile trajectory')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.title('Projectile Motion using Relaxation Method')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # Print maximum height and time
    max_height = np.max(x)
    max_time = t[np.argmax(x)]
    print(f"Maximum height: {max_height:.2f} m at t = {max_time:.2f} s")