import unittest
import numpy as np
import os
import sys

# Add parent directory to module search path to import student code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from solution.wave_equation_ftcs_solution import solve_wave_equation_ftcs, u_t
from wave_equation_ftcs_student import solve_wave_equation_ftcs, u_t

class TestWaveEquationFTCS(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            'a': 100,
            'L': 1,
            'd': 0.1,
            'C': 1,
            'sigma': 0.3,
            'dx': 0.01,
            'dt': 5e-5,
            'total_time': 0.1
        }
        # Pre-calculate expected solution using the reference solution
        self.expected_u, self.expected_x, self.expected_t = solve_wave_equation_ftcs(self.parameters)
        self.tolerance = 1e-6

    def test_u_t_function(self):
        """Test the initial velocity profile function u_t."""
        x_test = np.array([0.1, 0.5, 0.9])
        # Expected values for u_t at these points (calculated manually or with a known good implementation)
        # For C=1, d=0.1, sigma=0.3, L=1
        # x=0.1: 1 * 0.1 * 0.9 / 1 / 1 * exp(-(0.1-0.1)^2 / (2*0.3^2)) = 0.09 * exp(0) = 0.09
        # x=0.5: 1 * 0.5 * 0.5 / 1 / 1 * exp(-(0.5-0.1)^2 / (2*0.3^2)) = 0.25 * exp(-0.16 / 0.18) = 0.25 * exp(-0.888...) approx 0.0999
        # x=0.9: 1 * 0.9 * 0.1 / 1 / 1 * exp(-(0.9-0.1)^2 / (2*0.3^2)) = 0.09 * exp(-0.64 / 0.18) = 0.09 * exp(-3.555...) approx 0.0028
        expected_u_t_values = np.array([
            0.09 * np.exp(-(0.1-0.1)**2 / (2*0.3**2)),
            0.5 * 0.5 * np.exp(-(0.5-0.1)**2 / (2*0.3**2)),
            0.9 * 0.1 * np.exp(-(0.9-0.1)**2 / (2*0.3**2))
        ])
        u_t_params = {k: self.parameters[k] for k in ['C', 'd', 'sigma', 'L']}
        np.testing.assert_allclose(u_t(x_test, **u_t_params), expected_u_t_values, rtol=self.tolerance)

    def test_reference_solution(self):
        """Verify reference solution (0 points - validation)"""
        u_result, x_result, t_result = solve_wave_equation_ftcs(self.parameters)
        np.testing.assert_allclose(u_result, self.expected_u, rtol=self.tolerance)
        np.testing.assert_allclose(x_result, self.expected_x, rtol=self.tolerance)
        np.testing.assert_allclose(t_result, self.expected_t, rtol=self.tolerance)


if __name__ == '__main__':
    unittest.main()