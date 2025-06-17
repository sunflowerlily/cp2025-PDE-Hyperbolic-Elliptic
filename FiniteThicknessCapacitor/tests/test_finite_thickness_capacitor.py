#!/usr/bin/env python3
"""
Test suite for finite thickness parallel plate capacitor solver
"""

import unittest
import numpy as np
import os
import sys
import scipy.ndimage

# Add parent directory to module search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from student code
#from solution.finite_thickness_capacitor_solution import (
from finite_thickness_capacitor_student import (   
    solve_laplace_sor,
    calculate_charge_density,
    plot_results
)

class TestFiniteThicknessCapacitor(unittest.TestCase):
    
    def setUp(self):
        """Set up test parameters"""
        # Test parameters
        self.nx = 40
        self.ny = 30
        self.plate_thickness = 2
        self.plate_separation = 8
        self.omega = 1.8
        self.dx = 1.0 / (self.nx - 1)
        self.dy = 1.0 / (self.ny - 1)
        self.tolerance = 1e-4
        
    def test_student_basic_functionality_20pts(self):
        """Test basic SOR solver functionality (20 points)"""
        try:
            potential = solve_laplace_sor(
                self.nx, self.ny, self.plate_thickness, self.plate_separation, 
                self.omega, max_iter=1000, tolerance=1e-5
            )
            
            # Check return type and shape
            self.assertIsInstance(potential, np.ndarray)
            self.assertEqual(potential.shape, (self.ny, self.nx))
            
            # Check potential range
            potential_range = np.max(potential) - np.min(potential)
            self.assertGreater(potential_range, 150.0)
            self.assertLess(potential_range, 250.0)
            
        except NotImplementedError:
            self.fail("Student has not implemented solve_laplace_sor function")
        except Exception as e:
            self.fail(f"solve_laplace_sor function failed with error: {str(e)}")

    def test_student_boundary_conditions_15pts(self):
        """Test proper boundary condition implementation (15 points)"""
        try:
            potential = solve_laplace_sor(
                self.nx, self.ny, self.plate_thickness, self.plate_separation, 
                self.omega, max_iter=500, tolerance=1e-4
            )
            
            # Check boundary conditions
            np.testing.assert_allclose(potential[:, 0], 0.0, atol=1e-3)
            np.testing.assert_allclose(potential[:, -1], 0.0, atol=1e-3)
            np.testing.assert_allclose(potential[0, :], 0.0, atol=1e-3)
            np.testing.assert_allclose(potential[-1, :], 0.0, atol=1e-3)
            
        except NotImplementedError:
            self.fail("Student has not implemented solve_laplace_sor function")
        except Exception as e:
            self.fail(f"Boundary condition test failed: {str(e)}")
    
    def test_student_charge_density_calculation_15pts(self):
        """Test charge density calculation (15 points)"""
        try:
            # First get a potential solution
            potential = solve_laplace_sor(
                self.nx, self.ny, self.plate_thickness, self.plate_separation, 
                self.omega, max_iter=500, tolerance=1e-4
            )
            
            # Test charge density calculation
            charge_density = calculate_charge_density(potential, self.dx, self.dy)
            
            # Check return type and shape
            self.assertIsInstance(charge_density, np.ndarray)
            self.assertEqual(charge_density.shape, potential.shape)
            
            # Check that charge density is finite
            self.assertTrue(np.all(np.isfinite(charge_density)))
            
            # Check charge conservation (total charge should be approximately zero)
            total_charge = np.sum(charge_density) * self.dx * self.dy
            self.assertLess(abs(total_charge), 0.5)
            
            # Check that significant charge exists near conductors
            max_charge_density = np.max(np.abs(charge_density))
            self.assertGreater(max_charge_density, 1e-6)
            
        except NotImplementedError:
            self.fail("Student has not implemented calculate_charge_density function")
        except Exception as e:
            self.fail(f"Charge density calculation failed: {str(e)}")
    
    def test_student_plot_function_5pts(self):
        """Test plot function (5 points) - basic execution check"""
        nx, ny = 10, 10
        dx, dy = 1.0, 1.0
        
        # 创建坐标网格
        x_coords = np.arange(0, nx * dx, dx)
        y_coords = np.arange(0, ny * dy, dy)
        
        # 初始化电势和电荷密度（添加变化值）
        potential = np.random.rand(ny, nx) * 100  # 随机值0-100
        charge_density = np.zeros((ny, nx))
        
        try:
            plot_results(potential, charge_density, x_coords, y_coords)
        except Exception as e:
            self.fail(f"plot_results function failed with error: {str(e)}")
    
    def test_student_error_handling_5pts(self):
        """Test error handling and input validation (5 points)"""
        try:
            # Test with invalid parameters
            with self.assertRaises((ValueError, AssertionError, TypeError)):
                solve_laplace_sor(-10, self.ny, self.plate_thickness, 
                                self.plate_separation, self.omega)
            
            # Test with omega outside valid range
            potential, convergence, _ = solve_laplace_sor(
                self.nx, self.ny, self.plate_thickness, self.plate_separation, 
                0.5, max_iter=100, tolerance=1e-4  # omega < 1.0
            )
            # Should either raise error or handle gracefully
            self.assertTrue(len(convergence) > 0)
            
        except NotImplementedError:
            self.fail("Student has not implemented solve_laplace_sor function")
        except Exception:
            # Error handling is implemented (good)
            pass

if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)