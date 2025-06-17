import unittest
import numpy as np
import os
import sys

# Add parent directory to module search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from solution for testing reference implementation
#from solution.relaxation_method_solution import solve_ode
# Import from student file for testing student implementation
from relaxation_method_student import solve_ode

class TestRelaxationMethod(unittest.TestCase):
    def setUp(self):
        self.h = 10.0 / 100  # Time step
        self.g = 9.8         # Gravitational acceleration
        self.tol = 1e-6      # Tolerance
    
    def test_reference_solution_0pts(self):
        """Verify reference solution (0 points - validation)"""
        t, x = solve_ode(self.h, self.g, tol=self.tol)
        
        # Check boundary conditions
        self.assertAlmostEqual(x[0], 0.0, places=10, msg="Initial condition x(0) = 0 not satisfied")
        self.assertAlmostEqual(x[-1], 0.0, places=10, msg="Final condition x(10) = 0 not satisfied")
        
        # Check that solution is finite
        self.assertTrue(np.all(np.isfinite(x)), "Solution contains non-finite values")
        
        # Check that maximum height is positive (projectile goes up)
        max_height = np.max(x)
        self.assertGreater(max_height, 0, "Maximum height should be positive")
        
        # Check symmetry (projectile motion should be symmetric)
        mid_point = len(x) // 2
        left_half = x[:mid_point]
        right_half = x[-mid_point:][::-1]  # Reverse right half
        np.testing.assert_allclose(left_half, right_half, rtol=1e-3, 
                                 err_msg="Solution should be symmetric")
    
    def test_student_basic_functionality_15pts(self):
        """Test student basic functionality (15 points)"""
        try:
            # This would test student implementation
            # t, x = solve_ode(self.h, self.g, tol=self.tol)
            
            # For now, test reference solution
            t, x = solve_ode(self.h, self.g, tol=self.tol)
            
            # Check return format
            self.assertIsInstance(t, np.ndarray, "Time should be numpy array")
            self.assertIsInstance(x, np.ndarray, "Solution should be numpy array")
            self.assertEqual(len(t), len(x), "Time and solution arrays should have same length")
            
            # Check boundary conditions
            self.assertAlmostEqual(x[0], 0.0, places=6)
            self.assertAlmostEqual(x[-1], 0.0, places=6)
            
        except NotImplementedError:
            self.fail("Student has not implemented the solve_ode function")
    
    def test_student_convergence_10pts(self):
        """Test convergence properties (10 points)"""
        try:
            t, x = solve_ode(self.h, self.g, tol=1e-4)  # Looser tolerance
            
            # Check that solution converged to reasonable values
            max_height = np.max(x)
            self.assertGreater(max_height, 10, "Maximum height too low")
            self.assertLess(max_height, 200, "Maximum height too high")
            
            # Check time at maximum height (should be around 5 seconds)
            max_time = t[np.argmax(x)]
            self.assertGreater(max_time, 4, "Time at max height too early")
            self.assertLess(max_time, 6, "Time at max height too late")
            
        except NotImplementedError:
            self.fail("Student has not implemented the solve_ode function")
    
    def test_student_different_parameters_5pts(self):
        """Test with different parameters (5 points)"""
        try:
            # Test with different time step
            h_coarse = 10.0 / 50  # Coarser grid
            t_coarse, x_coarse = solve_ode(h_coarse, self.g, tol=1e-4)
            
            # Should still satisfy boundary conditions
            self.assertAlmostEqual(x_coarse[0], 0.0, places=4)
            self.assertAlmostEqual(x_coarse[-1], 0.0, places=4)
            
            # Test with different gravity
            g_moon = 1.6  # Moon gravity
            t_moon, x_moon = solve_ode(self.h, g_moon, tol=1e-4)
            
            # With same boundary conditions but lower gravity, trajectory should be lower
            # because less initial velocity is needed to return to zero in 10 seconds
            max_height_earth = np.max(solve_ode(self.h, self.g, tol=1e-4)[1])
            max_height_moon = np.max(x_moon)
            self.assertLess(max_height_moon, max_height_earth, 
                          "Moon trajectory should be lower than Earth for same boundary conditions")
            
        except NotImplementedError:
            self.fail("Student has not implemented the solve_ode function")

if __name__ == '__main__':
    unittest.main()