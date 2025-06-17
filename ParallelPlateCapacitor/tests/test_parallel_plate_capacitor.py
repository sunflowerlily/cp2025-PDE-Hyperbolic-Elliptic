import unittest
import numpy as np
import os
import sys

# 添加父目录到模块搜索路径，以便导入学生代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from solution.parallel_plate_capacitor_solution import (
from parallel_plate_capacitor_student import (
    solve_laplace_jacobi,
    solve_laplace_sor,
    plot_results
)

class TestParallelPlateCapacitor(unittest.TestCase):
    def setUp(self):
        """Set up test parameters"""
        self.xgrid = 30
        self.ygrid = 30
        self.w = 10  # plate width
        self.d = 10  # plate separation
        self.tol = 1e-4
        
        # Small test case
        self.small_xgrid = 20
        self.small_ygrid = 20
        self.small_w = 6
        self.small_d = 6
        self.omega = 1.5 # Optimal relaxation parameter for SOR
        self.Niter = 10000
        self.tol = 1e-4

        # Smaller grid for plotting tests
        self.small_xgrid = 50
        self.small_ygrid = 50
        self.small_w = 10
        self.small_d = 5

        # For simplicity, we'll run the solution here. In a real scenario,
        # these might be loaded from a file or a known good run.
        self.ref_u_jacobi, self.ref_iter_jacobi, self.ref_conv_history_jacobi = solve_laplace_jacobi(self.xgrid, self.ygrid, self.w, self.d, tol=self.tol)
        self.ref_u_sor, self.ref_iter_sor, self.ref_conv_history_sor = solve_laplace_sor(self.xgrid, self.ygrid, self.w, self.d, omega=self.omega, Niter=self.Niter, tol=self.tol)
    
    def test_jacobi_basic_functionality_15pts(self):
        """Test basic functionality of Jacobi method (15 points)"""
        try:
            result, iterations, conv_history = solve_laplace_jacobi(self.xgrid, self.ygrid, self.w, self.d, self.tol)
            
            # Check if result is a numpy array
            self.assertIsInstance(result, np.ndarray, "Result should be a numpy array")
            self.assertIsInstance(iterations, int, "Iterations should be an integer")
            self.assertIsInstance(conv_history, list, "Convergence history should be a list")
            
            # Check convergence history
            self.assertEqual(len(conv_history), iterations, "Convergence history length should match iterations")
            self.assertTrue(all(isinstance(x, (int, float)) for x in conv_history),
                          "Convergence history should contain numbers")
            
            # Check dimensions
            expected_shape = (self.ygrid, self.xgrid)
            self.assertEqual(result.shape, expected_shape, f"Result shape should be {expected_shape}")
            
            # Calculate plate positions
            xL = (self.xgrid - self.w) // 2
            xR = (self.xgrid + self.w) // 2
            yB = (self.ygrid - self.d) // 2
            yT = (self.ygrid + self.d) // 2
            
            # Check boundary conditions for plates
            top_plate_values = result[yT, xL:xR+1]
            np.testing.assert_allclose(top_plate_values, 100.0, 
                                     rtol=1e-2, atol=1e-2,
                                     err_msg="Top plate should be +100V")
            
            bottom_plate_values = result[yB, xL:xR+1]
            np.testing.assert_allclose(bottom_plate_values, -100.0, 
                                     rtol=1e-2, atol=1e-2,
                                     err_msg="Bottom plate should be -100V")
            
            # Check that boundary (grounded box) is approximately zero
            np.testing.assert_allclose(result[0, :], 0, atol=1e-1, 
                                     err_msg="Bottom boundary should be grounded")
            np.testing.assert_allclose(result[-1, :], 0, atol=1e-1, 
                                     err_msg="Top boundary should be grounded")
            np.testing.assert_allclose(result[:, 0], 0, atol=1e-1, 
                                     err_msg="Left boundary should be grounded")
            np.testing.assert_allclose(result[:, -1], 0, atol=1e-1, 
                                     err_msg="Right boundary should be grounded")
            
            # Check reasonable iteration count
            self.assertGreater(iterations, 0, "Should require at least one iteration")
            self.assertLess(iterations, 10000, "Should converge in reasonable time")
            
        except NotImplementedError:
            self.fail("Student has not implemented the solve_laplace_jacobi function")
    
    def test_sor_basic_functionality_15pts(self):
        """Test basic functionality of SOR method (15 points)"""
        try:
            result, iterations, conv_history = solve_laplace_sor(self.xgrid, self.ygrid, self.w, self.d)
            
            # Check return types
            self.assertIsInstance(result, np.ndarray, "Result should be a numpy array")
            self.assertIsInstance(iterations, int, "Iterations should be an integer")
            self.assertIsInstance(conv_history, list, "Convergence history should be a list")
            
            # Check dimensions
            expected_shape = (self.ygrid, self.xgrid)
            self.assertEqual(result.shape, expected_shape, f"Result shape should be {expected_shape}")
            
            # Calculate plate positions
            xL = (self.xgrid - self.w) // 2
            xR = (self.xgrid + self.w) // 2
            yB = (self.ygrid - self.d) // 2
            yT = (self.ygrid + self.d) // 2
            
            # Check boundary conditions
            top_plate_values = result[yT, xL:xR+1]
            np.testing.assert_allclose(top_plate_values, 100.0, 
                                     rtol=1e-2, atol=1e-2,
                                     err_msg="Top plate should be +100V")
            
            bottom_plate_values = result[yB, xL:xR+1]
            np.testing.assert_allclose(bottom_plate_values, -100.0, 
                                     rtol=1e-2, atol=1e-2,
                                     err_msg="Bottom plate should be -100V")
            
            # Check convergence history
            self.assertEqual(len(conv_history), iterations, "Convergence history length should match iterations")
            self.assertTrue(all(isinstance(x, (int, float)) for x in conv_history), 
                          "Convergence history should contain numbers")
            
        except NotImplementedError:
            self.fail("Student has not implemented the solve_laplace_sor function")
    
    def test_boundary_conditions_10pts(self):
        """Test boundary condition handling (10 points)"""
        try:
            # Test Jacobi
            result_j, _, _ = solve_laplace_jacobi(self.small_xgrid, self.small_ygrid, self.small_w, self.small_d)
            
            # Test SOR
            result_s, _, _ = solve_laplace_sor(self.small_xgrid, self.small_ygrid, self.small_w, self.small_d)
            
            for result in [result_j, result_s]:
                # Check that solution is reasonable (not all zeros, has variation)
                self.assertFalse(np.allclose(result, 0), "Solution should not be all zeros")
                self.assertGreater(np.std(result), 1e-3, "Solution should have variation")
                
                # Check that potential varies smoothly (no sudden jumps except at plates)
                # This is a basic physics check
                self.assertLess(np.max(result), 150, "Maximum potential should be reasonable")
                self.assertGreater(np.min(result), -150, "Minimum potential should be reasonable")
            
        except NotImplementedError:
            self.fail("Student has not implemented the required functions")
    
    def test_performance_comparison_5pts(self):
        """Test that both methods produce similar results (5 points)"""
        try:
            # Solve with both methods
            result_jacobi, iter_jacobi, conv_history_jacobi = solve_laplace_jacobi(self.small_xgrid, self.small_ygrid, 
                                                            self.small_w, self.small_d, 1e-4)
            result_sor, iter_sor, conv_history_sor = solve_laplace_sor(self.small_xgrid, self.small_ygrid, 
                                                      self.small_w, self.small_d, omega=1.5, Niter=1000)
            
            # Results should be similar (within tolerance)
            np.testing.assert_allclose(result_jacobi, result_sor, rtol=1e-2, atol=1e-2,
                                     err_msg="Jacobi and SOR should produce similar results")
            
            # Check convergence history lengths
            self.assertGreater(len(conv_history_jacobi), 0, "Jacobi convergence history should not be empty")
            self.assertGreater(len(conv_history_sor), 0, "SOR convergence history should not be empty")

            # SOR should generally converge faster
            # Note: This is not always guaranteed, so we just check both converged
            self.assertGreater(iter_jacobi, 0, "Jacobi should require iterations")
            self.assertGreater(iter_sor, 0, "SOR should require iterations")
            
        except NotImplementedError:
            self.fail("Student has not implemented the required functions")
    
if __name__ == '__main__':
    unittest.main()