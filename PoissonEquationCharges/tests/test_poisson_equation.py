#!/usr/bin/env python3
"""
Test module for Poisson equation solver
File: test_poisson_equation.py
"""

import unittest
import numpy as np
import os
import sys

# Add parent directory to module search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from student code
from poisson_equation_student import (
    solve_poisson_equation,
    visualize_solution,
    analyze_solution
)

class TestPoissonEquation(unittest.TestCase):
    
    def setUp(self):
        """Set up test cases"""
        self.M_small = 20  # Small grid for fast testing
        self.M_medium = 50  # Medium grid for accuracy testing
        self.target = 1e-4  # Relaxed tolerance for testing
        self.max_iter = 1000
        
    def test_reference_solution(self):
        """Verify reference solution works (0 points - validation)"""
        # Import reference solution for validation
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'solution'))
        from poisson_equation_solution import solve_poisson_equation as ref_solve
        
        phi, iterations, converged = ref_solve(self.M_small, self.target, self.max_iter)
        
        # Basic checks
        self.assertEqual(phi.shape, (self.M_small + 1, self.M_small + 1))
        self.assertTrue(converged)
        self.assertGreater(iterations, 0)
        self.assertLess(iterations, self.max_iter)
        
        # Physical checks
        self.assertTrue(np.all(np.isfinite(phi)))
        self.assertAlmostEqual(np.max(phi[0, :]), 0.0, places=10)  # Top boundary
        self.assertAlmostEqual(np.max(phi[-1, :]), 0.0, places=10)  # Bottom boundary
        self.assertAlmostEqual(np.max(phi[:, 0]), 0.0, places=10)  # Left boundary
        self.assertAlmostEqual(np.max(phi[:, -1]), 0.0, places=10)  # Right boundary
    
    def test_student_basic_functionality_20pts(self):
        """Test basic functionality (20 points)"""
        try:
            phi, iterations, converged = solve_poisson_equation(self.M_small, self.target, self.max_iter)
            
            # Check return types and shapes
            self.assertIsInstance(phi, np.ndarray)
            self.assertIsInstance(iterations, int)
            self.assertIsInstance(converged, bool)
            self.assertEqual(phi.shape, (self.M_small + 1, self.M_small + 1))
            
            # Check convergence
            self.assertTrue(converged, "Solution should converge")
            self.assertGreater(iterations, 0, "Should require at least one iteration")
            self.assertLess(iterations, self.max_iter, "Should converge before max iterations")
            
        except NotImplementedError:
            self.fail("Student has not implemented solve_poisson_equation function")
    
    def test_student_boundary_conditions_10pts(self):
        """Test boundary conditions (10 points)"""
        try:
            phi, _, converged = solve_poisson_equation(self.M_small, self.target, self.max_iter)
            
            if converged:
                # Check all boundaries are zero
                np.testing.assert_allclose(phi[0, :], 0.0, atol=1e-10, 
                                         err_msg="Top boundary should be zero")
                np.testing.assert_allclose(phi[-1, :], 0.0, atol=1e-10, 
                                         err_msg="Bottom boundary should be zero")
                np.testing.assert_allclose(phi[:, 0], 0.0, atol=1e-10, 
                                         err_msg="Left boundary should be zero")
                np.testing.assert_allclose(phi[:, -1], 0.0, atol=1e-10, 
                                         err_msg="Right boundary should be zero")
            
        except NotImplementedError:
            self.fail("Student has not implemented the function")
    
    def test_student_physical_properties_10pts(self):
        """Test physical properties of solution (10 points)"""
        try:
            phi, _, converged = solve_poisson_equation(self.M_medium, self.target, self.max_iter)
            
            if converged:
                # Solution should be finite everywhere
                self.assertTrue(np.all(np.isfinite(phi)), "Solution should be finite everywhere")
                
                # Should have both positive and negative regions due to charges
                self.assertGreater(np.max(phi), 0.01, "Should have positive potential regions")
                self.assertLess(np.min(phi), -0.01, "Should have negative potential regions")
                
                # Check charge regions have expected signs
                # For M=50, scale the charge positions proportionally
                # Original: pos (60:80, 20:40), neg (20:40, 60:80) for M=100
                # Scaled: pos (30:40, 10:20), neg (10:20, 30:40) for M=50
                pos_charge_region = phi[30:40, 10:20]  # Scaled for medium grid
                neg_charge_region = phi[10:20, 30:40]  # Scaled for medium grid
                
                self.assertGreater(np.mean(pos_charge_region), np.mean(neg_charge_region),
                                 "Positive charge region should have higher potential")
            
        except NotImplementedError:
            self.fail("Student has not implemented the function")
    
    def test_student_convergence_behavior_5pts(self):
        """Test convergence behavior (5 points)"""
        try:
            # Test with different tolerances
            phi1, iter1, conv1 = solve_poisson_equation(self.M_small, 1e-3, self.max_iter)
            phi2, iter2, conv2 = solve_poisson_equation(self.M_small, 1e-5, self.max_iter)
            
            if conv1 and conv2:
                # Stricter tolerance should require more iterations
                self.assertGreaterEqual(iter2, iter1, 
                                      "Stricter tolerance should require more iterations")
                
                # Solutions should be similar but more accurate with stricter tolerance
                diff = np.max(np.abs(phi2 - phi1))
                self.assertLess(diff, 0.1, "Solutions with different tolerances should be similar")
            
        except NotImplementedError:
            self.fail("Student has not implemented the function")
    
    def test_student_grid_size_effect_5pts(self):
        """Test effect of grid size (5 points)"""
        try:
            # Test with different grid sizes
            phi_small, _, conv_small = solve_poisson_equation(20, self.target, self.max_iter)
            phi_large, _, conv_large = solve_poisson_equation(40, self.target, self.max_iter)
            
            if conv_small and conv_large:
                # Both should converge
                self.assertTrue(conv_small and conv_large, "Both grid sizes should converge")
                
                # Check that solutions have correct shapes
                self.assertEqual(phi_small.shape, (21, 21))
                self.assertEqual(phi_large.shape, (41, 41))
                
                # Larger grid should provide more detailed solution
                self.assertGreater(phi_large.size, phi_small.size)
            
        except NotImplementedError:
            self.fail("Student has not implemented the function")
    
    def test_visualization_function(self):
        """Test visualization function (bonus - no points)"""
        try:
            phi, _, converged = solve_poisson_equation(self.M_small, self.target, self.max_iter)
            
            if converged:
                # This should not raise an exception
                visualize_solution(phi, self.M_small)
                
        except NotImplementedError:
            # This is acceptable for visualization function
            pass
        except Exception as e:
            # Other exceptions might indicate implementation issues
            self.fail(f"Visualization function raised unexpected error: {e}")
    
    def test_analysis_function(self):
        """Test analysis function (bonus - no points)"""
        try:
            phi, iterations, converged = solve_poisson_equation(self.M_small, self.target, self.max_iter)
            
            if converged:
                # This should not raise an exception
                analyze_solution(phi, iterations, converged)
                
        except NotImplementedError:
            # This is acceptable for analysis function
            pass
        except Exception as e:
            # Other exceptions might indicate implementation issues
            self.fail(f"Analysis function raised unexpected error: {e}")

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)