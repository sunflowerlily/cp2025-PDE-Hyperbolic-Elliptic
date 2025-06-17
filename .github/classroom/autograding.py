#!/usr/bin/env python3
"""Automated grading script for GitHub Classroom"""
import os, sys, json, subprocess, unittest
from io import StringIO

class AutoGrader:
    def __init__(self):
        self.results = {"tests": []}
        self.total_score = 0
        self.max_score = 0
    
    def run_project_tests(self, project_name, max_points):
        """Run tests for specific project"""
        test_file = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', f'test_{project_name}.py')
        
        # Temporarily redirect stdout to capture test results
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # Discover and run tests
            loader = unittest.TestLoader()
            suite = loader.discover(os.path.dirname(test_file), pattern=os.path.basename(test_file))
            runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
            run_results = runner.run(suite)
            
            output = sys.stdout.getvalue()
            
            # Parse results
            test_count = run_results.testsRun
            failures = len(run_results.failures)
            errors = len(run_results.errors)
            
            score = 0
            feedback = ""
            
            if errors > 0:
                feedback = "Test encountered errors. Check traceback.\n" + output
                score = 0
            elif failures > 0:
                feedback = "Some tests failed. Review your implementation.\n" + output
                score = 0
            else:
                feedback = "All tests passed!\n" + output
                score = max_points # Assign full points if all pass

            self.results["tests"].append({
                "name": f"Project: {project_name}",
                "score": score,
                "max_score": max_points,
                "output": feedback
            })
            self.total_score += score
            self.max_score += max_points
            
        except Exception as e:
            output = sys.stdout.getvalue()
            self.results["tests"].append({
                "name": f"Project: {project_name}",
                "score": 0,
                "max_score": max_points,
                "output": f"Error running tests: {e}\n{output}"
            })
        finally:
            sys.stdout = old_stdout # Restore stdout
        
    def generate_report(self):
        """Generate JSON report for GitHub Classroom"""
        # Final score calculation (optional, if you want to sum up all projects)
        # self.results["score"] = self.total_score
        # self.results["max_score"] = self.max_score
        
        print(json.dumps(self.results, indent=4))

if __name__ == "__main__":
    # Add the project root to the Python path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

    grader = AutoGrader()
    
    # Example usage for this project
    # For actual grading, you would call this for each project
    # For now, we'll just run the reference solution tests to ensure the autograder works
    # In a real scenario, you'd run the student's tests.
    
    # To test the autograder itself, we can run the reference solution tests
    # This assumes the student's solution file is named wave_equation_ftcs_student.py
    # and the tests are in tests/test_wave_equation_ftcs.py
    
    # For GitHub Classroom, the tests would typically be run against the student's code.
    # The test file itself would import the student's solution.
    
    # We will simulate running the student's tests by running the existing test file.
    # The test file already has logic to import student code (if uncommented).
    
    # For this setup, we assume the test file `test_wave_equation_ftcs.py`
    # will be responsible for testing the student's `wave_equation_ftcs_student.py`.
    # The `autograding.py` script just needs to run that test file.
    
    # Note: The current test file `test_wave_equation_ftcs.py` is set up to test
    # both the reference solution and to check if student functions raise NotImplementedError.
    # For actual grading, you'd modify `test_wave_equation_ftcs.py` to test the student's
    # implementation for correctness, not just for NotImplementedError.
    
    # For demonstration, let's run the existing test file.
    # In a real autograding scenario, you'd have specific tests for student code.
    
    # The `run_project_tests` method expects the project name (which maps to the test file name)
    # and max points. We'll use 'wave_equation_ftcs' as the project name.
    grader.run_project_tests('wave_equation_ftcs', 100) # Assuming 100 points for this project
    
    grader.generate_report()