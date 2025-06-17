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

    def run_all_tests(self):
        """Run tests for all projects"""
        projects = [
            ("ParallelPlateCapacitor", 100),
            ("WaveEquationFTCS", 100),
            ("FiniteThicknessCapacitor", 100)
        ]
        
        for project_name, max_points in projects:
            if os.path.exists(project_name):
                print(f"\nTesting {project_name}...")
                self.run_project_tests(project_name, max_points)
            else:
                print(f"\nProject {project_name} not found, skipping...")
        
        self.generate_report()

if __name__ == "__main__":
    grader = AutoGrader()
    grader.run_all_tests()
