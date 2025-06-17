#!/usr/bin/env python3
"""Automated grading script for GitHub Classroom"""
import os, sys, json, subprocess, unittest, traceback
from io import StringIO
from datetime import datetime

class AutoGrader:
    def __init__(self):
        self.results = {"tests": []}
        self.total_score = 0
        self.max_score = 0
        
    def run_project_tests(self, project_name, max_points):
        """Run tests for specific project with timeout"""
        test_file = f"{project_name}/tests/test_{project_name}.py"
        
        try:
            # 设置超时时间
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=60  # 60秒超时
            )
            
            # 解析测试结果
            if result.returncode == 0:
                score = max_points
                status = "passed"
            else:
                # 部分分数逻辑
                score = self._calculate_partial_score(result.stdout, max_points)
                status = "partial"
                
        except subprocess.TimeoutExpired:
            score = 0
            status = "timeout"
            
        self.results["tests"].append({
            "name": project_name,
            "score": score,
            "max_score": max_points,
            "status": status
        })
        
        self.total_score += score
        self.max_score += max_points
        
    def _calculate_partial_score(self, output, max_points):
        """Calculate partial scores based on test output"""
        # 解析unittest输出，计算通过的测试比例
        passed_tests = 0
        total_tests = 0
        for line in output.splitlines():
            if "... ok" in line:
                passed_tests += 1
            if line.startswith("test_") and "(" in line and ")" in line:
                total_tests += 1
        
        if total_tests > 0:
            return int((passed_tests / total_tests) * max_points)
        return 0
        
    def generate_report(self):
        """Generate JSON report for GitHub Classroom"""
        self.results["score"] = self.total_score
        self.results["max_score"] = self.max_score
        self.results["timestamp"] = datetime.now().isoformat()
        
        with open("autograding_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
            
        print(f"Total Score: {self.total_score}/{self.max_score}")

if __name__ == "__main__":
    grader = AutoGrader()
    
    # 项目配置
    projects = [
        ("ParallelPlateCapacitor", 100) # 假设这个项目总分100分
    ]
    
    for project, points in projects:
        grader.run_project_tests(project, points)
        
    grader.generate_report()