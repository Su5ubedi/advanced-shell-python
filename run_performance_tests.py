#!/usr/bin/env python3
"""
run_performance_tests.py - Script to run performance tests and generate metrics
"""

import subprocess
import sys
import os

def main():
    """Run the performance tests and generate metrics"""
    print("Advanced Shell - Performance Testing Suite")
    print("=" * 50)
    print()
    
    # Check if required files exist
    required_files = [
        "test_scheduling_with_metrics.py",
        "performance_metrics.py",
        "scheduler.py",
        "shell_types.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        return 1
    
    print("Running performance tests...")
    print()
    
    try:
        # Run the enhanced test script
        result = subprocess.run([
            sys.executable, "test_scheduling_with_metrics.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Performance tests completed successfully!")
            print()
            
            # Check if output files were created
            output_files = [
                "test_scheduling_with_metrics_output.txt",
                "performance_metrics_report.txt"
            ]
            
            for file in output_files:
                if os.path.exists(file):
                    print(f"✓ {file} generated successfully")
                else:
                    print(f"✗ {file} was not generated")
            
            print()
            print("Performance Metrics Summary:")
            print("-" * 30)
            
            # Display a summary of the performance report
            if os.path.exists("performance_metrics_report.txt"):
                with open("performance_metrics_report.txt", "r") as f:
                    lines = f.readlines()
                    
                # Find and display key metrics
                for i, line in enumerate(lines):
                    if "OVERALL SUMMARY" in line:
                        print("\nOverall Test Results:")
                        for j in range(i+1, min(i+10, len(lines))):
                            if lines[j].strip() and not lines[j].startswith("-"):
                                print(f"  {lines[j].strip()}")
                        break
                
                # Find algorithm comparison
                for i, line in enumerate(lines):
                    if "ALGORITHM COMPARISON" in line:
                        print("\nAlgorithm Performance:")
                        for j in range(i+1, min(i+20, len(lines))):
                            if lines[j].strip() and not lines[j].startswith("-"):
                                print(f"  {lines[j].strip()}")
                        break
                
                # Find recommendations
                for i, line in enumerate(lines):
                    if "PERFORMANCE RECOMMENDATIONS" in line:
                        print("\nPerformance Recommendations:")
                        for j in range(i+1, min(i+15, len(lines))):
                            if lines[j].strip() and not lines[j].startswith("-"):
                                print(f"  {lines[j].strip()}")
                        break
            
            print()
            print("Detailed reports available in:")
            print("  - test_scheduling_with_metrics_output.txt (Test execution log)")
            print("  - performance_metrics_report.txt (Performance analysis)")
            
        else:
            print("✗ Performance tests failed!")
            print("Error output:")
            print(result.stderr)
            return 1
            
    except Exception as e:
        print(f"✗ Error running performance tests: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())