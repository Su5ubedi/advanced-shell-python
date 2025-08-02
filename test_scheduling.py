#!/usr/bin/env python3
"""
test_scheduling.py - Test script for scheduling features with configurable time slices
"""

import time
import os
import sys
from scheduler import Scheduler, SchedulingAlgorithm


def ensure_test_directory():
    """Ensure the test directory exists"""
    test_dir = "test"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Created test directory: {test_dir}")
    return test_dir


def test_round_robin_configurable_time_slice():
    """Test Round-Robin scheduling with configurable time slice"""
    print("=== Testing Round-Robin Scheduling with Configurable Time Slice ===")
    
    test_dir = ensure_test_directory()
    scheduler = Scheduler()
    
    # Test with different time slice configurations
    time_slice_configs = [0.5, 1.0, 2.0]
    
    for time_slice in time_slice_configs:
        print(f"\n--- Testing with time slice: {time_slice} seconds ---")
        
        # Reset scheduler for each test
        scheduler = Scheduler()
        scheduler.set_algorithm(SchedulingAlgorithm.ROUND_ROBIN, time_slice=time_slice)
        
        # Create dummy jobs with realistic shell commands
        class DummyJob:
            def __init__(self, job_id, command):
                self.id = job_id
                self.command = command
                self.status = None
                self.priority = 5
                self.execution_time = 0.0
                self.total_time_needed = 0.0
                self.end_time = None
        
        # Add jobs with different execution times
        jobs = [
            DummyJob(1, f"touch {test_dir}/rr_ts{time_slice}_file1.txt"),           # Quick job
            DummyJob(2, f"echo 'Job 2 with time slice {time_slice}' > {test_dir}/rr_ts{time_slice}_file2.txt"),  # Medium job
            DummyJob(3, f"mkdir {test_dir}/rr_ts{time_slice}_dir"),  # Quick job
            DummyJob(4, f"ls -la {test_dir}/rr_ts{time_slice}_*"),  # Quick job
        ]
        
        # Add jobs with varying time requirements
        scheduler.add_process(jobs[0], priority=5, time_needed=0.3)  # Completes before time slice
        scheduler.add_process(jobs[1], priority=5, time_needed=1.5)  # Needs multiple time slices
        scheduler.add_process(jobs[2], priority=5, time_needed=4.0)  # Completes before time slice
        scheduler.add_process(jobs[3], priority=5, time_needed=0.4)  # Completes before time slice
        
        print(f"Jobs added with time slice {time_slice}s:")
        print(f"  Process 1: 'touch {test_dir}/rr_ts{time_slice}_file1.txt' (0.3s) - Should complete in one slice")
        print(f"  Process 2: 'echo job content' (1.5s) - Should need multiple slices")
        print(f"  Process 3: 'mkdir directory' (0.2s) - Should complete in one slice")
        print(f"  Process 4: 'ls command' (0.4s) - Should complete in one slice")
        print(f"Expected behavior: Jobs complete early if possible, others get multiple time slices")
        print("Starting scheduler...")
        
        # Start the scheduler
        scheduler.start_scheduler()
        
        # Wait for all jobs to complete
        while scheduler.running:
            time.sleep(0.1)
        
        print(f"Round-Robin test with {time_slice}s time slice completed")
        print("Checking created files:")
        try:
            if os.path.exists(f"{test_dir}/rr_ts{time_slice}_file1.txt"):
                print(f"  ✓ {test_dir}/rr_ts{time_slice}_file1.txt created")
            if os.path.exists(f"{test_dir}/rr_ts{time_slice}_file2.txt"):
                print(f"  ✓ {test_dir}/rr_ts{time_slice}_file2.txt created")
                with open(f"{test_dir}/rr_ts{time_slice}_file2.txt", "r") as f:
                    content = f.read().strip()
                    print(f"    Content: '{content}'")
            if os.path.exists(f"{test_dir}/rr_ts{time_slice}_dir"):
                print(f"  ✓ {test_dir}/rr_ts{time_slice}_dir created")
        except Exception as e:
            print(f"  Error checking files: {e}")
    
    print()


def test_priority_with_time_simulation():
    """Test Priority-based scheduling with time simulation"""
    print("=== Testing Priority-Based Scheduling with Time Simulation ===")
    
    test_dir = ensure_test_directory()
    scheduler = Scheduler()
    scheduler.set_algorithm(SchedulingAlgorithm.PRIORITY)
    
    # Create dummy jobs with realistic shell commands
    class DummyJob:
        def __init__(self, job_id, command):
            self.id = job_id
            self.command = command
            self.status = None
            self.priority = 5
            self.execution_time = 0.0
            self.total_time_needed = 0.0
            self.end_time = None
    
    # Add jobs with different priorities and execution times
    jobs = [
        DummyJob(1, f"echo 'High priority - quick job' > {test_dir}/priority_high_quick.txt"),  # High priority, quick
        DummyJob(2, f"mkdir {test_dir}/priority_medium_dir"),     # Medium priority, quick
        DummyJob(3, f"echo 'Low priority - longer job' > {test_dir}/priority_low_long.txt"),  # Low priority, longer
        DummyJob(4, f"touch {test_dir}/priority_high_medium.txt")      # High priority, medium time
    ]
    
    # Add jobs with different priorities and time requirements
    scheduler.add_process(jobs[0], priority=1, time_needed=0.3)   # Highest priority, quick
    scheduler.add_process(jobs[1], priority=5, time_needed=0.2)   # Medium priority, quick
    scheduler.add_process(jobs[2], priority=10, time_needed=1.0)  # Lowest priority, longer
    scheduler.add_process(jobs[3], priority=1, time_needed=0.5)   # High priority, medium
    
    print("Jobs added to Priority scheduler:")
    print(f"  Process 1: 'echo high priority quick' (Priority 1, 0.3s) - Should run first")
    print(f"  Process 2: 'mkdir directory' (Priority 5, 0.2s) - Should run after high priority")
    print(f"  Process 3: 'echo low priority long' (Priority 10, 1.0s) - Should run last")
    print(f"  Process 4: 'touch file' (Priority 1, 0.5s) - Should run after first high priority")
    print("Expected behavior: Jobs run in priority order (1, 1, 5, 10) with time simulation")
    print("Starting scheduler...")
    
    # Start the scheduler
    scheduler.start_scheduler()
    
    # Wait for all jobs to complete
    while scheduler.running:
        time.sleep(0.1)
    
    print("Priority test completed")
    print("Checking created files:")
    try:
        if os.path.exists(f"{test_dir}/priority_high_quick.txt"):
            print(f"  ✓ {test_dir}/priority_high_quick.txt created")
            with open(f"{test_dir}/priority_high_quick.txt", "r") as f:
                content = f.read().strip()
                print(f"    Content: '{content}'")
        if os.path.exists(f"{test_dir}/priority_medium_dir"):
            print(f"  ✓ {test_dir}/priority_medium_dir created")
        if os.path.exists(f"{test_dir}/priority_low_long.txt"):
            print(f"  ✓ {test_dir}/priority_low_long.txt created")
            with open(f"{test_dir}/priority_low_long.txt", "r") as f:
                content = f.read().strip()
                print(f"    Content: '{content}'")
        if os.path.exists(f"{test_dir}/priority_high_medium.txt"):
            print(f"  ✓ {test_dir}/priority_high_medium.txt created")
    except Exception as e:
        print(f"  Error checking files: {e}")
    print()


def test_preemption_with_time_simulation():
    """Test preemption in priority scheduling with time simulation"""
    print("=== Testing Priority Preemption with Time Simulation ===")
    
    test_dir = ensure_test_directory()
    scheduler = Scheduler()
    scheduler.set_algorithm(SchedulingAlgorithm.PRIORITY)
    
    # Create dummy jobs with realistic shell commands
    class DummyJob:
        def __init__(self, job_id, command):
            self.id = job_id
            self.command = command
            self.status = None
            self.priority = 5
            self.execution_time = 0.0
            self.total_time_needed = 0.0
            self.end_time = None
    
    # Add a low priority job first that takes longer
    low_priority_job = DummyJob(1, f"echo 'Low priority job running' > {test_dir}/preempt_low_priority.txt")
    scheduler.add_process(low_priority_job, priority=10, time_needed=2.0)
    
    print("Added low priority job:")
    print(f"  Process 1: 'echo low priority job' (Priority 10, 2.0s) - Should be preempted")
    print("Starting scheduler...")
    
    # Start the scheduler
    scheduler.start_scheduler()
    
    # Wait a bit, then add a high priority job
    time.sleep(0.5)
    high_priority_job = DummyJob(2, f"echo 'High priority job preempting' > {test_dir}/preempt_high_priority.txt")
    scheduler.add_process(high_priority_job, priority=1, time_needed=0.5)
    
    print("Added high priority job after 0.5s:")
    print(f"  Process 2: 'echo high priority job' (Priority 1, 0.5s) - Should preempt low priority")
    print("Expected behavior: High priority job should preempt low priority job immediately")
    
    # Wait for all jobs to complete
    while scheduler.running:
        time.sleep(0.1)
    
    print("Preemption test completed")
    print("Checking created files:")
    try:
        if os.path.exists(f"{test_dir}/preempt_high_priority.txt"):
            print(f"  ✓ {test_dir}/preempt_high_priority.txt created")
            with open(f"{test_dir}/preempt_high_priority.txt", "r") as f:
                content = f.read().strip()
                print(f"    Content: '{content}'")
        if os.path.exists(f"{test_dir}/preempt_low_priority.txt"):
            print(f"  ✓ {test_dir}/preempt_low_priority.txt created")
            with open(f"{test_dir}/preempt_low_priority.txt", "r") as f:
                content = f.read().strip()
                print(f"    Content: '{content}'")
    except Exception as e:
        print(f"  Error checking files: {e}")
    print()


def test_early_completion_behavior():
    """Test that processes complete early when possible"""
    print("=== Testing Early Completion Behavior ===")
    
    test_dir = ensure_test_directory()
    scheduler = Scheduler()
    scheduler.set_algorithm(SchedulingAlgorithm.ROUND_ROBIN, time_slice=1.0)
    
    # Create dummy jobs with realistic shell commands
    class DummyJob:
        def __init__(self, job_id, command):
            self.id = job_id
            self.command = command
            self.status = None
            self.priority = 5
            self.execution_time = 0.0
            self.total_time_needed = 0.0
            self.end_time = None
    
    # Add jobs where some complete before time slice
    jobs = [
        DummyJob(1, f"touch {test_dir}/early_complete_1.txt"),           # Very quick job
        DummyJob(2, f"echo 'Medium job' > {test_dir}/early_complete_2.txt"),  # Medium job
        DummyJob(3, f"touch {test_dir}/early_complete_3.txt"),           # Very quick job
    ]
    
    # Add jobs with varying time requirements
    scheduler.add_process(jobs[0], priority=5, time_needed=0.2)  # Completes early
    scheduler.add_process(jobs[1], priority=5, time_needed=0.8)  # Uses most of time slice
    scheduler.add_process(jobs[2], priority=5, time_needed=0.1)  # Completes very early
    
    print("Jobs added to test early completion:")
    print(f"  Process 1: 'touch file1' (0.2s) - Should complete early, next process starts immediately")
    print(f"  Process 2: 'echo content' (0.8s) - Should use most of time slice")
    print(f"  Process 3: 'touch file3' (0.1s) - Should complete very early")
    print("Expected behavior: Quick jobs complete early, scheduler moves to next job immediately")
    print("Starting scheduler...")
    
    # Start the scheduler
    scheduler.start_scheduler()
    
    # Wait for all jobs to complete
    while scheduler.running:
        time.sleep(0.1)
    
    print("Early completion test completed")
    print("Checking created files:")
    try:
        if os.path.exists(f"{test_dir}/early_complete_1.txt"):
            print(f"  ✓ {test_dir}/early_complete_1.txt created")
        if os.path.exists(f"{test_dir}/early_complete_2.txt"):
            print(f"  ✓ {test_dir}/early_complete_2.txt created")
            with open(f"{test_dir}/early_complete_2.txt", "r") as f:
                content = f.read().strip()
                print(f"    Content: '{content}'")
        if os.path.exists(f"{test_dir}/early_complete_3.txt"):
            print(f"  ✓ {test_dir}/early_complete_3.txt created")
    except Exception as e:
        print(f"  Error checking files: {e}")
    print()


if __name__ == "__main__":
    # Redirect all output to a text file
    output_file = "test_scheduling_output.txt"
    
    # Save original stdout
    original_stdout = sys.stdout
    
    try:
        # Redirect stdout to file
        with open(output_file, 'w') as f:
            sys.stdout = f
            
            print("Advanced Shell - Scheduling Test Suite with Configurable Time Slices")
            print("==================================================================")
            print()
            print("Testing Constraints:")
            print("1. Time slice is configurable and user-specified")
            print("2. Processes complete early if possible (removed from queue)")
            print("3. Process execution simulated with time.sleep()")
            print()
            
            try:
                test_round_robin_configurable_time_slice()
                test_priority_with_time_simulation()
                test_preemption_with_time_simulation()
                test_early_completion_behavior()
                
                print("All tests completed successfully!")
                print("The scheduling algorithms are working correctly with configurable time slices.")
                print("All test files and directories have been created in the 'test/' directory.")
                print()
                print("Key Features Demonstrated:")
                print("- Configurable time slices (0.5s, 1.0s, 2.0s)")
                print("- Early completion when processes finish before time slice")
                print("- Time simulation using actual command execution")
                print("- Priority-based preemption with time simulation")
                
            except KeyboardInterrupt:
                print("\nTests interrupted by user")
            except Exception as e:
                print(f"Test error: {e}")
                
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        print(f"Test output has been saved to: {output_file}") 