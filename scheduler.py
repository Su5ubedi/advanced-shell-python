#!/usr/bin/env python3
"""
scheduler.py - Process scheduling algorithms for Advanced Shell Simulation
"""

import time
import threading
import subprocess
import os
from typing import List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from queue import PriorityQueue
import heapq

from shell_types import Job, JobStatus


class SchedulingAlgorithm(Enum):
    """Scheduling algorithm types"""
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"


@dataclass
class ProcessInfo:
    """Information about a process for scheduling"""
    job: Job
    arrival_time: float
    priority: int
    time_needed: float
    time_executed: float = 0.0
    time_slice_remaining: float = 0.0

    def __lt__(self, other):
        """For priority queue ordering"""
        if self.priority != other.priority:
            return self.priority < other.priority  # Lower number = higher priority
        return self.arrival_time < other.arrival_time


class Scheduler:
    """Process scheduler implementation"""

    def __init__(self):
        self.algorithm = SchedulingAlgorithm.ROUND_ROBIN
        self.time_slice = 2.0  # Default time slice for Round-Robin (seconds)
        self.processes: List[ProcessInfo] = []
        self.running_process: Optional[ProcessInfo] = None
        self.scheduler_thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.Lock()
        self.on_process_complete: Optional[Callable[[Job], None]] = None

    def set_algorithm(self, algorithm: SchedulingAlgorithm, time_slice: float = 2.0):
        """Set the scheduling algorithm and parameters"""
        with self.lock:
            self.algorithm = algorithm
            if algorithm == SchedulingAlgorithm.ROUND_ROBIN:
                self.time_slice = time_slice
            print(f"Scheduler set to {algorithm.value}")
            if algorithm == SchedulingAlgorithm.ROUND_ROBIN:
                print(f"Time slice: {time_slice} seconds")

    def add_process(self, job: Job, priority: int = 5, time_needed: float = 5.0):
        """Add a process to the scheduler"""
        with self.lock:
            process_info = ProcessInfo(
                job=job,
                arrival_time=time.time(),
                priority=priority,
                time_needed=time_needed,
                time_slice_remaining=self.time_slice
            )
            
            # Set job status to waiting
            job.status = JobStatus.WAITING
            job.priority = priority
            job.total_time_needed = time_needed
            
            self.processes.append(process_info)
            print(f"Added process {job.id} (Priority: {priority}, Time needed: {time_needed}s)")
            
            # If no process is running, start scheduling
            if not self.running_process and not self.running:
                self.start_scheduler()

    def remove_process(self, job_id: int):
        """Remove a process from the scheduler"""
        with self.lock:
            # Remove from processes list
            self.processes = [p for p in self.processes if p.job.id != job_id]
            
            # If it's the currently running process, stop it
            if self.running_process and self.running_process.job.id == job_id:
                self.running_process.job.status = JobStatus.DONE
                self.running_process = None

    def start_scheduler(self):
        """Start the scheduler thread"""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            return

        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        print("Scheduler started")

    def stop_scheduler(self):
        """Stop the scheduler"""
        with self.lock:
            self.running = False
            if self.running_process:
                self.running_process.job.status = JobStatus.STOPPED
                self.running_process = None

    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                with self.lock:
                    if not self.processes:
                        self.running = False
                        break

                    if self.algorithm == SchedulingAlgorithm.ROUND_ROBIN:
                        self._round_robin_schedule()
                    elif self.algorithm == SchedulingAlgorithm.PRIORITY:
                        self._priority_schedule()

            except Exception as e:
                print(f"Scheduler error: {e}")
                time.sleep(0.1)

    def _execute_command(self, command: str, timeout: float) -> tuple[bool, str, float]:
        """
        Execute a shell command with timeout
        
        Returns:
            tuple: (success, output, execution_time)
        """
        try:
            start_time = time.time()
            
            # Execute the command with timeout
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                return True, result.stdout, execution_time
            else:
                return False, result.stderr, execution_time
                
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout} seconds", timeout
        except Exception as e:
            return False, str(e), 0.0

    def _round_robin_schedule(self):
        """Round-Robin scheduling algorithm"""
        if not self.processes:
            return

        # Get the next process
        process = self.processes.pop(0)
        self.running_process = process
        
        # Calculate how long to run this process
        time_to_run = min(process.time_slice_remaining, 
                         process.time_needed - process.time_executed)
        
        if time_to_run <= 0:
            # Process is complete
            self._complete_process(process)
            return

        # Run the process
        print(f"Running process {process.job.id} for {time_to_run:.1f}s (Round-Robin)")
        print(f"  Command: {process.job.command}")
        process.job.status = JobStatus.RUNNING
        
        # Execute the actual command
        success, output, actual_time = self._execute_command(process.job.command, time_to_run)
        
        if success:
            print(f"  Command executed successfully in {actual_time:.2f}s")
            if output.strip():
                print(f"  Output: {output.strip()}")
        else:
            print(f"  Command failed: {output}")
        
        # Update process state
        process.time_executed += actual_time
        process.time_slice_remaining = max(0, time_to_run - actual_time)
        
        # Check if process is complete
        if process.time_executed >= process.time_needed:
            self._complete_process(process)
        else:
            # Process needs more time, add back to queue with remaining time slice
            if process.time_slice_remaining > 0:
                self.processes.append(process)
            else:
                # Time slice used up, add back with fresh time slice
                process.time_slice_remaining = self.time_slice
                self.processes.append(process)
            process.job.status = JobStatus.WAITING

    def _priority_schedule(self):
        """Priority-based scheduling algorithm"""
        if not self.processes:
            return

        # Sort processes by priority (highest priority first)
        self.processes.sort(key=lambda p: (p.priority, p.arrival_time))
        
        # Get the highest priority process
        process = self.processes.pop(0)
        self.running_process = process
        
        # Run the process until completion or preemption
        print(f"Running process {process.job.id} (Priority: {process.priority})")
        print(f"  Command: {process.job.command}")
        process.job.status = JobStatus.RUNNING
        
        # Execute the actual command
        time_needed = process.time_needed - process.time_executed
        success, output, actual_time = self._execute_command(process.job.command, time_needed)
        
        if success:
            print(f"  Command executed successfully in {actual_time:.2f}s")
            if output.strip():
                print(f"  Output: {output.strip()}")
        else:
            print(f"  Command failed: {output}")
        
        # Process is complete
        process.time_executed += actual_time
        self._complete_process(process)

    def _complete_process(self, process: ProcessInfo):
        """Mark a process as complete"""
        process.job.status = JobStatus.DONE
        process.job.end_time = time.time()
        process.job.execution_time = process.time_executed
        
        print(f"Process {process.job.id} completed (Total time: {process.time_executed:.1f}s)")
        
        if self.on_process_complete:
            self.on_process_complete(process.job)
        
        self.running_process = None

    def get_scheduler_status(self) -> dict:
        """Get current scheduler status"""
        with self.lock:
            return {
                "algorithm": self.algorithm.value,
                "time_slice": self.time_slice if self.algorithm == SchedulingAlgorithm.ROUND_ROBIN else None,
                "total_processes": len(self.processes),
                "running_process": self.running_process.job.id if self.running_process else None,
                "processes": [
                    {
                        "id": p.job.id,
                        "priority": p.priority,
                        "time_executed": p.time_executed,
                        "time_needed": p.time_needed,
                        "status": p.job.status.value
                    }
                    for p in self.processes
                ]
            }

    def preempt_current_process(self):
        """Preempt the currently running process (for priority scheduling)"""
        with self.lock:
            if self.running_process:
                print(f"Preempting process {self.running_process.job.id}")
                self.running_process.job.status = JobStatus.WAITING
                # Add back to the front of the queue for immediate rescheduling
                self.processes.insert(0, self.running_process)
                self.running_process = None 