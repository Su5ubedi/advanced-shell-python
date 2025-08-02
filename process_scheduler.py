#!/usr/bin/env python3
"""
process_scheduler.py - Process scheduling algorithms implementation (Deliverable 2)
Enhanced with clear functionality for status and metrics
"""

import time
import heapq
from typing import List, Optional, Dict
from collections import deque

from scheduler_types import (
    ScheduledProcess, SchedulerConfig, SchedulingAlgorithm,
    ProcessState
)


class ProcessScheduler:
    """Process scheduler implementing Round-Robin and Priority-Based scheduling"""

    def __init__(self):
        self.config: Optional[SchedulerConfig] = None
        self.ready_queue = deque()  # For Round-Robin
        self.priority_queue = []    # For Priority-Based (heap)
        self.current_process: Optional[ScheduledProcess] = None
        self.completed_processes: List[ScheduledProcess] = []
        self.process_counter = 1
        self.running = False
        self.total_context_switches = 0

    def configure(self, algorithm: SchedulingAlgorithm, time_quantum: float = 1.0) -> None:
        """Configure the scheduler with algorithm and parameters"""
        if self.running:
            raise ValueError("Cannot reconfigure while scheduler is running")

        self.config = SchedulerConfig(
            algorithm=algorithm,
            time_quantum=time_quantum
        )

        # Don't clear anything - just set the configuration
        # Users can explicitly clear if they want to

    def clear_scheduler_state(self) -> None:
        """Clear all scheduler state including queues, processes, metrics, and configuration"""
        self.ready_queue.clear()
        self.priority_queue.clear()
        self.current_process = None
        self.completed_processes.clear()
        self.process_counter = 1
        self.total_context_switches = 0
        self.running = False
        self.config = None  # Clear configuration too

    def clear_metrics(self) -> None:
        """Clear only metrics while preserving configuration and queued processes"""
        if self.running:
            raise ValueError("Cannot clear metrics while scheduler is running")

        self.completed_processes.clear()
        self.total_context_switches = 0

        # Reset

    def add_process(self, name: str, duration: float, priority: int = 0) -> int:
        """Add a new process to the scheduler"""
        if not self.config:
            raise ValueError("Scheduler not configured. Use 'scheduler config <schedule_name> <optional time_slice_for_round_robin>' command first.")

        process = ScheduledProcess(
            pid=self.process_counter,
            name=name,
            duration=duration,
            priority=priority,
            state=ProcessState.READY
        )

        # Add to appropriate queue based on algorithm
        if self.config.algorithm == SchedulingAlgorithm.ROUND_ROBIN:
            self.ready_queue.append(process)
        else:  # Priority-Based
            heapq.heappush(self.priority_queue, process)

        self.process_counter += 1
        return process.pid

    def start_scheduler(self) -> None:
        """Start the scheduler (blocking execution)"""
        if self.running:
            raise ValueError("Scheduler is already running")

        if not self.config:
            raise ValueError("Scheduler not configured")

        # Check if there are any processes to schedule
        if not self.ready_queue and not self.priority_queue and not self.current_process:
            raise ValueError("No processes to schedule. Use 'addprocess' to add processes first.")

        self.running = True

        if self.config.algorithm == SchedulingAlgorithm.ROUND_ROBIN:
            self._round_robin_scheduler()
        else:
            self._priority_scheduler()

    def stop_scheduler(self) -> None:
        """Stop the scheduler without clearing state"""
        self.running = False

    def get_status(self) -> Dict:
        """Get current scheduler status"""
        status = {
            'running': self.running,
            'algorithm': self.config.algorithm.value if self.config else None,
            'time_quantum': self.config.time_quantum if self.config else None,
            'current_process': None,
            'ready_queue_size': 0,
            'completed_processes': len(self.completed_processes),
            'total_context_switches': self.total_context_switches
        }

        if self.current_process:
            status['current_process'] = {
                'pid': self.current_process.pid,
                'name': self.current_process.name,
                'remaining_time': round(self.current_process.remaining_time, 2),
                'priority': self.current_process.priority
            }

        if self.config:
            if self.config.algorithm == SchedulingAlgorithm.ROUND_ROBIN:
                status['ready_queue_size'] = len(self.ready_queue)
                status['ready_processes'] = [
                    {'pid': p.pid, 'name': p.name, 'remaining_time': round(p.remaining_time, 2)}
                    for p in self.ready_queue
                ]
            else:
                status['ready_queue_size'] = len(self.priority_queue)
                status['ready_processes'] = [
                    {'pid': p.pid, 'name': p.name, 'remaining_time': round(p.remaining_time, 2), 'priority': p.priority}
                    for p in sorted(self.priority_queue)
                ]

        return status

    def get_metrics(self) -> Dict:
        """Get performance metrics for completed processes"""
        if not self.completed_processes:
            return {'message': 'No completed processes yet'}

        total_waiting = sum(p.metrics.waiting_time for p in self.completed_processes if p.metrics)
        total_turnaround = sum(p.metrics.turnaround_time for p in self.completed_processes if p.metrics)
        total_response = sum(p.metrics.response_time for p in self.completed_processes if p.metrics and p.metrics.response_time)

        num_processes = len(self.completed_processes)

        metrics = {
            'completed_processes': num_processes,
            'average_waiting_time': round(total_waiting / num_processes, 2) if num_processes > 0 else 0,
            'average_turnaround_time': round(total_turnaround / num_processes, 2) if num_processes > 0 else 0,
            'average_response_time': round(total_response / num_processes, 2) if total_response > 0 and num_processes > 0 else 0,
            'total_context_switches': self.total_context_switches,
            'process_details': []
        }

        for process in self.completed_processes:
            if process.metrics:
                metrics['process_details'].append({
                    'pid': process.pid,
                    'name': process.name,
                    'waiting_time': round(process.metrics.waiting_time, 2),
                    'turnaround_time': round(process.metrics.turnaround_time, 2),
                    'response_time': round(process.metrics.response_time, 2) if process.metrics.response_time else 0
                })

        return metrics

    def _round_robin_scheduler(self) -> None:
        """Round-Robin scheduling algorithm implementation (blocking)"""
        print("Starting Round-Robin scheduler...")
        print(priority_queue := self.priority_queue)  # For debugging

        while self.running and (self.ready_queue or self.current_process):
            # Get next process from queue if needed
            if not self.current_process and self.ready_queue:
                self.current_process = self.ready_queue.popleft()
                if self.current_process:
                    assert self.current_process is not None
                    self.current_process.state = ProcessState.RUNNING

                    # Set start time for metrics
                    if self.current_process.metrics and self.current_process.metrics.start_time is None:
                        self.current_process.metrics.start_time = time.time()

                    self.total_context_switches += 1
                    print(f"[RR Scheduler] Running process {self.current_process.pid} ({self.current_process.name})")

            if not self.current_process:
                break  # No more processes

            # Execute process for time quantum
            if self.config:
                execution_time = min(self.config.time_quantum, self.current_process.remaining_time)
                time.sleep(execution_time)

                self.current_process.remaining_time -= execution_time

                # Check if process completed
                if self.current_process.remaining_time <= 0:
                    self.current_process.state = ProcessState.TERMINATED
                    if self.current_process.metrics:
                        self.current_process.metrics.completion_time = time.time()
                        self.current_process.metrics.calculate_metrics()

                    print(f"[RR Scheduler] Process {self.current_process.pid} ({self.current_process.name}) completed")
                    self.completed_processes.append(self.current_process)
                    self.current_process = None
                else:
                    # Time quantum expired, move to back of queue
                    self.current_process.state = ProcessState.READY
                    self.ready_queue.append(self.current_process)
                    print(f"[RR Scheduler] Process {self.current_process.pid} preempted, remaining time: {self.current_process.remaining_time:.2f}s")
                    self.current_process = None

        self.running = False
        print("Round-Robin scheduler completed!")

    def _priority_scheduler(self) -> None:
        """Priority-Based scheduling algorithm implementation (blocking)"""
        print("Starting Priority-Based scheduler...")

        while self.running and (self.priority_queue or self.current_process):
            # Get highest priority process if needed
            if not self.current_process and self.priority_queue:
                self.current_process = heapq.heappop(self.priority_queue)
                if self.current_process:
                    assert self.current_process is not None
                    self.current_process.state = ProcessState.RUNNING

                    # Set start time for metrics
                    if self.current_process.metrics and self.current_process.metrics.start_time is None:
                        self.current_process.metrics.start_time = time.time()

                    self.total_context_switches += 1
                    print(f"[Priority Scheduler] Running process {self.current_process.pid} ({self.current_process.name}) priority={self.current_process.priority}")

            if not self.current_process:
                break  # No more processes

            # Execute process to completion (no preemption in blocking mode)
            while self.current_process.remaining_time > 0:
                execution_slice = min(0.5, self.current_process.remaining_time)  # Execute in small chunks for visual feedback
                time.sleep(execution_slice)
                self.current_process.remaining_time -= execution_slice

                # Check for higher priority processes (simple check)
                if self.priority_queue and self.priority_queue[0].priority > self.current_process.priority:
                    # Preempt current process
                    self.current_process.state = ProcessState.READY
                    heapq.heappush(self.priority_queue, self.current_process)
                    print(f"[Priority Scheduler] Process {self.current_process.pid} preempted by higher priority process")
                    self.current_process = None
                    break

            # Process completed
            if self.current_process and self.current_process.remaining_time <= 0:
                self.current_process.state = ProcessState.TERMINATED
                if self.current_process.metrics:
                    self.current_process.metrics.completion_time = time.time()
                    self.current_process.metrics.calculate_metrics()

                print(f"[Priority Scheduler] Process {self.current_process.pid} ({self.current_process.name}) completed")
                self.completed_processes.append(self.current_process)
                self.current_process = None

        self.running = False
        print("Priority-Based scheduler completed!")

    def _preempt_current_process(self) -> None:
        """Preempt the currently running process (for priority scheduling)"""
        if self.current_process and self.current_process.state == ProcessState.RUNNING:
            self.current_process.state = ProcessState.READY
            heapq.heappush(self.priority_queue, self.current_process)
            print(f"[Priority Scheduler] Process {self.current_process.pid} preempted by higher priority process")
            self.current_process = None