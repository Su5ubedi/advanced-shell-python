#!/usr/bin/env python3
"""
scheduler_types.py - Data types and enums for process scheduling (Deliverable 2)
"""

import time
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class SchedulingAlgorithm(Enum):
    """Scheduling algorithm types"""
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"


class ProcessState(Enum):
    """Process state enumeration"""
    READY = "Ready"
    RUNNING = "Running"
    WAITING = "Waiting"
    TERMINATED = "Terminated"


@dataclass
class ProcessMetrics:
    """Process performance metrics"""
    arrival_time: float
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    waiting_time: float = 0.0
    turnaround_time: float = 0.0
    response_time: Optional[float] = None

    def calculate_metrics(self):
        """Calculate waiting time, turnaround time, and response time"""
        if self.completion_time and self.start_time and self.arrival_time:
            total_execution_time = self.completion_time - self.start_time
            self.turnaround_time = self.completion_time - self.arrival_time
            self.waiting_time = self.turnaround_time - total_execution_time
            if self.response_time is None:
                self.response_time = self.start_time - self.arrival_time
        else:
            # Set default values if times are not properly set
            self.waiting_time = 0.0
            self.turnaround_time = 0.0
            if self.response_time is None:
                self.response_time = 0.0


@dataclass
class ScheduledProcess:
    """Represents a process in the scheduler"""
    pid: int
    name: str
    duration: float  # Total execution time needed
    priority: int = 0  # Higher number = higher priority
    state: ProcessState = ProcessState.READY
    remaining_time: float = 0.0
    last_executed: float = 0.0
    metrics: Optional[ProcessMetrics] = None

    def __post_init__(self):
        if self.remaining_time == 0.0:
            self.remaining_time = self.duration
        if self.metrics is None:
            self.metrics = ProcessMetrics(arrival_time=time.time())

    def __lt__(self, other):
        """For priority queue comparison - higher priority first, then FCFS"""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first

        # FCFS for same priority - use arrival time if metrics exist
        self_arrival = self.metrics.arrival_time if self.metrics else 0.0
        other_arrival = other.metrics.arrival_time if other.metrics else 0.0
        return self_arrival < other_arrival


@dataclass
class SchedulerConfig:
    """Scheduler configuration"""
    algorithm: SchedulingAlgorithm
    time_quantum: float = 1.0  # For Round-Robin scheduling
    context_switch_time: float = 0.1  # Simulated context switch overhead