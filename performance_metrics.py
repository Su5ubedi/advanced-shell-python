#!/usr/bin/env python3
"""
performance_metrics.py - Performance metrics collection for Advanced Shell Scheduling Tests
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class MetricType(Enum):
    """Types of performance metrics"""
    EXECUTION_TIME = "execution_time"
    WAITING_TIME = "waiting_time"
    TURNAROUND_TIME = "turnaround_time"
    RESPONSE_TIME = "response_time"
    CPU_UTILIZATION = "cpu_utilization"
    THROUGHPUT = "throughput"
    CONTEXT_SWITCHES = "context_switches"
    PREEMPTIONS = "preemptions"


@dataclass
class JobMetrics:
    """Metrics for a single job"""
    job_id: int
    command: str
    priority: int
    arrival_time: float
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    total_time_needed: float = 0.0
    actual_execution_time: float = 0.0
    waiting_time: float = 0.0
    turnaround_time: float = 0.0
    response_time: float = 0.0
    context_switches: int = 0
    preemptions: int = 0
    time_slice_used: float = 0.0
    efficiency: float = 0.0  # actual_time / total_time_needed


@dataclass
class TestMetrics:
    """Metrics for a complete test"""
    test_name: str
    algorithm: str
    time_slice: Optional[float] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    total_execution_time: float = 0.0
    total_waiting_time: float = 0.0
    total_turnaround_time: float = 0.0
    total_response_time: float = 0.0
    total_context_switches: int = 0
    total_preemptions: int = 0
    cpu_utilization: float = 0.0
    throughput: float = 0.0  # jobs per second
    average_execution_time: float = 0.0
    average_waiting_time: float = 0.0
    average_turnaround_time: float = 0.0
    average_response_time: float = 0.0
    execution_time_variance: float = 0.0
    waiting_time_variance: float = 0.0
    turnaround_time_variance: float = 0.0
    response_time_variance: float = 0.0
    job_metrics: List[JobMetrics] = field(default_factory=list)


class PerformanceTracker:
    """Tracks performance metrics during scheduling tests"""

    def __init__(self):
        self.tests: List[TestMetrics] = []
        self.current_test: Optional[TestMetrics] = None
        self.job_start_times: Dict[int, float] = {}
        self.job_metrics: Dict[int, JobMetrics] = {}

    def start_test(self, test_name: str, algorithm: str, time_slice: Optional[float] = None):
        """Start tracking metrics for a new test"""
        self.current_test = TestMetrics(
            test_name=test_name,
            algorithm=algorithm,
            time_slice=time_slice,
            start_time=time.time()
        )
        self.job_metrics.clear()
        self.job_start_times.clear()
        print(f"Performance tracking started for: {test_name}")

    def end_test(self):
        """End the current test and calculate final metrics"""
        if not self.current_test:
            return

        self.current_test.end_time = time.time()
        self.current_test.total_jobs = len(self.job_metrics)
        self.current_test.completed_jobs = len([j for j in self.job_metrics.values() if j.completion_time])
        self.current_test.failed_jobs = self.current_test.total_jobs - self.current_test.completed_jobs

        # Calculate totals and averages
        completed_jobs = [j for j in self.job_metrics.values() if j.completion_time]
        if completed_jobs:
            self.current_test.total_execution_time = sum(j.actual_execution_time for j in completed_jobs)
            self.current_test.total_waiting_time = sum(j.waiting_time for j in completed_jobs)
            self.current_test.total_turnaround_time = sum(j.turnaround_time for j in completed_jobs)
            self.current_test.total_response_time = sum(j.response_time for j in completed_jobs)
            self.current_test.total_context_switches = sum(j.context_switches for j in completed_jobs)
            self.current_test.total_preemptions = sum(j.preemptions for j in completed_jobs)

            self.current_test.average_execution_time = self.current_test.total_execution_time / len(completed_jobs)
            self.current_test.average_waiting_time = self.current_test.total_waiting_time / len(completed_jobs)
            self.current_test.average_turnaround_time = self.current_test.total_turnaround_time / len(completed_jobs)
            self.current_test.average_response_time = self.current_test.total_response_time / len(completed_jobs)

            # Calculate variances
            execution_times = [j.actual_execution_time for j in completed_jobs]
            waiting_times = [j.waiting_time for j in completed_jobs]
            turnaround_times = [j.turnaround_time for j in completed_jobs]
            response_times = [j.response_time for j in completed_jobs]

            self.current_test.execution_time_variance = statistics.variance(execution_times) if len(execution_times) > 1 else 0
            self.current_test.waiting_time_variance = statistics.variance(waiting_times) if len(waiting_times) > 1 else 0
            self.current_test.turnaround_time_variance = statistics.variance(turnaround_times) if len(turnaround_times) > 1 else 0
            self.current_test.response_time_variance = statistics.variance(response_times) if len(response_times) > 1 else 0

        # Calculate throughput and CPU utilization
        test_duration = self.current_test.end_time - self.current_test.start_time
        if test_duration > 0:
            self.current_test.throughput = self.current_test.completed_jobs / test_duration
            self.current_test.cpu_utilization = self.current_test.total_execution_time / test_duration

        # Add job metrics to test
        self.current_test.job_metrics = list(self.job_metrics.values())

        # Add to tests list
        self.tests.append(self.current_test)
        print(f"Performance tracking completed for: {self.current_test.test_name}")

    def add_job(self, job_id: int, command: str, priority: int, time_needed: float):
        """Add a new job to tracking"""
        current_time = time.time()
        job_metric = JobMetrics(
            job_id=job_id,
            command=command,
            priority=priority,
            arrival_time=current_time,
            total_time_needed=time_needed
        )
        self.job_metrics[job_id] = job_metric
        self.job_start_times[job_id] = current_time

    def job_started(self, job_id: int):
        """Record when a job starts execution"""
        if job_id in self.job_metrics:
            current_time = time.time()
            self.job_metrics[job_id].start_time = current_time
            self.job_metrics[job_id].response_time = current_time - self.job_metrics[job_id].arrival_time

    def job_completed(self, job_id: int, actual_time: float):
        """Record when a job completes"""
        if job_id in self.job_metrics:
            current_time = time.time()
            job = self.job_metrics[job_id]
            job.completion_time = current_time
            job.actual_execution_time = actual_time
            job.turnaround_time = current_time - job.arrival_time
            job.waiting_time = job.turnaround_time - actual_time
            job.efficiency = actual_time / job.total_time_needed if job.total_time_needed > 0 else 1.0

    def context_switch(self, job_id: int):
        """Record a context switch for a job"""
        if job_id in self.job_metrics:
            self.job_metrics[job_id].context_switches += 1

    def preemption(self, job_id: int):
        """Record a preemption for a job"""
        if job_id in self.job_metrics:
            self.job_metrics[job_id].preemptions += 1

    def time_slice_used(self, job_id: int, time_used: float):
        """Record time slice usage for a job"""
        if job_id in self.job_metrics:
            self.job_metrics[job_id].time_slice_used += time_used

    def generate_report(self, filename: str = "performance_metrics_report.txt"):
        """Generate a comprehensive performance report"""
        with open(filename, 'w') as f:
            f.write("Advanced Shell - Performance Metrics Report\n")
            f.write("=" * 50 + "\n\n")

            # Overall summary
            f.write("OVERALL SUMMARY\n")
            f.write("-" * 20 + "\n")
            total_tests = len(self.tests)
            total_jobs = sum(t.total_jobs for t in self.tests)
            total_completed = sum(t.completed_jobs for t in self.tests)
            total_failed = sum(t.failed_jobs for t in self.tests)

            f.write(f"Total Tests Run: {total_tests}\n")
            f.write(f"Total Jobs Processed: {total_jobs}\n")
            f.write(f"Successfully Completed: {total_completed}\n")
            f.write(f"Failed: {total_failed}\n")
            f.write(f"Success Rate: {(total_completed/total_jobs*100):.2f}%\n\n")

            # Per-test breakdown
            for i, test in enumerate(self.tests, 1):
                f.write(f"TEST {i}: {test.test_name}\n")
                f.write("-" * (len(test.test_name) + 8) + "\n")
                f.write(f"Algorithm: {test.algorithm}\n")
                if test.time_slice:
                    f.write(f"Time Slice: {test.time_slice}s\n")
                f.write(f"Duration: {test.end_time - test.start_time:.3f}s\n")
                f.write(f"Jobs: {test.completed_jobs}/{test.total_jobs} completed\n")
                f.write(f"Success Rate: {(test.completed_jobs/test.total_jobs*100):.2f}%\n\n")

                # Performance metrics
                f.write("PERFORMANCE METRICS:\n")
                f.write(f"  Throughput: {test.throughput:.3f} jobs/second\n")
                f.write(f"  CPU Utilization: {test.cpu_utilization*100:.2f}%\n")
                f.write(f"  Average Execution Time: {test.average_execution_time:.3f}s\n")
                f.write(f"  Average Waiting Time: {test.average_waiting_time:.3f}s\n")
                f.write(f"  Average Turnaround Time: {test.average_turnaround_time:.3f}s\n")
                f.write(f"  Average Response Time: {test.average_response_time:.3f}s\n")
                f.write(f"  Total Context Switches: {test.total_context_switches}\n")
                f.write(f"  Total Preemptions: {test.total_preemptions}\n\n")

                # Variance analysis
                f.write("VARIANCE ANALYSIS:\n")
                f.write(f"  Execution Time Variance: {test.execution_time_variance:.6f}\n")
                f.write(f"  Waiting Time Variance: {test.waiting_time_variance:.6f}\n")
                f.write(f"  Turnaround Time Variance: {test.turnaround_time_variance:.6f}\n")
                f.write(f"  Response Time Variance: {test.response_time_variance:.6f}\n\n")

                # Job details
                f.write("JOB DETAILS:\n")
                for job in test.job_metrics:
                    f.write(f"  Job {job.job_id}: {job.command}\n")
                    f.write(f"    Priority: {job.priority}, Time Needed: {job.total_time_needed:.3f}s\n")
                    f.write(f"    Actual Time: {job.actual_execution_time:.3f}s, Efficiency: {job.efficiency:.2f}\n")
                    f.write(f"    Waiting: {job.waiting_time:.3f}s, Turnaround: {job.turnaround_time:.3f}s\n")
                    f.write(f"    Response: {job.response_time:.3f}s, Context Switches: {job.context_switches}\n")
                    f.write(f"    Preemptions: {job.preemptions}, Time Slice Used: {job.time_slice_used:.3f}s\n\n")

                f.write("\n" + "="*50 + "\n\n")

            # Algorithm comparison
            f.write("ALGORITHM COMPARISON\n")
            f.write("-" * 20 + "\n")

            round_robin_tests = [t for t in self.tests if t.algorithm == "round_robin"]
            priority_tests = [t for t in self.tests if t.algorithm == "priority"]

            if round_robin_tests:
                f.write("ROUND-ROBIN ALGORITHM:\n")
                avg_throughput = statistics.mean([t.throughput for t in round_robin_tests])
                avg_cpu_util = statistics.mean([t.cpu_utilization for t in round_robin_tests])
                avg_waiting = statistics.mean([t.average_waiting_time for t in round_robin_tests])
                avg_turnaround = statistics.mean([t.average_turnaround_time for t in round_robin_tests])
                f.write(f"  Average Throughput: {avg_throughput:.3f} jobs/second\n")
                f.write(f"  Average CPU Utilization: {avg_cpu_util*100:.2f}%\n")
                f.write(f"  Average Waiting Time: {avg_waiting:.3f}s\n")
                f.write(f"  Average Turnaround Time: {avg_turnaround:.3f}s\n\n")

            if priority_tests:
                f.write("PRIORITY ALGORITHM:\n")
                avg_throughput = statistics.mean([t.throughput for t in priority_tests])
                avg_cpu_util = statistics.mean([t.cpu_utilization for t in priority_tests])
                avg_waiting = statistics.mean([t.average_waiting_time for t in priority_tests])
                avg_turnaround = statistics.mean([t.average_turnaround_time for t in priority_tests])
                f.write(f"  Average Throughput: {avg_throughput:.3f} jobs/second\n")
                f.write(f"  Average CPU Utilization: {avg_cpu_util*100:.2f}%\n")
                f.write(f"  Average Waiting Time: {avg_waiting:.3f}s\n")
                f.write(f"  Average Turnaround Time: {avg_turnaround:.3f}s\n\n")

            # Recommendations
            f.write("PERFORMANCE RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")

            if self.tests:
                best_throughput = max(self.tests, key=lambda t: t.throughput)
                best_cpu_util = max(self.tests, key=lambda t: t.cpu_utilization)
                best_waiting = min(self.tests, key=lambda t: t.average_waiting_time)
                best_turnaround = min(self.tests, key=lambda t: t.average_turnaround_time)

                f.write(f"Best Throughput: {best_throughput.test_name} ({best_throughput.throughput:.3f} jobs/s)\n")
                f.write(f"Best CPU Utilization: {best_cpu_util.test_name} ({best_cpu_util.cpu_utilization*100:.2f}%)\n")
                f.write(f"Best Waiting Time: {best_waiting.test_name} ({best_waiting.average_waiting_time:.3f}s)\n")
                f.write(f"Best Turnaround Time: {best_turnaround.test_name} ({best_turnaround.average_turnaround_time:.3f}s)\n\n")

                # Time slice analysis for Round-Robin
                rr_tests = [t for t in self.tests if t.algorithm == "round_robin" and t.time_slice]
                if rr_tests:
                    f.write("TIME SLICE ANALYSIS (Round-Robin):\n")
                    for test in sorted(rr_tests, key=lambda t: t.time_slice):
                        f.write(f"  Time Slice {test.time_slice}s: Throughput={test.throughput:.3f}, "
                               f"CPU Util={test.cpu_utilization*100:.2f}%, "
                               f"Avg Wait={test.average_waiting_time:.3f}s\n")
                    f.write("\n")

        print(f"Performance report generated: {filename}")


# Global performance tracker instance
performance_tracker = PerformanceTracker()