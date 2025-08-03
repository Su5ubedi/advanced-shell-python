#!/usr/bin/env python3
"""
Advanced Shell Simulation - Deliverable 3: Memory Management and Process Synchronization
A comprehensive shell implementation demonstrating operating system concepts including:
- Memory management with paging and page replacement algorithms (FIFO, LRU)
- Process synchronization with mutexes, semaphores, and classical problems
- Integration with existing process management and scheduling features

Author: Advanced Operating Systems Course
Version: 3.0.0 (Deliverable 3)
Python Version: 3.8+

=== DELIVERABLE 3 TESTING STRATEGY ===

This implementation demonstrates all required Deliverable 3 features:

1. MEMORY MANAGEMENT:
   - Paging system with fixed-size page frames
   - Page fault handling and tracking
   - FIFO and LRU page replacement algorithms
   - Memory overflow simulation and statistics

2. PROCESS SYNCHRONIZATION:
   - Mutexes and semaphores implementation
   - Classical synchronization problems:
     * Producer-Consumer with bounded buffer
     * Dining Philosophers with deadlock prevention
   - Race condition prevention through proper locking

=== QUICK TEST COMMANDS ===

Memory Management:
  python3 deliverable_3_shell.py --test-memory    # Automated memory tests

  Manual tests in shell:
  memory create webapp 8        # Create process needing 8 pages
  memory alloc 1 0             # Allocate page (page fault)
  memory alloc 1 0             # Second access (page hit)
  memory algorithm lru         # Switch to LRU algorithm
  memory status                # View statistics

Process Synchronization:
  python3 deliverable_3_shell.py --test-sync      # Automated sync tests

  Manual tests in shell:
  sync prodcons start 2 3      # Start Producer-Consumer
  sync philosophers start 5    # Start Dining Philosophers
  sync status                  # View synchronization status

Integration Test:
  python3 deliverable_3_shell.py --debug          # Debug mode with status

  In shell - run both systems together:
  memory create app 6 && sync prodcons start 2 2
  memory status && sync status

Expected Results:
- Memory: Page faults → hits, FIFO/LRU replacements, 100% utilization
- Sync: Producer-Consumer balanced production/consumption, zero deadlocks
- Integration: Both systems working simultaneously without interference

=== IMPLEMENTATION NOTES ===

The implementation uses threading for concurrent operations, proper locking for
thread safety, and realistic simulation timings. All classical synchronization
problems include deadlock prevention mechanisms and comprehensive statistics.

Memory management simulates real OS paging with configurable frame sizes and
replacement algorithms. Page fault tracking demonstrates the difference between
FIFO and LRU algorithms under different access patterns.
"""

# ===== IMPORTS =====
import argparse
import os
import signal
import stat
import subprocess
import sys
import threading
import time
import shlex
import shutil
import tty
import termios
import queue
import random
import statistics
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ===== DATA TYPES AND ENUMS =====

class JobStatus(Enum):
    """Job status enumeration"""
    RUNNING = "Running"
    STOPPED = "Stopped"
    DONE = "Done"
    WAITING = "Waiting"  # New status for scheduling


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


class PageReplacementAlgorithm(Enum):
    FIFO = "fifo"
    LRU = "lru"


class PhilosopherState(Enum):
    THINKING = "thinking"
    HUNGRY = "hungry"
    EATING = "eating"


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
class Job:
    """Represents a background job"""
    id: int
    pid: int
    command: str
    args: List[str]
    status: JobStatus
    process: subprocess.Popen
    start_time: float
    end_time: Optional[float] = None
    background: bool = True
    priority: int = 5  # Default priority (1=highest, 10=lowest)
    execution_time: float = 0.0  # Time spent executing
    total_time_needed: float = 0.0  # Total time needed for completion
    time_slice_remaining: float = 0.0  # Remaining time slice for Round-Robin

    def is_alive(self) -> bool:
        """Check if the process is still running"""
        if self.process.poll() is None:
            return True
        else:
            if self.status != JobStatus.DONE:
                self.status = JobStatus.DONE
                self.end_time = time.time()
            return False


@dataclass
class ParsedCommand:
    """Represents a parsed command"""
    command: str
    args: List[str]
    background: bool = False
    pipes: List[List[str]] = field(default_factory=list)


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


# ===== MEMORY MANAGEMENT CLASSES =====

class Page:
    """Represents a memory page (NEW)"""

    def __init__(self, page_id: int, process_id: int):
        self.page_id = page_id
        self.process_id = process_id
        self.last_accessed = time.time()
        self.loaded_time = time.time()

    def access(self):
        """Mark page as accessed for LRU tracking (NEW)"""
        self.last_accessed = time.time()

    def __str__(self):
        return f"P{self.process_id}:Pg{self.page_id}"


class Process:
    """Process with memory requirements (NEW)"""

    def __init__(self, pid: int, name: str, pages_needed: int):
        self.pid = pid
        self.name = name
        self.pages_needed = pages_needed
        self.page_table: Dict[int, Optional[Page]] = {
            i: None for i in range(pages_needed)}


class MemoryManager:
    """Memory Manager with paging and page replacement (NEW - Deliverable 3)"""

    def __init__(self, total_frames: int = 8):
        # Physical memory simulation
        self.total_frames = total_frames
        self.physical_memory: List[Optional[Page]] = [None] * total_frames
        self.free_frames: List[int] = list(range(total_frames))
        self.used_frames: Dict[int, Page] = {}

        # Page replacement algorithm
        self.replacement_algorithm = PageReplacementAlgorithm.FIFO

        # Statistics (NEW)
        self.page_faults = 0
        self.page_hits = 0
        self.page_replacements = 0

        # Process management
        self.processes: Dict[int, Process] = {}
        self.next_pid = 1
        self.lock = threading.RLock()

        # Algorithm-specific data structures (NEW)
        self.fifo_queue: List[int] = []  # For FIFO replacement
        self.lru_access_order = OrderedDict()  # For LRU replacement

    def create_process(self, name: str, pages_needed: int) -> int:
        """Create a new process with memory requirements (NEW)"""
        with self.lock:
            pid = self.next_pid
            self.next_pid += 1
            self.processes[pid] = Process(pid, name, pages_needed)
            return pid

    def allocate_page(self, pid: int, page_number: int) -> Tuple[bool, str]:
        """Allocate a page for a process (NEW)"""
        with self.lock:
            if pid not in self.processes:
                return False, f"Process {pid} not found"

            process = self.processes[pid]
            if page_number >= process.pages_needed:
                return False, f"Invalid page number {page_number}"

            # Check if page already allocated (page hit)
            if process.page_table[page_number] is not None:
                self.page_hits += 1
                page = process.page_table[page_number]
                page.access()
                self._update_lru_access(page)
                return True, f"✓ Page hit: {page}"

            # Page fault occurred (NEW)
            self.page_faults += 1

            page = Page(page_number, pid)
            frame_number = self._allocate_frame(page)

            if frame_number == -1:
                return False, "No memory available"

            process.page_table[page_number] = page
            return True, f"✓ Page allocated: {page} -> Frame {frame_number}"

    def _allocate_frame(self, page: Page) -> int:
        """Allocate a physical frame (NEW)"""
        # Use free frame if available
        if self.free_frames:
            frame_number = self.free_frames.pop(0)
            self.physical_memory[frame_number] = page
            self.used_frames[frame_number] = page
            self.fifo_queue.append(frame_number)
            self._update_lru_access(page)
            return frame_number

        # Need page replacement (NEW)
        return self._replace_page(page)

    def _replace_page(self, new_page: Page) -> int:
        """Replace a page using selected algorithm (NEW)"""
        self.page_replacements += 1

        if self.replacement_algorithm == PageReplacementAlgorithm.FIFO:
            return self._fifo_replace(new_page)
        else:  # LRU
            return self._lru_replace(new_page)

    def _fifo_replace(self, new_page: Page) -> int:
        """FIFO page replacement algorithm (NEW)"""
        frame_to_replace = self.fifo_queue.pop(0)
        old_page = self.physical_memory[frame_to_replace]

        if old_page:
            # Update process page table
            if old_page.process_id in self.processes:
                self.processes[old_page.process_id].page_table[old_page.page_id] = None

        self.physical_memory[frame_to_replace] = new_page
        self.used_frames[frame_to_replace] = new_page
        self.fifo_queue.append(frame_to_replace)
        self._update_lru_access(new_page)

        return frame_to_replace

    def _lru_replace(self, new_page: Page) -> int:
        """LRU page replacement algorithm (NEW)"""
        # Find least recently used page
        lru_page = None
        lru_frame = -1

        for frame_num, page in self.used_frames.items():
            if lru_page is None or page.last_accessed < lru_page.last_accessed:
                lru_page = page
                lru_frame = frame_num

        if lru_page and lru_page.process_id in self.processes:
            self.processes[lru_page.process_id].page_table[lru_page.page_id] = None

        self.physical_memory[lru_frame] = new_page
        self.used_frames[lru_frame] = new_page
        self._update_lru_access(new_page)

        return lru_frame

    def _update_lru_access(self, page: Page):
        """Update LRU tracking (NEW)"""
        page_key = f"{page.process_id}:{page.page_id}"
        if page_key in self.lru_access_order:
            del self.lru_access_order[page_key]
        self.lru_access_order[page_key] = page

    def set_algorithm(self, algorithm: str) -> bool:
        """Set page replacement algorithm (NEW)"""
        if algorithm.lower() == "fifo":
            self.replacement_algorithm = PageReplacementAlgorithm.FIFO
            return True
        elif algorithm.lower() == "lru":
            self.replacement_algorithm = PageReplacementAlgorithm.LRU
            return True
        return False

    def get_status(self) -> Dict:
        """Get memory status (NEW)"""
        with self.lock:
            used_frames = len(self.used_frames)
            free_frames = len(self.free_frames)

            return {
                "total_frames": self.total_frames,
                "used_frames": used_frames,
                "free_frames": free_frames,
                "utilization": (used_frames / self.total_frames) * 100,
                "page_faults": self.page_faults,
                "page_hits": self.page_hits,
                "replacements": self.page_replacements,
                "hit_ratio": (self.page_hits / max(1, self.page_hits + self.page_faults)) * 100,
                "algorithm": self.replacement_algorithm.value,
                "processes": len(self.processes)
            }

    def deallocate_process(self, pid: int) -> bool:
        """Deallocate all pages for a process (NEW)"""
        with self.lock:
            if pid not in self.processes:
                return False

            process = self.processes[pid]
            freed_frames = 0

            # Free all allocated pages
            for page_num, page in process.page_table.items():
                if page is not None:
                    for frame_num, frame_page in self.used_frames.items():
                        if frame_page == page:
                            self.physical_memory[frame_num] = None
                            del self.used_frames[frame_num]
                            self.free_frames.append(frame_num)
                            if frame_num in self.fifo_queue:
                                self.fifo_queue.remove(frame_num)
                            freed_frames += 1
                            break

            del self.processes[pid]
            return True

    def simulate_memory_access(self, pid: int, pattern: str = "random", count: int = 8):
        """Simulate memory access patterns for testing (NEW)"""
        if pid not in self.processes:
            return

        process = self.processes[pid]

        for i in range(count):
            if pattern == "sequential":
                page_num = i % process.pages_needed
            else:  # random
                page_num = random.randint(0, process.pages_needed - 1)

            success, message = self.allocate_page(pid, page_num)
            time.sleep(0.1)


# ===== PROCESS SYNCHRONIZATION CLASSES =====

class ProcessSynchronizer:
    """Main synchronization manager (NEW - Deliverable 3)"""

    def __init__(self):
        self.mutexes: Dict[str, threading.Lock] = {}
        self.semaphores: Dict[str, threading.Semaphore] = {}
        self.shared_resources: Dict[str, any] = {}

        # Statistics (NEW)
        self.lock_acquisitions = 0
        self.lock_waits = 0

        self.main_lock = threading.RLock()

    def create_mutex(self, name: str) -> bool:
        """Create a named mutex (NEW)"""
        with self.main_lock:
            if name in self.mutexes:
                return False
            self.mutexes[name] = threading.Lock()
            return True

    def create_semaphore(self, name: str, initial_value: int = 1) -> bool:
        """Create a named semaphore (NEW)"""
        with self.main_lock:
            if name in self.semaphores:
                return False
            self.semaphores[name] = threading.Semaphore(initial_value)
            return True

    def acquire_mutex(self, name: str, timeout: float = None) -> bool:
        """Acquire a mutex (NEW)"""
        if name not in self.mutexes:
            return False

        self.lock_acquisitions += 1
        try:
            if timeout is None:
                acquired = self.mutexes[name].acquire(blocking=True)
            else:
                acquired = self.mutexes[name].acquire(timeout=timeout)
            if not acquired:
                self.lock_waits += 1
            return acquired
        except Exception:
            self.lock_waits += 1
            return False

    def release_mutex(self, name: str) -> bool:
        """Release a mutex (NEW)"""
        if name not in self.mutexes:
            return False
        try:
            self.mutexes[name].release()
            return True
        except:
            return False

    def acquire_semaphore(self, name: str, timeout: float = None) -> bool:
        """Acquire a semaphore (NEW)"""
        if name not in self.semaphores:
            return False
        try:
            if timeout is None:
                return self.semaphores[name].acquire(blocking=True)
            else:
                return self.semaphores[name].acquire(timeout=timeout)
        except Exception:
            return False

    def release_semaphore(self, name: str) -> bool:
        """Release a semaphore (NEW)"""
        if name not in self.semaphores:
            return False
        try:
            self.semaphores[name].release()
            return True
        except:
            return False

    def get_status(self) -> Dict:
        """Get synchronization status (NEW)"""
        return {
            "mutexes": len(self.mutexes),
            "semaphores": len(self.semaphores),
            "lock_acquisitions": self.lock_acquisitions,
            "lock_waits": self.lock_waits,
            "shared_resources": len(self.shared_resources)
        }


class ProducerConsumer:
    """Producer-Consumer synchronization problem (NEW - Deliverable 3)"""

    def __init__(self, buffer_size: int = 5):
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size
        self.running = False
        self.producers: List[threading.Thread] = []
        self.consumers: List[threading.Thread] = []

        # Synchronization primitives (NEW)
        self.mutex = threading.Lock()  # Protects buffer access
        self.not_full = threading.Semaphore(
            buffer_size)   # Producers wait when full
        self.not_empty = threading.Semaphore(
            0)            # Consumers wait when empty

        # Statistics (NEW)
        self.items_produced = 0
        self.items_consumed = 0
        self.producer_waits = 0
        self.consumer_waits = 0

    def producer_task(self, producer_id: int, items_to_produce: int = 10):
        """Producer thread function (NEW)"""
        for i in range(items_to_produce):
            if not self.running:
                break

            item = f"Item-P{producer_id}-{i}"

            # Wait if buffer is full
            if not self.not_full.acquire(timeout=1):
                self.producer_waits += 1
                continue

            try:
                with self.mutex:
                    if self.running:
                        self.buffer.put(item, block=False)
                        self.items_produced += 1

                self.not_empty.release()  # Signal consumers
                time.sleep(random.uniform(0.1, 0.5))

            except queue.Full:
                self.not_full.release()

    def consumer_task(self, consumer_id: int, items_to_consume: int = 10):
        """Consumer thread function (NEW)"""
        for i in range(items_to_consume):
            if not self.running:
                break

            # Wait if buffer is empty
            if not self.not_empty.acquire(timeout=1):
                self.consumer_waits += 1
                continue

            try:
                with self.mutex:
                    if self.running and not self.buffer.empty():
                        item = self.buffer.get(block=False)
                        self.items_consumed += 1

                self.not_full.release()  # Signal producers
                time.sleep(random.uniform(0.1, 0.3))

            except queue.Empty:
                self.not_empty.release()

    def start(self, num_producers: int = 2, num_consumers: int = 2, duration: int = 10):
        """Start the producer-consumer simulation (NEW)"""
        self.running = True
        self.items_produced = 0
        self.items_consumed = 0

        # Create producer threads
        for i in range(num_producers):
            thread = threading.Thread(
                target=self.producer_task, args=(i, duration))
            self.producers.append(thread)
            thread.start()

        # Create consumer threads
        for i in range(num_consumers):
            thread = threading.Thread(
                target=self.consumer_task, args=(i, duration))
            self.consumers.append(thread)
            thread.start()

    def stop(self):
        """Stop the simulation (NEW)"""
        self.running = False

        for thread in self.producers + self.consumers:
            thread.join(timeout=1)

        self.producers.clear()
        self.consumers.clear()

    def get_status(self) -> Dict:
        """Get simulation status (NEW)"""
        # Check if all threads have finished
        active_producers = len([t for t in self.producers if t.is_alive()])
        active_consumers = len([t for t in self.consumers if t.is_alive()])

        # Auto-update running status if no active threads
        if self.running and active_producers == 0 and active_consumers == 0 and len(self.producers) > 0:
            self.running = False

        return {
            "buffer_size": self.buffer_size,
            "current_buffer": self.buffer.qsize(),
            "items_produced": self.items_produced,
            "items_consumed": self.items_consumed,
            "producer_waits": self.producer_waits,
            "consumer_waits": self.consumer_waits,
            "active_producers": active_producers,
            "active_consumers": active_consumers,
            "running": self.running
        }


class DiningPhilosophers:
    """Dining Philosophers synchronization problem (NEW - Deliverable 3)"""

    def __init__(self, num_philosophers: int = 5):
        self.num_philosophers = num_philosophers
        self.philosophers: List[threading.Thread] = []
        self.forks = [threading.Lock() for _ in range(num_philosophers)]
        self.states = [
            PhilosopherState.THINKING for _ in range(num_philosophers)]
        self.running = False

        # Statistics (NEW)
        self.meals_eaten = [0] * num_philosophers
        self.deadlock_prevention_count = 0

        self.state_lock = threading.Lock()

    def philosopher_task(self, philosopher_id: int):
        """Philosopher thread function with deadlock prevention (NEW)"""
        left_fork = philosopher_id
        right_fork = (philosopher_id + 1) % self.num_philosophers

        while self.running:
            # Think
            self._set_state(philosopher_id, PhilosopherState.THINKING)
            time.sleep(random.uniform(1, 3))

            # Get hungry
            self._set_state(philosopher_id, PhilosopherState.HUNGRY)

            # Deadlock prevention: even philosophers pick left first, odd pick right first
            if philosopher_id % 2 == 0:
                first_fork, second_fork = left_fork, right_fork
            else:
                first_fork, second_fork = right_fork, left_fork
                self.deadlock_prevention_count += 1

            # Try to acquire forks
            if self.forks[first_fork].acquire(timeout=2):
                if self.forks[second_fork].acquire(timeout=2):
                    # Eat
                    self._set_state(philosopher_id, PhilosopherState.EATING)
                    self.meals_eaten[philosopher_id] += 1
                    time.sleep(random.uniform(1, 2))

                    # Release forks
                    self.forks[second_fork].release()
                    self.forks[first_fork].release()
                else:
                    # Couldn't get second fork
                    self.forks[first_fork].release()

    def _set_state(self, philosopher_id: int, state: PhilosopherState):
        """Thread-safe state update (NEW)"""
        with self.state_lock:
            self.states[philosopher_id] = state

    def start(self, duration: int = 20):
        """Start the dining philosophers simulation (NEW)"""
        self.running = True

        # Reset statistics
        self.meals_eaten = [0] * self.num_philosophers
        self.deadlock_prevention_count = 0

        # Create philosopher threads
        for i in range(self.num_philosophers):
            thread = threading.Thread(target=self.philosopher_task, args=(i,))
            self.philosophers.append(thread)
            thread.start()

        # Run for specified duration
        time.sleep(duration)
        self.stop()

    def stop(self):
        """Stop the simulation (NEW)"""
        self.running = False

        for thread in self.philosophers:
            thread.join(timeout=1)

        self.philosophers.clear()

    def get_status(self) -> Dict:
        """Get simulation status (NEW)"""
        with self.state_lock:
            return {
                "num_philosophers": self.num_philosophers,
                "states": [state.value for state in self.states],
                "meals_eaten": self.meals_eaten.copy(),
                "total_meals": sum(self.meals_eaten),
                "avg_meals": sum(self.meals_eaten) / self.num_philosophers if self.num_philosophers > 0 else 0.0,
                "deadlock_preventions": self.deadlock_prevention_count,
                "running": self.running,
                "active_threads": len([t for t in self.philosophers if t.is_alive()])
            }


# ===== SHELL INTEGRATION CLASSES =====

class MemorySyncCommands:
    """Command handler class for Deliverable 3 integration (NEW)"""

    def __init__(self):
        self.memory_manager = MemoryManager(total_frames=12)
        self.synchronizer = ProcessSynchronizer()
        self.producer_consumer = None
        self.dining_philosophers = None

    def handle_memory(self, args: List[str]) -> str:
        """Handle memory management commands (NEW)"""
        if not args:
            return self._memory_help()

        command = args[0].lower()

        if command == "status":
            return self._memory_status()
        elif command == "create" and len(args) >= 3:
            return self._create_process(args[1], int(args[2]))
        elif command == "alloc" and len(args) >= 3:
            return self._allocate_page(int(args[1]), int(args[2]))
        elif command == "dealloc" and len(args) >= 2:
            return self._deallocate_process(int(args[1]))
        elif command == "algorithm" and len(args) >= 2:
            return self._set_algorithm(args[1])
        elif command == "test" and len(args) >= 2:
            return self._test_memory(args[1])
        else:
            return self._memory_help()

    def handle_sync(self, args: List[str]) -> str:
        """Handle synchronization commands (NEW)"""
        if not args:
            return self._sync_help()

        command = args[0].lower()

        if command == "status":
            return self._sync_status()
        elif command == "mutex" and len(args) >= 3:
            return self._handle_mutex(args[1], args[2:])
        elif command == "semaphore" and len(args) >= 3:
            return self._handle_semaphore(args[1], args[2:])
        elif command == "prodcons":
            return self._handle_producer_consumer(args[1:])
        elif command == "philosophers":
            return self._handle_dining_philosophers(args[1:])
        else:
            return self._sync_help()

    def _memory_help(self) -> str:
        """Memory management help (NEW)"""
        return """✓ Memory Management Commands (NEW - Deliverable 3):
memory status                     - Show memory status and statistics
memory create <name> <pages>      - Create process with pages needed
memory alloc <pid> <page_num>     - Allocate specific page for process
memory dealloc <pid>              - Deallocate all pages for process
memory algorithm <fifo|lru>       - Set page replacement algorithm
memory test <sequential|random>   - Run memory access pattern test"""

    def _sync_help(self) -> str:
        """Synchronization help (NEW)"""
        return """✓ Process Synchronization Commands (NEW - Deliverable 3):
sync status                               - Show synchronization status
sync mutex <create|acquire|release> <name>    - Mutex operations
sync semaphore <create|acquire|release> <name> [value] - Semaphore operations
sync prodcons <start|stop|status> [producers] [consumers] - Producer-Consumer
sync philosophers <start|stop|status> [num_philosophers] - Dining Philosophers"""

    def _memory_status(self) -> str:
        """Get memory status (NEW)"""
        status = self.memory_manager.get_status()

        result = f"""=== MEMORY MANAGEMENT STATUS (NEW) ===
Total Frames: {status['total_frames']}
Used Frames: {status['used_frames']}
Free Frames: {status['free_frames']}
✓ Memory Utilization: {status['utilization']:.1f}%
✓ Page Replacement: {status['algorithm'].upper()}

=== PAGING STATISTICS (NEW) ===
✓ Page Faults: {status['page_faults']}
✓ Page Hits: {status['page_hits']}
✓ Page Replacements: {status['replacements']}
✓ Hit Ratio: {status['hit_ratio']:.1f}%
Active Processes: {status['processes']}"""

        if self.memory_manager.processes:
            result += "\n\n=== ACTIVE PROCESSES (NEW) ==="
            for pid, process in self.memory_manager.processes.items():
                allocated = len(
                    [p for p in process.page_table.values() if p is not None])
                result += f"\nPID {pid} ({process.name}): {allocated}/{process.pages_needed} pages"

        return result

    def _create_process(self, name: str, pages: int) -> str:
        """Create a new process (NEW)"""
        try:
            pid = self.memory_manager.create_process(name, pages)
            return f"✓ Process {pid} ({name}) created with {pages} pages needed"
        except Exception as e:
            return f"Error creating process: {e}"

    def _allocate_page(self, pid: int, page_num: int) -> str:
        """Allocate a page for a process (NEW)"""
        try:
            success, message = self.memory_manager.allocate_page(pid, page_num)
            return message
        except Exception as e:
            return f"Error allocating page: {e}"

    def _deallocate_process(self, pid: int) -> str:
        """Deallocate a process (NEW)"""
        try:
            if self.memory_manager.deallocate_process(pid):
                return f"✓ Process {pid} deallocated successfully"
            else:
                return f"Process {pid} not found"
        except Exception as e:
            return f"Error deallocating process: {e}"

    def _set_algorithm(self, algorithm: str) -> str:
        """Set page replacement algorithm (NEW)"""
        if self.memory_manager.set_algorithm(algorithm):
            return f"✓ Page replacement algorithm set to {algorithm.upper()}"
        else:
            return f"Invalid algorithm: {algorithm}. Use 'fifo' or 'lru'"

    def _test_memory(self, pattern: str) -> str:
        """Test memory with different access patterns (NEW)"""
        pid = self.memory_manager.create_process("TestProcess", 6)

        result = f"✓ Testing {pattern} memory access pattern...\n"

        if pattern == "sequential":
            for i in range(8):
                page_num = i % 6
                success, message = self.memory_manager.allocate_page(
                    pid, page_num)
                result += f"Access {i+1}: {message}\n"
        elif pattern == "random":
            for i in range(8):
                page_num = random.randint(0, 5)
                success, message = self.memory_manager.allocate_page(
                    pid, page_num)
                result += f"Access {i+1}: {message}\n"

        return result + "\n" + self._memory_status()

    def _sync_status(self) -> str:
        """Get synchronization status (NEW)"""
        status = self.synchronizer.get_status()

        result = f"""=== PROCESS SYNCHRONIZATION STATUS (NEW) ===
✓ Mutexes: {status['mutexes']}
✓ Semaphores: {status['semaphores']}
✓ Lock Acquisitions: {status['lock_acquisitions']}
✓ Lock Waits: {status['lock_waits']}
Shared Resources: {status['shared_resources']}"""

        # Producer-Consumer status (NEW)
        if self.producer_consumer:
            pc_status = self.producer_consumer.get_status()
            result += f"""

=== PRODUCER-CONSUMER STATUS (NEW) ===
Running: {pc_status['running']}
Buffer: {pc_status['current_buffer']}/{pc_status['buffer_size']}
✓ Items Produced: {pc_status['items_produced']}
✓ Items Consumed: {pc_status['items_consumed']}
Active Producers: {pc_status['active_producers']}
Active Consumers: {pc_status['active_consumers']}"""

        # Dining Philosophers status (NEW)
        if self.dining_philosophers:
            dp_status = self.dining_philosophers.get_status()
            result += f"""

=== DINING PHILOSOPHERS STATUS (NEW) ===
Running: {dp_status['running']}
Philosophers: {dp_status['num_philosophers']}
✓ Total Meals: {dp_status['total_meals']}
✓ Average Meals: {dp_status['avg_meals']:.1f}
✓ Deadlock Preventions: {dp_status['deadlock_preventions']}
States: {', '.join(dp_status['states'])}"""

        return result

    def _handle_mutex(self, operation: str, args: List[str]) -> str:
        """Handle mutex operations (NEW)"""
        if operation == "create" and args:
            name = args[0]
            if self.synchronizer.create_mutex(name):
                return f"✓ Mutex '{name}' created"
            else:
                return f"Mutex '{name}' already exists"
        elif operation == "acquire" and args:
            name = args[0]
            timeout = float(args[1]) if len(args) > 1 else None
            if self.synchronizer.acquire_mutex(name, timeout):
                return f"✓ Mutex '{name}' acquired"
            else:
                return f"Failed to acquire mutex '{name}'"
        elif operation == "release" and args:
            name = args[0]
            if self.synchronizer.release_mutex(name):
                return f"✓ Mutex '{name}' released"
            else:
                return f"Failed to release mutex '{name}'"
        else:
            return "Usage: sync mutex <create|acquire|release> <name> [timeout]"

    def _handle_semaphore(self, operation: str, args: List[str]) -> str:
        """Handle semaphore operations (NEW)"""
        if operation == "create" and args:
            name = args[0]
            value = int(args[1]) if len(args) > 1 else 1
            if self.synchronizer.create_semaphore(name, value):
                return f"✓ Semaphore '{name}' created with value {value}"
            else:
                return f"Semaphore '{name}' already exists"
        elif operation == "acquire" and args:
            name = args[0]
            timeout = float(args[1]) if len(args) > 1 else None
            if self.synchronizer.acquire_semaphore(name, timeout):
                return f"✓ Semaphore '{name}' acquired"
            else:
                return f"Failed to acquire semaphore '{name}'"
        elif operation == "release" and args:
            name = args[0]
            if self.synchronizer.release_semaphore(name):
                return f"✓ Semaphore '{name}' released"
            else:
                return f"Failed to release semaphore '{name}'"
        else:
            return "Usage: sync semaphore <create|acquire|release> <name> [value|timeout]"

    def _handle_producer_consumer(self, args: List[str]) -> str:
        """Handle Producer-Consumer problem (NEW)"""
        if not args:
            return "Usage: sync prodcons <start|stop|status> [producers] [consumers] [duration]"

        operation = args[0].lower()

        if operation == "start":
            if self.producer_consumer and self.producer_consumer.running:
                return "Producer-Consumer already running. Stop it first."

            producers = int(args[1]) if len(args) > 1 else 2
            consumers = int(args[2]) if len(args) > 2 else 2
            duration = int(args[3]) if len(args) > 3 else 10

            self.producer_consumer = ProducerConsumer(buffer_size=5)
            self.producer_consumer.start(producers, consumers, duration)

            return f"✓ Producer-Consumer started: {producers} producers, {consumers} consumers"

        elif operation == "stop":
            if self.producer_consumer:
                self.producer_consumer.stop()
                return "✓ Producer-Consumer stopped"
            else:
                return "No Producer-Consumer running"

        elif operation == "status":
            if self.producer_consumer:
                status = self.producer_consumer.get_status()
                return f"""Producer-Consumer Status:
Running: {status['running']}
Buffer: {status['current_buffer']}/{status['buffer_size']}
✓ Produced: {status['items_produced']}
✓ Consumed: {status['items_consumed']}
Active Producers: {status['active_producers']}
Active Consumers: {status['active_consumers']}
Producer Waits: {status['producer_waits']}
Consumer Waits: {status['consumer_waits']}"""
            else:
                return "No Producer-Consumer instance"
        else:
            return "Usage: sync prodcons <start|stop|status>"

    def _handle_dining_philosophers(self, args: List[str]) -> str:
        """Handle Dining Philosophers problem (NEW)"""
        if not args:
            return "Usage: sync philosophers <start|stop|status> [num_philosophers] [duration]"

        operation = args[0].lower()

        if operation == "start":
            if self.dining_philosophers and self.dining_philosophers.running:
                return "Dining Philosophers already running. Stop it first."

            num_philosophers = int(args[1]) if len(args) > 1 else 5
            duration = int(args[2]) if len(args) > 2 else 15

            self.dining_philosophers = DiningPhilosophers(num_philosophers)

            # Start in background thread
            def start_philosophers():
                self.dining_philosophers.start(duration)

            thread = threading.Thread(target=start_philosophers)
            thread.daemon = True
            thread.start()

            return f"✓ Dining Philosophers started: {num_philosophers} philosophers for {duration}s"

        elif operation == "stop":
            if self.dining_philosophers:
                self.dining_philosophers.stop()
                return "✓ Dining Philosophers stopped"
            else:
                return "No Dining Philosophers running"

        elif operation == "status":
            if self.dining_philosophers:
                status = self.dining_philosophers.get_status()
                meals_str = ", ".join(
                    [f"P{i}:{meals}" for i, meals in enumerate(status['meals_eaten'])])
                return f"""Dining Philosophers Status:
Running: {status['running']}
Philosophers: {status['num_philosophers']}
States: {', '.join([f"P{i}:{state}" for i, state in enumerate(status['states'])])}
✓ Total Meals: {status['total_meals']}
✓ Average Meals: {status['avg_meals']:.1f}
Meals: {meals_str}
✓ Deadlock Preventions: {status['deadlock_preventions']}
Active Threads: {status['active_threads']}"""
            else:
                return "No Dining Philosophers instance"
        else:
            return "Usage: sync philosophers <start|stop|status>"

    def get_memory_manager(self):
        """Get memory manager instance (NEW)"""
        return self.memory_manager

    def get_synchronizer(self):
        """Get synchronizer instance (NEW)"""
        return self.synchronizer


# ===== INPUT HANDLER =====

class InputHandler:
    """Handles enhanced input with keyboard navigation support"""

    def __init__(self):
        self.history = []
        self.history_index = 0
        self.max_history = 100

    def get_char(self) -> str:
        """Get a single character from stdin"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def clear_line(self):
        """Clear the current line"""
        sys.stdout.write('\r\033[K')
        sys.stdout.flush()

    def move_cursor_left(self, current_pos: int) -> int:
        """Move cursor left if possible"""
        if current_pos > 0:
            sys.stdout.write('\b')
            sys.stdout.flush()
            return current_pos - 1
        return current_pos

    def move_cursor_right(self, current_pos: int, text: str) -> int:
        """Move cursor right if possible"""
        if current_pos < len(text):
            sys.stdout.write(text[current_pos])
            sys.stdout.flush()
            return current_pos + 1
        return current_pos

    def clear_current_command(self, text: str, cursor_pos: int) -> tuple[str, int]:
        """Clear the current command (Ctrl+C)"""
        # Clear the entire line
        self.clear_line()
        return "", 0

    def backspace(self, text: str, pos: int) -> tuple[str, int]:
        """Handle backspace key"""
        if pos > 0:
            # Remove the character
            text = text[:pos-1] + text[pos:]
            pos -= 1

            # Clear the character and redraw the rest of the line
            sys.stdout.write('\b \b')
            if pos < len(text):
                sys.stdout.write(text[pos:])
                sys.stdout.write(' ')
                # Move cursor back to the correct position
                sys.stdout.write('\b' * (len(text) - pos + 1))
            sys.stdout.flush()

            return text, pos
        return text, pos

    def get_input(self, prompt: str = "") -> str:
        """Get input with keyboard navigation support"""
        if prompt:
            sys.stdout.write(prompt)
            sys.stdout.flush()

        text = ""
        cursor_pos = 0

        while True:
            ch = self.get_char()

            # Handle special keys
            if ch == '\x1b':  # ESC - might be arrow key
                ch2 = self.get_char()
                if ch2 == '[':
                    ch3 = self.get_char()
                    if ch3 == 'A':  # UP arrow
                        # History navigation (future enhancement)
                        continue
                    elif ch3 == 'B':  # DOWN arrow
                        # History navigation (future enhancement)
                        continue
                    elif ch3 == 'C':  # RIGHT arrow
                        cursor_pos = self.move_cursor_right(cursor_pos, text)
                        continue
                    elif ch3 == 'D':  # LEFT arrow
                        cursor_pos = self.move_cursor_left(cursor_pos)
                        continue
                # If not an arrow key, ignore the ESC sequence
                continue

            # Handle control characters
            elif ch == '\x03':  # Ctrl+C - clear current command
                text, cursor_pos = self.clear_current_command(text, cursor_pos)
                print('^C')  # Show Ctrl+C was pressed
                continue
            elif ch == '\x04':  # Ctrl+D
                print('^D')
                raise EOFError()
            elif ch == '\x7f':  # Backspace
                text, cursor_pos = self.backspace(text, cursor_pos)
                continue
            elif ch == '\r':  # Enter
                print()
                if text.strip():
                    self.add_to_history(text.strip())
                return text
            elif ch == '\t':  # Tab
                # Tab completion (future enhancement)
                continue
            elif ch < ' ' or ch > '~':  # Non-printable characters
                continue

            # Handle regular characters
            else:
                text = self.insert_char(text, cursor_pos, ch)
                # Display the character and any text that follows
                sys.stdout.write(ch)
                if cursor_pos < len(text) - 1:
                    sys.stdout.write(text[cursor_pos + 1:])
                    # Move cursor back to the correct position
                    sys.stdout.write('\b' * (len(text) - cursor_pos - 1))
                cursor_pos += 1
                sys.stdout.flush()

    def insert_char(self, text: str, pos: int, char: str) -> str:
        """Insert a character at the specified position"""
        return text[:pos] + char + text[pos:]

    def add_to_history(self, command: str):
        """Add command to history"""
        if command and (not self.history or command != self.history[-1]):
            self.history.append(command)
            if len(self.history) > self.max_history:
                self.history.pop(0)
        self.history_index = len(self.history)


# ===== COMMAND PARSER =====

class CommandParser:
    """Handles parsing of command line input"""

    BUILTIN_COMMANDS = {
        # Deliverable 1: Basic shell commands
        'cd', 'pwd', 'exit', 'echo', 'clear', 'ls', 'cat',
        'mkdir', 'rmdir', 'rm', 'touch', 'kill', 'jobs',
        'fg', 'bg', 'stop', 'help',
        # Deliverable 2: Process scheduling commands
        'schedule', 'addprocess', 'scheduler',
        # Deliverable 3: NEW - Memory management and synchronization commands
        'memory', 'sync'
    }

    def parse(self, input_str: str) -> Optional[ParsedCommand]:
        """Parse a command line input string"""
        input_str = input_str.strip()
        if not input_str:
            return None

        # Check for background execution
        background = False
        if input_str.endswith('&'):
            background = True
            input_str = input_str[:-1].strip()

        # Tokenize the input
        try:
            args = shlex.split(input_str)
        except ValueError as e:
            raise ValueError(f"Parse error: {e}")

        if not args:
            return None

        return ParsedCommand(
            command=args[0],
            args=args,
            background=background,
            pipes=[args]  # Single command for now
        )

    def is_builtin_command(self, command: str) -> bool:
        """Check if a command is a built-in command"""
        return command in self.BUILTIN_COMMANDS

    def validate_command(self, parsed: ParsedCommand) -> None:
        """Perform comprehensive validation on parsed commands"""
        if not parsed or not parsed.command:
            return

        # Check for dangerous command patterns
        if '..' in parsed.command:
            raise ValueError(
                f"Potentially dangerous path detected: {parsed.command}")

        # Validate command name
        dangerous_chars = '|;&<>(){}[]'
        if any(char in parsed.command for char in dangerous_chars):
            raise ValueError(
                f"Invalid characters in command name: {parsed.command}")

        # Check for excessively long commands
        if len(parsed.command) > 256:
            raise ValueError("Command name too long (max 256 characters)")

        # Validate arguments
        for i, arg in enumerate(parsed.args):
            if len(arg) > 1024:
                raise ValueError(
                    f"Argument {i} too long (max 1024 characters)")

        # Check total argument count
        if len(parsed.args) > 100:
            raise ValueError("Too many arguments (max 100)")


# ===== PERFORMANCE TRACKING =====

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
        self.current_test.completed_jobs = len(
            [j for j in self.job_metrics.values() if j.completion_time])
        self.current_test.failed_jobs = self.current_test.total_jobs - \
            self.current_test.completed_jobs

        # Calculate totals and averages
        completed_jobs = [
            j for j in self.job_metrics.values() if j.completion_time]
        if completed_jobs:
            self.current_test.total_execution_time = sum(
                j.actual_execution_time for j in completed_jobs)
            self.current_test.total_waiting_time = sum(
                j.waiting_time for j in completed_jobs)
            self.current_test.total_turnaround_time = sum(
                j.turnaround_time for j in completed_jobs)
            self.current_test.total_response_time = sum(
                j.response_time for j in completed_jobs)
            self.current_test.total_context_switches = sum(
                j.context_switches for j in completed_jobs)
            self.current_test.total_preemptions = sum(
                j.preemptions for j in completed_jobs)

            self.current_test.average_execution_time = self.current_test.total_execution_time / \
                len(completed_jobs)
            self.current_test.average_waiting_time = self.current_test.total_waiting_time / \
                len(completed_jobs)
            self.current_test.average_turnaround_time = self.current_test.total_turnaround_time / \
                len(completed_jobs)
            self.current_test.average_response_time = self.current_test.total_response_time / \
                len(completed_jobs)

            # Calculate variances
            execution_times = [j.actual_execution_time for j in completed_jobs]
            waiting_times = [j.waiting_time for j in completed_jobs]
            turnaround_times = [j.turnaround_time for j in completed_jobs]
            response_times = [j.response_time for j in completed_jobs]

            self.current_test.execution_time_variance = statistics.variance(
                execution_times) if len(execution_times) > 1 else 0
            self.current_test.waiting_time_variance = statistics.variance(
                waiting_times) if len(waiting_times) > 1 else 0
            self.current_test.turnaround_time_variance = statistics.variance(
                turnaround_times) if len(turnaround_times) > 1 else 0
            self.current_test.response_time_variance = statistics.variance(
                response_times) if len(response_times) > 1 else 0

        # Calculate throughput and CPU utilization
        test_duration = self.current_test.end_time - self.current_test.start_time
        if test_duration > 0:
            self.current_test.throughput = self.current_test.completed_jobs / test_duration
            self.current_test.cpu_utilization = self.current_test.total_execution_time / test_duration

        # Add job metrics to test
        self.current_test.job_metrics = list(self.job_metrics.values())

        # Add to tests list
        self.tests.append(self.current_test)
        print(
            f"Performance tracking completed for: {self.current_test.test_name}")

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
            self.job_metrics[job_id].response_time = current_time - \
                self.job_metrics[job_id].arrival_time

    def job_completed(self, job_id: int, actual_time: float):
        """Record when a job completes"""
        if job_id in self.job_metrics:
            current_time = time.time()
            job = self.job_metrics[job_id]
            job.completion_time = current_time
            job.actual_execution_time = actual_time
            job.turnaround_time = current_time - job.arrival_time
            job.waiting_time = job.turnaround_time - actual_time
            job.efficiency = actual_time / \
                job.total_time_needed if job.total_time_needed > 0 else 1.0

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
            success_rate = (total_completed/total_jobs*100) if total_jobs > 0 else 0.0
            f.write(f"Success Rate: {success_rate:.2f}%\n\n")

            # Per-test breakdown
            for i, test in enumerate(self.tests, 1):
                f.write(f"TEST {i}: {test.test_name}\n")
                f.write("-" * (len(test.test_name) + 8) + "\n")
                f.write(f"Algorithm: {test.algorithm}\n")
                if test.time_slice:
                    f.write(f"Time Slice: {test.time_slice}s\n")
                f.write(f"Duration: {test.end_time - test.start_time:.3f}s\n")
                f.write(
                    f"Jobs: {test.completed_jobs}/{test.total_jobs} completed\n")
                test_success_rate = (test.completed_jobs/test.total_jobs*100) if test.total_jobs > 0 else 0.0
                f.write(f"Success Rate: {test_success_rate:.2f}%\n\n")

                # Performance metrics
                f.write("PERFORMANCE METRICS:\n")
                f.write(f"  Throughput: {test.throughput:.3f} jobs/second\n")
                f.write(
                    f"  CPU Utilization: {test.cpu_utilization*100:.2f}%\n")
                f.write(
                    f"  Average Execution Time: {test.average_execution_time:.3f}s\n")
                f.write(
                    f"  Average Waiting Time: {test.average_waiting_time:.3f}s\n")
                f.write(
                    f"  Average Turnaround Time: {test.average_turnaround_time:.3f}s\n")
                f.write(
                    f"  Average Response Time: {test.average_response_time:.3f}s\n")
                f.write(
                    f"  Total Context Switches: {test.total_context_switches}\n")
                f.write(f"  Total Preemptions: {test.total_preemptions}\n\n")

                # Variance analysis
                f.write("VARIANCE ANALYSIS:\n")
                f.write(
                    f"  Execution Time Variance: {test.execution_time_variance:.6f}\n")
                f.write(
                    f"  Waiting Time Variance: {test.waiting_time_variance:.6f}\n")
                f.write(
                    f"  Turnaround Time Variance: {test.turnaround_time_variance:.6f}\n")
                f.write(
                    f"  Response Time Variance: {test.response_time_variance:.6f}\n\n")

                # Job details
                f.write("JOB DETAILS:\n")
                for job in test.job_metrics:
                    f.write(f"  Job {job.job_id}: {job.command}\n")
                    f.write(
                        f"    Priority: {job.priority}, Time Needed: {job.total_time_needed:.3f}s\n")
                    f.write(
                        f"    Actual Time: {job.actual_execution_time:.3f}s, Efficiency: {job.efficiency:.2f}\n")
                    f.write(
                        f"    Waiting: {job.waiting_time:.3f}s, Turnaround: {job.turnaround_time:.3f}s\n")
                    f.write(
                        f"    Response: {job.response_time:.3f}s, Context Switches: {job.context_switches}\n")
                    f.write(
                        f"    Preemptions: {job.preemptions}, Time Slice Used: {job.time_slice_used:.3f}s\n\n")

                f.write("\n" + "="*50 + "\n\n")

        print(f"Performance report generated: {filename}")


# Global performance tracker instance
performance_tracker = PerformanceTracker()


# ===== JOB MANAGER =====

class JobManager:
    """Handles job control operations"""

    def __init__(self):
        self.jobs: Dict[int, Job] = {}
        self.job_counter = 0

    def add_job(self, command: str, args: List[str], process: subprocess.Popen, background: bool = True) -> Job:
        """Add a new job to the manager"""
        self.job_counter += 1
        job = Job(
            id=self.job_counter,
            pid=process.pid,
            command=command,
            args=args,
            status=JobStatus.RUNNING,
            process=process,
            start_time=time.time(),
            background=background
        )
        self.jobs[self.job_counter] = job
        return job

    def get_job(self, job_id: int) -> Optional[Job]:
        """Retrieve a job by ID"""
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> List[Job]:
        """Return all jobs"""
        return list(self.jobs.values())

    def list_jobs(self):
        """List all jobs with their status"""
        if not self.jobs:
            print("No active jobs")
            return

        print("Active jobs:")
        for job in self.jobs.values():
            job.is_alive()  # Update status
            duration = time.time() - job.start_time
            if job.end_time:
                duration = job.end_time - job.start_time

            status_info = job.status.value
            print(
                f"[{job.id}] {status_info} {job.command} (PID: {job.pid}, Duration: {int(duration)}s)")

    def kill_job(self, job_id: int) -> bool:
        """Kill a job by sending SIGTERM"""
        job = self.get_job(job_id)
        if not job:
            return False

        if job.status == JobStatus.DONE:
            return False

        print(f"Terminating job [{job.id}]: {job.command}")
        try:
            job.process.terminate()
            try:
                job.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                job.process.kill()

            job.status = JobStatus.DONE
            job.end_time = time.time()
            return True
        except Exception as e:
            print(f"Failed to kill job: {e}")
            return False

    def cleanup_completed_jobs(self):
        """Remove completed jobs from the manager"""
        completed_jobs = []
        for job_id, job in list(self.jobs.items()):
            job.is_alive()  # Update status

            if job.status == JobStatus.DONE:
                completed_jobs.append(job_id)
                print(f"[{job_id}]+ Done\t\t{job.command}")

        for job_id in completed_jobs:
            if job_id in self.jobs:
                del self.jobs[job_id]


# ===== COMMAND HANDLER =====

class CommandHandler:
    """Handles built-in shell commands"""

    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        # Deliverable 3: NEW - Add memory management and synchronization
        self.memory_sync_commands = MemorySyncCommands()

    def handle_command(self, parsed: ParsedCommand) -> None:
        """Execute a built-in command"""
        if not parsed or not parsed.command:
            return

        command_map = {
            'cd': self.handle_cd,
            'pwd': self.handle_pwd,
            'exit': self.handle_exit,
            'echo': self.handle_echo,
            'clear': self.handle_clear,
            'ls': self.handle_ls,
            'cat': self.handle_cat,
            'mkdir': self.handle_mkdir,
            'rmdir': self.handle_rmdir,
            'rm': self.handle_rm,
            'touch': self.handle_touch,
            'kill': self.handle_kill,
            'jobs': self.handle_jobs,
            'help': self.handle_help,
            # Deliverable 3: NEW - Memory and synchronization commands
            'memory': self.handle_memory,
            'sync': self.handle_sync,
        }

        handler = command_map.get(parsed.command)
        if handler:
            handler(parsed.args)
        else:
            raise ValueError(f"Unknown built-in command: {parsed.command}")

    # Deliverable 3: NEW command handlers
    def handle_memory(self, args: List[str]) -> None:
        """Handle memory management commands (NEW)"""
        try:
            # Skip the command name 'memory' and pass the rest
            memory_args = args[1:] if len(args) > 1 else []
            result = self.memory_sync_commands.handle_memory(memory_args)
            if result:
                print(result)
        except Exception as e:
            raise ValueError(f"memory: {e}")

    def handle_sync(self, args: List[str]) -> None:
        """Handle synchronization commands (NEW)"""
        try:
            # Skip the command name 'sync' and pass the rest
            sync_args = args[1:] if len(args) > 1 else []
            result = self.memory_sync_commands.handle_sync(sync_args)
            if result:
                print(result)
        except Exception as e:
            raise ValueError(f"sync: {e}")

    def handle_cd(self, args: List[str]) -> None:
        """Change directory command"""
        if len(args) < 2:
            # Change to home directory
            try:
                home_dir = Path.home()
                os.chdir(home_dir)
            except Exception as e:
                raise ValueError(f"cd: cannot determine home directory: {e}")
        elif len(args) > 2:
            raise ValueError("cd: too many arguments")
        else:
            target_dir = args[1]

            if not target_dir:
                raise ValueError("cd: empty directory name")

            # Handle special cases
            if target_dir == "~":
                target_dir = str(Path.home())
            elif target_dir.startswith("~/"):
                target_dir = str(Path.home() / target_dir[2:])

            # Check if directory exists before trying to change
            target_path = Path(target_dir)
            if not target_path.exists():
                raise ValueError(
                    f"cd: {target_dir}: no such file or directory")
            elif not target_path.is_dir():
                raise ValueError(f"cd: {target_dir}: not a directory")

            try:
                os.chdir(target_dir)
            except PermissionError:
                raise ValueError(f"cd: {target_dir}: permission denied")
            except Exception as e:
                raise ValueError(f"cd: {target_dir}: {e}")

    def handle_pwd(self, args: List[str]) -> None:
        """Print working directory command"""
        try:
            pwd = os.getcwd()
            print(pwd)
        except Exception as e:
            raise ValueError(f"pwd: {e}")

    def handle_exit(self, args: List[str]) -> None:
        """Exit shell command"""
        # Deliverable 3: NEW - Clean shutdown with memory/sync cleanup
        if hasattr(self, 'memory_sync_commands'):
            if self.memory_sync_commands.producer_consumer:
                self.memory_sync_commands.producer_consumer.stop()
            if self.memory_sync_commands.dining_philosophers:
                self.memory_sync_commands.dining_philosophers.stop()

        # Clean shutdown - kill remaining jobs
        jobs = self.job_manager.get_all_jobs()
        for job in jobs:
            if job.status.value != "Done":
                self.job_manager.kill_job(job.id)
        print("Goodbye!")
        sys.exit(0)

    def handle_echo(self, args: List[str]) -> None:
        """Echo command"""
        if len(args) > 1:
            # Join all arguments except the command itself
            output = " ".join(args[1:])
            # Handle basic escape sequences
            output = output.replace("\\n", "\n")
            output = output.replace("\\t", "\t")
            print(output)

    def handle_clear(self, args: List[str]) -> None:
        """Clear screen command"""
        try:
            # Try different clear commands based on OS
            if os.name == 'nt':  # Windows
                subprocess.run(['cls'], shell=True, check=True)
            else:  # Unix-like
                subprocess.run(['clear'], check=True)
        except subprocess.CalledProcessError:
            # Fallback: print newlines
            print('\n' * 50)

    def handle_ls(self, args: List[str]) -> None:
        """List files command"""
        target_dir = "."
        show_hidden = False
        long_format = False

        # Parse flags and directory
        for i in range(1, len(args)):
            arg = args[i]
            if arg.startswith("-"):
                # Handle flags
                for flag in arg[1:]:
                    if flag == 'a':
                        show_hidden = True
                    elif flag == 'l':
                        long_format = True
            else:
                if target_dir != ".":
                    raise ValueError("ls: too many directory arguments")
                target_dir = arg

        # Check if directory exists and is accessible
        target_path = Path(target_dir)
        if not target_path.exists():
            raise ValueError(f"ls: {target_dir}: no such file or directory")
        elif not target_path.is_dir():
            raise ValueError(f"ls: {target_dir}: not a directory")

        try:
            entries = list(target_path.iterdir())
        except PermissionError:
            raise ValueError(f"ls: {target_dir}: permission denied")
        except Exception as e:
            raise ValueError(f"ls: {target_dir}: {e}")

        # Sort entries by name
        entries.sort(key=lambda x: x.name.lower())

        for entry in entries:
            # Skip hidden files unless -a flag is used
            if not show_hidden and entry.name.startswith("."):
                continue

            if long_format:
                try:
                    entry_stat = entry.stat()
                    mode = stat.filemode(entry_stat.st_mode)
                    size = entry_stat.st_size
                    mod_time = time.strftime(
                        "%b %d %H:%M", time.localtime(entry_stat.st_mtime))
                    print(f"{mode} {size:>8} {mod_time} {entry.name}")
                except Exception:
                    print(f"? {entry.name}")
            else:
                if entry.is_dir():
                    print(f"{entry.name}/")
                else:
                    print(entry.name)

    def handle_cat(self, args: List[str]) -> None:
        """Cat command"""
        if len(args) < 2:
            raise ValueError(
                "cat: missing filename\nUsage: cat [file1] [file2] ...")

        for filename in args[1:]:
            if not filename:
                raise ValueError("cat: empty filename")

            file_path = Path(filename)
            if not file_path.exists():
                raise ValueError(f"cat: {filename}: no such file or directory")
            elif file_path.is_dir():
                raise ValueError(f"cat: {filename}: is a directory")

            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    print(f.read(), end='')
            except PermissionError:
                raise ValueError(f"cat: {filename}: permission denied")
            except Exception as e:
                raise ValueError(f"cat: {filename}: {e}")

    def handle_mkdir(self, args: List[str]) -> None:
        """Make directory command"""
        if len(args) < 2:
            raise ValueError("mkdir: missing directory name")

        create_parents = False
        dirs = []

        # Parse flags and directories
        for i in range(1, len(args)):
            arg = args[i]
            if arg.startswith("-"):
                if 'p' in arg:
                    create_parents = True
            else:
                dirs.append(arg)

        if not dirs:
            raise ValueError("mkdir: missing directory name")

        for dirname in dirs:
            try:
                dir_path = Path(dirname)
                if create_parents:
                    dir_path.mkdir(parents=True, exist_ok=True)
                else:
                    dir_path.mkdir()
            except FileExistsError:
                raise ValueError(f"mkdir: {dirname}: file exists")
            except Exception as e:
                raise ValueError(f"mkdir: {dirname}: {e}")

    def handle_rmdir(self, args: List[str]) -> None:
        """Remove directory command"""
        if len(args) < 2:
            raise ValueError("rmdir: missing directory name")

        for dirname in args[1:]:
            try:
                dir_path = Path(dirname)
                dir_path.rmdir()
            except FileNotFoundError:
                raise ValueError(
                    f"rmdir: {dirname}: no such file or directory")
            except OSError as e:
                if e.errno == 39:  # Directory not empty
                    raise ValueError(f"rmdir: {dirname}: directory not empty")
                else:
                    raise ValueError(f"rmdir: {dirname}: {e}")
            except Exception as e:
                raise ValueError(f"rmdir: {dirname}: {e}")

    def handle_rm(self, args: List[str]) -> None:
        """Remove file command"""
        if len(args) < 2:
            raise ValueError("rm: missing filename")

        recursive = False
        force = False
        files = []

        # Parse flags and files
        for i in range(1, len(args)):
            arg = args[i]
            if arg.startswith("-"):
                if 'r' in arg or 'R' in arg:
                    recursive = True
                if 'f' in arg:
                    force = True
            else:
                files.append(arg)

        if not files:
            raise ValueError("rm: missing filename")

        for filename in files:
            try:
                file_path = Path(filename)
                if recursive and file_path.is_dir():
                    shutil.rmtree(file_path)
                else:
                    file_path.unlink()
            except FileNotFoundError:
                if not force:
                    raise ValueError(
                        f"rm: {filename}: no such file or directory")
            except IsADirectoryError:
                if not force:
                    raise ValueError(f"rm: {filename}: is a directory")
            except Exception as e:
                if not force:
                    raise ValueError(f"rm: {filename}: {e}")

    def handle_touch(self, args: List[str]) -> None:
        """Touch command"""
        if len(args) < 2:
            raise ValueError("touch: missing filename")

        for filename in args[1:]:
            try:
                file_path = Path(filename)
                file_path.touch()
            except Exception as e:
                raise ValueError(f"touch: {filename}: {e}")

    def handle_kill(self, args: List[str]) -> None:
        """Kill command"""
        if len(args) < 2:
            raise ValueError(
                "kill: missing PID\nUsage: kill [pid1] [pid2] ...")

        errors = []
        killed = 0

        for pid_str in args[1:]:
            if not pid_str:
                errors.append("empty PID")
                continue

            try:
                pid = int(pid_str)
            except ValueError:
                errors.append(f"invalid PID '{pid_str}': not a number")
                continue

            if pid <= 0:
                errors.append(f"invalid PID {pid}: must be positive")
                continue

            # Don't allow killing init process or shell itself
            if pid == 1:
                errors.append("cannot kill init process (PID 1)")
                continue

            if pid == os.getpid():
                errors.append("cannot kill shell process itself")
                continue

            try:
                os.kill(pid, 9)  # SIGKILL
                print(f"Process {pid} killed")
                killed += 1
            except ProcessLookupError:
                errors.append(f"process {pid} not found")
            except PermissionError:
                errors.append(f"permission denied to kill process {pid}")
            except Exception as e:
                errors.append(f"failed to kill process {pid}: {e}")

        # Report any errors
        if errors:
            if killed == 0:
                raise ValueError(f"kill: {'; '.join(errors)}")
            else:
                print(f"kill: warnings: {'; '.join(errors)}")

    def handle_jobs(self, args: List[str]) -> None:
        """Jobs command"""
        self.job_manager.cleanup_completed_jobs()
        self.job_manager.list_jobs()

    def handle_help(self, args: List[str]) -> None:
        """Help command"""
        print("Advanced Shell - Available Commands:")
        print()
        print("Built-in Commands:")
        print("  cd [directory]     - Change directory (supports ~)")
        print("  pwd               - Print working directory")
        print(
            "  echo [text]       - Print text (supports \\n, \\t escape sequences)")
        print("  clear             - Clear screen")
        print(
            "  ls [options] [dir] - List files (-a for hidden, -l for long format)")
        print("  cat [files...]    - Display file contents")
        print(
            "  mkdir [options] [dirs...] - Create directories (-p for parents)")
        print("  rmdir [dirs...]   - Remove empty directories")
        print("  rm [options] [files...] - Remove files (-r recursive, -f force)")
        print("  touch [files...]  - Create empty files or update timestamps")
        print("  kill [pids...]    - Kill processes by PID")
        print("  exit              - Exit shell")
        print("  help              - Show this help")
        print()
        print("Job Control:")
        print("  jobs              - List background jobs")
        print()
        # Deliverable 3: NEW help section
        print("✓ Memory Management (Deliverable 3 - NEW):")
        print("  memory status             - Show memory status and statistics")
        print("  memory create <n> <pages> - Create process with memory requirements")
        print("  memory alloc <pid> <page> - Allocate specific page for process")
        print("  memory dealloc <pid>      - Deallocate all pages for process")
        print("  memory algorithm <fifo|lru> - Set page replacement algorithm")
        print("  memory test <sequential|random> - Run memory access pattern test")
        print()
        print("✓ Process Synchronization (Deliverable 3 - NEW):")
        print("  sync status               - Show synchronization status")
        print("  sync mutex <create|acquire|release> <name> - Mutex operations")
        print(
            "  sync semaphore <create|acquire|release> <name> [value] - Semaphore operations")
        print(
            "  sync prodcons <start|stop|status> [producers] [consumers] - Producer-Consumer")
        print(
            "  sync philosophers <start|stop|status> [philosophers] - Dining Philosophers")
        print()
        print("Usage:")
        print("  command &         - Run command in background")
        print("  Ctrl+C            - Interrupt current foreground process")
        print("  Arrow Keys        - Navigate cursor left/right")
        print()
        print("✓ Deliverable 3 Examples (NEW):")
        print("  memory create webapp 8         # Create process needing 8 pages")
        print("  memory alloc 1 0               # Allocate page 0 for process 1")
        print("  memory algorithm lru           # Switch to LRU replacement")
        print("  memory test random             # Test random access pattern")
        print("  sync mutex create mylock       # Create a mutex")
        print("  sync prodcons start 2 3        # Start Producer-Consumer")
        print("  sync philosophers start 5      # Start Dining Philosophers")

    # Deliverable 3: NEW - Access to memory manager and synchronizer for integration
    def get_memory_manager(self):
        """Get memory manager instance for integration"""
        return self.memory_sync_commands.get_memory_manager()

    def get_synchronizer(self):
        """Get synchronizer instance for integration"""
        return self.memory_sync_commands.get_synchronizer()


# ===== MAIN SHELL CLASS =====

class Shell:
    """Main shell implementation"""

    def __init__(self):
        self.job_manager = JobManager()
        self.command_handler = CommandHandler(self.job_manager)
        self.parser = CommandParser()
        self.input_handler = InputHandler()
        self.running = True
        self.prompt = "[shell]$ "
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def sigint_handler(signum, frame):
            print("\nReceived interrupt signal. Use 'exit' to quit the shell.")
            # Don't exit immediately, let user decide

        signal.signal(signal.SIGINT, sigint_handler)

    def run(self):
        """Start the main shell loop"""
        self.print_welcome()

        while self.running:
            try:
                self.display_prompt()

                try:
                    # Use input handler if available, otherwise fallback to input()
                    if self.input_handler:
                        user_input = self.input_handler.get_input(
                            self.prompt).strip()
                    else:
                        user_input = input().strip()
                except EOFError:
                    # Handle Ctrl+D
                    print("\nGoodbye!")
                    break
                except KeyboardInterrupt:
                    # Handle Ctrl+C
                    print()
                    continue

                # Handle empty input
                if not user_input:
                    continue

                # Process the input and handle errors gracefully
                try:
                    self.process_input(user_input)
                except Exception as e:
                    print(f"\033[31mError:\033[0m {e}")

                # Clean up completed jobs after each command
                self.job_manager.cleanup_completed_jobs()

            except Exception as e:
                print(f"\033[31mShell Error:\033[0m {e}")

        self.shutdown()

    def process_input(self, input_str: str) -> None:
        """Process a single line of input"""
        # Parse the command
        try:
            parsed = self.parser.parse(input_str)
        except ValueError as e:
            raise ValueError(f"Parse error: {e}")

        if not parsed:
            return  # Empty command

        # Validate the command
        try:
            self.parser.validate_command(parsed)
        except ValueError as e:
            raise ValueError(f"Validation error: {e}")

        # Check if it's a built-in command
        if self.parser.is_builtin_command(parsed.command):
            try:
                self.command_handler.handle_command(parsed)
            except ValueError as e:
                raise ValueError(str(e))
            except Exception as e:
                raise ValueError(f"{parsed.command}: {e}")
        else:
            # Try to execute as external command
            self.execute_external_command(parsed)

    def execute_external_command(self, parsed: ParsedCommand) -> None:
        """Execute external command"""
        # Check if external command exists
        if not shutil.which(parsed.command):
            raise ValueError(f"{parsed.command}: command not found")

        try:
            if parsed.background:
                # Background execution
                process = subprocess.Popen(
                    parsed.args,
                    cwd=os.getcwd(),
                    preexec_fn=os.setsid  # Create new process group
                )

                job = self.job_manager.add_job(
                    command=' '.join(parsed.args),
                    args=parsed.args,
                    process=process,
                    background=True
                )

                print(f"[{job.id}] {process.pid}")

                # Deliverable 3: NEW - Integrate with memory management for external commands
                try:
                    memory_manager = self.command_handler.get_memory_manager()
                    if memory_manager:
                        # Estimate memory needs for external command
                        pages_needed = self._estimate_memory_needs(
                            parsed.command)
                        memory_pid = memory_manager.create_process(
                            f"ExtCmd-{parsed.command}", pages_needed)
                except:
                    pass  # Don't fail if memory management unavailable

            else:
                # Foreground execution
                process = subprocess.Popen(
                    parsed.args,
                    cwd=os.getcwd()
                )

                try:
                    process.wait()
                except KeyboardInterrupt:
                    # Forward Ctrl+C to child process
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    print("\n^C")

        except FileNotFoundError:
            raise ValueError(f"{parsed.command}: command not found")
        except PermissionError:
            raise ValueError(f"{parsed.command}: permission denied")
        except Exception as e:
            raise ValueError(f"Error executing {parsed.command}: {e}")

    def _estimate_memory_needs(self, command: str) -> int:
        """Estimate memory needs for external commands (NEW)"""
        # Simple heuristic for memory estimation
        memory_estimates = {
            'ls': 2, 'grep': 3, 'sort': 5, 'cat': 2, 'find': 4,
            'python': 8, 'gcc': 10, 'make': 6, 'vim': 4, 'nano': 2
        }
        return memory_estimates.get(command, 4)  # Default to 4 pages

    def display_prompt(self) -> None:
        """Show the shell prompt"""
        try:
            pwd = os.getcwd()
            dir_name = os.path.basename(pwd)
            if dir_name == "":
                dir_name = pwd
        except Exception:
            dir_name = "unknown"

        # Get current time for enhanced prompt
        current_time = time.strftime("%H:%M:%S")
        self.prompt = f"[shell:{dir_name} {current_time}]$ "

    def print_welcome(self) -> None:
        """Print the welcome message"""
        print("==========================================")
        print("  Advanced Shell Simulation - Deliverable 3")
        print("==========================================")
        print()
        print("Features implemented:")
        print("✓ Built-in commands (cd, pwd, ls, cat, etc.)")
        print("✓ External command execution")
        print("✓ Process management (foreground/background)")
        print("✓ Job control (jobs, kill)")
        print("✓ Signal handling")
        print("✓ Error handling")
        print("✓ Memory management with paging (NEW)")
        print("  • FIFO and LRU page replacement algorithms")
        print("  • Page fault handling and tracking")
        print("  • Memory overflow simulation")
        print("✓ Process synchronization (NEW)")
        print("  • Mutexes and semaphores")
        print("  • Producer-Consumer problem")
        print("  • Dining Philosophers problem")
        print("  • Race condition prevention")
        print("  • Deadlock avoidance")
        print()
        print("Type 'help' for available commands")
        print("Type 'exit' to quit")
        print()
        print("✓ Quick Start - Memory Management (NEW):")
        print("  memory create webapp 8        # Create process needing 8 pages")
        print("  memory alloc 1 0              # Allocate page 0 for process 1")
        print("  memory algorithm lru          # Switch to LRU replacement")
        print("  memory status                 # Show memory statistics")
        print()
        print("✓ Quick Start - Synchronization (NEW):")
        print("  sync prodcons start 2 3       # Start Producer-Consumer")
        print("  sync philosophers start 5     # Start Dining Philosophers")
        print("  sync status                   # Show sync statistics")
        print()

    def shutdown(self) -> None:
        """Perform cleanup before exiting"""
        print("\nShutting down shell...")

        # Deliverable 3: NEW - Stop synchronization problems
        try:
            memory_sync_commands = getattr(
                self.command_handler, 'memory_sync_commands', None)
            if memory_sync_commands:
                if memory_sync_commands.producer_consumer:
                    print("Stopping Producer-Consumer...")
                    memory_sync_commands.producer_consumer.stop()

                if memory_sync_commands.dining_philosophers:
                    print("Stopping Dining Philosophers...")
                    memory_sync_commands.dining_philosophers.stop()
        except:
            pass

        # Get all active jobs
        jobs = self.job_manager.get_all_jobs()
        if jobs:
            print(f"Terminating {len(jobs)} active job(s)...")

            for job in jobs:
                if job.status.value != "Done":
                    print(f"Killing job [{job.id}]: {job.command}")
                    self.job_manager.kill_job(job.id)

            # Give processes time to terminate
            time.sleep(0.1)

        print("Goodbye!")


# ===== TEST FUNCTIONS =====

def test_memory_management():
    """Test memory management features (NEW)"""
    print("=== TESTING MEMORY MANAGEMENT FEATURES ===")
    print()

    try:
        # Test 1: Memory Usage Tracking per Process
        print("✓ Testing Memory Usage Tracking per Process:")
        mm = MemoryManager(total_frames=6)  # Small frames to force overflow

        # Create multiple processes with different memory needs
        pid1 = mm.create_process("WebBrowser", 4)
        pid2 = mm.create_process("TextEditor", 3)
        pid3 = mm.create_process("MediaPlayer", 5)

        print("  - Created processes with different memory requirements")
        print(f"    * WebBrowser (PID {pid1}): needs 4 pages")
        print(f"    * TextEditor (PID {pid2}): needs 3 pages")
        print(f"    * MediaPlayer (PID {pid3}): needs 5 pages")

        # Allocate pages for each process
        print("  - Allocating pages for each process...")
        mm.allocate_page(pid1, 0)  # WebBrowser page 0
        mm.allocate_page(pid1, 1)  # WebBrowser page 1
        mm.allocate_page(pid2, 0)  # TextEditor page 0
        mm.allocate_page(pid3, 0)  # MediaPlayer page 0
        mm.allocate_page(pid3, 1)  # MediaPlayer page 1

        status = mm.get_status()
        print(
            f"  - Memory Usage: {status['used_frames']}/{status['total_frames']} frames ({status['utilization']:.1f}%)")
        print(f"  - Per-Process Allocation:")
        for pid, process in mm.processes.items():
            allocated = len(
                [p for p in process.page_table.values() if p is not None])
            print(
                f"    * PID {pid} ({process.name}): {allocated}/{process.pages_needed} pages allocated")
        print()

        # Test 2: Memory Overflow Simulation
        print("✓ Testing Memory Overflow Scenarios:")
        mm.set_algorithm("fifo")
        print("  - Set algorithm to FIFO for overflow testing")

        # Force memory overflow by allocating more pages than available frames
        print("  - Forcing memory overflow (6 frames available, allocating more)...")
        overflow_results = []

        # Fill remaining frame
        success, msg = mm.allocate_page(pid3, 2)  # Frame 6 (last free frame)
        overflow_results.append(f"    Frame 6: {msg}")

        # Now force page replacements
        success, msg = mm.allocate_page(pid1, 2)  # Should trigger replacement
        overflow_results.append(f"    Overflow 1: {msg}")

        # Should trigger another replacement
        success, msg = mm.allocate_page(pid2, 1)
        overflow_results.append(f"    Overflow 2: {msg}")

        # Should trigger another replacement
        success, msg = mm.allocate_page(pid3, 3)
        overflow_results.append(f"    Overflow 3: {msg}")

        for result in overflow_results:
            print(result)

        status = mm.get_status()
        print(f"  - Memory overflow results:")
        print(f"    * Total page faults: {status['page_faults']}")
        print(f"    * Page replacements triggered: {status['replacements']}")
        print(
            f"    * Memory utilization: {status['utilization']:.1f}% (should be 100%)")
        print()

        # Test 3: Page Fault Tracking
        print("✓ Testing Page Fault Tracking:")
        print("  - Testing page fault vs page hit patterns...")

        # Access same pages multiple times to show fault vs hit tracking
        test_accesses = [
            (pid1, 0, "First access (should be hit - already allocated)"),
            (pid1, 3, "New page (should be fault)"),
            (pid1, 0, "Repeat access (should be hit)"),
            (pid2, 2, "New page (should be fault)"),
            (pid1, 3, "Repeat access (should be hit)"),
        ]

        for pid, page_num, description in test_accesses:
            old_faults = mm.page_faults
            old_hits = mm.page_hits
            success, msg = mm.allocate_page(pid, page_num)
            new_faults = mm.page_faults
            new_hits = mm.page_hits

            fault_occurred = new_faults > old_faults
            hit_occurred = new_hits > old_hits
            result_type = "FAULT" if fault_occurred else "HIT" if hit_occurred else "ERROR"

            print(f"    * {description}: {result_type}")

        final_status = mm.get_status()
        print(f"  - Final page fault statistics:")
        print(f"    * Total page faults: {final_status['page_faults']}")
        print(f"    * Total page hits: {final_status['page_hits']}")
        print(f"    * Hit ratio: {final_status['hit_ratio']:.1f}%")
        print()

        # Test 4: FIFO vs LRU Comparison
        print("✓ Testing FIFO vs LRU Page Replacement:")

        # Test FIFO
        mm_fifo = MemoryManager(total_frames=4)
        mm_fifo.set_algorithm("fifo")
        pid_fifo = mm_fifo.create_process("FIFOTest", 6)

        print("  - FIFO Algorithm Test:")
        fifo_pattern = [0, 1, 2, 3, 0, 1, 4, 5]  # Will cause replacements
        for i, page_num in enumerate(fifo_pattern):
            success, msg = mm_fifo.allocate_page(pid_fifo, page_num)

        fifo_status = mm_fifo.get_status()
        print(
            f"    * FIFO Results: {fifo_status['page_faults']} faults, {fifo_status['page_hits']} hits, {fifo_status['replacements']} replacements")

        # Test LRU
        mm_lru = MemoryManager(total_frames=4)
        mm_lru.set_algorithm("lru")
        pid_lru = mm_lru.create_process("LRUTest", 6)

        print("  - LRU Algorithm Test:")
        # Same pattern, different results
        lru_pattern = [0, 1, 2, 3, 0, 1, 4, 5]
        for i, page_num in enumerate(lru_pattern):
            success, msg = mm_lru.allocate_page(pid_lru, page_num)

        lru_status = mm_lru.get_status()
        print(
            f"    * LRU Results: {lru_status['page_faults']} faults, {lru_status['page_hits']} hits, {lru_status['replacements']} replacements")

        print(f"  - Algorithm Comparison:")
        print(f"    * FIFO hit ratio: {fifo_status['hit_ratio']:.1f}%")
        print(f"    * LRU hit ratio: {lru_status['hit_ratio']:.1f}%")
        print(
            f"    * LRU is {'better' if lru_status['hit_ratio'] > fifo_status['hit_ratio'] else 'same as'} FIFO for this pattern")
        print()

        print("✓ Memory management tests completed successfully!")
        print("✓ Demonstrated: Per-process tracking, memory overflow, page fault tracking, algorithm comparison")

    except Exception as e:
        print(f"Error during memory tests: {e}")


def run_scheduling_tests():
    """Run the scheduling tests with performance metrics"""
    print("Running scheduling tests with performance metrics...")

    # Import test functions
    from test_scheduling_with_metrics import (
        test_round_robin_configurable_time_slice,
        test_priority_with_time_simulation,
        test_preemption_with_time_simulation,
        test_early_completion_behavior
    )

    # Redirect all output to a text file
    output_file = "test_scheduling_with_metrics_output.txt"

    # Save original stdout
    original_stdout = sys.stdout

    try:
        # Redirect stdout to file
        with open(output_file, 'w') as f:
            sys.stdout = f

            print("Advanced Shell - Scheduling Test Suite with Performance Metrics")
            print("==============================================================")
            print()
            print("Testing Constraints:")
            print("1. Time slice is configurable and user-specified")
            print("2. Processes complete early if possible (removed from queue)")
            print("3. Process execution simulated with time.sleep()")
            print("4. Performance metrics tracked for all tests")
            print()

            try:
                test_round_robin_configurable_time_slice()
                test_priority_with_time_simulation()
                test_preemption_with_time_simulation()
                test_early_completion_behavior()

                print("All tests completed successfully!")
                print(
                    "The scheduling algorithms are working correctly with configurable time slices.")
                print(
                    "All test files and directories have been created in the 'test/' directory.")
                print()
                print("Key Features Demonstrated:")
                print("- Configurable time slices (0.5s, 1.0s, 2.0s)")
                print("- Early completion when processes finish before time slice")
                print("- Time simulation using actual command execution")
                print("- Priority-based preemption with time simulation")
                print("- Comprehensive performance metrics collection")

            except KeyboardInterrupt:
                print("\nTests interrupted by user")
            except Exception as e:
                print(f"Test error: {e}")

    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        print(f"Test output has been saved to: {output_file}")

        # Generate performance report
        performance_tracker.generate_report("performance_metrics_report.txt")
        print(
            "Performance metrics report has been saved to: performance_metrics_report.txt")


def test_synchronization():
    """Test synchronization features (NEW)"""
    print("=== TESTING SYNCHRONIZATION FEATURES ===")
    print()

    try:
        # Test basic synchronization primitives
        print("✓ Testing Synchronization Primitives:")
        sync = ProcessSynchronizer()

        # Test mutex
        sync.create_mutex("test_mutex")
        print("  - Created mutex 'test_mutex'")

        success = sync.acquire_mutex("test_mutex")
        print(f"  - Acquired mutex: {success}")

        if success:
            success = sync.release_mutex("test_mutex")
            print(f"  - Released mutex: {success}")
        else:
            print("  - Skipping release (acquire failed)")

        # Test semaphore
        sync.create_semaphore("test_sem", 3)
        print("  - Created semaphore 'test_sem' with value 3")

        acquired_count = 0
        for i in range(2):
            success = sync.acquire_semaphore("test_sem")
            if success:
                acquired_count += 1
            print(f"  - Acquired semaphore {i+1}: {success}")

        print(f"  - Successfully acquired {acquired_count} semaphore permits")

        status = sync.get_status()
        print(
            f"  - Synchronization status: {status['mutexes']} mutexes, {status['semaphores']} semaphores")
        print()

        # Test Producer-Consumer
        print("✓ Testing Producer-Consumer Problem:")
        pc = ProducerConsumer(buffer_size=3)
        print("Starting Producer-Consumer simulation... (Number of producers: 2, consumers: 3)")

        pc.start(num_producers=2, num_consumers=3, duration=5)
        time.sleep(6)  # Let it run briefly

        pc_status = pc.get_status()
        print(
            f"  - Buffer: {pc_status['current_buffer']}/{pc_status['buffer_size']}")
        print(f"  - Items Produced: {pc_status['items_produced']}")
        print(f"  - Items Consumed: {pc_status['items_consumed']}")
        print(
            f"  - Producer Waits: {pc_status['producer_waits']} (times producers waited for space)")
        print(
            f"  - Consumer Waits: {pc_status['consumer_waits']} (times consumers waited for items)")
        print(
            f"  - Active: {pc_status['active_producers']} producers, {pc_status['active_consumers']} consumers")

        pc.stop()
        print("Producer-Consumer stopped")
        print()

        # Test Dining Philosophers
        print("✓ Testing Dining Philosophers Problem: (Number of philosophers: 5) (Time duration: 3 seconds)")
        dp = DiningPhilosophers(num_philosophers=5)
        print("Starting Dining Philosophers simulation...")

        # Run in background thread for brief test
        def run_philosophers():
            dp.start(duration=3)

        phil_thread = threading.Thread(target=run_philosophers)
        phil_thread.daemon = True
        phil_thread.start()

        time.sleep(4)  # Let it run briefly

        dp_status = dp.get_status()
        print(f"  - Philosophers: {dp_status['num_philosophers']}")
        print(f"  - Total meals eaten: {dp_status['total_meals']}")
        print(f"  - Deadlock preventions: {dp_status['deadlock_preventions']}")
        print(f"  - States: {', '.join(dp_status['states'])}")

        dp.stop()
        print("Dining Philosophers stopped")
        print()

        print("✓ Synchronization tests completed successfully!")

    except Exception as e:
        print(f"Error during synchronization tests: {e}")


# ===== MAIN FUNCTION =====

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced Shell Simulation - Deliverable 3",
        add_help=False  # We'll handle help ourselves
    )

    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version information'
    )

    parser.add_argument(
        '--help',
        action='store_true',
        help='Show help information'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    # Deliverable 3: NEW test options
    parser.add_argument(
        '--test-memory',
        action='store_true',
        help='Run memory management tests and exit (NEW)'
    )

    parser.add_argument(
        '--test-sync',
        action='store_true',
        help='Run synchronization tests and exit (NEW)'
    )

    parser.add_argument(
        '--test-scheduling-with-metrics',
        action='store_true',
        help='Run scheduling tests with performance metrics'
    )

    args = parser.parse_args()

    if args.version:
        print("Advanced Shell Simulation")
        print("Version: 3.0.0 (Deliverable 3)")
        print("Build: Development")
        print()
        print("Features:")
        print("- Basic shell functionality (Deliverable 1)")
        print("- Built-in commands")
        print("- Process management")
        print("- Job control")
        print("- Process scheduling algorithms (Deliverable 2)")
        print("  * Round-Robin Scheduling")
        print("  * Priority-Based Scheduling")
        print("- Performance metrics")
        print("- Real-time process monitoring")
        print("✓ Memory management with paging (Deliverable 3 - NEW)")
        print("  * FIFO and LRU page replacement algorithms")
        print("  * Page fault handling and tracking")
        print("  * Memory overflow simulation")
        print("✓ Process synchronization (Deliverable 3 - NEW)")
        print("  * Mutexes and semaphores")
        print("  * Producer-Consumer problem")
        print("  * Dining Philosophers problem")
        print("  * Race condition prevention")
        print("  * Deadlock avoidance")
        return

    if args.help:
        print("Advanced Shell Simulation - Deliverable 3")
        print()
        print("Usage:")
        print("  python3 deliverable_3_shell.py [options]")
        print()
        print("Options:")
        print("  --version    Show version information")
        print("  --help       Show this help message")
        print("  --debug      Enable debug mode")
        print("  --test-memory    Run memory management tests (NEW)")
        print("  --test-sync      Run synchronization tests (NEW)")
        print("  --test-scheduling-with-metrics  Run scheduling tests with performance metrics")
        print()
        print("Once started, type 'help' for available shell commands")
        print()
        print("✓ Quick Start - Memory Management (NEW):")
        print("  1. Create process:         memory create webapp 8")
        print("  2. Allocate pages:         memory alloc 1 0")
        print("  3. Set algorithm:          memory algorithm lru")
        print("  4. View status:            memory status")
        print("  5. Test patterns:          memory test random")
        print()
        print("✓ Quick Start - Synchronization (NEW):")
        print("  1. Create mutex:           sync mutex create mylock")
        print("  2. Start Producer-Consumer: sync prodcons start 2 3")
        print("  3. Start Dining Philosophers: sync philosophers start 5")
        print("  4. View status:            sync status")
        return

    if args.test_scheduling_with_metrics:
        run_scheduling_tests()
        return

    # Deliverable 3: NEW - Handle test options
    if args.test_memory:
        test_memory_management()
        return
    if args.test_sync:
        test_synchronization()
        return

    # Create and start the shell
    try:
        shell = Shell()

        if args.debug:
            print("Debug mode enabled")
            # Deliverable 3: NEW debug info
            print("Memory management available:", hasattr(
                shell.command_handler, 'memory_sync_commands'))
            if hasattr(shell.command_handler, 'memory_sync_commands'):
                memory_manager = shell.command_handler.get_memory_manager()
                synchronizer = shell.command_handler.get_synchronizer()
                print(
                    f"  - Memory manager: {memory_manager.total_frames} frames")
                print(f"  - Synchronizer: initialized")

        # Run the shell
        shell.run()

    except KeyboardInterrupt:
        print("\nShell interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()