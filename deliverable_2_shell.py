"""
Advanced Shell Simulation - Deliverable 2
A custom shell implementation in Python that simulates Unix-like Operating System environment
with process management, job control, built-in commands, and advanced process scheduling algorithms.

HOW TO RUN:
Prerequisites:
- Python 3.8 or higher

Running the Shell:
# Start the shell
python3 deliverable_2_shell.py

# Command-line options
python3 deliverable_2_shell.py --help     # Show help
python3 deliverable_2_shell.py --version  # Show version info
python3 deliverable_2_shell.py --debug    # Enable debug mode

Example Usage:
# Navigation and file operations (from Deliverable 1)
ls -la
pwd
cd /tmp
mkdir -p test/nested/path
touch test_file.txt
cat test_file.txt
rm -rf test

# Process and job management (from Deliverable 1)
kill 1234          # Kill process by PID
sleep 60 &         # Start background job
stop 1             # Stop job 1 (testing feature)
bg 1               # Resume job 1 in background
fg 1               # Bring job 1 to foreground

# Process Scheduling (NEW in Deliverable 2)
# Round-Robin Scheduling
scheduler config rr 2.5                    # Configure Round-Robin with 2.5s quantum
scheduler addprocess task1 8               # Add 8-second process
scheduler addprocess task2 5               # Add 5-second process
scheduler addprocess task3 3               # Add 3-second process
scheduler start                            # Begin scheduling
scheduler status                           # Monitor execution
scheduler metrics                          # View performance data

# Priority-Based Scheduling
scheduler config priority                  # Configure Priority scheduling
scheduler addprocess low_task 10 1         # Low priority (1)
scheduler addprocess med_task 6 5          # Medium priority (5)
scheduler addprocess high_task 4 10        # High priority (10)
scheduler start                            # High priority runs first

# Test Preemption
scheduler config priority
scheduler addprocess background 15 2       # Long, low priority task
scheduler start                            # Starts background task
# Add high priority task while running:
scheduler addprocess urgent 3 8            # Will preempt background task!

# Scheduler Management
scheduler stop                             # Stop scheduler
scheduler clear                            # Clear all state and metrics
scheduler test                             # Run automated preemption test

# Shell operations
help
exit

GitHub Repository: https://github.com/Su5ubedi/advanced-shell-python
"""

import os
import sys
import signal
import subprocess
import stat
import time
import tty
import termios
import shlex
import shutil
import heapq
import threading
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


# =====================================================
# DATA TYPES AND ENUMS
# =====================================================

class JobStatus(Enum):
    """Job status enumeration"""
    RUNNING = "Running"
    STOPPED = "Stopped"
    DONE = "Done"


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
            self.waiting_time = 0.0
            self.turnaround_time = 0.0
            if self.response_time is None:
                self.response_time = 0.0


@dataclass
class ScheduledProcess:
    """Represents a process in the scheduler"""
    pid: int
    name: str
    duration: float
    priority: int = 0
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
            return self.priority > other.priority
        self_arrival = self.metrics.arrival_time if self.metrics else 0.0
        other_arrival = other.metrics.arrival_time if other.metrics else 0.0
        return self_arrival < other_arrival


@dataclass
class SchedulerConfig:
    """Scheduler configuration"""
    algorithm: SchedulingAlgorithm
    time_quantum: float = 1.0
    context_switch_time: float = 0.1


# =====================================================
# INPUT HANDLER
# =====================================================

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

    def clear_current_command(self, text: str, cursor_pos: int) -> Tuple[str, int]:
        """Clear the current command (Ctrl+C)"""
        self.clear_line()
        return "", 0

    def backspace(self, text: str, pos: int) -> Tuple[str, int]:
        """Handle backspace key"""
        if pos > 0:
            text = text[:pos-1] + text[pos:]
            pos -= 1
            sys.stdout.write('\b \b')
            if pos < len(text):
                sys.stdout.write(text[pos:])
                sys.stdout.write(' ')
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

            if ch == '\x1b':  # ESC - might be arrow key
                ch2 = self.get_char()
                if ch2 == '[':
                    ch3 = self.get_char()
                    if ch3 == 'A':  # UP arrow
                        continue
                    elif ch3 == 'B':  # DOWN arrow
                        continue
                    elif ch3 == 'C':  # RIGHT arrow
                        cursor_pos = self.move_cursor_right(cursor_pos, text)
                        continue
                    elif ch3 == 'D':  # LEFT arrow
                        cursor_pos = self.move_cursor_left(cursor_pos)
                        continue
                continue

            elif ch == '\x03':  # Ctrl+C
                text, cursor_pos = self.clear_current_command(text, cursor_pos)
                print('^C')
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
                continue
            elif ch < ' ' or ch > '~':
                continue
            else:
                text = self.insert_char(text, cursor_pos, ch)
                sys.stdout.write(ch)
                if cursor_pos < len(text) - 1:
                    sys.stdout.write(text[cursor_pos + 1:])
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


# =====================================================
# COMMAND PARSER
# =====================================================

class CommandParser:
    """Handles parsing of command line input"""

    BUILTIN_COMMANDS = {
        'cd', 'pwd', 'exit', 'echo', 'clear', 'ls', 'cat',
        'mkdir', 'rmdir', 'rm', 'touch', 'kill', 'jobs',
        'fg', 'bg', 'stop', 'help', 'scheduler'
    }

    def parse(self, input_str: str) -> Optional[ParsedCommand]:
        """Parse a command line input string"""
        input_str = input_str.strip()
        if not input_str:
            return None

        background = False
        if input_str.endswith('&'):
            background = True
            input_str = input_str[:-1].strip()

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
            pipes=[args]
        )

    def is_builtin_command(self, command: str) -> bool:
        """Check if a command is a built-in command"""
        return command in self.BUILTIN_COMMANDS

    def validate_command(self, parsed: ParsedCommand) -> None:
        """Perform comprehensive validation on parsed commands"""
        if not parsed or not parsed.command:
            return

        if '..' in parsed.command:
            raise ValueError(f"Potentially dangerous path detected: {parsed.command}")

        dangerous_chars = '|;&<>(){}[]'
        if any(char in parsed.command for char in dangerous_chars):
            raise ValueError(f"Invalid characters in command name: {parsed.command}")

        if len(parsed.command) > 256:
            raise ValueError("Command name too long (max 256 characters)")

        for i, arg in enumerate(parsed.args):
            if len(arg) > 1024:
                raise ValueError(f"Argument {i} too long (max 1024 characters)")

        if len(parsed.args) > 100:
            raise ValueError("Too many arguments (max 100)")


# =====================================================
# JOB MANAGER
# =====================================================

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
            job.is_alive()
            duration = time.time() - job.start_time
            if job.end_time:
                duration = job.end_time - job.start_time
            print(f"[{job.id}] {job.status.value} {job.command} (PID: {job.pid}, Duration: {int(duration)}s)")

    def bring_to_foreground(self, job_id: int) -> bool:
        """Bring a background job to the foreground"""
        job = self.get_job(job_id)
        if not job:
            print(f"fg: job {job_id} not found")
            return False

        if job.status == JobStatus.DONE:
            print(f"fg: job {job_id} has already completed")
            return False

        print(f"Bringing job [{job.id}] to foreground: {job.command}")

        try:
            if job.status == JobStatus.STOPPED:
                os.kill(job.pid, signal.SIGCONT)

            job.status = JobStatus.RUNNING
            job.background = False

            try:
                job.process.wait()
                print(f"Job [{job.id}] completed")
            except KeyboardInterrupt:
                print(f"\nJob [{job.id}] interrupted")
                job.process.terminate()
                try:
                    job.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    job.process.kill()

            job.status = JobStatus.DONE
            job.end_time = time.time()
            del self.jobs[job_id]
            return True

        except ProcessLookupError:
            print(f"fg: process {job.pid} no longer exists")
            job.status = JobStatus.DONE
            if job_id in self.jobs:
                del self.jobs[job_id]
            return False
        except Exception as e:
            print(f"fg: failed to bring job to foreground: {e}")
            return False

    def resume_in_background(self, job_id: int) -> bool:
        """Resume a stopped job in the background"""
        job = self.get_job(job_id)
        if not job:
            print(f"bg: job {job_id} not found")
            return False

        if job.status == JobStatus.DONE:
            print(f"bg: job {job_id} has already completed")
            return False

        if job.status != JobStatus.STOPPED:
            print(f"bg: job {job_id} is not stopped")
            return False

        print(f"Resuming job [{job.id}] in background: {job.command}")

        try:
            os.kill(job.pid, signal.SIGCONT)
            job.status = JobStatus.RUNNING
            job.background = True
            return True
        except ProcessLookupError:
            print(f"bg: process {job.pid} no longer exists")
            job.status = JobStatus.DONE
            if job_id in self.jobs:
                del self.jobs[job_id]
            return False
        except Exception as e:
            print(f"bg: failed to resume job: {e}")
            return False

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
            job.is_alive()
            if job.status == JobStatus.DONE:
                completed_jobs.append(job_id)
                print(f"[{job_id}]+ Done\t\t{job.command}")

        for job_id in completed_jobs:
            if job_id in self.jobs:
                del self.jobs[job_id]


# =====================================================
# PROCESS SCHEDULER
# =====================================================

class ProcessScheduler:
    """Process scheduler implementing Round-Robin and Priority-Based scheduling"""

    def __init__(self):
        self.config: Optional[SchedulerConfig] = None
        self.ready_queue = deque()
        self.priority_queue = []
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

    def clear_scheduler_state(self) -> None:
        """Clear all scheduler state including queues, processes, metrics, and configuration"""
        self.ready_queue.clear()
        self.priority_queue.clear()
        self.current_process = None
        self.completed_processes.clear()
        self.process_counter = 1
        self.total_context_switches = 0
        self.running = False
        self.config = None

    def add_process(self, name: str, duration: float, priority: int = 0) -> int:
        """Add a new process to the scheduler"""
        if not self.config:
            raise ValueError("Scheduler not configured. Use 'scheduler config' command first.")

        process = ScheduledProcess(
            pid=self.process_counter,
            name=name,
            duration=duration,
            priority=priority,
            state=ProcessState.READY
        )

        if self.config.algorithm == SchedulingAlgorithm.ROUND_ROBIN:
            self.ready_queue.append(process)
        else:
            heapq.heappush(self.priority_queue, process)

        self.process_counter += 1
        return process.pid

    def start_scheduler(self) -> None:
        """Start the scheduler (blocking execution)"""
        if self.running:
            raise ValueError("Scheduler is already running")

        if not self.config:
            raise ValueError("Scheduler not configured")

        if not self.ready_queue and not self.priority_queue and not self.current_process:
            raise ValueError("No processes to schedule. Use 'scheduler addprocess' to add processes first.")

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
        """Round-Robin scheduling algorithm implementation"""
        print("Starting Round-Robin scheduler...")

        while self.running and (self.ready_queue or self.current_process):
            if not self.current_process and self.ready_queue:
                self.current_process = self.ready_queue.popleft()
                if self.current_process:
                    self.current_process.state = ProcessState.RUNNING
                    if self.current_process.metrics and self.current_process.metrics.start_time is None:
                        self.current_process.metrics.start_time = time.time()
                    self.total_context_switches += 1
                    print(f"[RR Scheduler] Running process {self.current_process.pid} ({self.current_process.name})")

            if not self.current_process:
                break

            if self.config:
                execution_time = min(self.config.time_quantum, self.current_process.remaining_time)
                time.sleep(execution_time)
                self.current_process.remaining_time -= execution_time

                if self.current_process.remaining_time <= 0:
                    self.current_process.state = ProcessState.TERMINATED
                    if self.current_process.metrics:
                        self.current_process.metrics.completion_time = time.time()
                        self.current_process.metrics.calculate_metrics()
                    print(f"[RR Scheduler] Process {self.current_process.pid} ({self.current_process.name}) completed")
                    self.completed_processes.append(self.current_process)
                    self.current_process = None
                else:
                    self.current_process.state = ProcessState.READY
                    self.ready_queue.append(self.current_process)
                    print(f"[RR Scheduler] Process {self.current_process.pid} preempted, remaining time: {self.current_process.remaining_time:.2f}s")
                    self.current_process = None

        self.running = False
        print("Round-Robin scheduler completed!")

    def _priority_scheduler(self) -> None:
        """Priority-Based scheduling algorithm implementation with preemption support"""
        print("Starting Priority-Based scheduler...")

        while self.running and (self.priority_queue or self.current_process):
            if not self.current_process and self.priority_queue:
                self.current_process = heapq.heappop(self.priority_queue)
                if self.current_process:
                    self.current_process.state = ProcessState.RUNNING
                    if self.current_process.metrics and self.current_process.metrics.start_time is None:
                        self.current_process.metrics.start_time = time.time()
                    self.total_context_switches += 1
                    print(f"[Priority Scheduler] Running process {self.current_process.pid} ({self.current_process.name}) priority={self.current_process.priority}")

            if not self.current_process:
                break

            while self.current_process.remaining_time > 0 and self.running:
                execution_slice = min(0.5, self.current_process.remaining_time)
                time.sleep(execution_slice)
                self.current_process.remaining_time -= execution_slice

                if self.priority_queue and self.priority_queue[0].priority > self.current_process.priority:
                    self.current_process.state = ProcessState.READY
                    heapq.heappush(self.priority_queue, self.current_process)
                    print(f"[Priority Scheduler] Process {self.current_process.pid} preempted by higher priority process")
                    self.current_process = None
                    break

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


# =====================================================
# SCHEDULER COMMANDS
# =====================================================

class SchedulerCommands:
    """Command handlers for process scheduling functionality"""

    def __init__(self, scheduler: ProcessScheduler):
        self.scheduler = scheduler

    def handle_scheduler(self, args: List[str]) -> None:
        """Handle scheduler command for all scheduler operations"""
        if len(args) < 2:
            raise ValueError("scheduler: missing subcommand\nUsage: scheduler <config|start|stop|status|metrics|clear|addprocess|test>")

        subcommand = args[1].lower()

        if subcommand == "config" or subcommand == "configure":
            self._handle_scheduler_config(args[2:])
        elif subcommand == "addprocess" or subcommand == "add":
            self._handle_scheduler_addprocess(args[2:])
        elif subcommand == "start":
            try:
                self.scheduler.start_scheduler()
                algorithm_name = "Round-Robin" if self.scheduler.config and self.scheduler.config.algorithm == SchedulingAlgorithm.ROUND_ROBIN else "Priority-Based"
                print(f"Started {algorithm_name} scheduler")
            except ValueError as e:
                raise ValueError(f"scheduler start: {e}")
        elif subcommand == "stop":
            if self.scheduler.running:
                self.scheduler.stop_scheduler()
                print("Scheduler stopped")
            else:
                print("Scheduler is not running")
        elif subcommand == "clear":
            if self.scheduler.running:
                raise ValueError("scheduler clear: Cannot clear state while scheduler is running. Stop scheduler first.")

            self.scheduler.clear_scheduler_state()
            print("Scheduler state and metrics cleared")
            print("Use 'scheduler config <algorithm>' to configure the scheduler")
        elif subcommand == "status":
            status = self.scheduler.get_status()
            self._print_scheduler_status(status)
        elif subcommand == "metrics":
            metrics = self.scheduler.get_metrics()
            self._print_scheduler_metrics(metrics)
        elif subcommand == "test" or subcommand == "testpreemption":
            self._handle_scheduler_test(args[2:])
        else:
            raise ValueError(f"scheduler: unknown subcommand '{subcommand}'\nAvailable subcommands: config, addprocess, start, stop, status, metrics, clear, test")

    def _handle_scheduler_config(self, args: List[str]) -> None:
        """Handle scheduler config subcommand"""
        if len(args) < 1:
            raise ValueError("scheduler config: missing algorithm\nUsage: scheduler config <rr|priority> [time_quantum]")

        algorithm_str = args[0].lower()

        if algorithm_str == "rr" or algorithm_str == "round_robin":
            algorithm = SchedulingAlgorithm.ROUND_ROBIN
            time_quantum = 1.0  # Default time quantum

            if len(args) > 1:
                try:
                    time_quantum = float(args[1])
                    if time_quantum <= 0:
                        raise ValueError("scheduler config: time quantum must be positive")
                except ValueError as e:
                    if "time quantum must be positive" in str(e):
                        raise e
                    raise ValueError(f"scheduler config: invalid time quantum '{args[1]}': must be a number")

            self.scheduler.configure(algorithm, time_quantum)
            print(f"Configured Round-Robin scheduling with time quantum: {time_quantum}s")

            if self.scheduler.priority_queue:
                print("Warning: There are existing processes in priority queue. Use 'scheduler clear' if you want to start fresh.")

        elif algorithm_str == "priority":
            algorithm = SchedulingAlgorithm.PRIORITY
            self.scheduler.configure(algorithm)
            print("Configured Priority-Based scheduling")

            if self.scheduler.ready_queue:
                print("Warning: There are existing processes in round-robin queue. Use 'scheduler clear' if you want to start fresh.")

        else:
            raise ValueError(f"scheduler config: unknown algorithm '{algorithm_str}'\nSupported algorithms: rr (round_robin), priority")

    def _handle_scheduler_addprocess(self, args: List[str]) -> None:
        """Handle scheduler addprocess subcommand"""
        if len(args) < 2:
            raise ValueError("scheduler addprocess: missing arguments\nUsage: scheduler addprocess <name> <duration> [priority]")

        name = args[0]
        if not name:
            raise ValueError("scheduler addprocess: process name cannot be empty")

        try:
            duration = float(args[1])
            if duration <= 0:
                raise ValueError("scheduler addprocess: duration must be positive")
        except ValueError as e:
            if "duration must be positive" in str(e):
                raise e
            raise ValueError(f"scheduler addprocess: invalid duration '{args[1]}': must be a number")

        priority = 0  # Default priority
        if len(args) > 2:
            try:
                priority = int(args[2])
            except ValueError:
                raise ValueError(f"scheduler addprocess: invalid priority '{args[2]}': must be an integer")

        try:
            pid = self.scheduler.add_process(name, duration, priority)
            if self.scheduler.config and self.scheduler.config.algorithm == SchedulingAlgorithm.PRIORITY:
                print(f"Added process '{name}' (PID: {pid}, Duration: {duration}s, Priority: {priority})")
            else:
                print(f"Added process '{name}' (PID: {pid}, Duration: {duration}s)")

        except ValueError as e:
            raise ValueError(str(e))

    def _handle_scheduler_test(self, args: List[str]) -> None:
        """Handle scheduler test subcommand for preemption testing"""
        if not args:
            print("Running preemption test...")
            print("This will demonstrate priority-based preemption")

            try:
                self.scheduler.configure(SchedulingAlgorithm.PRIORITY)
                print("Configured Priority-Based scheduling")

                pid1 = self.scheduler.add_process("long_low_task", 8.0, 1)
                print(f"Added long low-priority process (PID: {pid1}, Duration: 8s, Priority: 1)")

                print("Starting scheduler...")
                print("The scheduler will run the low-priority task first.")
                print("After 3 seconds, a high-priority task will be added to demonstrate preemption.")

                self.scheduler.start_scheduler()

                def add_high_priority():
                    time.sleep(3)
                    try:
                        pid2 = self.scheduler.add_process("high_priority_task", 2.0, 10)
                        print(f"\n[TEST] Added high-priority process (PID: {pid2}, Duration: 2s, Priority: 10)")
                        print("[TEST] This should preempt the low-priority task!")
                    except Exception as e:
                        print(f"[TEST] Error adding high-priority process: {e}")

                thread = threading.Thread(target=add_high_priority, daemon=True)
                thread.start()

            except Exception as e:
                raise ValueError(f"Test failed: {e}")
        else:
            raise ValueError("scheduler test: no arguments needed\nUsage: scheduler test")

    def _print_scheduler_status(self, status: dict) -> None:
        """Print scheduler status information"""
        print("=== Scheduler Status ===")

        if not status['algorithm']:
            print("Status: Not configured")
            print("Use 'scheduler config <rr|priority>' to configure the scheduler")
            return

        algorithm_name = "Round-Robin" if status['algorithm'] == 'round_robin' else "Priority-Based"
        print(f"Algorithm: {algorithm_name}")

        if status['algorithm'] == 'round_robin':
            print(f"Time Quantum: {status['time_quantum']}s")

        print(f"Status: {'Running' if status['running'] else 'Stopped'}")
        print(f"Completed Processes: {status['completed_processes']}")
        print(f"Total Context Switches: {status['total_context_switches']}")

        if status['current_process']:
            cp = status['current_process']
            if status['algorithm'] == 'priority':
                print(f"Current Process: PID {cp['pid']} ({cp['name']}) - Remaining: {cp['remaining_time']}s, Priority: {cp['priority']}")
            else:
                print(f"Current Process: PID {cp['pid']} ({cp['name']}) - Remaining: {cp['remaining_time']}s")
        else:
            print("Current Process: None")

        print(f"Ready Queue Size: {status['ready_queue_size']}")

        if status.get('ready_processes'):
            print("Ready Processes:")
            for proc in status['ready_processes']:
                if 'priority' in proc:
                    print(f"  PID {proc['pid']} ({proc['name']}) - Remaining: {proc['remaining_time']}s, Priority: {proc['priority']}")
                else:
                    print(f"  PID {proc['pid']} ({proc['name']}) - Remaining: {proc['remaining_time']}s")

        if status['ready_queue_size'] == 0 and not status['current_process'] and not status['running']:
            print()
            print("Tip: Use 'scheduler addprocess <name> <duration> [priority]' to add processes")
            print("     Use 'scheduler start' to begin scheduling")

    def _print_scheduler_metrics(self, metrics: dict) -> None:
        """Print scheduler performance metrics"""
        print("=== Scheduler Performance Metrics ===")

        if 'message' in metrics:
            print(metrics['message'])
            if not metrics.get('completed_processes', 0):
                print()
                print("Tip: Add processes with 'scheduler addprocess' and run 'scheduler start'")
                print("     to generate performance metrics")
            return

        print(f"Completed Processes: {metrics['completed_processes']}")
        print(f"Total Context Switches: {metrics['total_context_switches']}")
        print()
        print("Average Times:")
        print(f"  Waiting Time: {metrics['average_waiting_time']}s")
        print(f"  Turnaround Time: {metrics['average_turnaround_time']}s")
        print(f"  Response Time: {metrics['average_response_time']}s")

        if metrics['process_details']:
            print()
            print("Individual Process Metrics:")
            print("PID | Name        | Waiting | Turnaround | Response")
            print("----|-------------|---------|------------|----------")
            for proc in metrics['process_details']:
                name = proc['name'][:11]  # Truncate long names
                print(f"{proc['pid']:3} | {name:<11} | {proc['waiting_time']:7.2f} | {proc['turnaround_time']:10.2f} | {proc['response_time']:8.2f}")

        print()
        print("Use 'scheduler clear' to reset metrics and start fresh")


# =====================================================
# COMMAND HANDLER
# =====================================================

class CommandHandler:
    """Handles built-in shell commands"""

    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        self.process_scheduler = ProcessScheduler()
        self.scheduler_commands = SchedulerCommands(self.process_scheduler)

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
            'fg': self.handle_fg,
            'bg': self.handle_bg,
            'stop': self.handle_stop,
            'help': self.handle_help,
            'scheduler': self.scheduler_commands.handle_scheduler
        }

        handler = command_map.get(parsed.command)
        if handler:
            handler(parsed.args)
        else:
            raise ValueError(f"Unknown built-in command: {parsed.command}")

    def handle_cd(self, args: List[str]) -> None:
        """Change directory command"""
        if len(args) < 2:
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

            if target_dir == "-":
                raise ValueError("cd: previous directory functionality not implemented yet")
            elif target_dir == "~":
                target_dir = str(Path.home())
            elif target_dir.startswith("~/"):
                target_dir = str(Path.home() / target_dir[2:])

            target_path = Path(target_dir)
            if not target_path.exists():
                raise ValueError(f"cd: {target_dir}: no such file or directory")
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
        if hasattr(self, 'process_scheduler'):
            self.process_scheduler.stop_scheduler()

        jobs = self.job_manager.get_all_jobs()
        for job in jobs:
            if job.status.value != "Done":
                self.job_manager.kill_job(job.id)
        print("Goodbye!")
        sys.exit(0)

    def handle_echo(self, args: List[str]) -> None:
        """Echo command"""
        if len(args) > 1:
            output = " ".join(args[1:])
            output = output.replace("\\n", "\n")
            output = output.replace("\\t", "\t")
            print(output)

    def handle_clear(self, args: List[str]) -> None:
        """Clear screen command"""
        try:
            if os.name == 'nt':
                subprocess.run(['cls'], shell=True, check=True)
            else:
                subprocess.run(['clear'], check=True)
        except subprocess.CalledProcessError:
            print('\n' * 50)

    def handle_ls(self, args: List[str]) -> None:
        """List files command"""
        target_dir = "."
        show_hidden = False
        long_format = False
        invalid_flags = []

        for i in range(1, len(args)):
            arg = args[i]
            if arg.startswith("-"):
                for flag in arg[1:]:
                    if flag == 'a':
                        show_hidden = True
                    elif flag == 'l':
                        long_format = True
                    else:
                        invalid_flags.append(flag)
            else:
                if target_dir != ".":
                    raise ValueError("ls: too many directory arguments")
                target_dir = arg

        if invalid_flags:
            raise ValueError(f"ls: invalid option(s): {', '.join(invalid_flags)}")

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

        entries.sort(key=lambda x: x.name.lower())

        for entry in entries:
            if not show_hidden and entry.name.startswith("."):
                continue

            if long_format:
                try:
                    entry_stat = entry.stat()
                    mode = stat.filemode(entry_stat.st_mode)
                    size = entry_stat.st_size
                    mod_time = time.strftime("%b %d %H:%M", time.localtime(entry_stat.st_mtime))
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
            raise ValueError("cat: missing filename\nUsage: cat [file1] [file2] ...")

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
                raise ValueError(f"rmdir: {dirname}: no such file or directory")
            except OSError as e:
                if e.errno == 39:
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
                    raise ValueError(f"rm: {filename}: no such file or directory")
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
            raise ValueError("kill: missing PID\nUsage: kill [pid1] [pid2] ...")

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

            if pid == 1:
                errors.append("cannot kill init process (PID 1)")
                continue

            if pid == os.getpid():
                errors.append("cannot kill shell process itself")
                continue

            try:
                os.kill(pid, 9)
                print(f"Process {pid} killed")
                killed += 1
            except ProcessLookupError:
                errors.append(f"process {pid} not found")
            except PermissionError:
                errors.append(f"permission denied to kill process {pid}")
            except Exception as e:
                errors.append(f"failed to kill process {pid}: {e}")

        if errors:
            if killed == 0:
                raise ValueError(f"kill: {'; '.join(errors)}")
            else:
                print(f"kill: warnings: {'; '.join(errors)}")

    def handle_jobs(self, args: List[str]) -> None:
        """Jobs command"""
        self.job_manager.cleanup_completed_jobs()
        self.job_manager.list_jobs()

    def handle_fg(self, args: List[str]) -> None:
        """Foreground command"""
        if len(args) < 2:
            raise ValueError("fg: missing job ID\nUsage: fg [job_id]\nUse 'jobs' to see available jobs")

        if len(args) > 2:
            raise ValueError("fg: too many arguments")

        job_id_str = args[1]
        if not job_id_str:
            raise ValueError("fg: empty job ID")

        try:
            job_id = int(job_id_str)
        except ValueError:
            raise ValueError(f"fg: invalid job ID '{job_id_str}': not a number")

        if job_id <= 0:
            raise ValueError(f"fg: invalid job ID {job_id}: must be positive")

        all_jobs = self.job_manager.get_all_jobs()
        if not all_jobs:
            raise ValueError("fg: no jobs to bring to foreground")

        if not self.job_manager.bring_to_foreground(job_id):
            raise ValueError(f"fg: failed to bring job {job_id} to foreground")

    def handle_bg(self, args: List[str]) -> None:
        """Background command"""
        if len(args) < 2:
            raise ValueError("bg: missing job ID\nUsage: bg [job_id]\nUse 'jobs' to see available jobs")

        if len(args) > 2:
            raise ValueError("bg: too many arguments")

        job_id_str = args[1]
        if not job_id_str:
            raise ValueError("bg: empty job ID")

        try:
            job_id = int(job_id_str)
        except ValueError:
            raise ValueError(f"bg: invalid job ID '{job_id_str}': not a number")

        if job_id <= 0:
            raise ValueError(f"bg: invalid job ID {job_id}: must be positive")

        all_jobs = self.job_manager.get_all_jobs()
        if not all_jobs:
            raise ValueError("bg: no jobs to resume in background")

        if not self.job_manager.resume_in_background(job_id):
            raise ValueError(f"bg: failed to resume job {job_id} in background")

    def handle_stop(self, args: List[str]) -> None:
        """Stop a job for testing bg command"""
        if len(args) < 2:
            raise ValueError("stop: missing job ID\nUsage: stop [job_id]")

        try:
            job_id = int(args[1])
        except ValueError:
            raise ValueError(f"stop: invalid job ID '{args[1]}': not a number")

        job = self.job_manager.get_job(job_id)
        if not job:
            raise ValueError(f"stop: job {job_id} not found")

        if not job.is_alive():
            raise ValueError(f"stop: job {job_id} has already completed")

        try:
            os.kill(job.pid, signal.SIGSTOP)
            job.status = JobStatus.STOPPED
            print(f"Job [{job_id}] stopped")
        except ProcessLookupError:
            raise ValueError(f"stop: process {job.pid} no longer exists")
        except Exception as e:
            raise ValueError(f"stop: failed to stop job {job_id}: {e}")

    def handle_help(self, args: List[str]) -> None:
        """Help command"""
        print("Advanced Shell - Available Commands:")
        print()
        print("Built-in Commands:")
        print("  cd [directory]     - Change directory (supports ~, relative paths)")
        print("  pwd               - Print working directory")
        print("  echo [text]       - Print text (supports \\n, \\t escape sequences)")
        print("  clear             - Clear screen")
        print("  ls [options] [dir] - List files (-a for hidden, -l for long format)")
        print("  cat [files...]    - Display file contents")
        print("  mkdir [options] [dirs...] - Create directories (-p for parents)")
        print("  rmdir [dirs...]   - Remove empty directories")
        print("  rm [options] [files...] - Remove files (-r recursive, -f force)")
        print("  touch [files...]  - Create empty files or update timestamps")
        print("  kill [pids...]    - Kill processes by PID")
        print("  stop [job_id]     - Stop a running job (for testing bg command)")
        print("  exit              - Exit shell")
        print("  help              - Show this help")
        print()
        print("Job Control:")
        print("  jobs              - List background jobs")
        print("  fg [job_id]       - Bring job to foreground")
        print("  bg [job_id]       - Resume job in background")
        print()
        print("Process Scheduling (Deliverable 2):")
        print("  scheduler config rr [quantum]     - Configure Round-Robin scheduling")
        print("  scheduler config priority         - Configure Priority-Based scheduling")
        print("  scheduler addprocess <name> <duration> [priority] - Add process to scheduler")
        print("  scheduler start                   - Start the scheduler")
        print("  scheduler stop                    - Stop the scheduler")
        print("  scheduler status                  - Show scheduler status")
        print("  scheduler metrics                 - Show performance metrics")
        print("  scheduler clear                   - Clear scheduler state and metrics")
        print("  scheduler test                    - Run automated preemption test")
        print()
        print("Usage:")
        print("  command &         - Run command in background")
        print("  Ctrl+C            - Interrupt current foreground process")
        print("  Arrow Keys        - Navigate cursor left/right")
        print("  Ctrl+C (in input) - Clear current command line")
        print("  Backspace         - Delete character to the left")
        print()
        print("Process Scheduling Examples:")
        print("  # Round-Robin Scheduling")
        print("  scheduler config rr 2.5")
        print("  scheduler addprocess task1 8")
        print("  scheduler addprocess task2 5")
        print("  scheduler addprocess task3 3")
        print("  scheduler start")
        print()
        print("  # Priority-Based Scheduling (higher number = higher priority)")
        print("  scheduler config priority")
        print("  scheduler addprocess low_task 10 1      # Low priority")
        print("  scheduler addprocess med_task 6 5       # Medium priority")
        print("  scheduler addprocess high_task 4 10     # High priority")
        print("  scheduler start                         # High priority runs first")
        print()
        print("  # Test preemption with priorities")
        print("  scheduler config priority")
        print("  scheduler addprocess long_low 10 2      # Long, low priority")
        print("  scheduler addprocess short_high 2 8     # Short, high priority")
        print("  scheduler start                         # Watch preemption")
        print()
        print("  # Same priority (FCFS order)")
        print("  scheduler config priority")
        print("  scheduler addprocess first 3 5")
        print("  scheduler addprocess second 3 5")
        print("  scheduler addprocess third 3 5")
        print("  scheduler start                         # Runs in FCFS order")
        print()
        print("Scheduler Workflow:")
        print("  1. Configure:     scheduler config rr 2")
        print("  2. Add processes: scheduler addprocess task1 5")
        print("  3. Start:         scheduler start")
        print("  4. Monitor:       scheduler status")
        print("  5. View metrics:  scheduler metrics")
        print("  6. Clear state:   scheduler clear")


# =====================================================
# MAIN SHELL CLASS
# =====================================================

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

        signal.signal(signal.SIGINT, sigint_handler)

    def run(self):
        """Start the main shell loop"""
        self.print_welcome()

        while self.running:
            try:
                self.display_prompt()

                try:
                    if self.input_handler:
                        user_input = self.input_handler.get_input(self.prompt).strip()
                    else:
                        user_input = input().strip()
                except EOFError:
                    print("\nGoodbye!")
                    break
                except KeyboardInterrupt:
                    print()
                    continue

                if not user_input:
                    continue

                try:
                    self.process_input(user_input)
                except Exception as e:
                    print(f"\033[31mError:\033[0m {e}")

                self.job_manager.cleanup_completed_jobs()

            except Exception as e:
                print(f"\033[31mShell Error:\033[0m {e}")

        self.shutdown()

    def process_input(self, input_str: str) -> None:
        """Process a single line of input"""
        try:
            parsed = self.parser.parse(input_str)
        except ValueError as e:
            raise ValueError(f"Parse error: {e}")

        if not parsed:
            return

        try:
            self.parser.validate_command(parsed)
        except ValueError as e:
            raise ValueError(f"Validation error: {e}")

        if self.parser.is_builtin_command(parsed.command):
            try:
                self.command_handler.handle_command(parsed)
            except ValueError as e:
                raise ValueError(str(e))
            except Exception as e:
                raise ValueError(f"{parsed.command}: {e}")
        else:
            self.execute_external_command(parsed)

    def execute_external_command(self, parsed: ParsedCommand) -> None:
        """Execute external command"""
        if not shutil.which(parsed.command):
            raise ValueError(f"{parsed.command}: command not found")

        try:
            if parsed.background:
                process = subprocess.Popen(
                    parsed.args,
                    cwd=os.getcwd(),
                    preexec_fn=os.setsid
                )

                job = self.job_manager.add_job(
                    command=' '.join(parsed.args),
                    args=parsed.args,
                    process=process,
                    background=True
                )

                print(f"[{job.id}] {process.pid}")

            else:
                process = subprocess.Popen(
                    parsed.args,
                    cwd=os.getcwd()
                )

                try:
                    process.wait()
                except KeyboardInterrupt:
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

    def display_prompt(self) -> None:
        """Show the shell prompt"""
        try:
            pwd = os.getcwd()
            dir_name = os.path.basename(pwd)
            if dir_name == "":
                dir_name = pwd
        except Exception:
            dir_name = "unknown"

        current_time = time.strftime("%H:%M:%S")
        self.prompt = f"[shell:{dir_name} {current_time}]$ "

    def print_welcome(self) -> None:
        """Print the welcome message"""
        print("==========================================")
        print("  Advanced Shell Simulation - Deliverable 2")
        print("==========================================")
        print()
        print("Features implemented:")
        print("✓ Built-in commands (cd, pwd, ls, cat, etc.)")
        print("✓ External command execution")
        print("✓ Process management (foreground/background)")
        print("✓ Job control (jobs, fg, bg, stop)")
        print("✓ Keyboard navigation (arrow keys, Ctrl+C to clear)")
        print("✓ Signal handling")
        print("✓ Error handling")
        print("✓ Process scheduling algorithms (NEW)")
        print("  • Round-Robin Scheduling")
        print("  • Priority-Based Scheduling")
        print("✓ Performance metrics and monitoring (NEW)")
        print()
        print("Type 'help' for available commands")
        print("Type 'exit' to quit")
        print()
        print("Quick Start - Process Scheduling:")
        print("  scheduler config rr 2         # Configure Round-Robin")
        print("  scheduler addprocess task1 5  # Add 5-second process")
        print("  scheduler start               # Start scheduling")
        print("  scheduler status              # Monitor execution")
        print()

    def shutdown(self) -> None:
        """Perform cleanup before exiting"""
        print("\nShutting down shell...")

        if hasattr(self.command_handler, 'process_scheduler') and self.command_handler.process_scheduler:
            print("Stopping process scheduler...")
            self.command_handler.process_scheduler.stop_scheduler()

        jobs = self.job_manager.get_all_jobs()
        if jobs:
            print(f"Terminating {len(jobs)} active job(s)...")

            for job in jobs:
                if job.status.value != "Done":
                    print(f"Killing job [{job.id}]: {job.command}")
                    self.job_manager.kill_job(job.id)

            time.sleep(0.1)

        print("Goodbye!")


# =====================================================
# MAIN ENTRY POINT
# =====================================================

def show_version():
    """Show version information"""
    print("Advanced Shell Simulation")
    print("Version: 2.0.0 (Deliverable 2)")
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


def show_help():
    """Show help information"""
    print("Advanced Shell Simulation - Deliverable 2")
    print()
    print("Usage:")
    print("  python3 deliverable_2_shell.py [options]")
    print()
    print("Options:")
    print("  --version    Show version information")
    print("  --help       Show this help message")
    print("  --debug      Enable debug mode")
    print()
    print("Once started, type 'help' for available shell commands")
    print()
    print("Quick Start - Process Scheduling:")
    print("  1. Configure algorithm:    scheduler config rr 2")
    print("  2. Add processes:          scheduler addprocess task1 5")
    print("  3. Start scheduler:        scheduler start")
    print("  4. Monitor status:         scheduler status")
    print("  5. View metrics:           scheduler metrics")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced Shell Simulation - Deliverable 2",
        add_help=False
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

    args = parser.parse_args()

    if args.version:
        show_version()
        return

    if args.help:
        show_help()
        return

    try:
        shell = Shell()

        if args.debug:
            print("Debug mode enabled")
            print("Scheduler modules available:", hasattr(shell.command_handler, 'process_scheduler'))

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
    main()#!/usr/bin/env python3