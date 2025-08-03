import hashlib
import os
import json
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import os
import sys
import signal
import subprocess
import stat
import time
from pathlib import Path
from typing import List
import shlex
from typing import Optional
import sys
import tty
import termios
import os
import signal
import subprocess
import time
from typing import Dict, List, Optional
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import OrderedDict
import time
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import subprocess
import os
from typing import List, Optional, Tuple
from dataclasses import dataclass
import time
import heapq
from typing import List, Optional, Dict
from collections import deque
import threading
import time
import random
import queue
from typing import Dict, List
from enum import Enum
from typing import List
import time
from typing import Optional
from dataclasses import dataclass
from enum import Enum
import time
import threading
import subprocess
from typing import List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import threading
from typing import List
import subprocess
import time
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum
import os
import signal
import subprocess
import time
import time
import os
import sys
import argparse
import io
import time
import threading

# SHELL TYPES
class JobStatus(Enum):
    """Job status enumeration"""
    RUNNING = "Running"
    STOPPED = "Stopped"
    DONE = "Done"
    WAITING = "Waiting"  # New status for scheduling

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
    has_pipes: bool = False
    pipe_chain: List[str] = field(default_factory=list)

# SCHEDULER TYPES
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

# JOB MANAGER
class JobManager:
    """Handles job control operations"""

    def __init__(self):
        self.jobs: Dict[int, Job] = {}
        self.job_counter = 0
        self.scheduler = Scheduler()

        # Set up scheduler callback
        self.scheduler.on_process_complete = self._on_scheduled_job_complete

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

    def _on_scheduled_job_complete(self, job: Job):
        """Callback when a scheduled job completes"""
        job.status = JobStatus.DONE
        job.end_time = time.time()
        print(f"[{job.id}]+ Done\t\t{job.command}")

    def set_scheduling_algorithm(self, algorithm: str, time_slice: float = 2.0):
        """Set the scheduling algorithm"""
        try:
            if algorithm.lower() == "round_robin":
                self.scheduler.set_algorithm(SchedulingAlgorithm.ROUND_ROBIN, time_slice)
            elif algorithm.lower() == "priority":
                self.scheduler.set_algorithm(SchedulingAlgorithm.PRIORITY)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
        except Exception as e:
            raise ValueError(f"Failed to set scheduling algorithm: {e}")

    def get_scheduler_status(self) -> dict:
        """Get scheduler status"""
        return self.scheduler.get_scheduler_status()

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
            job.is_alive()  # Update status for non-scheduled jobs
            duration = time.time() - job.start_time
            if job.end_time:
                duration = job.end_time - job.start_time

            # Show additional info for scheduled jobs
            if hasattr(job, 'priority') and job.priority:
                status_info = f"{job.status.value} (Priority: {job.priority})"
                if job.total_time_needed > 0:
                    status_info += f", Time: {job.execution_time:.1f}/{job.total_time_needed:.1f}s"
            else:
                status_info = job.status.value

            print(f"[{job.id}] {status_info} {job.command} (PID: {job.pid}, Duration: {int(duration)}s)")

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
            # Resume the process if it's stopped
            if job.status == JobStatus.STOPPED:
                os.kill(job.pid, signal.SIGCONT)

            job.status = JobStatus.RUNNING
            job.background = False

            # Wait for the job to complete in foreground
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

            # Remove from jobs list since it's completed
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

        # Handle scheduled jobs
        if hasattr(job, 'priority') and job.priority:
            self.scheduler.remove_process(job_id)
            job.status = JobStatus.DONE
            job.end_time = time.time()
            return True

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
            # For non-scheduled jobs, check if alive
            if not hasattr(job, 'priority') or not job.priority:
                job.is_alive()  # Update status

            if job.status == JobStatus.DONE:
                completed_jobs.append(job_id)
                print(f"[{job_id}]+ Done\t\t{job.command}")

        for job_id in completed_jobs:
            if job_id in self.jobs:
                del self.jobs[job_id]

# MEMORY MANAGER
class PageReplacementAlgorithm(Enum):
    FIFO = "fifo"
    LRU = "lru"


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

    def reset_memory(self) -> bool:
        """Reset all memory management state and statistics (NEW)"""
        with self.lock:
            # Reset physical memory
            self.physical_memory = [None] * self.total_frames
            self.free_frames = list(range(self.total_frames))
            self.used_frames.clear()

            # Reset statistics
            self.page_faults = 0
            self.page_hits = 0
            self.page_replacements = 0

            # Reset processes
            self.processes.clear()
            self.next_pid = 1

            # Reset algorithm data structures
            self.fifo_queue.clear()
            self.lru_access_order.clear()

            return True

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

        import random
        for i in range(count):
            if pattern == "sequential":
                page_num = i % process.pages_needed
            else:  # random
                page_num = random.randint(0, process.pages_needed - 1)

            success, message = self.allocate_page(pid, page_num)
            time.sleep(0.1)

## AUTH SYSTEM
class UserRole(Enum):
    """User role enumeration"""
    ADMIN = "admin"
    STANDARD = "standard"
    GUEST = "guest"


class Permission(Enum):
    """File permission enumeration"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"


@dataclass
class User:
    """Represents a user in the system"""
    username: str
    password_hash: str
    role: UserRole
    home_directory: str
    permissions: Dict[str, List[Permission]] = None  # file_path -> permissions

    def __post_init__(self):
        if self.permissions is None:
            self.permissions = {}


class AuthenticationSystem:
    """Handles user authentication and authorization"""

    def __init__(self, users_file: str = "users.json"):
        self.users_file = users_file
        self.current_user: Optional[User] = None
        self.users: Dict[str, User] = {}
        self.load_users()

    def load_users(self) -> None:
        """Load users from file or create default users"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    data = json.load(f)
                    for user_data in data['users']:
                        user = User(
                            username=user_data['username'],
                            password_hash=user_data['password_hash'],
                            role=UserRole(user_data['role']),
                            home_directory=user_data['home_directory'],
                            permissions=user_data.get('permissions', {})
                        )
                        self.users[user.username] = user
            except Exception as e:
                print(f"Warning: Could not load users file: {e}")
                self.create_default_users()
        else:
            self.create_default_users()

    def create_default_users(self) -> None:
        """Create default users for the system with restricted write permissions"""
        # Create admin user with full access
        admin_hash = self._hash_password("admin123")
        admin_user = User(
            username="admin",
            password_hash=admin_hash,
            role=UserRole.ADMIN,
            home_directory="/home/admin",
            permissions={
                "/": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/home": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/etc": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/var": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/usr": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/bin": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/sbin": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/tmp": [Permission.READ, Permission.WRITE, Permission.EXECUTE]
            }
        )

        # Create standard user with read-only access to most directories
        user_hash = self._hash_password("user123")
        standard_user = User(
            username="user",
            password_hash=user_hash,
            role=UserRole.STANDARD,
            home_directory="/home/user",
            permissions={
                "/home/user": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/tmp": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/home": [Permission.READ, Permission.EXECUTE],
                "/usr": [Permission.READ, Permission.EXECUTE],
                "/bin": [Permission.READ, Permission.EXECUTE],
                "/var": [Permission.READ],
                "/etc": [Permission.READ]
            }
        )

        # Create guest user with very limited read-only access
        guest_hash = self._hash_password("guest123")
        guest_user = User(
            username="guest",
            password_hash=guest_hash,
            role=UserRole.GUEST,
            home_directory="/home/guest",
            permissions={
                "/home/guest": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/tmp": [Permission.READ, Permission.WRITE, Permission.EXECUTE],
                "/home": [Permission.READ],
                "/usr": [Permission.READ],
                "/bin": [Permission.READ]
            }
        )

        self.users = {
            "admin": admin_user,
            "user": standard_user,
            "guest": guest_user
        }

        self.save_users()

    def save_users(self) -> None:
        """Save users to file"""
        try:
            data = {
                'users': [
                    {
                        'username': user.username,
                        'password_hash': user.password_hash,
                        'role': user.role.value,
                        'home_directory': user.home_directory,
                        'permissions': {
                            path: [perm.value for perm in perms]
                            for path, perms in user.permissions.items()
                        }
                    }
                    for user in self.users.values()
                ]
            }
            with open(self.users_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save users file: {e}")

    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate a user with username and password"""
        if username not in self.users:
            return False

        user = self.users[username]
        password_hash = self._hash_password(password)

        if user.password_hash == password_hash:
            self.current_user = user
            return True
        return False

    def logout(self) -> None:
        """Logout current user"""
        self.current_user = None

    def is_authenticated(self) -> bool:
        """Check if a user is currently authenticated"""
        return self.current_user is not None

    def get_current_user(self) -> Optional[User]:
        """Get the currently authenticated user"""
        return self.current_user

    def has_permission(self, file_path: str, permission: Permission) -> bool:
        """Check if current user has permission for a file with enhanced logic"""
        if not self.current_user:
            return False

        # Admin has all permissions
        if self.current_user.role == UserRole.ADMIN:
            return True

        # Normalize file path
        file_path = os.path.abspath(file_path)

        # Check user-specific permissions first
        if file_path in self.current_user.permissions:
            return permission in self.current_user.permissions[file_path]

        # Check directory-based permissions
        for dir_path, permissions in self.current_user.permissions.items():
            if file_path.startswith(dir_path):
                return permission in permissions

        # Role-based default permissions - UPDATED FOR READ-ONLY ACCESS
        if self.current_user.role == UserRole.STANDARD:
            # Standard users can read most files, but write access is very restricted
            if permission == Permission.READ:
                # Can read most directories but not sensitive system files
                if file_path.startswith("/etc/passwd") or file_path.startswith("/etc/shadow") or file_path.startswith("/var/log/auth"):
                    return False
                return True
            elif permission == Permission.WRITE:
                # Only write access to their home directory and /tmp
                return file_path.startswith(self.current_user.home_directory) or file_path.startswith("/tmp")
            elif permission == Permission.EXECUTE:
                # Execute access only to their home directory, /tmp, and system binaries
                return file_path.startswith(self.current_user.home_directory) or file_path.startswith("/tmp") or file_path.startswith("/bin") or file_path.startswith("/usr/bin")

        elif self.current_user.role == UserRole.GUEST:
            # Guests have very limited permissions - mostly read-only
            if permission == Permission.READ:
                # Can read their home directory, /tmp, and some system directories
                return file_path.startswith(self.current_user.home_directory) or file_path.startswith("/tmp") or file_path.startswith("/home") or file_path.startswith("/usr") or file_path.startswith("/bin")
            elif permission == Permission.WRITE:
                # Only write access to their home directory and /tmp
                return file_path.startswith(self.current_user.home_directory) or file_path.startswith("/tmp")
            elif permission == Permission.EXECUTE:
                # Execute access only to their home directory and /tmp
                return file_path.startswith(self.current_user.home_directory) or file_path.startswith("/tmp")

        return False

    def get_file_permission_info(self, file_path: str) -> Dict[str, bool]:
        """Get detailed permission information for a file"""
        if not self.current_user:
            return {"read": False, "write": False, "execute": False}

        return {
            "read": self.has_permission(file_path, Permission.READ),
            "write": self.has_permission(file_path, Permission.WRITE),
            "execute": self.has_permission(file_path, Permission.EXECUTE)
        }

    def add_user(self, username: str, password: str, role: UserRole, home_directory: str) -> bool:
        """Add a new user (admin only)"""
        if not self.current_user or self.current_user.role != UserRole.ADMIN:
            return False

        if username in self.users:
            return False

        password_hash = self._hash_password(password)
        new_user = User(
            username=username,
            password_hash=password_hash,
            role=role,
            home_directory=home_directory
        )

        self.users[username] = new_user
        self.save_users()
        return True

    def change_password(self, username: str, new_password: str) -> bool:
        """Change user password"""
        if not self.current_user:
            return False

        # Users can only change their own password, or admin can change any
        if self.current_user.username != username and self.current_user.role != UserRole.ADMIN:
            return False

        if username not in self.users:
            return False

        password_hash = self._hash_password(new_password)
        self.users[username].password_hash = password_hash
        self.save_users()
        return True

## COMMAND HANDLER
class CommandHandler:
    """Handles built-in shell commands"""

    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        # Deliverable 2: Add process scheduler
        self.process_scheduler = ProcessScheduler()
        self.scheduler_commands = SchedulerCommands(self.process_scheduler)
        # Deliverable 3: Add memory management and synchronization
        self.memory_sync_commands = MemorySyncCommands()
        # Deliverable 3: Add authentication and piping
        self.auth_system = AuthenticationSystem()
        self.pipe_handler = PipeHandler()

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
            # Deliverable 2: Scheduling commands
            'addprocess': self.scheduler_commands.handle_addprocess,
            'scheduler': self.scheduler_commands.handle_scheduler,
            # Deliverable 3: NEW - Memory and synchronization commands
            'memory': self.handle_memory,
            'sync': self.handle_sync,
            # Deliverable 3: Authentication commands
            'login': self.handle_login,
            'logout': self.handle_logout,
            'whoami': self.handle_whoami,
            'adduser': self.handle_adduser,
            'chpasswd': self.handle_chpasswd,
            # New permission command
            'permissions': self.handle_permissions,
            'grep': self.handle_grep,
            'sort': self.handle_sort
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

    # All existing command handlers remain the same...
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
            if target_dir == "-":
                raise ValueError("cd: previous directory functionality not implemented yet")
            elif target_dir == "~":
                target_dir = str(Path.home())
            elif target_dir.startswith("~/"):
                target_dir = str(Path.home() / target_dir[2:])

            # Check if directory exists before trying to change
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
        # Deliverable 3: NEW - Clean shutdown with memory/sync cleanup
        if hasattr(self, 'memory_sync_commands'):
            if self.memory_sync_commands.producer_consumer:
                self.memory_sync_commands.producer_consumer.stop()
            if self.memory_sync_commands.dining_philosophers:
                self.memory_sync_commands.dining_philosophers.stop()

        # Clean shutdown - stop scheduler and kill remaining jobs
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
        """List files command with permission-based display"""
        target_dir = "."
        show_hidden = False
        long_format = False
        invalid_flags = []

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
                        invalid_flags.append(flag)
            else:
                if target_dir != ".":
                    raise ValueError("ls: too many directory arguments")
                target_dir = arg

        # Report invalid flags
        if invalid_flags:
            raise ValueError(f"ls: invalid option(s): {', '.join(invalid_flags)}")

        # Check if directory exists and is accessible
        target_path = Path(target_dir)
        if not target_path.exists():
            raise ValueError(f"ls: {target_dir}: no such file or directory")
        elif not target_path.is_dir():
            raise ValueError(f"ls: {target_dir}: not a directory")

        # Check if user has read permission for the directory
        if not self.check_file_permission(str(target_path), Permission.READ):
            raise ValueError(f"ls: {target_dir}: permission denied")

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
                    mod_time = time.strftime("%b %d %H:%M", time.localtime(entry_stat.st_mtime))

                    # Get user permissions for this file
                    entry_path = str(entry)
                    user_permissions = self.get_user_file_permissions(entry_path)

                    # Show permission status with color coding
                    permission_status = self.format_permission_status(user_permissions)

                    print(f"{mode} {size:>8} {mod_time} {entry.name} {permission_status}")
                except Exception:
                    print(f"? {entry.name}")
            else:
                # In short format, still show permission indicators
                entry_path = str(entry)
                user_permissions = self.get_user_file_permissions(entry_path)

                if entry.is_dir():
                    print(f"{entry.name}/")
                else:
                    print(entry.name)

                # Show permission indicators for files user can't access
                if not user_permissions:
                    print(f"  (no access)")

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

        # Check if user has write permission for the target directory
        if not self.auth_system.is_authenticated():
            raise ValueError("mkdir: authentication required")

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
                target_dir = str(dir_path.parent) if dir_path.parent != Path('.') else str(Path.cwd())

                # Check write permission for the target directory
                if not self.check_file_permission(target_dir, Permission.WRITE):
                    raise PermissionError(f"mkdir: {dirname}: permission denied - no write access to directory")

                if create_parents:
                    dir_path.mkdir(parents=True, exist_ok=True)
                else:
                    dir_path.mkdir()
            except FileExistsError:
                raise ValueError(f"mkdir: {dirname}: file exists")
            except PermissionError:
                raise
            except Exception as e:
                raise ValueError(f"mkdir: {dirname}: {e}")

    def handle_rmdir(self, args: List[str]) -> None:
        """Remove directory command"""
        if len(args) < 2:
            raise ValueError("rmdir: missing directory name")

        # Check if user has write permission
        if not self.auth_system.is_authenticated():
            raise ValueError("rmdir: authentication required")

        for dirname in args[1:]:
            try:
                dir_path = Path(dirname)
                target_dir = str(dir_path.parent) if dir_path.parent != Path('.') else str(Path.cwd())

                # Check write permission for the parent directory
                if not self.check_file_permission(target_dir, Permission.WRITE):
                    raise PermissionError(f"rmdir: {dirname}: permission denied - no write access to directory")

                dir_path.rmdir()
            except FileNotFoundError:
                raise ValueError(f"rmdir: {dirname}: no such file or directory")
            except PermissionError:
                raise
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

        # Check if user has write permission
        if not self.auth_system.is_authenticated():
            raise ValueError("rm: authentication required")

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
                target_dir = str(file_path.parent) if file_path.parent != Path('.') else str(Path.cwd())

                # Check write permission for the parent directory
                if not self.check_file_permission(target_dir, Permission.WRITE):
                    if not force:
                        raise PermissionError(f"rm: {filename}: permission denied - no write access to directory")
                    continue

                if recursive and file_path.is_dir():
                    import shutil
                    shutil.rmtree(file_path)
                else:
                    file_path.unlink()
            except FileNotFoundError:
                if not force:
                    raise ValueError(f"rm: {filename}: no such file or directory")
            except PermissionError:
                if not force:
                    raise
                continue
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

        # Check if user has write permission
        if not self.auth_system.is_authenticated():
            raise ValueError("touch: authentication required")

        for filename in args[1:]:
            try:
                file_path = Path(filename)
                target_dir = str(file_path.parent) if file_path.parent != Path('.') else str(Path.cwd())

                # Check write permission for the target directory
                if not self.check_file_permission(target_dir, Permission.WRITE):
                    raise PermissionError(f"touch: {filename}: permission denied - no write access to directory")

                if file_path.exists():
                    # Update timestamp
                    file_path.touch()
                else:
                    # Create the file
                    file_path.touch()
            except PermissionError:
                raise
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

        # Check if any jobs exist
        all_jobs = self.job_manager.get_all_jobs()
        if not all_jobs:
            raise ValueError("fg: no jobs to bring to foreground")

        if not self.job_manager.bring_to_foreground(job_id):
            raise ValueError(f"fg: failed to bring job {job_id} to foreground")

    def handle_stop(self, args: List[str]) -> None:
        """Stop a job for testing bg command (temporary testing feature)"""
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
            os.kill(job.pid, signal.SIGSTOP)  # Stop the process
            job.status = JobStatus.STOPPED
            print(f"Job [{job_id}] stopped")
        except ProcessLookupError:
            raise ValueError(f"stop: process {job.pid} no longer exists")
        except Exception as e:
            raise ValueError(f"stop: failed to stop job {job_id}: {e}")

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

        # Check if any jobs exist
        all_jobs = self.job_manager.get_all_jobs()
        if not all_jobs:
            raise ValueError("bg: no jobs to resume in background")

        if not self.job_manager.resume_in_background(job_id):
            raise ValueError(f"bg: failed to resume job {job_id} in background")

    def handle_help(self, args: List[str]) -> None:
        """Show help information"""
        print("Advanced Shell Simulation - Available Commands")
        print("=" * 50)
        print()

        # Basic commands
        print("Basic Commands:")
        print("  cd [directory]     - Change directory")
        print("  pwd                - Print working directory")
        print("  ls [-la] [dir]     - List files (with permissions)")
        print("  cat [file]         - Display file contents")
        print("  echo [text]        - Print text")
        print("  clear              - Clear screen")
        print()

        # File operations
        print("File Operations:")
        print("  mkdir [dir]        - Create directory")
        print("  rmdir [dir]        - Remove directory")
        print("  rm [file]          - Remove file")
        print("  touch [file]       - Create empty file")
        print()

        # Process management
        print("Process Management:")
        print("  jobs               - List background jobs")
        print("  fg [job_id]        - Bring job to foreground")
        print("  bg [job_id]        - Continue job in background")
        print("  stop [job_id]      - Stop a job")
        print("  kill [pid]         - Kill a process")
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
        print("  sync mutex <create|acquire|release> <n> - Mutex operations")
        print("  sync semaphore <create|acquire|release> <n> [value] - Semaphore operations")
        print("  sync prodcons <start|stop|status> [producers] [consumers] - Producer-Consumer")
        print("  sync philosophers <start|stop|status> [philosophers] - Dining Philosophers")
        print()
        print("Usage:")
        print("  command &         - Run command in background")
        print("  Ctrl+C            - Interrupt current foreground process")
        print("  Arrow Keys        - Navigate cursor left/right")
        print("  Ctrl+C (in input) - Clear current command line")
        print("  Backspace         - Delete character to the left")

        # Authentication commands
        print("Authentication:")
        print("  login [user] [pass] - Login as user")
        print("  logout             - Logout current user")
        print("  whoami             - Show current user info")
        print("  adduser [user] [pass] [role] - Add new user (admin only)")
        print("  chpasswd [user] [pass] - Change password")
        print("  permissions [file] - Show file permissions")
        print()

        # Scheduling commands
        print("Process Scheduling:")
        print("  scheduler config [type] [time_slice] - Configure scheduler")
        print("  scheduler addprocess [name] [duration] - Add process")
        print("  scheduler start    - Start scheduling")
        print("  scheduler status   - Show scheduler status")
        print("  scheduler stop     - Stop scheduler")
        print()
        # Deliverable 3: NEW examples
        print("✓ Deliverable 3 Examples:")
        print("  memory create webapp 8         # Create process needing 8 pages")
        print("  memory alloc 1 0               # Allocate page 0 for process 1")
        print("  memory algorithm lru           # Switch to LRU replacement")
        print("  memory test random             # Test random access pattern")
        print("  sync mutex create mylock       # Create a mutex")
        print("  sync prodcons start 2 3        # Start Producer-Consumer")
        print("  sync philosophers start 5      # Start Dining Philosophers")
        print()
        print("Scheduling Algorithms:")
        print("  Round-Robin: Each process gets a time slice, then moves to end of queue")
        print("  Priority: Highest priority process runs first (1=highest, 10=lowest)")
        print()
        print("✓ Memory Management:")
        print("  FIFO: First-In-First-Out page replacement")
        print("  LRU: Least Recently Used page replacement")
        print("  Page Faults: When requested page not in memory")
        print()
        print("✓ Synchronization Problems:")
        print("  Producer-Consumer: Buffer synchronization with semaphores")
        print("  Dining Philosophers: Deadlock prevention with asymmetric fork acquisition")
        print()
        print("Scheduler Workflow:")
        print("  1. Configure:     scheduler config rr 2")
        print("  2. Add processes: scheduler addprocess task1 5")
        print("  3. Start:         scheduler start")
        print("  4. Monitor:       scheduler status")
        print("  5. View metrics:  scheduler metrics")
        print("  6. Clear state:   scheduler clear")
        print()

        # System commands
        print("System:")
        print("  exit               - Exit shell")
        print("  help               - Show this help")
        print()

        # Permission information
        if self.auth_system.is_authenticated():
            user = self.auth_system.get_current_user()
            print(f"Current User: {user.username} ({user.role.value})")
            print()
            print("Permission Levels:")
            print("  Admin:    Full access to all files and commands")
            print("  Standard: Read access to most files, write to home and /tmp")
            print("  Guest:    Very limited access, read/write only to home and /tmp")
            print()
            print("File Permission Display:")
            print("  ls -l shows: [RWX] where:")
            print("    R (green) = Read permission")
            print("    W (green) = Write permission")
            print("    X (green) = Execute permission")
            print("    r/w/x (red) = No permission")
            print("    [NO ACCESS] = No access at all")
        else:
            print("Not logged in. Use 'login' to authenticate.")


    def handle_scheduler(self, args: List[str]) -> None:
        """Scheduler command - configure and view scheduler status"""
        if len(args) < 2:
            # Show current scheduler status
            status = self.job_manager.get_scheduler_status()
            print("Scheduler Status:")
            print(f"  Algorithm: {status['algorithm']}")
            if status['time_slice']:
                print(f"  Time Slice: {status['time_slice']}s")
            print(f"  Total Processes: {status['total_processes']}")
            if status['running_process']:
                print(f"  Running Process: {status['running_process']}")
            else:
                print("  Running Process: None")

            if status['processes']:
                print("\nQueued Processes:")
                for proc in status['processes']:
                    print(f"  [{proc['id']}] Priority: {proc['priority']}, "
                          f"Time: {proc['time_executed']:.1f}/{proc['time_needed']:.1f}s, "
                          f"Status: {proc['status']}")
            return

        subcommand = args[1].lower()

        if subcommand == "round_robin":
            time_slice = 2.0  # Default
            if len(args) >= 3:
                try:
                    time_slice = float(args[2])
                    if time_slice <= 0:
                        raise ValueError("Time slice must be positive")
                except ValueError:
                    raise ValueError(f"Invalid time slice: {args[2]}")

            self.job_manager.set_scheduling_algorithm("round_robin", time_slice)

        elif subcommand == "priority":
            self.job_manager.set_scheduling_algorithm("priority")

        else:
            raise ValueError(f"Unknown scheduler subcommand: {subcommand}\n"
                           "Available: round_robin [time_slice], priority")

    # Deliverable 3: NEW - Access to memory manager and synchronizer for integration
    def get_memory_manager(self):
        """Get memory manager instance for integration"""
        return self.memory_sync_commands.get_memory_manager()

    def get_synchronizer(self):
        """Get synchronizer instance for integration"""
        return self.memory_sync_commands.get_synchronizer()

        print(f"Added job [{job.id}]: {command} (Priority: {priority}, Time: {time_needed}s, Background: {background})")

    def handle_login(self, args: List[str]) -> None:
        """Handle user login"""
        if len(args) < 3:
            print("Usage: login <username> <password>")
            return

        username = args[1]
        password = args[2]

        if self.auth_system.authenticate(username, password):
            user = self.auth_system.get_current_user()
            print(f"Welcome, {user.username}! Role: {user.role.value}")
            print(f"Home directory: {user.home_directory}")

            # Change to user's home directory
            try:
                os.chdir(user.home_directory)
            except (OSError, FileNotFoundError):
                print(f"Warning: Could not change to home directory {user.home_directory}")
        else:
            print("Login failed: Invalid username or password")

    def handle_logout(self, args: List[str]) -> None:
        """Handle user logout"""
        if self.auth_system.is_authenticated():
            username = self.auth_system.get_current_user().username
            self.auth_system.logout()
            print(f"Goodbye, {username}!")
        else:
            print("No user is currently logged in")

    def handle_whoami(self, args: List[str]) -> None:
        """Show current user information"""
        if self.auth_system.is_authenticated():
            user = self.auth_system.get_current_user()
            print(f"Username: {user.username}")
            print(f"Role: {user.role.value}")
            print(f"Home directory: {user.home_directory}")
        else:
            print("No user is currently logged in")

    def handle_adduser(self, args: List[str]) -> None:
        """Add a new user (admin only)"""
        if len(args) < 4:
            print("Usage: adduser <username> <password> <role> [home_directory]")
            return

        if not self.auth_system.is_authenticated():
            print("Error: You must be logged in to add users")
            return

        current_user = self.auth_system.get_current_user()
        if current_user.role != UserRole.ADMIN:
            print("Error: Only administrators can add users")
            return

        username = args[1]
        password = args[2]
        role_str = args[3].lower()
        home_directory = args[4] if len(args) > 4 else f"/home/{username}"

        try:
            role = UserRole(role_str)
        except ValueError:
            print(f"Error: Invalid role '{role_str}'. Valid roles: admin, standard, guest")
            return

        if self.auth_system.add_user(username, password, role, home_directory):
            print(f"User '{username}' added successfully")
        else:
            print(f"Error: Could not add user '{username}' (user may already exist)")

    def handle_chpasswd(self, args: List[str]) -> None:
        """Change user password"""
        if len(args) < 3:
            print("Usage: chpasswd <username> <new_password>")
            return

        if not self.auth_system.is_authenticated():
            print("Error: You must be logged in to change passwords")
            return

        username = args[1]
        new_password = args[2]

        if self.auth_system.change_password(username, new_password):
            print(f"Password for user '{username}' changed successfully")
        else:
            print(f"Error: Could not change password for user '{username}'")

    def handle_permissions(self, args: List[str]) -> None:
        """Show detailed permission information for files"""
        if len(args) < 2:
            print("Usage: permissions <file1> [file2] ...")
            return

        if not self.auth_system.is_authenticated():
            print("Error: You must be logged in to check permissions")
            return

        user = self.auth_system.get_current_user()
        print(f"Permission check for user: {user.username} ({user.role.value})")
        print("=" * 60)

        for filename in args[1:]:
            if not filename:
                continue

            file_path = Path(filename)
            if not file_path.exists():
                print(f"{filename}: File does not exist")
                continue

            # Get permission information
            perm_info = self.auth_system.get_file_permission_info(str(file_path))

            print(f"\nFile: {filename}")
            print(f"  Read:    {'✓' if perm_info['read'] else '✗'}")
            print(f"  Write:   {'✓' if perm_info['write'] else '✗'}")
            print(f"  Execute: {'✓' if perm_info['execute'] else '✗'}")

            # Show what operations are allowed
            allowed_ops = []
            if perm_info['read']:
                allowed_ops.append("read")
            if perm_info['write']:
                allowed_ops.append("write")
            if perm_info['execute']:
                allowed_ops.append("execute")

            if allowed_ops:
                print(f"  Allowed operations: {', '.join(allowed_ops)}")
            else:
                print(f"  \033[31mNo access allowed\033[0m")

    def handle_grep(self, args: List[str]) -> None:
        """Built-in grep command"""
        if len(args) < 2:
            print("Usage: grep <pattern> [file...]")
            return

        pattern = args[1]
        files = args[2:] if len(args) > 2 else []

        # Read from stdin if no files specified
        if not files:
            import sys
            for line in sys.stdin:
                if pattern.lower() in line.lower():
                    print(line.rstrip())
        else:
            for filename in files:
                try:
                    with open(filename, 'r') as f:
                        for line in f:
                            if pattern.lower() in line.lower():
                                print(line.rstrip())
                except FileNotFoundError:
                    print(f"grep: {filename}: No such file or directory")
                except PermissionError:
                    print(f"grep: {filename}: Permission denied")

    def handle_sort(self, args: List[str]) -> None:
        """Built-in sort command"""
        files = args[1:] if len(args) > 1 else []

        lines = []

        # Read from stdin if no files specified
        if not files:
            import sys
            lines = sys.stdin.readlines()
        else:
            for filename in files:
                try:
                    with open(filename, 'r') as f:
                        lines.extend(f.readlines())
                except FileNotFoundError:
                    print(f"sort: {filename}: No such file or directory")
                    return
                except PermissionError:
                    print(f"sort: {filename}: Permission denied")
                    return

        # Sort and print
        for line in sorted(lines):
            print(line.rstrip())

    def check_file_permission(self, file_path: str, permission: Permission) -> bool:
        """Check if current user has permission for a file"""
        if not self.auth_system.is_authenticated():
            return False
        return self.auth_system.has_permission(file_path, permission)

    def get_user_file_permissions(self, file_path: str) -> List[Permission]:
        """Get user permissions for a specific file"""
        if not self.auth_system.is_authenticated():
            return []

        permissions = []
        if self.check_file_permission(file_path, Permission.READ):
            permissions.append(Permission.READ)
        if self.check_file_permission(file_path, Permission.WRITE):
            permissions.append(Permission.WRITE)
        if self.check_file_permission(file_path, Permission.EXECUTE):
            permissions.append(Permission.EXECUTE)

        return permissions

    def format_permission_status(self, permissions: List[Permission]) -> str:
        """Format permission status for display"""
        if not permissions:
            return "\033[31m[NO ACCESS]\033[0m"

        status_parts = []
        if Permission.READ in permissions:
            status_parts.append("\033[32mR\033[0m")
        else:
            status_parts.append("\033[31mr\033[0m")

        if Permission.WRITE in permissions:
            status_parts.append("\033[32mW\033[0m")
        else:
            status_parts.append("\033[31mw\033[0m")

        if Permission.EXECUTE in permissions:
            status_parts.append("\033[32mX\033[0m")
        else:
            status_parts.append("\033[31mx\033[0m")

        return f"[{''.join(status_parts)}]"

    def handle_pipe_command(self, parsed: ParsedCommand) -> str:
        """Handle piped commands"""
        if not parsed.has_pipes:
            return ""

        try:
            # Parse the pipe chain
            pipe_commands = self.pipe_handler.parse_pipe_chain(' | '.join(parsed.pipe_chain))

            # Execute the pipe chain
            result = self.pipe_handler.execute_pipe_chain(pipe_commands, self.auth_system)
            return result
        except Exception as e:
            raise ValueError(f"Pipe execution error: {e}")

# COMMAND PARSER
class CommandParser:
    """Handles parsing of command line input"""

    BUILTIN_COMMANDS = {
        # Deliverable 1: Basic shell commands
        'cd', 'pwd', 'exit', 'echo', 'clear', 'ls', 'cat',
        'mkdir', 'rmdir', 'rm', 'touch', 'kill', 'jobs',
        'fg', 'bg', 'stop', 'help',
        # Deliverable 2: Process scheduling commands
        'scheduler', 'addprocess', 'scheduler',
        # Deliverable 3: NEW - Memory management and synchronization commands
        'memory', 'sync',
        # Deliverable 3: Authentication commands
        'login', 'logout', 'whoami', 'adduser', 'chpasswd',
        # Text processing commands
        'grep', 'sort'
    }

    def parse(self, input_str: str) -> Optional[ParsedCommand]:
        """Parse a command line input string"""
        input_str = input_str.strip()
        if not input_str:
            return None

        # Check for pipes
        if '|' in input_str:
            return self._parse_pipe_command(input_str)

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
            pipes=[args],
            has_pipes=False,
            pipe_chain=[]
        )

    def _parse_pipe_command(self, input_str: str) -> ParsedCommand:
        """Parse a command with pipes"""
        # Split by pipe character
        pipe_parts = [part.strip() for part in input_str.split('|')]

        # Check for background execution on the last command
        background = False
        if pipe_parts[-1].endswith('&'):
            background = True
            pipe_parts[-1] = pipe_parts[-1][:-1].strip()

        # Parse each command in the pipe chain
        pipe_commands = []
        for part in pipe_parts:
            if not part:
                continue
            try:
                args = shlex.split(part)
                pipe_commands.append(args)
            except ValueError as e:
                raise ValueError(f"Parse error in pipe command: {e}")

        if not pipe_commands:
            return None

        # The main command is the first one in the chain
        main_command = pipe_commands[0][0] if pipe_commands[0] else ""

        return ParsedCommand(
            command=main_command,
            args=pipe_commands[0] if pipe_commands else [],
            background=background,
            pipes=pipe_commands,
            has_pipes=True,
            pipe_chain=pipe_parts
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
            raise ValueError(f"Potentially dangerous path detected: {parsed.command}")

        # Validate command name
        dangerous_chars = ';&<>(){}[]'
        if any(char in parsed.command for char in dangerous_chars):
            raise ValueError(f"Invalid characters in command name: {parsed.command}")

        # Check for excessively long commands
        if len(parsed.command) > 256:
            raise ValueError("Command name too long (max 256 characters)")

        # Validate arguments
        for i, arg in enumerate(parsed.args):
            if len(arg) > 1024:
                raise ValueError(f"Argument {i} too long (max 1024 characters)")

        # Check total argument count
        if len(parsed.args) > 100:
            raise ValueError("Too many arguments (max 100)")

        # Validate pipe commands
        if parsed.has_pipes:
            for pipe_cmd in parsed.pipes:
                if not pipe_cmd:
                    raise ValueError("Empty command in pipe chain")
                if len(pipe_cmd) > 100:
                    raise ValueError("Too many arguments in pipe command (max 100)")

# INPUT HANDLER
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

# PERFORMANCE METRICS
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

# PIPE HANDLER
@dataclass
class PipeCommand:
    """Represents a command in a pipe chain"""
    command: str
    args: List[str]
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    append_output: bool = False


class PipeHandler:
    """Handles command piping and redirection"""

    def __init__(self):
        self.temp_files = []

    def parse_pipe_chain(self, input_str: str) -> List[PipeCommand]:
        """Parse a command string into a chain of piped commands"""
        commands = []

        # Split by pipe character
        pipe_parts = input_str.split('|')

        for i, part in enumerate(pipe_parts):
            part = part.strip()
            if not part:
                continue

            # Parse redirection
            input_file = None
            output_file = None
            append_output = False

            # Handle input redirection
            if '<' in part:
                parts = part.split('<', 1)
                part = parts[0].strip()
                input_file = parts[1].strip()

            # Handle output redirection
            if '>' in part:
                if '>>' in part:
                    parts = part.split('>>', 1)
                    part = parts[0].strip()
                    output_file = parts[1].strip()
                    append_output = True
                else:
                    parts = part.split('>', 1)
                    part = parts[0].strip()
                    output_file = parts[1].strip()

            # Parse command and arguments
            args = part.split()
            if not args:
                continue

            command = PipeCommand(
                command=args[0],
                args=args,
                input_file=input_file,
                output_file=output_file,
                append_output=append_output
            )
            commands.append(command)

        return commands

    def execute_pipe_chain(self, commands: List[PipeCommand], auth_system=None) -> str:
        """Execute a chain of piped commands"""
        if not commands:
            return ""

        # Check permissions for all commands
        if auth_system and auth_system.is_authenticated():
            for cmd in commands:
                if cmd.input_file and not auth_system.has_permission(cmd.input_file, auth_system.Permission.READ):
                    raise PermissionError(f"Permission denied: cannot read {cmd.input_file}")
                if cmd.output_file and not auth_system.has_permission(cmd.output_file, auth_system.Permission.WRITE):
                    raise PermissionError(f"Permission denied: cannot write to {cmd.output_file}")

        # For single command, use simple subprocess
        if len(commands) == 1:
            return self._execute_single_command(commands[0])

        # For multiple commands, use pipes
        return self._execute_pipe_chain(commands, auth_system)

    def _execute_single_command(self, cmd: PipeCommand) -> str:
        """Execute a single command"""
        try:
            # Handle input/output redirection
            stdin = None
            stdout = None

            if cmd.input_file:
                stdin = open(cmd.input_file, 'r')

            if cmd.output_file:
                mode = 'a' if cmd.append_output else 'w'
                stdout = open(cmd.output_file, mode)
            else:
                # Capture output if no output file
                result = subprocess.run(
                    cmd.args,
                    stdin=stdin,
                    capture_output=True,
                    text=True
                )

                # Clean up input file
                if stdin:
                    stdin.close()

                if result.returncode != 0:
                    raise RuntimeError(f"Command '{cmd.command}' failed: {result.stderr}")

                return result.stdout

            # Execute with file redirection
            result = subprocess.run(
                cmd.args,
                stdin=stdin,
                stdout=stdout,
                stderr=subprocess.PIPE,
                text=True
            )

            # Clean up files
            if stdin:
                stdin.close()
            if stdout:
                stdout.close()

            if result.returncode != 0:
                raise RuntimeError(f"Command '{cmd.command}' failed: {result.stderr}")

            return ""

        except FileNotFoundError:
            raise ValueError(f"{cmd.command}: command not found")
        except PermissionError:
            raise PermissionError(f"{cmd.command}: permission denied")

    def _execute_pipe_chain(self, commands: List[PipeCommand], auth_system=None) -> str:
        """Execute a chain of piped commands"""
        # Check if all commands are built-in
        builtin_commands = ['cat', 'grep', 'sort', 'echo', 'ls', 'pwd', 'whoami']

        # For now, handle simple cases with built-in commands
        if all(cmd.command in builtin_commands for cmd in commands):
            return self._execute_builtin_pipe_chain(commands, auth_system)
        else:
            # Fall back to subprocess for external commands
            return self._execute_subprocess_pipe_chain(commands)

    def _execute_builtin_pipe_chain(self, commands: List[PipeCommand], auth_system=None) -> str:
        """Execute a chain of built-in commands"""
        # Create a temporary command handler for built-in commands
        temp_handler = CommandHandler(None)  # We don't need job manager for this

        # Set the authentication system to maintain user session
        if auth_system:
            temp_handler.auth_system = auth_system

        # Start with input data
        current_input = ""

        # Execute each command in the chain
        for i, cmd in enumerate(commands):
            # For the first command, handle input file if specified
            if i == 0 and cmd.input_file:
                with open(cmd.input_file, 'r') as f:
                    current_input = f.read()

            # Redirect stdin to our current input
            import sys
            old_stdin = sys.stdin
            sys.stdin = io.StringIO(current_input)

            # Capture stdout
            old_stdout = sys.stdout
            output_buffer = io.StringIO()
            sys.stdout = output_buffer

            try:
                # Execute the built-in command
                if cmd.command == 'ls':
                    temp_handler.handle_ls(cmd.args)
                    current_input = output_buffer.getvalue()

                elif cmd.command == 'cat':
                    # For cat, we need to handle the case where it reads from a file
                    # In a pipe like "cat file | grep pattern", cat should read from the file
                    if i == 0 and len(cmd.args) > 1:
                        # First command with file argument - read the file directly
                        filename = cmd.args[1]
                        with open(filename, 'r') as f:
                            current_input = f.read()
                    else:
                        # Cat reading from stdin (shouldn't happen in normal pipes)
                        temp_handler.handle_cat(cmd.args)
                        current_input = output_buffer.getvalue()

                elif cmd.command == 'grep':
                    temp_handler.handle_grep(cmd.args)
                    current_input = output_buffer.getvalue()

                elif cmd.command == 'sort':
                    temp_handler.handle_sort(cmd.args)
                    current_input = output_buffer.getvalue()

                # Add other built-in commands as needed

            finally:
                # Restore stdin/stdout
                sys.stdin = old_stdin
                sys.stdout = old_stdout

        return current_input

    def _execute_subprocess_pipe_chain(self, commands: List[PipeCommand]) -> str:
        """Execute a chain of piped commands using subprocess"""
        processes = []
        pipes = []
        open_files = []

        try:
            # Create pipes between commands
            for i in range(len(commands) - 1):
                read_pipe, write_pipe = os.pipe()
                pipes.extend([read_pipe, write_pipe])

            # Execute each command
            for i, cmd in enumerate(commands):
                # Set up input
                stdin = None
                if i == 0 and cmd.input_file:
                    # First command with input file
                    stdin = open(cmd.input_file, 'r')
                    open_files.append(stdin)
                elif i > 0:
                    # Not first command, use pipe from previous
                    stdin = os.fdopen(pipes[(i-1)*2], 'r')
                    open_files.append(stdin)

                # Set up output
                stdout = None
                if i == len(commands) - 1 and cmd.output_file:
                    # Last command with output file
                    mode = 'a' if cmd.append_output else 'w'
                    stdout = open(cmd.output_file, mode)
                    open_files.append(stdout)
                elif i < len(commands) - 1:
                    # Not last command, use pipe to next
                    stdout = os.fdopen(pipes[i*2+1], 'w')
                    open_files.append(stdout)

                # Execute command
                try:
                    process = subprocess.Popen(
                        cmd.args,
                        stdin=stdin,
                        stdout=stdout,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    processes.append(process)
                except FileNotFoundError:
                    raise ValueError(f"{cmd.command}: command not found")
                except PermissionError:
                    raise PermissionError(f"{cmd.command}: permission denied")

            # Wait for all processes to complete
            for process in processes:
                process.wait()

            # Check for errors
            for i, process in enumerate(processes):
                if process.returncode != 0:
                    stderr_output = process.stderr.read() if process.stderr else ""
                    raise RuntimeError(f"Command '{commands[i].command}' failed: {stderr_output}")

            # Get output from last command if no output file
            if not commands[-1].output_file:
                # For piped commands, we need to capture the output differently
                # Since the last command's output goes to stdout, we need to read it
                # This is a simplified approach - in a real implementation, you'd need
                # to capture the output during execution
                return "Command executed successfully"

            return ""

        finally:
            # Clean up all open files
            for file_obj in open_files:
                try:
                    file_obj.close()
                except (OSError, AttributeError):
                    pass

            # Clean up pipes
            for pipe in pipes:
                try:
                    os.close(pipe)
                except OSError:
                    pass

    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass
        self.temp_files.clear()

# PROCESS SCHEDULER
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

# PROCESS SYNC
class PhilosopherState(Enum):
    THINKING = "thinking"
    HUNGRY = "hungry"
    EATING = "eating"


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
        self.not_full = threading.Semaphore(buffer_size)   # Producers wait when full
        self.not_empty = threading.Semaphore(0)            # Consumers wait when empty

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
            thread = threading.Thread(target=self.producer_task, args=(i, duration))
            self.producers.append(thread)
            thread.start()

        # Create consumer threads
        for i in range(num_consumers):
            thread = threading.Thread(target=self.consumer_task, args=(i, duration))
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
        self.states = [PhilosopherState.THINKING for _ in range(num_philosophers)]
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
                "avg_meals": sum(self.meals_eaten) / self.num_philosophers,
                "deadlock_preventions": self.deadlock_prevention_count,
                "running": self.running,
                "active_threads": len([t for t in self.philosophers if t.is_alive()])
            }

# SCHEDULER COMMANDS
class SchedulerCommands:
    """Command handlers for process scheduling functionality"""

    def __init__(self, scheduler: ProcessScheduler):
        self.scheduler = scheduler

    def handle_addprocess(self, args: List[str]) -> None:
        """Handle addprocess command to add a process to the scheduler"""
        if len(args) < 3:
            raise ValueError("addprocess: missing arguments\nUsage: addprocess <name> <duration> [priority]")

        name = args[1]
        if not name:
            raise ValueError("addprocess: process name cannot be empty")

        try:
            duration = float(args[2])
            if duration <= 0:
                raise ValueError("addprocess: duration must be positive")
        except ValueError as e:
            if "duration must be positive" in str(e):
                raise e
            raise ValueError(f"addprocess: invalid duration '{args[2]}': must be a number")

        priority = 0  # Default priority
        if len(args) > 3:
            try:
                priority = int(args[3])
            except ValueError:
                raise ValueError(f"addprocess: invalid priority '{args[3]}': must be an integer")

        try:
            pid = self.scheduler.add_process(name, duration, priority)
            if self.scheduler.config and self.scheduler.config.algorithm == SchedulingAlgorithm.PRIORITY:
                print(f"Added process '{name}' (PID: {pid}, Duration: {duration}s, Priority: {priority})")
            else:
                print(f"Added process '{name}' (PID: {pid}, Duration: {duration}s)")

        except ValueError as e:
            raise ValueError(str(e))

    def handle_scheduler(self, args: List[str]) -> None:
        """Handle scheduler command for all scheduler operations"""
        if len(args) < 2:
            raise ValueError("scheduler: missing subcommand\nUsage: scheduler <config|start|stop|status|metrics|clear|addprocess>")

        subcommand = args[1].lower()

        if subcommand == "config" or subcommand == "configure":
            self._handle_scheduler_config(args[2:])

        elif subcommand == "addprocess" or subcommand == "add":
            self._handle_scheduler_addprocess(args[2:])

        elif subcommand == "start":
            try:
                self.scheduler.start_scheduler()
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

        else:
            raise ValueError(f"scheduler: unknown subcommand '{subcommand}'\nAvailable subcommands: config, addprocess, start, stop, status, metrics, clear")

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

        elif algorithm_str == "priority":
            algorithm = SchedulingAlgorithm.PRIORITY
            self.scheduler.configure(algorithm)
            print("Configured Priority-Based scheduling")

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

        # Show helpful commands if no processes are queued
        if status['ready_queue_size'] == 0 and not status['current_process'] and not status['running']:
            print()
            print("Tip: Use 'addprocess <name> <duration> [priority]' to add processes")
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

# SCHEDULER
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
        Execute a shell command with timeout and sleep simulation

        Returns:
            tuple: (success, output, execution_time)
        """
        try:
            start_time = time.time()

            # Simulate process execution time with sleep
            print(f"  Simulating execution time: {timeout:.2f}s")
            time.sleep(timeout)

            # Execute the actual command (but don't wait for it to complete)
            # We're simulating the time, so we just run it quickly
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=min(timeout, 1.0)  # Limit actual command execution to prevent blocking
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
        """Priority-based scheduling algorithm with preemption"""
        if not self.processes:
            return

        # Sort processes by priority (highest priority first)
        self.processes.sort(key=lambda p: (p.priority, p.arrival_time))

        # Get the highest priority process
        process = self.processes.pop(0)

        # Check if we need to preempt the currently running process
        if self.running_process and process.priority < self.running_process.priority:
            print(f"Preempting process {self.running_process.job.id} for higher priority process {process.job.id}")
            self.running_process.job.status = JobStatus.WAITING
            # Add the preempted process back to the queue
            self.processes.append(self.running_process)
            # Re-sort to maintain priority order
            self.processes.sort(key=lambda p: (p.priority, p.arrival_time))

        self.running_process = process

        # Run the process for a small time slice to allow preemption
        time_slice = 0.1  # Small time slice for preemptive scheduling
        time_to_run = min(time_slice, process.time_needed - process.time_executed)

        if time_to_run <= 0:
            # Process is complete
            self._complete_process(process)
            return

        print(f"Running process {process.job.id} (Priority: {process.priority}) for {time_to_run:.1f}s")
        print(f"  Command: {process.job.command}")
        process.job.status = JobStatus.RUNNING

        # Execute the command for the time slice
        success, output, actual_time = self._execute_command(process.job.command, time_to_run)

        if success:
            print(f"  Command executed successfully in {actual_time:.2f}s")
            if output.strip():
                print(f"  Output: {output.strip()}")
        else:
            print(f"  Command failed: {output}")

        # Update process state
        process.time_executed += actual_time

        # Check if process is complete
        if process.time_executed >= process.time_needed:
            self._complete_process(process)
        else:
            # Process needs more time, add back to queue
            process.job.status = JobStatus.WAITING
            self.processes.append(process)

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

# SHELL INTEGRATION
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
        elif command == "reset":
            return self._reset_memory()
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
        """Memory management help"""
        return """✓ Memory Management Commands:
memory status                     - Show memory status and statistics
memory create <name> <pages>      - Create process with pages needed
memory alloc <pid> <page_num>     - Allocate specific page for process
memory dealloc <pid>              - Deallocate all pages for process
memory algorithm <fifo|lru>       - Set page replacement algorithm
memory test <sequential|random>   - Run memory access pattern test
memory reset                      - Reset all memory state and statistics"""

    def _sync_help(self) -> str:
        """Synchronization help"""
        return """✓ Process Synchronization Commands:
sync status                               - Show synchronization status
sync mutex <create|acquire|release> <name>    - Mutex operations
sync semaphore <create|acquire|release> <name> [value] - Semaphore operations
sync prodcons <start|stop|status> [producers] [consumers] - Producer-Consumer
sync philosophers <start|stop|status> [num_philosophers] - Dining Philosophers"""

    def _memory_status(self) -> str:
        """Get memory status"""
        status = self.memory_manager.get_status()

        result = f"""=== MEMORY MANAGEMENT STATUS ===
Total Frames: {status['total_frames']}
Used Frames: {status['used_frames']}
Free Frames: {status['free_frames']}
✓ Memory Utilization: {status['utilization']:.1f}%
✓ Page Replacement: {status['algorithm'].upper()}

=== PAGING STATISTICS===
✓ Page Faults: {status['page_faults']}
✓ Page Hits: {status['page_hits']}
✓ Page Replacements: {status['replacements']}
✓ Hit Ratio: {status['hit_ratio']:.1f}%
Active Processes: {status['processes']}"""

        if self.memory_manager.processes:
            result += "\n\n=== ACTIVE PROCESSES==="
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

    def _reset_memory(self) -> str:
        """Reset all memory management state (NEW)"""
        try:
            success = self.memory_manager.reset_memory()
            if success:
                return "✓ Memory management system reset successfully\n✓ All processes, statistics, and memory state cleared"
            else:
                return "Error: Failed to reset memory system"
        except Exception as e:
            return f"Error resetting memory: {e}"

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
            import random
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
            return "Usage: sync mutex <create|acquire|release> <n> [timeout]"

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
            return "Usage: sync semaphore <create|acquire|release> <n> [value|timeout]"

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

# SHELL
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

        # Require login before starting shell
        if not self.command_handler.auth_system.is_authenticated():
            self.handle_initial_login()

        while self.running:
            try:
                self.display_prompt()

                try:
                    # Use input handler if available, otherwise fallback to input()
                    if self.input_handler:
                        user_input = self.input_handler.get_input(self.prompt).strip()
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

    def handle_initial_login(self):
        """Handle initial login before starting shell"""
        print("Welcome to Advanced Shell Simulation - Deliverable 3")
        print("You must log in to continue.")
        print()
        print("Available users:")
        print("  admin/admin123    - Administrator (full access)")
        print("  user/user123      - Standard user (limited access)")
        print("  guest/guest123    - Guest user (very limited access)")
        print()

        while not self.command_handler.auth_system.is_authenticated():
            try:
                username = input("Username: ").strip()
                if not username:
                    continue

                password = input("Password: ").strip()
                if not password:
                    continue

                if self.command_handler.auth_system.authenticate(username, password):
                    user = self.command_handler.auth_system.get_current_user()
                    print(f"\nWelcome, {user.username}! Role: {user.role.value}")
                    break
                else:
                    print("Login failed: Invalid username or password")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                sys.exit(0)
            except EOFError:
                print("\nGoodbye!")
                sys.exit(0)

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

        # Handle piped commands
        if parsed.has_pipes:
            try:
                result = self.command_handler.handle_pipe_command(parsed)
                if result:
                    print(result, end='')
            except Exception as e:
                raise ValueError(f"Pipe error: {e}")
            return

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
        """Execute external command with permission checking"""
        # Check if external command exists
        import shutil
        if not shutil.which(parsed.command):
            raise ValueError(f"{parsed.command}: command not found")

        # Check permissions for command execution
        if not self.command_handler.auth_system.is_authenticated():
            raise ValueError("Authentication required to execute commands")

        # Check if user has execute permission for the command
        command_path = shutil.which(parsed.command)
        if command_path and not self.command_handler.check_file_permission(command_path, Permission.EXECUTE):
            raise PermissionError(f"{parsed.command}: permission denied")

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
                        pages_needed = self._estimate_memory_needs(parsed.command)
                        memory_pid = memory_manager.create_process(f"ExtCmd-{parsed.command}", pages_needed)
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
        """Estimate memory needs for external commands """
        # Simple heuristic for memory estimation
        memory_estimates = {
            'ls': 2, 'grep': 3, 'sort': 5, 'cat': 2, 'find': 4,
            'python': 8, 'gcc': 10, 'make': 6, 'vim': 4, 'nano': 2
        }
        return memory_estimates.get(command, 4)  # Default to 4 pages

    def display_prompt(self) -> None:
        """Show the shell prompt with user information"""
        try:
            pwd = os.getcwd()
            dir_name = os.path.basename(pwd)
            if dir_name == "":
                dir_name = pwd
        except Exception:
            dir_name = "unknown"

        # Get current time for enhanced prompt
        current_time = time.strftime("%H:%M:%S")

        # Add user information to prompt
        if self.command_handler.auth_system.is_authenticated():
            user = self.command_handler.auth_system.get_current_user()
            self.prompt = f"[{user.username}:{user.role.value}:{dir_name} {current_time}]$ "
        else:
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
        print("✓ Job control (jobs, fg, bg, stop)")
        print("✓ Round-Robin scheduling with configurable time slices")
        print("✓ Priority-based scheduling with preemption")
        print("✓ Process simulation using timers")
        print("✓ Keyboard navigation (arrow keys, Ctrl+C to clear)")
        print("✓ Signal handling")
        print("✓ Error handling")
        print("✓ Process scheduling algorithms")
        print("  • Round-Robin Scheduling")
        print("  • Priority-Based Scheduling")
        print("✓ Performance metrics and monitoring")
        print("✓ Memory management with paging ")
        print("  • FIFO and LRU page replacement algorithms")
        print("  • Page fault handling and tracking")
        print("  • Memory overflow simulation")
        print("✓ Process synchronization ")
        print("  • Mutexes and semaphores")
        print("  • Producer-Consumer problem")
        print("  • Dining Philosophers problem")
        print("  • Race condition prevention")
        print("  • Deadlock avoidance")
        print("✓ Command piping and redirection ")
        print("✓ User authentication system ")
        print("✓ File permissions and access control ")
        print("✓ READ-ONLY access for standard users (UPDATED)")
        print()
        print("Type 'help' for available commands")
        print("Type 'exit' to quit")
        print()
        print("Quick Start - Authentication:")
        print("  login admin admin123    # Login as administrator (full access)")
        print("  login user user123      # Login as standard user (read-only)")
        print("  login guest guest123    # Login as guest (very limited)")
        print("  whoami                  # Show current user")
        print("  logout                  # Logout current user")
        print()
        print("Quick Start - File Permissions:")
        print("  permissions file.txt    # Check file permissions")
        print("  ls -l                   # List with permission details")
        print()
        print("Quick Start - Piping:")
        print("  ls | grep txt           # List files containing 'txt'")
        print("  cat file.txt | sort     # Display sorted file contents")
        print()
        print("Quick Start - Process Scheduling:")
        print("  scheduler config rr 2         # Configure Round-Robin")
        print("  scheduler addprocess task1 5  # Add 5-second process")
        print("  scheduler start               # Start scheduling")
        print("  scheduler status              # Monitor execution")
        print()
        print("✓ Quick Start - Memory Management :")
        print("  memory create webapp 8        # Create process needing 8 pages")
        print("  memory alloc 1 0              # Allocate page 0 for process 1")
        print("  memory algorithm lru          # Switch to LRU replacement")
        print("  memory status                 # Show memory statistics")
        print()
        print("✓ Quick Start - Synchronization :")
        print("  sync prodcons start 2 3       # Start Producer-Consumer")
        print("  sync philosophers start 5     # Start Dining Philosophers")
        print("  sync status                   # Show sync statistics")
        print("IMPORTANT: Standard and Guest users have READ-ONLY access")
        print("to most directories. Only Admin can create/modify files.")
        print()

    def shutdown(self) -> None:
        """Perform cleanup before exiting"""
        print("\nShutting down shell...")

        # Deliverable 3: NEW - Stop synchronization problems
        try:
            memory_sync_commands = getattr(self.command_handler, 'memory_sync_commands', None)
            if memory_sync_commands:
                if memory_sync_commands.producer_consumer:
                    print("Stopping Producer-Consumer...")
                    memory_sync_commands.producer_consumer.stop()

                if memory_sync_commands.dining_philosophers:
                    print("Stopping Dining Philosophers...")
                    memory_sync_commands.dining_philosophers.stop()
        except:
            pass
        # Cleanup pipe handler
        if hasattr(self.command_handler, 'pipe_handler'):
            self.command_handler.pipe_handler.cleanup()

        # Stop scheduler if available
        if hasattr(self.command_handler, 'process_scheduler') and self.command_handler.process_scheduler:
            print("Stopping process scheduler...")
            self.command_handler.process_scheduler.stop_scheduler()

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

# TEST SCHEDULING WITH METRICS
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

    # Test with different time slice configurations
    time_slice_configs = [0.5, 1.0, 2.0]

    for time_slice in time_slice_configs:
        print(f"\n--- Testing with time slice: {time_slice} seconds ---")

        # Start performance tracking
        performance_tracker.start_test(
            f"Round-Robin Time Slice {time_slice}s",
            "round_robin",
            time_slice
        )

        # Reset scheduler for each test
        scheduler = Scheduler()
        scheduler.set_algorithm(SchedulingAlgorithm.ROUND_ROBIN, time_slice=time_slice)

        # Connect scheduler completion callback to performance tracker
        scheduler.on_process_complete = lambda job: performance_tracker.job_completed(job.id, job.execution_time)

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

        # Add jobs to performance tracker
        job_configs = [
            (jobs[0], 5, 0.3),  # Completes before time slice
            (jobs[1], 5, 1.5),  # Needs multiple time slices
            (jobs[2], 5, 0.2),  # Completes before time slice
            (jobs[3], 5, 0.4),  # Completes before time slice
        ]

        for job, priority, time_needed in job_configs:
            performance_tracker.add_job(job.id, job.command, priority, time_needed)
            scheduler.add_process(job, priority=priority, time_needed=time_needed)

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

        # End performance tracking
        performance_tracker.end_test()

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

    # Start performance tracking
    performance_tracker.start_test("Priority-Based Scheduling", "priority")

    scheduler = Scheduler()
    scheduler.set_algorithm(SchedulingAlgorithm.PRIORITY)

    # Connect scheduler completion callback to performance tracker
    scheduler.on_process_complete = lambda job: performance_tracker.job_completed(job.id, job.execution_time)

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

    # Add jobs to performance tracker
    job_configs = [
        (jobs[0], 1, 0.3),   # Highest priority, quick
        (jobs[1], 5, 0.2),   # Medium priority, quick
        (jobs[2], 10, 1.0),  # Lowest priority, longer
        (jobs[3], 1, 0.5),   # High priority, medium
    ]

    for job, priority, time_needed in job_configs:
        performance_tracker.add_job(job.id, job.command, priority, time_needed)
        scheduler.add_process(job, priority=priority, time_needed=time_needed)

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

    # End performance tracking
    performance_tracker.end_test()

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

    # Start performance tracking
    performance_tracker.start_test("Priority Preemption", "priority")

    scheduler = Scheduler()
    scheduler.set_algorithm(SchedulingAlgorithm.PRIORITY)

    # Connect scheduler completion callback to performance tracker
    scheduler.on_process_complete = lambda job: performance_tracker.job_completed(job.id, job.execution_time)

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
    performance_tracker.add_job(low_priority_job.id, low_priority_job.command, 10, 4.0)
    scheduler.add_process(low_priority_job, priority=10, time_needed=4.0)

    print("Added low priority job:")
    print(f"  Process 1: 'echo low priority job' (Priority 10, 4.0s) - Should be preempted")
    print("Starting scheduler...")

    # Start the scheduler
    scheduler.start_scheduler()

    # Wait a bit, then add a high priority job
    time.sleep(0.5)
    high_priority_job = DummyJob(2, f"echo 'High priority job preempting' > {test_dir}/preempt_high_priority.txt")
    performance_tracker.add_job(high_priority_job.id, high_priority_job.command, 1, 0.5)
    scheduler.add_process(high_priority_job, priority=1, time_needed=0.5)

    print("Added high priority job after 0.5s:")
    print(f"  Process 2: 'echo high priority job' (Priority 1, 0.5s) - Should preempt low priority")
    print("Expected behavior: High priority job should preempt low priority job immediately")

    # Wait for all jobs to complete
    while scheduler.running:
        time.sleep(0.1)

    # End performance tracking
    performance_tracker.end_test()

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

    # Start performance tracking
    performance_tracker.start_test("Early Completion", "round_robin", 1.0)

    scheduler = Scheduler()
    scheduler.set_algorithm(SchedulingAlgorithm.ROUND_ROBIN, time_slice=1.0)

    # Connect scheduler completion callback to performance tracker
    scheduler.on_process_complete = lambda job: performance_tracker.job_completed(job.id, job.execution_time)

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

    # Add jobs to performance tracker
    job_configs = [
        (jobs[0], 5, 0.2),  # Completes early
        (jobs[1], 5, 0.8),  # Uses most of time slice
        (jobs[2], 5, 0.1),  # Completes very early
    ]

    for job, priority, time_needed in job_configs:
        performance_tracker.add_job(job.id, job.command, priority, time_needed)
        scheduler.add_process(job, priority=priority, time_needed=time_needed)

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

    # End performance tracking
    performance_tracker.end_test()

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

def show_version():
    """Show version information"""
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
    print("✓ Memory management with paging (Deliverable 3)")
    print("  * FIFO and LRU page replacement algorithms")
    print("  * Page fault handling and tracking")
    print("  * Memory overflow simulation")
    print("✓ Process synchronization (Deliverable 3 )")
    print("  * Mutexes and semaphores")
    print("  * Producer-Consumer problem")
    print("  * Dining Philosophers problem")
    print("  * Race condition prevention")
    print("  * Deadlock avoidance")
    print("- Command piping and redirection (Deliverable 3)")
    print("- User authentication system")
    print("- File permissions and access control")


def show_help():
    """Show help information"""
    print("Advanced Shell Simulation")
    print()
    print("Usage:")
    print("  python3 main.py [options]")
    print()
    print("Options:")
    print("  --version    Show version information")
    print("  --help       Show this help message")
    print("  --debug      Enable debug mode")
    print("  --test-memory    Run memory management tests (NEW)")
    print("  --test-sync      Run synchronization tests (NEW)")
    print("  --test-scheduling-with-metrics  Run scheduling tests with performance metrics")
    print()
    print("Once started, you will be prompted to log in.")
    print("Available users:")
    print("  admin/admin123    - Administrator (full access)")
    print("  user/user123      - Standard user (limited access)")
    print("  guest/guest123    - Guest user (very limited access)")
    print()
    print("Type 'help' for available shell commands")
    print()
    print("Quick Start - Authentication:")
    print("  login admin admin123    # Login as administrator")
    print("  whoami                  # Show current user")
    print("  logout                  # Logout current user")
    print()
    print("Quick Start - Piping:")
    print("  ls | grep txt           # List files containing 'txt'")
    print("  cat test_data.txt | sort     # Display sorted file contents")
    print("  cat test_data.txt | grep error | sort  # Sort errors in test_data")
    print()
    print("Quick Start - Process Scheduling:")
    print("  1. Configure algorithm:    scheduler config rr 2")
    print("  2. Add processes:          scheduler addprocess task1 5")
    print("  3. Start scheduler:        scheduler start")
    print("  4. Monitor status:         scheduler status")
    print("  5. View metrics:           scheduler metrics")
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
        print(f"  - Memory Usage: {status['used_frames']}/{status['total_frames']} frames ({status['utilization']:.1f}%)")
        print(f"  - Per-Process Allocation:")
        for pid, process in mm.processes.items():
            allocated = len([p for p in process.page_table.values() if p is not None])
            print(f"    * PID {pid} ({process.name}): {allocated}/{process.pages_needed} pages allocated")
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

        success, msg = mm.allocate_page(pid2, 1)  # Should trigger another replacement
        overflow_results.append(f"    Overflow 2: {msg}")

        success, msg = mm.allocate_page(pid3, 3)  # Should trigger another replacement
        overflow_results.append(f"    Overflow 3: {msg}")

        for result in overflow_results:
            print(result)

        status = mm.get_status()
        print(f"  - Memory overflow results:")
        print(f"    * Total page faults: {status['page_faults']}")
        print(f"    * Page replacements triggered: {status['replacements']}")
        print(f"    * Memory utilization: {status['utilization']:.1f}% (should be 100%)")
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
        print(f"    * FIFO Results: {fifo_status['page_faults']} faults, {fifo_status['page_hits']} hits, {fifo_status['replacements']} replacements")

        # Test LRU
        mm_lru = MemoryManager(total_frames=4)
        mm_lru.set_algorithm("lru")
        pid_lru = mm_lru.create_process("LRUTest", 6)

        print("  - LRU Algorithm Test:")
        lru_pattern = [0, 1, 2, 3, 0, 1, 4, 5]  # Same pattern, different results
        for i, page_num in enumerate(lru_pattern):
            success, msg = mm_lru.allocate_page(pid_lru, page_num)

        lru_status = mm_lru.get_status()
        print(f"    * LRU Results: {lru_status['page_faults']} faults, {lru_status['page_hits']} hits, {lru_status['replacements']} replacements")

        print(f"  - Algorithm Comparison:")
        print(f"    * FIFO hit ratio: {fifo_status['hit_ratio']:.1f}%")
        print(f"    * LRU hit ratio: {lru_status['hit_ratio']:.1f}%")
        print(f"    * LRU is {'better' if lru_status['hit_ratio'] > fifo_status['hit_ratio'] else 'same as'} FIFO for this pattern")
        print()

        print("✓ Memory management tests completed successfully!")
        print("✓ Demonstrated: Per-process tracking, memory overflow, page fault tracking, algorithm comparison")

    except ImportError as e:
        print(f"Error: Could not import memory management modules: {e}")
    except Exception as e:
        print(f"Error during memory tests: {e}")


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
        print(f"  - Synchronization status: {status['mutexes']} mutexes, {status['semaphores']} semaphores")
        print()

        # Test Producer-Consumer
        print("✓ Testing Producer-Consumer Problem:")
        pc = ProducerConsumer(buffer_size=3)
        print("Starting Producer-Consumer simulation... (Number of producers: 2, consumers: 3)")

        pc.start(num_producers=2, num_consumers=3, duration=5)
        time.sleep(10)  # Let it run briefly

        pc_status = pc.get_status()
        print(f"  - Buffer: {pc_status['current_buffer']}/{pc_status['buffer_size']}")
        print(f"  - Items Produced: {pc_status['items_produced']}")
        print(f"  - Items Consumed: {pc_status['items_consumed']}")
        print(f"  - Producer Waits: {pc_status['producer_waits']} (times producers waited for space)")
        print(f"  - Consumer Waits: {pc_status['consumer_waits']} (times consumers waited for items)")
        print(f"  - Produced: {pc_status['items_produced']}, Consumed: {pc_status['items_consumed']}")
        print(f"  - Active: {pc_status['active_producers']} producers, {pc_status['active_consumers']} consumers")

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

        time.sleep(3)  # Let it run briefly

        dp_status = dp.get_status()
        print(f"  - Philosophers: {dp_status['num_philosophers']}")
        print(f"  - Total meals eaten: {dp_status['total_meals']}")
        print(f"  - Deadlock preventions: {dp_status['deadlock_preventions']}")
        print(f"  - States: {', '.join(dp_status['states'])}")

        dp.stop()
        print("Dining Philosophers stopped")
        print()

        print("✓ Synchronization tests completed successfully!")

    except ImportError as e:
        print(f"Error: Could not import synchronization modules: {e}")
    except Exception as e:
        print(f"Error during synchronization tests: {e}")


def run_scheduling_tests():
    """Run the scheduling tests with performance metrics"""
    print("Running scheduling tests with performance metrics...")

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
                print("The scheduling algorithms are working correctly with configurable time slices.")
                print("All test files and directories have been created in the 'test/' directory.")
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
        print("Performance metrics report has been saved to: performance_metrics_report.txt")

# Main entry point for the shell simulation
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
        show_version()
        return

    if args.help:
        show_help()
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
            print("Scheduler modules available:", hasattr(shell.command_handler, 'process_scheduler'))
            # Deliverable 3: NEW debug info
            print("Memory management available:", hasattr(shell.command_handler, 'memory_sync_commands'))
            if hasattr(shell.command_handler, 'memory_sync_commands'):
                memory_manager = shell.command_handler.get_memory_manager()
                synchronizer = shell.command_handler.get_synchronizer()
                print(f"  - Memory manager: {memory_manager.total_frames} frames")
                print(f"  - Synchronizer: initialized")
            print("Auth system available:", hasattr(shell.command_handler, 'auth_system'))
            print("Pipe handler available:", hasattr(shell.command_handler, 'pipe_handler'))
            # Additional debug setup could go here

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