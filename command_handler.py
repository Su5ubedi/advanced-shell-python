#!/usr/bin/env python3
"""
command_handler.py - Built-in command implementations for Advanced Shell Simulation
Updated for Deliverable 3: Memory Management and Process Synchronization
"""

import os
import sys
import signal
import subprocess
import stat
import time
from pathlib import Path
from typing import List

from shell_types import ParsedCommand, JobStatus
from job_manager import JobManager
from process_scheduler import ProcessScheduler
from scheduler_commands import SchedulerCommands
# Deliverable 3: NEW imports
from shell_integration import MemorySyncCommands
from auth_system import AuthenticationSystem, UserRole, Permission
from pipe_handler import PipeHandler


class CommandHandler:
    """Handles built-in shell commands"""

    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        # Deliverable 2: Add process scheduler
        self.process_scheduler = ProcessScheduler()
        self.scheduler_commands = SchedulerCommands(self.process_scheduler)
        # Deliverable 3: NEW - Add memory management and synchronization
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
            'schedule': self.scheduler_commands.handle_schedule,
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
        print("✓ Deliverable 3 Examples (NEW):")
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
        print("✓ Memory Management (NEW):")
        print("  FIFO: First-In-First-Out page replacement")
        print("  LRU: Least Recently Used page replacement")
        print("  Page Faults: When requested page not in memory")
        print()
        print("✓ Synchronization Problems (NEW):")
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

    def handle_schedule(self, args: List[str]) -> None:
        """Schedule command - add a job to the scheduler"""
        if len(args) < 2:
            raise ValueError("schedule: missing command\nUsage: schedule <command> [priority] [time_needed]")

        command = args[1]
        priority = 5  # Default priority
        time_needed = 5.0  # Default time needed

        # Parse optional arguments
        if len(args) >= 3:
            try:
                priority = int(args[2])
                if priority < 1 or priority > 10:
                    raise ValueError("Priority must be between 1 (highest) and 10 (lowest)")
            except ValueError:
                raise ValueError(f"Invalid priority: {args[2]}")

        if len(args) >= 4:
            try:
                time_needed = float(args[3])
                if time_needed <= 0:
                    raise ValueError("Time needed must be positive")
            except ValueError:
                raise ValueError(f"Invalid time: {args[3]}")

        # Create a scheduled job
        job = self.job_manager.add_scheduled_job(
            command=command,
            args=[command],  # Simple command without arguments for now
            priority=priority,
            time_needed=time_needed
        )

        print(f"Scheduled job [{job.id}]: {command} (Priority: {priority}, Time: {time_needed}s)")

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

    def handle_addjob(self, args: List[str]) -> None:
        """Addjob command - add a job with specific parameters"""
        if len(args) < 2:
            raise ValueError("addjob: missing command\n"
                           "Usage: addjob <command> [priority] [time_needed] [background]")

        command = args[1]
        priority = 5
        time_needed = 5.0
        background = True

        # Parse optional arguments
        if len(args) >= 3:
            try:
                priority = int(args[2])
                if priority < 1 or priority > 10:
                    raise ValueError("Priority must be between 1 (highest) and 10 (lowest)")
            except ValueError:
                raise ValueError(f"Invalid priority: {args[2]}")

        if len(args) >= 4:
            try:
                time_needed = float(args[3])
                if time_needed <= 0:
                    raise ValueError("Time needed must be positive")
            except ValueError:
                raise ValueError(f"Invalid time: {args[3]}")

        if len(args) >= 5:
            bg_str = args[4].lower()
            if bg_str in ['true', '1', 'yes']:
                background = True
            elif bg_str in ['false', '0', 'no']:
                background = False
            else:
                raise ValueError(f"Invalid background value: {args[4]}")

        # Create a scheduled job
        job = self.job_manager.add_scheduled_job(
            command=command,
            args=[command],
            priority=priority,
            time_needed=time_needed,
            background=background
        )

        print(f"Added job [{job.id}]: {command} (Priority: {priority}, Time: {time_needed}s, Background: {background})")

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
