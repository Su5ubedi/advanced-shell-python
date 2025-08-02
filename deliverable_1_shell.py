#!/usr/bin/env python3
"""
Advanced Shell Simulation - Deliverable 1
A custom shell implementation in Python that simulates Unix-like Operating System environment
with process management, job control, and built-in commands.

HOW TO RUN:
Prerequisites:
- Python 3.8 or higher

Running the Shell:
# Start the shell
python3 deliverable_1_shell.py

# Command-line options
python3 deliverable_1_shell.py --help     # Show help
python3 deliverable_1_shell.py --version  # Show version info
python3 deliverable_1_shell.py --debug    # Enable debug mode

Example Usage:
# Navigation and file operations
ls -la
pwd
cd /tmp
mkdir -p test/nested/path
touch test_file.txt
cat test_file.txt
rm -rf test

# Process and job management
kill 1234          # Kill process by PID
sleep 60 &         # Start background job
stop 1             # Stop job 1 (testing feature)
bg 1               # Resume job 1 in background
fg 1               # Bring job 1 to foreground

# Shell operations
help
exit

GitHub Repository: https://github.com/Su5ubedi/advanced-shell-python
"""

import os
import sys
import subprocess
import shlex
import signal
import time
import argparse
import stat
import shutil
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# ============================================================================
# Data Types and Enums
# ============================================================================

class JobStatus(Enum):
    """Job status enumeration"""
    RUNNING = "Running"
    STOPPED = "Stopped"
    DONE = "Done"


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


# ============================================================================
# Job Manager - Job Control Operations
# ============================================================================

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

            print(f"[{job.id}] {job.status.value} {job.command} (PID: {job.pid}, Duration: {int(duration)}s)")

    def bring_to_foreground(self, job_id: int) -> bool:
        """Bring a background job to the foreground"""
        job = self.get_job(job_id)
        if not job:
            print(f"fg: job {job_id} not found")
            return False

        # Check if job is still alive before proceeding
        if not job.is_alive():
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

        # Check if job is still alive before proceeding
        if not job.is_alive():
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
            job.is_alive()  # Update status
            if job.status == JobStatus.DONE:
                completed_jobs.append(job_id)
                print(f"[{job_id}]+ Done\t\t{job.command}")

        for job_id in completed_jobs:
            if job_id in self.jobs:
                del self.jobs[job_id]


# ============================================================================
# Command Parser - Command Parsing and Validation
# ============================================================================

class CommandParser:
    """Handles parsing of command line input"""

    BUILTIN_COMMANDS = {
        'cd', 'pwd', 'exit', 'echo', 'clear', 'ls', 'cat',
        'mkdir', 'rmdir', 'rm', 'touch', 'kill', 'jobs',
        'fg', 'bg', 'help', 'stop'
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
            raise ValueError(f"Potentially dangerous path detected: {parsed.command}")

        # Validate command name
        dangerous_chars = '|;&<>(){}[]'
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


# ============================================================================
# Command Handler - Built-in Command Implementations
# ============================================================================

class CommandHandler:
    """Handles built-in shell commands"""

    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager

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
            'help': self.handle_help,
            'stop': self.handle_stop
        }

        handler = command_map.get(parsed.command)
        if handler:
            handler(parsed.args)
        else:
            raise ValueError(f"Unknown built-in command: {parsed.command}")

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
        # Clean shutdown - kill any remaining jobs
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
                raise ValueError(f"rmdir: {dirname}: no such file or directory")
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
                if file_path.exists():
                    # Update timestamp
                    file_path.touch()
                else:
                    # Create the file
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

    def handle_help(self, args: List[str]) -> None:
        """Help command"""
        print("Advanced Shell - Available Commands:")
        print()
        print("Built-in Commands:")
        print("  cd [directory]     - Change directory (supports ~, -, and relative paths)")
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
        print("Usage:")
        print("  command &         - Run command in background (for external commands)")
        print("  Ctrl+C            - Interrupt current operation")
        print()
        print("Note: This shell supports both built-in and external commands.")
        print()
        print("Examples:")
        print("  ls -la")
        print("  mkdir -p path/to/dir")
        print("  rm -rf unwanted_dir")
        print("  cat file1.txt file2.txt")
        print("  echo \"Hello\\nWorld\"")
        print("  sleep 30 &        # External command in background")
        print("  stop 1            # Stop job 1")
        print("  bg 1              # Resume job 1 in background")
        print("  fg 1              # Bring job 1 to foreground")
        print()
        print("Advanced Features (Future Deliverables):")
        print("  - Process scheduling algorithms")
        print("  - Memory management simulation")
        print("  - Process synchronization")
        print("  - Command piping")
        print("  - User authentication and file permissions")
        print()


# ============================================================================
# Main Shell Class
# ============================================================================

class Shell:
    """Main shell implementation"""

    def __init__(self):
        self.job_manager = JobManager()
        self.command_handler = CommandHandler(self.job_manager)
        self.parser = CommandParser()
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
            # Determine preexec_fn for background processes
            preexec_fn = None
            creation_flags = 0

            if parsed.background:
                try:
                    # Try Unix approach first
                    preexec_fn = os.setsid
                except AttributeError:
                    # Windows doesn't have os.setsid, use creation flags instead
                    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

            if parsed.background:
                # Background execution
                if creation_flags:
                    # Windows
                    process = subprocess.Popen(
                        parsed.args,
                        cwd=os.getcwd(),
                        creationflags=creation_flags
                    )
                else:
                    # Unix
                    process = subprocess.Popen(
                        parsed.args,
                        cwd=os.getcwd(),
                        preexec_fn=preexec_fn
                    )

                job = self.job_manager.add_job(
                    command=' '.join(parsed.args),
                    args=parsed.args,
                    process=process,
                    background=True
                )

                print(f"[{job.id}] {process.pid}")

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
        prompt = f"[shell:{dir_name} {current_time}]$ "
        print(prompt, end='', flush=True)

    def print_welcome(self) -> None:
        """Print the welcome message"""
        print("==========================================")
        print("  Advanced Shell Simulation - Deliverable 1")
        print("==========================================")
        print()
        print("Features implemented:")
        print("✓ Built-in commands (cd, pwd, ls, cat, etc.)")
        print("✓ Process management (foreground/background)")
        print("✓ Job control (jobs, fg, bg)")
        print("✓ Signal handling")
        print("✓ Error handling")
        print()
        print("Type 'help' for available commands")
        print("Type 'exit' to quit")
        print()

    def shutdown(self) -> None:
        """Perform cleanup before exiting"""
        print("\nShutting down shell...")

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


# ============================================================================
# Main Entry Point
# ============================================================================

def show_version():
    """Show version information"""
    print("Advanced Shell Simulation")
    print("Version: 1.0.0 (Deliverable 1)")
    print("Build: Development")
    print()
    print("Features:")
    print("- Basic shell functionality")
    print("- Built-in commands")
    print("- Process management")
    print("- Job control")


def show_help():
    """Show help information"""
    print("Advanced Shell Simulation")
    print()
    print("Usage:")
    print("  python3 deliverable_1_shell.py [options]")
    print()
    print("Options:")
    print("  --version    Show version information")
    print("  --help       Show this help message")
    print("  --debug      Enable debug mode")
    print()
    print("Once started, type 'help' for available shell commands")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced Shell Simulation - Deliverable 1",
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

    args = parser.parse_args()

    if args.version:
        show_version()
        return

    if args.help:
        show_help()
        return

    # Create and start the shell
    try:
        shell = Shell()

        if args.debug:
            print("Debug mode enabled")
            # Additional debug setup could go here

        # Run the shell
        shell.run()

    except KeyboardInterrupt:
        print("\nShell interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()