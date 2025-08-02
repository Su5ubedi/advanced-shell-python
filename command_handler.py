
#!/usr/bin/env python3
"""
command_handler.py - Built-in command implementations for Advanced Shell Simulation
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


class CommandHandler:
    """Handles built-in shell commands"""

    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        # Deliverable 2: Add process scheduler
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
            'stop': self.handle_stop,  # Temporary for testing bg command
            'help': self.handle_help,
            # Deliverable 2: Scheduling commands
            'schedule': self.scheduler_commands.handle_schedule,
            'addprocess': self.scheduler_commands.handle_addprocess,
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
                    import shutil
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
        print("Process Scheduling (Deliverable 2):")
        print("  schedule rr [quantum]     - Configure Round-Robin scheduling")
        print("  schedule priority         - Configure Priority-Based scheduling")
        print("  addprocess <n> <duration> [priority] - Add process to scheduler")
        print("  scheduler start           - Start the scheduler")
        print("  scheduler stop            - Stop and clear scheduler")
        print("  scheduler status          - Show scheduler status")
        print("  scheduler metrics         - Show performance metrics")
        print("  scheduler clear           - Clear scheduler state and metrics")
        print()
        print("Usage:")
        print("  command &         - Run command in background")
        print("  Ctrl+C            - Interrupt current foreground process")
        print("  Arrow Keys        - Navigate cursor left/right")
        print("  Ctrl+C (in input) - Clear current command line")
        print("  Backspace         - Delete character to the left")
        print()
        print("Examples:")
        print("  ls -la")
        print("  mkdir -p path/to/dir")
        print("  rm -rf unwanted_dir")
        print("  sleep 10 &")
        print("  scheduler config rr 2          # Round-Robin with 2s quantum")
        print("  scheduler addprocess task1 5 3  # Add process with priority 3")
        print("  scheduler start                 # Start scheduling")
        print("  scheduler stop                  # Stop scheduling")
        print("  scheduler clear                 # Clear metrics and queues")
        print("  jobs")
        print("  fg 1")
        print("  cat file1.txt file2.txt")
        print("  echo \"Hello\\nWorld\"")
        print("  schedule task1 3 10")
        print("  scheduler round_robin 2.5")
        print("  addjob task2 1 5.0 true")
        print()
        print("Scheduling Algorithms:")
        print("  Round-Robin: Each process gets a time slice, then moves to end of queue")
        print("  Priority: Highest priority process runs first (1=highest, 10=lowest)")
        print()
        print("Scheduler Workflow:")
        print("  1. Configure:     scheduler config rr 2")
        print("  2. Add processes: scheduler addprocess task1 5")
        print("  3. Start:         scheduler start")
        print("  4. Monitor:       scheduler status")
        print("  5. View metrics:  scheduler metrics")
        print("  6. Clear state:   scheduler clear")
        print()
        print("Future Deliverables:")
        print("  - Memory management simulation")
        print("  - Process synchronization")
        print("  - Command piping")
        print("  - User authentication and file permissions")
        print()

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